from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import torch

from pathlib import Path

import trimesh
from doubletake.options import OptionsHandler
from doubletake.tools.tsdf import TSDF
from doubletake.utils.dataset_utils import get_dataset
from doubletake.utils.generic_utils import readlines
import os

import numpy as np
import open3d as o3d
import torch
import tqdm
from PIL import Image
from doubletake.utils.geometry_utils import BackprojectDepth
from doubletake.utils.rendering_utils import PyTorch3DMeshDepthRenderer


from doubletake.utils.visualization_utils import colormap_image, save_viz_video_frames


class SimpleScanNetDataset(torch.utils.data.Dataset):
    """Simple Dataset for loading ScanNet frames."""

    def __init__(self, scan_name: str, scan_data_root: Path, tuple_filepath: Path):
        self.scan_name = scan_name
        self.scan_data_root = scan_data_root

        metadata_filename = self.scan_data_root / self.scan_name / f"{self.scan_name}.txt"

        # load in basic intrinsics for the full size depth map.
        lines_str = readlines(metadata_filename)
        lines = [line.split(" = ") for line in lines_str]
        self.scan_metadata = {key: val for key, val in lines}
        self._get_available_frames(tuple_filepath=tuple_filepath)

    def _get_available_frames(self, tuple_filepath: str):
        # get a list of available frames
        self.frame_tuples = readlines(tuple_filepath)
        self.available_frames = [
            int(frame_tuple.split(" ")[1])
            for frame_tuple in self.frame_tuples
            if frame_tuple.split(" ")[0] == self.scan_name
        ]

    def load_pose(self, frame_ind) -> dict[str, torch.Tensor]:
        """loads pose for a frame from the scan's directory"""
        pose_path = (
            self.scan_data_root / self.scan_name / "sensor_data" / f"frame-{frame_ind:06d}.pose.txt"
        )
        world_T_cam_44 = torch.tensor(np.genfromtxt(pose_path).astype(np.float32))
        cam_T_world_44 = torch.linalg.inv(world_T_cam_44)

        pose_dict = {}
        pose_dict["world_T_cam_b44"] = world_T_cam_44
        pose_dict["cam_T_world_b44"] = cam_T_world_44

        return pose_dict

    def load_intrinsics(self) -> dict[str, torch.Tensor]:
        """Loads normalized intrinsics. Align corners false!"""
        intrinsics_filepath = (
            self.scan_data_root / self.scan_name / "intrinsic" / "intrinsic_depth.txt"
        )
        K_44 = torch.tensor(np.genfromtxt(intrinsics_filepath).astype(np.float32))

        K_44[0] /= int(self.scan_metadata["depthWidth"])
        K_44[1] /= int(self.scan_metadata["depthHeight"])

        invK_44 = torch.linalg.inv(K_44)

        intrinsics = {}
        intrinsics["K_b44"] = K_44
        intrinsics["invK_b44"] = invK_44

        return intrinsics

    def load_mesh(self) -> o3d.geometry.TriangleMesh:
        """Loads the mesh for the scan."""
        meshes_paths = self.scan_data_root / self.scan_name / f"{self.scan_name}_vh_clean_2.ply"
        mesh = o3d.io.read_triangle_mesh(str(meshes_paths))
        return mesh

    def __len__(self):
        """Returns the number of frames in the scan."""
        return len(self.available_frames)

    def __getitem__(self, idx):
        """Loads a rendered depth map and the corresponding pose and intrinsics."""
        frame_ind = self.available_frames[idx]
        pose_dict = self.load_pose(frame_ind)
        intrinsics = self.load_intrinsics()

        item_dict = {}
        item_dict.update(pose_dict)
        item_dict.update(intrinsics)
        item_dict["frame_id_str"] = str(frame_ind)

        return item_dict


def render_scene_meshes(
    scan_id: str,
    dataset_root: Path,
    tuple_filepath: Path,
    render_output_path: Path,
    mesh_path: Path,
    data_to_render: str,
    tsdf_path: Path = None,
    batch_size: int = 4,
):
    """Renders a scene's mesh with a depth renderer.

    Args:
        scan_id (str): The scan id.
        dataset_root (Path): The path to the dataset folder with scans data.
        tuple_filepath (Path): The path to the tuple file.
        render_output_path (Path): The path to save the rendered depth maps.
        mesh_path (Path): The path to the mesh ply file.

    """
    render_output_path.mkdir(exist_ok=True, parents=True)

    dataset = SimpleScanNetDataset(
        scan_name=scan_id,
        scan_data_root=dataset_root,
        tuple_filepath=tuple_filepath,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

    height = 192
    width = 256

    mesh_trimesh = trimesh.exchange.load.load(mesh_path)
    if data_to_render == "both":
        tsdf = TSDF.from_file(tsdf_path)
        backprojector = BackprojectDepth(height=height, width=width).cuda()

    mesh = Meshes(
        verts=[torch.tensor(mesh_trimesh.vertices).float()],
        faces=[torch.tensor(mesh_trimesh.faces).float()],
        textures=TexturesVertex(
            torch.tensor(mesh_trimesh.visual.vertex_colors).unsqueeze(0).float() / 255.0
        ),
    ).cuda()

    mesh_renderer = PyTorch3DMeshDepthRenderer(height=height, width=width)

    image_list = []
    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            depth_b1hw = mesh_renderer.render(mesh, batch["cam_T_world_b44"], batch["K_b44"])

        for elem_ind, depth_1hw in enumerate(depth_b1hw):
            # save the depth map
            depth_path = (
                render_output_path / f"rendered_depth_{batch['frame_id_str'][elem_ind]}.png"
            )
            depth_1hw[depth_1hw == -1] = 0
            mask_1hw = depth_1hw != 0

            numpy_depth = (depth_1hw.cpu().numpy().squeeze() * 256).astype("uint16")
            Image.fromarray(numpy_depth).save(depth_path)

            sampled_weights_1hw = None
            if data_to_render == "both":
                K_b44 = batch["K_b44"][elem_ind][None].cuda()
                K_b44[:, 0] *= width
                K_b44[:, 1] *= height
                invK_b44 = torch.linalg.inv(K_b44)
                cam_points_b4N = backprojector(depth_1hw[None], invK_b44)
                world_points_b4N = (
                    batch["world_T_cam_b44"][elem_ind][None].cuda() @ cam_points_b4N.cuda()
                )
                sampled_weights_N = tsdf.sample_tsdf(
                    world_points_b4N[:, :3, :].squeeze(0).transpose(0, 1), what_to_sample="weights"
                )
                sampled_weights_1hw = sampled_weights_N.view(1, 192, 256)
                sampled_weights_1hw[~mask_1hw] = 0.0

                weights_path = (
                    render_output_path / f"sampled_weights_{batch['frame_id_str'][elem_ind]}.png"
                )

                numpy_weights = (sampled_weights_1hw.cpu().numpy().squeeze() * 256).astype("uint16")
                Image.fromarray(numpy_weights).save(weights_path)

            # for debug. make and dump a video.
            colormapped_depth = colormap_image(depth_1hw, mask_1hw.float(), vmin=0.0, vmax=4)
            if sampled_weights_1hw is not None:
                colormapped_weights = colormap_image(
                    sampled_weights_1hw,
                    vmin=0,
                    vmax=1,
                    colormap="magma",
                    flip="False",
                )
                colormapped_depth = torch.cat(
                    [colormapped_depth.cpu(), colormapped_weights.cpu()], dim=2
                )

            numpy_depth = np.uint8(colormapped_depth.permute(1, 2, 0).cpu().detach().numpy() * 255)
            image_list.append(numpy_depth)

    save_viz_video_frames(
        image_list,
        str(render_output_path / f"{scan_id}.mp4"),
        fps=10,
    )


def render_scenes(
    dataset_root: Path,
    scan_list: list[str],
    tuple_filepath: Path,
    output_root: Path,
    meshes_paths: Path,
    batch_size: int,
    data_to_render: str,
):
    for scan_id in tqdm.tqdm(scan_list):
        mesh_path = meshes_paths / f"{scan_id}.ply"
        tsdf_path = meshes_paths / f"{scan_id}_tsdf.npz"
        render_output_path = output_root / scan_id
        render_scene_meshes(
            scan_id=scan_id,
            dataset_root=dataset_root,
            tuple_filepath=tuple_filepath,
            render_output_path=render_output_path,
            mesh_path=mesh_path,
            tsdf_path=tsdf_path,
            batch_size=batch_size,
            data_to_render=data_to_render,
        )


if __name__ == "__main__":
    # load options file
    option_handler = OptionsHandler()

    option_handler.parser.add_argument(
        "--meshes_paths",
        type=Path,
        required=True,
        help="Path to the ScanNet dataset.",
    )
    option_handler.parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Path to the ScanNet dataset.",
    )
    option_handler.parser.add_argument(
        "--data_to_render",
        type=str,
        required=True,
        help="Data to render and save. Can be 'depth' or 'both' for mesh depth renders and sampled TSDF weights.",
    )

    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options

    # get the dataset
    dataset_class, scan_names = get_dataset(
        opts.dataset,
        opts.dataset_scan_split_file,
        opts.single_debug_scan_id,
    )

    dataset_root = Path(opts.dataset_path) / dataset_class.get_sub_folder_dir(opts.split)

    render_scenes(
        dataset_root=dataset_root,
        scan_list=scan_names,
        tuple_filepath=Path(opts.tuple_info_file_location)
        / f"{opts.split}{opts.mv_tuple_file_suffix}",
        output_root=opts.output_root,
        meshes_paths=opts.meshes_paths,
        batch_size=opts.batch_size,
        data_to_render=opts.data_to_render,
    )
