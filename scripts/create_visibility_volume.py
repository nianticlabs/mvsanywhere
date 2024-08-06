import os
from pathlib import Path

import click
import numpy as np
import open3d as o3d
import torch
import torchvision
import tqdm
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from doubletake.utils.generic_utils import readlines, to_gpu
from doubletake.utils.volume_utils import SimpleVolume, VisibilityAggregator
from pytorch3d.utils import cameras_from_opencv_projection

import trimesh


"""
This script creates visibility volumes for ScanNet scenes. It does this by 
loading rednered depth maps from GT meshes, backprojecting those meshes into 
a volume, and marking any voxels up to that depth (plus some buffer) as visible.

The script doesn't render the GT meshes, it assumes that the rendered depth maps
are already available. See `load_rendered_depth`.
"""


class SimpleScanNetDataset(torch.utils.data.Dataset):
    """Simple Dataset for loading ScanNet frames and rendered mesh depths."""

    def __init__(
        self, scan_name: str, scan_data_root: Path, rendered_depths_root: Path, frame_interval: int
    ):
        self.scan_name = scan_name
        self.scan_data_root = scan_data_root
        self.rendered_depths_root = rendered_depths_root

        metadata_filename = self.scan_data_root / self.scan_name / f"{self.scan_name}.txt"

        # load in basic intrinsics for the full size depth map.
        lines_str = readlines(metadata_filename)
        lines = [line.split(" = ") for line in lines_str]
        self.scan_metadata = {key: val for key, val in lines}
        self._get_available_frames(frame_interval=frame_interval)

    def _get_available_frames(self, frame_interval: int):
        # get image files under each scan in /mnt/scannet/scans_test/scene0707_00/sensor_data
        self.available_frames = []
        frames = os.listdir(self.scan_data_root / self.scan_name / "sensor_data")
        frames.sort()
        for frame in frames:
            if frame.endswith(".color.jpg"):
                self.available_frames.append(int(frame.split(".")[0].split("-")[1]))

        # skip frames
        self.available_frames = self.available_frames[::frame_interval]

    def load_rendered_depth(self, frame_ind: int) -> torch.Tensor:
        """loads a rendered depth map from the rendered depths root.
        Assumes the invalid value is 65280.
        """
        render_path = (
            self.rendered_depths_root / f"{self.scan_name}/frames/{frame_ind:06d}_depth.png"
        )
        depth_1hw = torchvision.transforms.functional.to_tensor(Image.open(render_path))
        depth_1hw[depth_1hw == 65280] = -1
        depth_1hw = depth_1hw.float()
        depth_1hw /= 256
        return depth_1hw

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
        mesh_path = self.scan_data_root / self.scan_name / f"{self.scan_name}_vh_clean_2.ply"
        # mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = trimesh.exchange.load.load(str(mesh_path))
        return mesh

    def __len__(self):
        """Returns the number of frames in the scan."""
        return len(self.available_frames)

    def __getitem__(self, idx):
        """Loads a rendered depth map and the corresponding pose and intrinsics."""
        frame_ind = self.available_frames[idx]
        # depth_1hw = self.load_rendered_depth(frame_ind)
        pose_dict = self.load_pose(frame_ind)
        intrinsics = self.load_intrinsics()

        item_dict = {}
        # item_dict["depth_b1hw"] = depth_1hw
        item_dict.update(pose_dict)
        item_dict.update(intrinsics)

        return item_dict


def create_all_occlusion_masks(
    scan_data_root: Path,
    scan_list_path: Path,
    output_dir: Path,
    rendered_depths_root: Path,
    voxel_size: float = 0.02,
    buffer_size: float = 0.5,
    additional_extent: float = 0.3,
    batch_size: int = 4,
    use_gpu: bool = True,
    frame_interval: int = 1,
):
    np.random.seed(1)
    torch.manual_seed(1)

    output_dir.mkdir(exist_ok=True, parents=True)

    # load scan names from val file
    scan_names = readlines(scan_list_path)
    scan_names.sort()

    for ind, scan_name in tqdm.tqdm(enumerate(scan_names)):
        dataset = SimpleScanNetDataset(
            scan_name=scan_name,
            scan_data_root=scan_data_root,
            rendered_depths_root=rendered_depths_root,
            frame_interval=frame_interval,
        )
        mesh_gt = dataset.load_mesh()
        bounds: dict[str, float] = {}
        min_bounds = np.array(mesh_gt.vertices).min(0) - buffer_size
        max_bounds = np.array(mesh_gt.vertices).max(0) + buffer_size

        bounds = {}
        bounds["xmin"] = min_bounds[0]
        bounds["ymin"] = min_bounds[1]
        bounds["zmin"] = min_bounds[2]
        bounds["xmax"] = max_bounds[0]
        bounds["ymax"] = max_bounds[1]
        bounds["zmax"] = max_bounds[2]

        # build volume
        volume = SimpleVolume.from_bounds(bounds=bounds, voxel_size=voxel_size)
        if use_gpu:
            volume.cuda()

        # get aggregator
        visiblity_aggregator = VisibilityAggregator(volume, additional_extent=additional_extent)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

        mesh = Meshes(
            verts=[torch.tensor(mesh_gt.vertices).float()],
            faces=[torch.tensor(mesh_gt.faces).float()],
        )
        raster_settings = RasterizationSettings(
            image_size=(192, 256),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        for batch in tqdm.tqdm(dataloader):
            R = batch["cam_T_world_b44"][:, :3, :3]
            T = batch["cam_T_world_b44"][:, :3, 3]
            for_render_K_b44 = batch["K_b44"].clone()
            for_render_K_b44[:, 0] *= 256
            for_render_K_b44[:, 1] *= 192
            K = for_render_K_b44[:, :3, :3]
            cams = cameras_from_opencv_projection(
                R=R, tvec=T, camera_matrix=K, image_size=torch.tensor((192, 256)).unsqueeze(0)
            )

            mesh = mesh.cuda()
            cams = cams.cuda()

            rasterizer = MeshRasterizer(
                cameras=cams,
                raster_settings=raster_settings,
            )

            _mesh = mesh.extend(len(cams))
            fragments = rasterizer(_mesh)

            depth_bhw1 = fragments.zbuf
            depth_b1hw = depth_bhw1.permute(0, 3, 1, 2)

            if use_gpu:
                batch = to_gpu(batch)

            visiblity_aggregator.integrate_into_volume(
                depth_b1hw, batch["cam_T_world_b44"], batch["K_b44"]
            )

        (output_dir / scan_name).mkdir(exist_ok=True, parents=True)

        volume.save(output_dir / scan_name / f"{scan_name}_volume.npz")

        # save pcd
        pcd = visiblity_aggregator.volume.to_point_cloud(threshold=0.5, num_points=500000)
        o3d.io.write_point_cloud(
            str(output_dir / scan_name / f"{scan_name}_visibility_pcd.ply"), pcd
        )


@click.command()
@click.option(
    "--scan_data_root",
    type=Path,
    default=Path("/mnt/scannet-clean/scans"),
    help="Root directory of scans (scans or scans_test)",
)
@click.option(
    "--scan_list_path",
    type=Path,
    default=Path("src/ar_planes/data_splits/ScanNetv2/standard_split/scannetv2_val.txt"),
    help="Path to a text file with a list of scans to process.",
)
@click.option(
    "--output_dir",
    type=Path,
    default=Path("/home/mohameds/ar_planes_data/plane_gt_meshes/"),
    help="Output directory",
)
@click.option(
    "--rendered_depths_root",
    type=Path,
    default=Path("/home/mohameds/ar_planes_data/plane_gt_distance_renders"),
    help="Directory with mesh renders.",
)
@click.option(
    "--voxel_size",
    type=float,
    default=0.02,
    help="Voxel size in meters.",
)
@click.option(
    "--buffer_size",
    type=float,
    default=0.5,
    help="Buffer size around the extents of the mesh in meters. Used for volume creation.",
)
@click.option(
    "--additional_extent",
    type=float,
    default=0.3,
    help="How far ahead of each camera's depth to mark visible.",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size. Keep this low.",
)
@click.option(
    "--use_gpu",
    type=bool,
    default=True,
    help="Use gpu or not.",
)
@click.option(
    "--frame_interval",
    type=int,
    default=1,
    help="Number of frames to skip between each frame.",
)
def cli(
    scan_data_root: Path,
    scan_list_path: Path,
    output_dir: Path,
    rendered_depths_root: Path,
    voxel_size: float,
    buffer_size: float,
    additional_extent: float,
    batch_size: int,
    use_gpu: bool,
    frame_interval: int,
):
    create_all_occlusion_masks(
        scan_data_root=scan_data_root,
        scan_list_path=scan_list_path,
        output_dir=output_dir,
        rendered_depths_root=rendered_depths_root,
        voxel_size=voxel_size,
        buffer_size=buffer_size,
        additional_extent=additional_extent,
        batch_size=batch_size,
        use_gpu=use_gpu,
        frame_interval=frame_interval,
    )


if __name__ == "__main__":
    cli()  # type: ignore
