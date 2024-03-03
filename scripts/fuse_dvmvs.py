import os
import torch

from pathlib import Path

import trimesh

from geometryhints.datasets.scannet_dataset import ScannetDataset
from geometryhints.options import OptionsHandler
from geometryhints.tools.fusers_helper import OurFuser
from geometryhints.tools.partial_fuser import PartialFuser
from geometryhints.utils.dataset_utils import get_dataset
from geometryhints.utils.generic_utils import readlines, to_gpu

import numpy as np
import open3d as o3d
import torch
import tqdm
from PIL import Image
from geometryhints.utils.geometry_utils import BackprojectDepth
from geometryhints.utils.rendering_utils import PyTorch3DMeshDepthRenderer


from geometryhints.utils.visualization_utils import colormap_image, save_viz_video_frames


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
        self.available_frames.sort()

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

def fuse_depth_maps(
    scan_id: str,
    dataset_root: Path,
    cached_depth_path: Path,
    output_path: Path,
    tuple_filepath: Path,
    batch_size: int = 4,
):


    dataset = SimpleScanNetDataset(
        scan_name=scan_id,
        scan_data_root=dataset_root,
        tuple_filepath=tuple_filepath,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False
    )
    
    cached_depths_list = np.load(cached_depth_path, allow_pickle=True)['arr_0']
    cached_depths_N1hw = torch.tensor(np.stack(cached_depths_list)).float().cuda().unsqueeze(1)

    assert len(cached_depths_N1hw) == len(dataset), f"{len(cached_depths_N1hw)} != {len(dataset)}"

    get_mesh_path = ScannetDataset.get_gt_mesh_path(opts.dataset_path, opts.split, scan_id)
    fuser = OurFuser(gt_path=get_mesh_path, fusion_resolution=0.04, max_fusion_depth=3.0)

    with torch.no_grad():
        for batch_ind, batch in tqdm.tqdm(enumerate(dataloader)):
            depths_b1hw = cached_depths_N1hw[batch_ind * batch_size : (batch_ind + 1) * batch_size]
            
            # upsample nearest to 640x480
            depths_b1hw = torch.nn.functional.interpolate(
                depths_b1hw, size=(480, 640), mode="nearest"
            )
            
            K_b44 = batch["K_b44"]
            K_b44[:,0] *= 640
            K_b44[:,1] *= 480
            
            batch = to_gpu(batch, key_ignores=["frame_id_str"])
            fuser.fuse_frames(
                depths_b1hw=depths_b1hw,
                K_b44=K_b44.cuda(),
                cam_T_world_b44=batch["cam_T_world_b44"].cuda(),
                color_b3hw=None,
            )
    
    fuser.export_mesh(output_path / f"{scan_id}.ply")



def fuse_cached_depths_for_scans(
    scan_list: list[str],
    dataset_root: Path,
    cached_depth_wild_card_path: str,
    output_path: Path,
    tuple_filepath: Path,
    batch_size: int = 4,
):
    output_path.mkdir(exist_ok=True, parents=True)
    for scan_id in tqdm.tqdm(scan_list):
        fuse_depth_maps(
            scan_id = scan_id,
            dataset_root = dataset_root,
            cached_depth_path = cached_depth_wild_card_path.replace("SCAN_NAME", scan_id),
            output_path = output_path,
            tuple_filepath = tuple_filepath,
            batch_size = batch_size,
        )


if __name__ == "__main__":
    # python scripts/fuse_dvmvs.py --data_config configs/data/scannet_default_test.yaml \
    # --cached_depth_wild_card_path /mnt/nas3/personal/mohameds/geometry_hints/outputs/dvmvs_results/fusion_depths/keyframe__320_256_2_dvmvs_fusionnet_predictions_SCAN_NAME.npz  \
    # --output_path /mnt/nas/personal/mohameds/geometry_hints/outputs/
    # load options file
    option_handler = OptionsHandler()

    option_handler.parser.add_argument(
        "--cached_depth_wild_card_path",
        type=str,
        required=True,
        help="Path to the cached depths.",
    )
    option_handler.parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to output directory.",
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

    dataset_root = Path(opts.dataset_path) / dataset_class.get_sub_folder_dir("test")

    fuse_cached_depths_for_scans(
        scan_list=scan_names,
        dataset_root=dataset_root,
        cached_depth_wild_card_path=opts.cached_depth_wild_card_path,
        tuple_filepath=Path(opts.tuple_info_file_location)
        / f"{opts.split}{opts.mv_tuple_file_suffix}",
        output_path=opts.output_path,
        batch_size=opts.batch_size,
    )
