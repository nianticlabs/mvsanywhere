import os
import pickle
from collections import OrderedDict
from pathlib import Path

import torch

from doubletake.tools.fusers_helper import OurFuser


class PartialFuser:
    def __init__(
        self,
        gt_mesh_path,
        cached_depth_path: Path,
        depth_noise: float = 0.0,
    ) -> None:
        """Fuses depths for a scan at a time."""
        self.fuser = OurFuser(gt_path=gt_mesh_path, fusion_resolution=0.04, max_fusion_depth=4.0)
        self.cached_depth_path = cached_depth_path

        self.cached_depths = OrderedDict()
        # get pickle files from the cached_depth_path
        pickle_files = os.listdir(cached_depth_path)
        # sort
        pickle_files.sort()

        for file in os.listdir(cached_depth_path):
            if file.endswith(".pickle"):
                # load the pickle file
                with open(cached_depth_path / file, "rb") as f:
                    self.cached_depths[int(file.split(".")[0])] = pickle.load(f)

        self.next_frame_ind_to_fuse = 0
        self.mesh = None

        self.frame_ids = list(self.cached_depths.keys())
        self.frame_ids.sort()

        self.depth_noise = depth_noise

    def get_mesh(self, query_frame_id: int):
        """Returns partial mesh for the current frame_id. Will fuse frames up
        to that point."""

        updated_mesh = False

        if self.next_frame_ind_to_fuse >= len(self.cached_depths):
            return self.mesh

        frame_id_to_fuse = self.frame_ids[self.next_frame_ind_to_fuse]
        # check if we need to fuse.
        if query_frame_id > frame_id_to_fuse:
            # fuse
            while frame_id_to_fuse < query_frame_id:
                # pick the right depth
                cached_data = self.cached_depths[frame_id_to_fuse]
                # fuse
                if self.depth_noise > 0:
                    noise = torch.rand(1) * self.depth_noise
                    noise = noise * (-1 if torch.rand(1) > 0.5 else 1)
                    noise = 1 + noise
                else:
                    noise = torch.tensor(1)
                self.fuser.fuse_frames(
                    depths_b1hw=cached_data["depth_pred_s0_b1hw"] * noise.cuda(),
                    K_b44=cached_data["K_s0_b44"],
                    cam_T_world_b44=cached_data["cam_T_world_b44"],
                    color_b3hw=None,
                )
                updated_mesh = True

                self.next_frame_ind_to_fuse += 1
                if self.next_frame_ind_to_fuse >= len(self.cached_depths):
                    break
                else:
                    frame_id_to_fuse = self.frame_ids[self.next_frame_ind_to_fuse]

        # if the mesh got updated, run marching cubes
        if updated_mesh:
            self.mesh, _, _ = self.fuser.get_mesh_pytorch3d(scale_to_world=True)

        return self.mesh

    def fuse_all_frames(self):
        """Returns partial mesh for the current frame_id. Will fuse frames up
        to that point."""

        for frame_id_to_fuse in self.frame_ids:
            # pick the right depth
            cached_data = self.cached_depths[frame_id_to_fuse]
            # fuse
            if self.depth_noise > 0:
                noise = torch.rand(1) * self.depth_noise
                noise = noise * (-1 if torch.rand(1) > 0.5 else 1)
                noise = 1 + noise
            else:
                noise = torch.tensor(1)
            self.fuser.fuse_frames(
                depths_b1hw=cached_data["depth_pred_s0_b1hw"] * noise.cuda(),
                K_b44=cached_data["K_s0_b44"],
                cam_T_world_b44=cached_data["cam_T_world_b44"],
                color_b3hw=None,
            )

        self.mesh, _, _ = self.fuser.get_mesh_pytorch3d(scale_to_world=True)

        return self.mesh
