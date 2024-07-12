import os
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as TF
import trimesh
from pytorch3d.structures import Meshes
from skimage import measure

from doubletake.utils.pytorch3d_extras import marching_cubes


def get_frustum_bounds(
    invK_144, world_T_cam_144, min_depth=0.1, max_depth=10.0, img_h=480, img_w=640
):
    """
    Gets the frustum bounds for a camera.
    """

    corner_uv_144 = torch.tensor(
        [
            [0, 0, 1, 1],
            [img_w, 0, 1, 1],
            [0, img_h, 1, 1],
            [img_w, img_h, 1, 1],
        ],
        dtype=invK_144.dtype,
        device=invK_144.device,
    ).T.unsqueeze(0)

    corner_points_144 = torch.matmul(invK_144, corner_uv_144)
    min_corner_points_144 = corner_points_144.clone()
    min_corner_points_144[:, :3] *= min_depth
    max_corner_points_144 = corner_points_144.clone()
    max_corner_points_144[:, :3] *= max_depth

    corner_points_148 = torch.cat(
        (
            min_corner_points_144,
            max_corner_points_144,
        ),
        dim=2,
    )
    corner_points_148 = torch.matmul(world_T_cam_144, corner_points_148)
    minbounds_3 = corner_points_148[0].amin(dim=1)[:3]
    maxbounds_3 = corner_points_148[0].amax(dim=1)[:3]

    return minbounds_3, maxbounds_3


class TSDF:

    """
    Class for housing and data handling TSDF volumes.
    """

    # Ensures the final voxel volume dimensions are multiples of 8
    VOX_MOD = 8

    def __init__(
        self,
        voxel_coords_3hwd: torch.tensor,
        tsdf_values: torch.tensor,
        tsdf_weights: torch.tensor,
        voxel_size: float,
        origin: torch.tensor,
    ):
        """
        Sets interal class attributes.
        """
        self.voxel_coords_3hwd = voxel_coords_3hwd.half()
        self.tsdf_values = tsdf_values.half()
        self.tsdf_weights = tsdf_weights.half()
        self.voxel_size = voxel_size
        self.origin = origin.half()

        self.voxel_hashset = o3d.core.HashSet(
            10000,
            key_dtype=o3d.core.int64,
            key_element_shape=(3,),
            device=o3d.core.Device("CUDA:0"),
        )

    @classmethod
    def from_file(cls, tsdf_file):
        """Loads a tsdf from a numpy file."""
        tsdf_data = np.load(tsdf_file)

        tsdf_values = torch.from_numpy(tsdf_data["tsdf_values"])
        tsdf_weights = torch.from_numpy(tsdf_data["tsdf_weights"])
        origin = torch.from_numpy(tsdf_data["origin"])
        voxel_coords_3hwd = torch.from_numpy(tsdf_data["voxel_coords_3hwd"])
        voxel_size = tsdf_data["voxel_size"].item()

        return TSDF(voxel_coords_3hwd, tsdf_values, tsdf_weights, voxel_size, origin)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, voxel_size: float):
        """Gets TSDF bounds from a mesh file."""
        xmax, ymax, zmax = mesh.vertices.max(0)
        xmin, ymin, zmin = mesh.vertices.min(0)

        bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

        # create a buffer around bounds
        for key, val in bounds.items():
            if "min" in key:
                bounds[key] = val - 3 * voxel_size
            else:
                bounds[key] = val + 3 * voxel_size
        return cls.from_bounds(bounds, voxel_size)

    @classmethod
    def from_bounds(cls, bounds: dict, voxel_size: float):
        """Creates a TSDF volume with bounds at a specific voxel size."""

        expected_keys = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        for key in expected_keys:
            if key not in bounds.keys():
                raise KeyError(
                    "Provided bounds dict need to have keys"
                    "'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'!"
                )

        num_voxels_x = (
            int(np.ceil((bounds["xmax"] - bounds["xmin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )
        num_voxels_y = (
            int(np.ceil((bounds["ymax"] - bounds["ymin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )
        num_voxels_z = (
            int(np.ceil((bounds["zmax"] - bounds["zmin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )

        origin = torch.FloatTensor([bounds["xmin"], bounds["ymin"], bounds["zmin"]])

        voxel_coords_3hwd = cls.generate_voxel_coords(
            origin, (num_voxels_x, num_voxels_y, num_voxels_z), voxel_size
        ).half()

        # init to -1s
        tsdf_values = -torch.ones_like(voxel_coords_3hwd[0]).half()
        tsdf_weights = torch.zeros_like(voxel_coords_3hwd[0]).half()

        return TSDF(voxel_coords_3hwd, tsdf_values, tsdf_weights, voxel_size, origin)

    @classmethod
    def generate_voxel_coords(
        cls, origin: torch.tensor, volume_dims: Tuple[int, int, int], voxel_size: float
    ):
        """Gets world coordinates for each location in the TSDF."""

        grid = torch.meshgrid([torch.arange(vd) for vd in volume_dims])

        voxel_coords_3hwd = origin.view(3, 1, 1, 1) + torch.stack(grid, 0) * voxel_size

        return voxel_coords_3hwd

    def cuda(self):
        """Moves TSDF to gpu memory."""
        self.voxel_coords_3hwd = self.voxel_coords_3hwd.cuda()
        self.tsdf_values = self.tsdf_values.cuda()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cuda()

    def cpu(self):
        """Moves TSDF to cpu memory."""
        self.voxel_coords_3hwd = self.voxel_coords_3hwd.cpu()
        self.tsdf_values = self.tsdf_values.cpu()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cpu()

    def to_mesh(self, scale_to_world=True, export_single_mesh=False):
        """Extracts a mesh from the TSDF volume using marching cubes.

        Args:
            scale_to_world: should we scale vertices from TSDF voxel coords
                to world coordinates?
            export_single_mesh: returns a single walled mesh from marching
                cubes. Requires a custom implementation of
                measure.marching_cubes that supports single_mesh

        """
        tsdf = self.tsdf_values.detach().cpu().clone().float()
        tsdf_np = tsdf.clamp(-1, 1).cpu().numpy()

        if export_single_mesh:
            verts, faces, norms, _ = measure.marching_cubes(
                tsdf_np,
                level=0,
                allow_degenerate=False,
                single_mesh=True,
            )
        else:
            verts, faces, norms, _ = measure.marching_cubes(
                tsdf_np,
                level=0,
                allow_degenerate=False,
            )

        if scale_to_world:
            verts = self.origin.cpu().view(1, 3) + verts * self.voxel_size

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=norms)
        return mesh

    def to_mesh_pytorch3d(self, scale_to_world=True, min_bounds_3=None, max_bounds_3=None):
        active_buf_indices = self.voxel_hashset.active_buf_indices().to(o3d.core.int64)
        active_keys = self.voxel_hashset.key_tensor()[active_buf_indices]
        active_keys = (
            torch.utils.dlpack.from_dlpack(active_keys.to_dlpack()).to(torch.int32).contiguous()
        )

        tsdf_vals = self.tsdf_values

        # convert bounds to indices
        min_bounds_3 = (
            torch.floor((min_bounds_3 - self.origin.cuda().float()) / self.voxel_size).int()
            if min_bounds_3 is not None
            else None
        )
        max_bounds_3 = (
            torch.ceil((max_bounds_3 - self.origin.cuda().float()) / self.voxel_size).int()
            if max_bounds_3 is not None
            else None
        )

        batched_verts, batched_faces = marching_cubes(
            tsdf_vals[None].float().cuda(),
            active_keys,
            isolevel=0.0,
            return_local_coords=False,
            min_bounds=min_bounds_3,
            max_bounds=max_bounds_3,
        )

        verts = batched_verts[0]
        faces = batched_faces[0]

        if len(verts) == 0:
            verts = torch.zeros(1, 3).cuda()
            faces = torch.zeros(1, 3).cuda()
        if scale_to_world:
            verts = self.origin.view(1, 3).cuda() + verts * self.voxel_size

        return Meshes(verts=[verts], faces=[faces]), verts, faces

    def save_mesh(self, savepath, filename):
        """Saves a mesh to disk."""
        self.cpu()
        os.makedirs(savepath, exist_ok=True)

        mesh = self.to_mesh()
        trimesh.exchange.export.export_mesh(
            mesh, os.path.join(savepath, filename).replace(".bin", ".ply"), "ply"
        )

    def save_tsdf(self, filepath):
        np.savez_compressed(
            filepath,
            tsdf_values=self.tsdf_values.cpu().numpy().astype(np.float16),
            tsdf_weights=self.tsdf_weights.cpu().numpy().astype(np.float16),
            origin=self.origin.cpu().numpy().astype(np.float16),
            voxel_coords_3hwd=self.voxel_coords_3hwd.cpu().numpy().astype(np.float16),
            voxel_size=self.voxel_size,
        )

    def sample_tsdf(self, world_points_N3, what_to_sample="tsdf", sampling_method="bilinear"):
        """Samples the TSDF volume at world coordinates provided.
        Args:
            world_points_N3 (torch.Tensor): Tensor of shape (N, 3) containing
                world coordinates to sample the volume at.
            what_to_sample (str): what to sample from the TSDF volume. Can be one of
                "tsdf", "weights", ...
            sampling_method (str): sampling method to use. Can be one of
                "nearest", "bilinear", "trilinear".
        Returns:
            torch.Tensor: Tensor of shape (N,) containing the values of the
                volume at the provided world coordinates.
        """

        if not (world_points_N3.shape[1] == 3 and world_points_N3.ndim == 2):
            raise ValueError(
                "world_points_N3 must have shape (N, 3)! Instead got shape {}".format(
                    world_points_N3.shape
                )
            )

        world_points_N3 = world_points_N3.to(self.voxel_coords_3hwd.device)

        # convert world coordinates to voxel coordinates
        voxel_coords_N3 = world_points_N3 - self.origin.view(1, 3).to(world_points_N3.device)
        voxel_coords_N3 = voxel_coords_N3 / self.voxel_size

        # divide by the volume_size - 1 for align corners True!
        dim_size_3 = torch.tensor(
            self.voxel_coords_3hwd.shape[1:],
            dtype=world_points_N3.dtype,
            device=world_points_N3.device,
        )
        voxel_coords_N3 = voxel_coords_N3 / (dim_size_3.view(1, 3) - 1)
        # convert from 0-1 to [-1, 1] range
        voxel_coords_N3 = voxel_coords_N3 * 2 - 1
        voxel_coords_111N3 = voxel_coords_N3[None, None, None]

        # sample the volume
        # grid_sample expects y, x, z and we have x, y, z
        # swap the axes of the coords to match the pytorch grid_sample convention
        voxel_coords_111N3 = voxel_coords_111N3[:, :, :, :, [2, 1, 0]]

        if what_to_sample == "tsdf":
            volume_to_sample_chwd = self.tsdf_values.unsqueeze(0)
        elif what_to_sample == "weights":
            volume_to_sample_chwd = self.tsdf_weights.unsqueeze(0)

        # in case we're asked to support fp16 and cpu, we need to cast to fp32 for the
        # grid_sample call
        if volume_to_sample_chwd.device == torch.device("cpu"):
            tensor_dtype = torch.float32
        else:
            tensor_dtype = volume_to_sample_chwd.dtype

        values_N = torch.nn.functional.grid_sample(
            volume_to_sample_chwd.unsqueeze(0).type(tensor_dtype),
            voxel_coords_111N3.type(tensor_dtype),
            align_corners=True,
            mode=sampling_method,
        ).squeeze()

        return values_N


class TSDFFuser:
    """
    Class for fusing depth maps into TSDF volumes.
    """

    def __init__(self, tsdf, min_depth=0.5, max_depth=5.0, use_gpu=True):
        """
        Inits the fuser with fusing parameters.

        Args:
            tsdf: a TSDF volume object.
            min_depth: minimum depth to limit inomcing depth maps to.
            max_depth: maximum depth to limit inomcing depth maps to.
            use_gpu: use cuda?

        """
        self.tsdf = tsdf
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_gpu = use_gpu
        self.truncation_size = 3.0
        self.maxW = 100.0
        self.vox_indices = torch.stack(
            torch.meshgrid([torch.arange(vd) for vd in self.shape])
        ).long()

        # Create homogeneous coords once only
        self.hom_voxel_coords_14hwd = torch.cat(
            (self.voxel_coords_3hwd, torch.ones_like(self.voxel_coords_3hwd[:1])), 0
        ).unsqueeze(0)

    @property
    def voxel_coords_3hwd(self):
        return self.tsdf.voxel_coords_3hwd

    @property
    def voxel_hashset(self):
        return self.tsdf.voxel_hashset

    @property
    def tsdf_values(self):
        return self.tsdf.tsdf_values

    @property
    def tsdf_weights(self):
        return self.tsdf.tsdf_weights

    @property
    def voxel_size(self):
        return self.tsdf.voxel_size

    @property
    def shape(self):
        return self.voxel_coords_3hwd.shape[1:]

    @property
    def truncation(self):
        return self.truncation_size * self.voxel_size

    def project_to_camera(self, cam_T_world_T_144, K_144, valid_voxels_14N):
        if self.use_gpu:
            cam_T_world_T_144 = cam_T_world_T_144.cuda()
            K_144 = K_144.cuda()
            valid_voxels_14N = valid_voxels_14N.cuda()

        world_to_pix_P_134 = torch.matmul(K_144, cam_T_world_T_144)[:, :3]

        cam_points_13N = torch.matmul(world_to_pix_P_134, valid_voxels_14N)
        cam_points_13N[:, :2] = cam_points_13N[:, :2] / cam_points_13N[:, 2, None]

        return cam_points_13N

    def integrate_depth(
        self,
        depth_b1hw,
        cam_T_world_T_b44,
        K_b44,
        depth_mask_b1hw=None,
        extended_neg_truncation=False,
    ):
        """
        Integrates depth maps into the volume. Supports batching.

        depth_b1hw: tensor with depth map
        cam_T_world_T_b44: camera extrinsics (not pose!).
        K_b44: camera intrinsics.
        depth_mask_b1hw: an optional boolean mask for valid depth points in
            the depth map.
        """
        img_h, img_w = depth_b1hw.shape[2:]
        img_size = torch.tensor([img_w, img_h], dtype=torch.float16).view(1, 1, 1, 2)
        if self.use_gpu:
            depth_b1hw = depth_b1hw.cuda()
            img_size = img_size.cuda()
            self.tsdf.cuda()
            self.hom_voxel_coords_14hwd = self.hom_voxel_coords_14hwd.cuda()
            self.vox_indices = self.vox_indices.cuda()

        if depth_mask_b1hw is not None:
            depth_b1hw = depth_b1hw.clone()
            depth_b1hw[~depth_mask_b1hw] = -1

        for batch_idx in range(len(depth_b1hw)):
            cam_T_world_T_144 = cam_T_world_T_b44[batch_idx : batch_idx + 1]
            K_144 = K_b44[batch_idx : batch_idx + 1]
            depth_11hw = depth_b1hw[batch_idx : batch_idx + 1]

            # get voxels which are visible in the camera
            depth_min = 0.01
            depth_max = self.max_depth + self.truncation + 0.1
            invK_144 = torch.inverse(K_144.float()).half()
            world_T_cam_T_144 = torch.inverse(cam_T_world_T_144.float()).half()

            minbounds_3, maxbounds_3 = get_frustum_bounds(
                invK_144[0], world_T_cam_T_144[0], depth_min, depth_max, img_h, img_w
            )

            valid_voxel_mask_N = (
                torch.logical_and(
                    self.hom_voxel_coords_14hwd[0, :3] > minbounds_3.view(1, -1, 1, 1, 1),
                    self.hom_voxel_coords_14hwd[0, :3] < maxbounds_3.view(1, -1, 1, 1, 1),
                )
                .all(1)
                .view(-1)
            )

            valid_voxels_14N = self.hom_voxel_coords_14hwd.flatten(2)[..., valid_voxel_mask_N]

            # Project voxel coordinates into images
            cam_points_b3N = self.project_to_camera(cam_T_world_T_144, K_144, valid_voxels_14N)
            vox_depth_b1N = cam_points_b3N[:, 2:3]
            pixel_coords_b2N = cam_points_b3N[:, :2]

            # Reshape the projected voxel coords to a 2D view of shape Hx(WxD)
            pixel_coords_bhw2 = pixel_coords_b2N.reshape(1, 2, 1, -1).permute(0, 2, 3, 1)
            pixel_coords_bhw2 = 2 * pixel_coords_bhw2 / img_size - 1

            # Sample the depth using grid sample
            sampled_depth_b1hw = TF.grid_sample(
                input=depth_11hw,
                grid=pixel_coords_bhw2,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_depth_b1N = sampled_depth_b1hw.flatten(start_dim=2)

            # Confidence from InfiniTAM
            confidence_b1N = (
                torch.clamp(
                    1.0 - (sampled_depth_b1N - self.min_depth) / (self.max_depth - self.min_depth),
                    min=0.25,
                    max=1.0,
                )
                ** 2
            )

            # Calculate TSDF values from depth difference by normalizing to [-1, 1]
            dist_b1N = sampled_depth_b1N - vox_depth_b1N
            tsdf_vals_b1N = torch.clamp(dist_b1N / self.truncation, min=-1.0, max=1.0)

            if extended_neg_truncation:
                trunc_check = -self.truncation * 1.5
            else:
                trunc_check = -self.truncation

            # Get the valid points mask
            valid_points_b1N = (
                (vox_depth_b1N > 0)
                & (dist_b1N > trunc_check)
                & (sampled_depth_b1N > 0)
                & (vox_depth_b1N > 0)
                & (vox_depth_b1N < self.max_depth)
                & (confidence_b1N > 0)
            )

            valid_points_1N = valid_points_b1N[0]
            tsdf_val_1N = tsdf_vals_b1N[0]
            confidence_1N = confidence_b1N[0]

            # Reshape the valid mask to the TSDF's shape and read the old values
            valid_indices = self.vox_indices.flatten(1)[:, valid_voxel_mask_N][
                :, valid_points_1N[0]
            ]
            old_tsdf_vals = self.tsdf_values[valid_indices[0], valid_indices[1], valid_indices[2]]
            old_weights = self.tsdf_weights[valid_indices[0], valid_indices[1], valid_indices[2]]

            # add active voxels to the hashset
            active_mask = torch.logical_and(valid_points_1N, dist_b1N < self.truncation)
            active_indices_N3 = self.vox_indices.flatten(1)[:, valid_voxel_mask_N][
                :, active_mask[0, 0]
            ].T.contiguous()
            active_indices_N3 = o3d.core.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(active_indices_N3)
            )
            if active_indices_N3.shape[0] > 0:
                self.voxel_hashset.insert(active_indices_N3)

            # Fetch the new tsdf values and the confidence
            new_tsdf_vals = tsdf_val_1N[valid_points_1N]
            confidence = confidence_1N[valid_points_1N]

            # simplified now.
            # update_rate = torch.where(confidence < old_weights, 2.0, 5.0).half()
            update_rate = 2.5

            # Compute the new weight and the normalization factor
            new_weights = confidence * update_rate / self.maxW
            total_weights = old_weights + new_weights

            # Update the tsdf and the weights
            self.tsdf_values[valid_indices[0], valid_indices[1], valid_indices[2]] = (
                old_tsdf_vals * old_weights + new_tsdf_vals * new_weights
            ) / total_weights
            self.tsdf_weights[valid_indices[0], valid_indices[1], valid_indices[2]] = torch.clamp(
                total_weights, max=1.0
            )
