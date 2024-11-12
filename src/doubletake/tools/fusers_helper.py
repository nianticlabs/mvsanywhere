import numpy as np
from open3d import core as o3c
import open3d as o3d
import torch
import trimesh
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from doubletake.datasets.scannet_dataset import ScannetDataset
from doubletake.datasets.threer_scan_dataset import ThreeRScanDataset
from doubletake.tools.tsdf import TSDF, TSDFFuser
from doubletake.utils.generic_utils import reverse_imagenet_normalize


class DepthFuser:
    def __init__(self, gt_path="", fusion_resolution=0.04, max_fusion_depth=3.0, fuse_color=False):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth


class OurFuser(DepthFuser):
    """
    This is the fuser used for scores in the SimpleRecon paper. Note that
    unlike open3d's fuser this implementation does not do voxel hashing. If a
    path to a known mehs reconstruction is provided, this function will limit
    bounds to that mesh's extent, otherwise it'll use a wide volume to prevent
    clipping.

    It's best not to use this fuser unless you need to recreate numbers from the
    paper.

    """

    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3,
        fuse_color=False,
        extended_neg_truncation=False,
    ):
        super().__init__(
            gt_path,
            fusion_resolution,
            max_fusion_depth,
            fuse_color,
        )

        if gt_path is not None:
            gt_mesh = trimesh.load(gt_path, force="mesh")
            tsdf_pred = TSDF.from_mesh(gt_mesh, voxel_size=fusion_resolution)
        else:
            bounds = {}
            bounds["xmin"] = -10.0
            bounds["xmax"] = 10.0
            bounds["ymin"] = -10.0
            bounds["ymax"] = 10.0
            bounds["zmin"] = -10.0
            bounds["zmax"] = 10.0

            tsdf_pred = TSDF.from_bounds(bounds, voxel_size=fusion_resolution)
        self.extended_neg_truncation = extended_neg_truncation
        self.tsdf_fuser_pred = TSDFFuser(tsdf_pred, max_depth=max_fusion_depth)

    def fuse_frames(self, depths_b1hw, K_b44, cam_T_world_b44, color_b3hw):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
            extended_neg_truncation=self.extended_neg_truncation,
        )

    def export_mesh(self, path, export_single_mesh=True):
        _ = trimesh.exchange.export.export_mesh(
            self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh),
            path,
        )

    def save_tsdf(self, path):
        self.tsdf_fuser_pred.tsdf.save_tsdf(path)

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
        return self.tsdf_fuser_pred.tsdf.sample_tsdf(
            world_points_N3, what_to_sample=what_to_sample, sampling_method=sampling_method
        )

    def get_mesh(self, export_single_mesh=True, convert_to_trimesh=True):
        return self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh)

    def get_mesh_pytorch3d(self, scale_to_world=True, min_bounds_3=None, max_bounds_3=None):
        return self.tsdf_fuser_pred.tsdf.to_mesh_pytorch3d(
            scale_to_world=scale_to_world, min_bounds_3=min_bounds_3, max_bounds_3=max_bounds_3
        )


class Open3DFuser(DepthFuser):
    """
    Wrapper class for the open3d fuser.

    This wrapper does not support fusion of tensors with higher than batch 1.
    """

    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3,
        fuse_color=False,
        use_upsample_depth=False,
    ):
        super().__init__(
            gt_path,
            fusion_resolution,
            max_fusion_depth,
            fuse_color,
        )

        self.fuse_color = fuse_color
        self.use_upsample_depth = use_upsample_depth
        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depths_b1hw,
        K_b44,
        cam_T_world_b44,
        color_b3hw,
    ):
        width = depths_b1hw.shape[-1]
        height = depths_b1hw.shape[-2]

        if self.fuse_color:
            color_b3hw = torch.nn.functional.interpolate(
                color_b3hw,
                size=(height, width),
            )
            color_b3hw = reverse_imagenet_normalize(color_b3hw)

        for batch_index in range(depths_b1hw.shape[0]):
            if self.fuse_color:
                image_i = color_b3hw[batch_index].permute(1, 2, 0)

                color_im = (image_i * 255).cpu().numpy().astype(np.uint8).copy(order="C")
            else:
                # mesh will now be grey
                color_im = (
                    0.7 * torch.ones_like(depths_b1hw[batch_index]).squeeze().cpu().clone().numpy()
                )
                color_im = np.repeat(color_im[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)

            depth_pred = depths_b1hw[batch_index].squeeze().cpu().clone().numpy()
            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_im,
                depth_pred,
                depth_scale=1.0,
                depth_trunc=self.fusion_max_depth,
                convert_rgb_to_intensity=False,
            )
            cam_intr = K_b44[batch_index].cpu().clone().numpy()
            cam_T_world_44 = cam_T_world_b44[batch_index].cpu().clone().numpy()

            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    fx=cam_intr[0, 0],
                    fy=cam_intr[1, 1],
                    cx=cam_intr[0, 2],
                    cy=cam_intr[1, 2],
                ),
                cam_T_world_44,
            )

    def export_mesh(self, path, use_marching_cubes_mask=None):
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False):
        mesh = self.volume.extract_triangle_mesh()

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)

        return mesh

    def save_tsdf(self, path):
        return


def get_fuser(opts, scan):
    """Returns the depth fuser required. Our fuser doesn't allow for"""

    dataset_opts = opts.datasets[0]

    if dataset_opts.dataset == "scannet":
        gt_path = ScannetDataset.get_gt_mesh_path(dataset_opts.dataset_path, dataset_opts.split, scan)
    elif dataset_opts.dataset == "3rscan":
        gt_path = ThreeRScanDataset.get_gt_mesh_path(dataset_opts.dataset_path, dataset_opts.split, scan)
    elif dataset_opts.dataset == "7scenes":
        gt_path = "/outputs/fused_gt/7scenes/default/meshes/0.04_8.0_ours/SCAN_NAME.ply".replace(
            "SCAN_NAME", scan.replace("/", "_")
        )
    else:
        gt_path = None

    if opts.depth_fuser == "ours":
        if opts.fuse_color:
            print(
                "WARNING: fusing color using 'ours' fuser is not supported, "
                "Color will not be fused."
            )

        fuser = OurFuser(
            gt_path=gt_path,
            fusion_resolution=opts.fusion_resolution,
            max_fusion_depth=opts.fusion_max_depth,
            fuse_color=False,
            extended_neg_truncation=opts.extended_neg_truncation,
        )
        fuser.tsdf_fuser_pred.tsdf.cuda()
        return fuser
    if opts.depth_fuser == "open3d":
        return Open3DFuser(
            gt_path=gt_path,
            fusion_resolution=opts.fusion_resolution,
            max_fusion_depth=opts.fusion_max_depth,
            fuse_color=opts.fuse_color,
        )
    elif opts.depth_fuser == "custom_open3d":
        return CustomOpen3dFuser(
            gt_path=gt_path,
            fusion_resolution=opts.fusion_resolution,
            max_fusion_depth=opts.fusion_max_depth,
            fuse_color=opts.fuse_color,
            extended_neg_truncation=opts.extended_neg_truncation,
        )
    else:
        raise ValueError(f"Unrecognized fuser {opts.depth_fuser}!")


class CustomOpen3dFuser(Open3DFuser):
    def __init__(self, extended_neg_truncation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sdf_trunc = 3 * self.fusion_resolution
        self.weight_threshold = 0.00000001

        attr_names = ("tsdf", "weight", "color")
        attr_dtypes = (o3c.float32, o3c.float32, o3c.float32)
        attr_channels = ((1), (1), (3))

        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=attr_names,
            attr_dtypes=attr_dtypes,
            attr_channels=attr_channels,
            voxel_size=self.fusion_resolution,
            block_resolution=16,
            block_count=1000,
            device=o3c.Device("cuda:0"),
        )

        self.counts = 0
        self.cached_colors = None
        self.prev_voxel_coords = []

        self.extended_neg_truncation = extended_neg_truncation

    def fuse_frames(
        self,
        depths_b1hw,
        K_b44,
        cam_T_world_b44,
        color_b3hw,
        hint_b1hw=None,
    ):
        width = depths_b1hw.shape[-1]
        height = depths_b1hw.shape[-2]

        if self.fuse_color:
            color_b3hw = torch.nn.functional.interpolate(
                color_b3hw,
                size=(height, width),
            )
            color_b3hw = reverse_imagenet_normalize(color_b3hw)

        # convert to open3d tensors
        depths_b1hw = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depths_b1hw))
        K_b33 = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(K_b44[:, :3, :3]))
        cam_T_world_b44 = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(cam_T_world_b44))
        if self.fuse_color:
            color_bhw3 = o3c.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(color_b3hw.permute(0, 2, 3, 1))
            )
        else:
            color_bhw3 = None

        if hint_b1hw is not None:
            if hint_b1hw.max() <= 0.0 or hint_b1hw.min() >= self.max_fusion_depth or True:
                hint_b1hw = None
            else:
                hint_b1hw = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(hint_b1hw))

        for batch_index in range(depths_b1hw.shape[0]):
            depth_pred_hw = depths_b1hw[batch_index, 0]
            depth_pred_img = o3d.t.geometry.Image(depth_pred_hw)
            K_33 = K_b33[batch_index]
            cam_T_world_44 = cam_T_world_b44[batch_index]

            # Find which voxels are to be updated and activate in the hashmap ready for update
            # For some reason even if the depth map and volume are on the gpu, we need the
            # camera intrinsics and extrinsics on the cpu
            frustum_block_coords = self.volume.compute_unique_block_coordinates(
                depth_pred_img,
                K_33.to(o3c.Device("cpu:0")).to(o3c.float64),
                cam_T_world_44.to(o3c.Device("cpu:0")).to(o3c.float64),
                1.0,
                self.max_fusion_depth,
                trunc_voxel_multiplier=3.0,
            )
            self.volume.hashmap().activate(frustum_block_coords)

            # Ideally we'd use the following piece of code to find the indices of the activated
            # voxels and use that in the call to `voxel_coordinates_and_flattened_indices`. Unfortunately,
            # `compute_unique_block_coordinates` only returns those blocks at the depth map, so we can't
            # clean up free space with those indices. We'll update all voxels that have ever been seen instead.
            # That means not passing in `buf_indices`.

            # # Find indices of the activated voxels in the hashmap
            # buf_indices, masks = self.volume.hashmap().find(frustum_block_coords)

            # Extract voxel coordinates and indices of the activated voxels
            voxel_coords, voxel_indices = self.volume.voxel_coordinates_and_flattened_indices()
            o3d.core.cuda.synchronize()

            self.update_tsdf_for_voxels(
                voxel_coords,
                voxel_indices,
                depth_pred_hw,
                K_33,
                cam_T_world_44,
                color_bhw3,
                batch_index,
                width,
                height,
            )

        o3d.core.cuda.release_cache()
        # torch.cuda.empty_cache()

    def update_tsdf_for_voxels(
        self,
        voxel_coords,
        voxel_indices,
        depth_pred_hw,
        K_33,
        cam_T_world_44,
        color_bhw3,
        batch_index,
        width,
        height,
    ):
        # project the voxel coordinates to the camera frame
        cam_coords = cam_T_world_44[:3, :3] @ voxel_coords.T() + cam_T_world_44[:3, 3:]
        cam_coords = K_33[:3, :3] @ cam_coords

        proj_depth = cam_coords[2]
        x_pix = (cam_coords[0] / proj_depth).round().to(o3c.int64)
        y_pix = (cam_coords[1] / proj_depth).round().to(o3c.int64)
        o3d.core.cuda.synchronize()

        mask_proj = (
            (proj_depth > 0) & (x_pix >= 0) & (y_pix >= 0) & (x_pix < width) & (y_pix < height)
        )

        # sample the depth map to get new tsdf values
        x_pix = x_pix[mask_proj]
        y_pix = y_pix[mask_proj]
        proj_depth = proj_depth[mask_proj]
        sampled_depths = depth_pred_hw[y_pix, x_pix]
        tsdf_vals = sampled_depths - proj_depth

        min_sdf_value = -self.sdf_trunc * 1.5 if self.extended_neg_truncation else -self.sdf_trunc
        mask_inlier = (
            (sampled_depths > 0)
            & (sampled_depths < self.max_fusion_depth)
            & (tsdf_vals >= min_sdf_value)
        )

        tsdf_vals[tsdf_vals >= self.sdf_trunc] = self.sdf_trunc
        tsdf_vals = tsdf_vals / self.sdf_trunc
        o3d.core.cuda.synchronize()

        # infinTAM confidence
        confidence = (
            1.0 - (sampled_depths[mask_inlier] - 0.5) / (self.max_fusion_depth - 0.5)
        ).clip(0.25, 1.0)
        confidence = (confidence * confidence).reshape((-1, 1))

        voxel_weights = self.volume.attribute("weight").reshape((-1, 1))
        voxel_tsdf_vals = self.volume.attribute("tsdf").reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        old_weights = voxel_weights[valid_voxel_indices]

        # More infiniTAM magic: update faster when the new samples are more confident
        update_rate = 2.5

        # Compute the new weight and the normalization factor
        new_weights = confidence * update_rate / 100.0
        total_weights = old_weights + new_weights

        voxel_tsdf_vals[valid_voxel_indices] = (
            voxel_tsdf_vals[valid_voxel_indices] * old_weights
            + tsdf_vals[mask_inlier].reshape(old_weights.shape) * new_weights
        ) / total_weights

        if self.fuse_color:
            color_hw3 = color_bhw3[batch_index]
            sampled_colors = color_hw3[y_pix, x_pix].to(o3c.float32)

            voxel_colors = self.volume.attribute("color").reshape((-1, 3))
            voxel_colors[valid_voxel_indices] = (
                voxel_colors[valid_voxel_indices] * old_weights
                + sampled_colors[mask_inlier] * new_weights
            ) / total_weights

        voxel_weights[valid_voxel_indices] = total_weights.clip(0.0, 1.0)
        o3d.core.cuda.synchronize()

    def export_mesh(self, path, use_marching_cubes_mask=None, trim_tsdf_using_confience=False):
        mesh = self.get_mesh(
            convert_to_trimesh=False, trim_tsdf_using_confience=trim_tsdf_using_confience
        )
        o3d.t.io.write_triangle_mesh(path, mesh)

    def get_mesh(
        self,
        export_single_mesh=None,
        convert_to_trimesh=False,
        get_confidence=False,
        trim_tsdf_using_confience=False,
    ):
        voxel_weights = self.volume.attribute("weight")
        voxel_sdfs = self.volume.attribute("tsdf")

        if trim_tsdf_using_confience:
            voxel_sdfs[voxel_weights < 0.02] = 0

        if get_confidence:
            voxel_colors = self.volume.attribute("color").reshape((-1, 3))
            voxel_weights = self.volume.attribute("weight").reshape((-1, 1))

            # save the colors so we can get them back if required
            self.cached_colors = voxel_colors.clone()
            voxel_colors[:, 0:1] = voxel_weights
        elif self.cached_colors is not None:
            voxel_colors = self.cached_colors

        mesh = self.volume.extract_triangle_mesh(weight_threshold=self.weight_threshold)  # .cpu()

        # Open3D MC assumes TSDF colors are in [0-255] so divides by 255 -> undo this
        mesh.vertex.colors = mesh.vertex.colors * 255.0

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)

        return mesh

    def get_mesh_pytorch3d(self, scale_to_world=True):
        mesh = self.get_mesh(convert_to_trimesh=False, get_confidence=True)

        # convert to pytorch3d mesh
        verts = torch.from_dlpack(o3c.Tensor.to_dlpack(mesh.vertex.positions)).clone()
        faces = torch.from_dlpack(o3c.Tensor.to_dlpack(mesh.triangle.indices)).clone()
        textures = TexturesVertex(
            torch.from_dlpack(o3c.Tensor.to_dlpack(mesh.vertex.colors))  # .cpu()
            .unsqueeze(0)
            .clone()
        )

        pytorch3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures).cuda()

        self.counts += 1
        if self.counts % 25 == 0:
            self.counts = 0
            o3d.core.cuda.release_cache()
            torch.cuda.empty_cache()

        return pytorch3d_mesh, verts, faces