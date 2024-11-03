import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops

from doubletake.utils.geometry_utils import BackprojectDepth, Project3D


class CostVolumeManager(nn.Module):

    """
    Class to build a cost volume from extracted features of an input
    reference image and N source images.

    Achieved by backwarping source features onto current features using
    hypothesised depths between min_depth_bin and max_depth_bin, and then
    collapsing over views by taking a dot product between each source and
    reference feature, before summing over source views at each pixel location.
    The final tensor is size batch_size x num_depths x H x  W tensor.
    """

    def __init__(
        self,
        matching_height,
        matching_width,
        num_depth_bins=64,
        matching_dim_size=None,
        num_source_views=None,
    ):
        """
        matching_dim_size and num_source_views are not used for the standard
        cost volume.

        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            matching_dim_size: number of channels per visual feature; the basic
                dot product cost volume does not need this information at init.
            num_source_views: number of source views; the basic dot product cost
                volume does not need this information at init.
        """
        super().__init__()

        self.num_depth_bins = num_depth_bins
        self.matching_height = matching_height
        self.matching_width = matching_width

        self.initialise_for_projection()

    def initialise_for_projection(self, device=None):
        """
        Set up for backwarping and projection of feature maps

        Args:
            batch_height: height of the current batch of features
            batch_width: width of the current batch of features
        """

        linear_ramp = torch.linspace(0, 1, self.num_depth_bins).view(1, self.num_depth_bins, 1, 1)
        self.register_buffer("linear_ramp_1d11", linear_ramp)

        self.backprojector = BackprojectDepth(
            height=self.matching_height, width=self.matching_width
        )
        self.projector = Project3D()

        if device is not None:
            self.projector = self.projector.to(device)
            self.backprojector = self.backprojector.to(device)
            self.linear_ramp_1d11 = self.linear_ramp_1d11.to(device)

    def get_mask(self, pix_coords_bk2hw):
        """
        Create a mask to ignore features from the edges or outside of source
        images.

        Args:
            pix_coords_bk2hw: sampling locations of source features

        Returns:
            mask: a binary mask indicating whether to ignore a pixels
        """

        mask = torch.logical_and(
            torch.logical_and(
                pix_coords_bk2hw[:, :, 0] > 2, pix_coords_bk2hw[:, :, 0] < self.matching_width - 2
            ),
            torch.logical_and(
                pix_coords_bk2hw[:, :, 1] > 2, pix_coords_bk2hw[:, :, 1] < self.matching_height - 2
            ),
        )

        return mask

    def generate_depth_planes(
        self, batch_size: int, min_depth: Tensor, max_depth: Tensor
    ) -> Tensor:
        """
        Creates a depth planes tensor of size batch_size x number of depth planes
        x matching height x matching width. Every plane contains the same depths
        and depths will vary with a log scale from min_depth to max_depth.

        Args:
            batch_size: number of these view replications to make for each
                element in the batch.
            min_depth: minimum depth tensor defining the starting point for
                depth planes.
            max_depth: maximum depth tensor defining the end point for
                depth planes.

        Returns:
            depth_planes_bdhw: depth planes tensor.
        """
        linear_ramp_bd11 = self.linear_ramp_1d11.expand(
            batch_size,
            self.num_depth_bins,
            1,
            1,
        )
        log_depth_planes_bd11 = (
            torch.log(min_depth) + torch.log(max_depth / min_depth) * linear_ramp_bd11
        )
        depth_planes_bd11 = torch.exp(log_depth_planes_bd11)

        depth_planes_bdhw = depth_planes_bd11.expand(
            batch_size, self.num_depth_bins, self.matching_height, self.matching_width
        )

        return depth_planes_bdhw

    def warp_features(
        self,
        src_feats,
        src_extrinsics,
        src_Ks,
        cur_invK,
        depth_plane_b1hw,
        batch_size,
        num_src_frames,
        num_feat_channels,
        uv_scale,
    ):
        """
        Warps every soruce view feature to the current view at the depth
        plane defined by depth_plane_b1hw.

        Args:
            src_feats: source image matching features - B x num_src_frames x C x
                H x W where H and W should be self.matching_height and
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            depth_plane_b1hw: depth plane to use for every spatial location. For
                DoubleTake, this will be the same value at each location.
            batch_size: the batch size.
            num_src_frames: number of source views.
            num_feat_channels: number of feature channels for feature maps.
            uv_scale: normalization for image space coords before grid_sample.

        Returns:
            world_points_B4N: the world points at every backprojected depth
                point in depth_plane_b1hw.
            depths: depths for each projected point in every source views.
            src_feat_warped: warped source view for every spatial location at
                the depth plane.
            mask: depth mask where 1.0 indicated that the point projected to the
                source view is infront of the view.
        """

        # backproject points at that depth plane to the world, where the
        # world is really the current view.
        world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
        world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, dim=0)

        # project these points down to each source frame
        cam_points_B3N = self.projector(
            world_points_B4N, src_Ks.view(-1, 4, 4), src_extrinsics.view(-1, 4, 4)
        )

        cam_points_B3hw = cam_points_B3N.view(-1, 3, self.matching_height, self.matching_width)
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        depths = cam_points_B3hw[:, 2:]

        uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1

        src_feat_warped = F.grid_sample(
            input=src_feats.view(-1, num_feat_channels, self.matching_height, self.matching_width),
            grid=uv_coords.type_as(src_feats),
            padding_mode="zeros",
            mode="bilinear",
            align_corners=False,
        )

        # Reshape tensors to "unbatch"
        src_feat_warped = src_feat_warped.view(
            batch_size,
            num_src_frames,
            num_feat_channels,
            self.matching_height,
            self.matching_width,
        )

        depths = depths.view(
            batch_size,
            num_src_frames,
            self.matching_height,
            self.matching_width,
        )

        mask_b = depths > 0
        mask = mask_b.type_as(src_feat_warped)

        return world_points_B4N, depths, src_feat_warped, mask

    def build_cost_volume(
        self,
        cur_feats: Tensor,
        src_feats: Tensor,
        src_extrinsics: Tensor,
        src_poses: Tensor,
        src_Ks: Tensor,
        cur_invK: Tensor,
        min_depth: Tensor,
        max_depth: Tensor,
        depth_planes_bdhw: Tensor = None,
        return_mask: bool = False,
    ):
        """
        Build the cost volume. Using hypothesised depths, we backwarp src_feats
        onto cur_feats using known intrinsics and take the dot product.
        We sum the dot over all src_feats.

        Args:
            cur_feats: current image matching features - B x C x H x W where H
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x
                H x W where H and W should be self.matching_height and
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead
                of constructing one here.
            return_mask: should we return a mask for source view information
                w.r.t to the current image's view. When true overall_mask_bhw is
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor
                of size BxHxW where True indicates a there is some valid source
                view feature information that was used to match the current
                view's feature against.
        """

        del src_poses, return_mask

        batch_size, num_src_frames, num_feat_channels, _, _ = src_feats.shape

        uv_scale = torch.tensor(
            [1 / self.matching_width, 1 / self.matching_height],
            dtype=src_extrinsics.dtype,
            device=src_extrinsics.device,
        ).view(1, 1, 1, 2)

        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, min_depth, max_depth)

        # Intialize the cost volume and the counts
        all_dps = []

        # loop through depth planes
        for depth_id in range(self.num_depth_bins):
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            _, _, src_feat_warped, mask = self.warp_features(
                src_feats,
                src_extrinsics,
                src_Ks,
                cur_invK,
                depth_plane_b1hw,
                batch_size,
                num_src_frames,
                num_feat_channels,
                uv_scale,
            )

            # Compute the dot product between cur and src features
            dot_product_bkhw = (
                torch.sum(
                    src_feat_warped * cur_feats.unsqueeze(1),
                    dim=2,
                )
                * mask
            )

            # Sum over the frames
            dot_product_b1hw = dot_product_bkhw.sum(dim=1, keepdim=True)

            all_dps.append(dot_product_b1hw)

        cost_volume = torch.cat(all_dps, dim=1)

        return cost_volume, depth_planes_bdhw, None

    def indices_to_disparity(self, indices, depth_planes_bdhw):
        """Convert cost volume indices to 1/depth for visualisation"""
        depth = torch.gather(depth_planes_bdhw, dim=1, index=indices.unsqueeze(1)).squeeze(1)
        return depth

    def forward(
        self,
        cur_feats,
        src_feats,
        src_extrinsics,
        src_poses,
        src_Ks,
        cur_invK,
        min_depth,
        max_depth,
        depth_planes_bdhw=None,
        return_mask=False,
    ):
        """Runs the cost volume and gets the lowest cost result"""

        # change to portrait if we need to.
        if self.matching_height != cur_feats.shape[-2] or self.matching_width != cur_feats.shape[-1]:
            self.matching_width = cur_feats.shape[-1]
            self.matching_height = cur_feats.shape[-2]
            self.initialise_for_projection(device=cur_feats.device)

        cost_volume, depth_planes_bdhw, overall_mask_bhw = self.build_cost_volume(
            cur_feats=cur_feats,
            src_feats=src_feats,
            src_extrinsics=src_extrinsics,
            src_Ks=src_Ks,
            cur_invK=cur_invK,
            src_poses=src_poses,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_planes_bdhw=depth_planes_bdhw,
            return_mask=return_mask,
        )

        # for visualisation - ignore 0s in cost volume for minimum
        with torch.no_grad():
            lowest_cost = self.indices_to_disparity(
                torch.argmax(cost_volume.detach(), 1),
                depth_planes_bdhw,
            )

            cost_volume = torch.cat([
                torch.log(lowest_cost.unsqueeze(1)),
                cost_volume.detach().mean(dim=1, keepdim=True),
                cost_volume.detach().std(dim=1, keepdim=True),
            ], dim=1)

        return cost_volume, lowest_cost, depth_planes_bdhw, overall_mask_bhw,
    

class EfficientCostVolumeManager(CostVolumeManager):
    def __init__(self, matching_height, matching_width, num_depth_bins=64, matching_dim_size=None,
        num_source_views=None,):
        super().__init__(matching_height, matching_width, num_depth_bins=num_depth_bins, matching_dim_size=None,
        num_source_views=None,)

    def warp_features(
        self,
        src_feats,
        src_extrinsics,
        src_Ks,
        cur_invK,
        depth_plane_bkhw,
        num_src_frames,
        num_feat_channels,
        uv_scale,
    ):
        num_depth_planes = int(depth_plane_bkhw.shape[1])
        depth_plane_B1hw = einops.rearrange(depth_plane_bkhw, "b k h w -> (b k) 1 h w")
        world_points_B4N = self.backprojector(
            depth_plane_B1hw, einops.repeat(cur_invK, "b i j -> (b k) i j", k=num_depth_planes)
        )

        world_points_B4N = einops.repeat(
            world_points_B4N, "(b k) i N -> (b r k) i N", r=num_src_frames, k=num_depth_planes
        )

        # Warp features and reshape
        cam_points_B3N = self.projector(
            world_points_B4N,
            einops.repeat(src_Ks, "b r i j -> (b r k) i j", r=num_src_frames, k=num_depth_planes),
            einops.repeat(
                src_extrinsics, "b r i j -> (b r k) i j", r=num_src_frames, k=num_depth_planes
            ),
        )

        cam_points_B3hw = einops.rearrange(
            cam_points_B3N, "B i (h w) -> B i h w", h=self.matching_height, w=self.matching_width
        )
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        pix_coords_brk2hw = einops.rearrange(
            pix_coords_B2hw, "(b r k) i h w -> b r k i h w", r=num_src_frames, k=num_depth_planes
        )
        depths_B1hw = cam_points_B3hw[:, 2:]

        uv_coords = 2 * einops.rearrange(pix_coords_B2hw, "B i h w -> B h w i") * uv_scale - 1

        src_feats_Bfhw = einops.repeat(src_feats, "b r f h w -> (b r k) f h w", k=num_depth_planes)

        src_feat_warped_Bfhw = F.grid_sample(
            input=src_feats_Bfhw,
            grid=uv_coords.type_as(src_feats_Bfhw),
            padding_mode="zeros",
            mode="bilinear",
            align_corners=False,
        )

        # Reshape tensors to "unbatch"
        src_feat_warped_brkfhw = einops.rearrange(
            src_feat_warped_Bfhw,
            "(b r k) f h w -> b r k f h w",
            r=num_src_frames,
            k=num_depth_planes,
            f=num_feat_channels,
            h=self.matching_height,
            w=self.matching_width,
        )

        depths_brkhw = einops.rearrange(
            depths_B1hw, "(b r k) i h w -> b r (k i) h w", r=num_src_frames, k=num_depth_planes
        )

        # mask values landing outside the image and optionally near the border
        # mask_b = torch.logical_and(self.get_mask(pix_coords_bk2hw), depths > 0)
        mask_b = depths_brkhw > 0
        mask_brkhw = mask_b.type_as(src_feat_warped_brkfhw)

        world_points_brk4hw = einops.rearrange(
            world_points_B4N,
            "(b r k) i (h w) -> b r k i h w",
            r=num_src_frames,
            k=num_depth_planes,
            h=self.matching_height,
            w=self.matching_width,
        )

        return (
            world_points_brk4hw,
            depths_brkhw,
            src_feat_warped_brkfhw,
            mask_brkhw,
            pix_coords_brk2hw,
        )

    def build_cost_volume(
        self,
        cur_feats: Tensor,
        src_feats: Tensor,
        src_extrinsics: Tensor,
        src_poses: Tensor,
        src_Ks: Tensor,
        cur_invK: Tensor,
        min_depth: Tensor,
        max_depth: Tensor,
        depth_planes_bdhw: Tensor = None,
        return_mask: bool = False,
    ):
        """
        Build the cost volume. Using hypothesised depths, we backwarp src_feats onto cur_feats
        using known intrinsics and take the abs difference. We average costs over src_feats.

        :param cur_feats: input image features - B x C x H x W
        :param src_feats: srcerence image features - B x num_src_frames x C x H x W
        :param src_poses: srcerence image camera poses - B x num_src_frames x 4 x 4
        :param src_Ks: srcerence image inverse intrinsics - B x num_src_frames x 4 x 4
        :param cur_invK: current image intrinsics - B x 4 x 4
        """

        # to deal with different sized batches (e.g. landscape/portrait) set up projection
        # here, not in __init__
        num_src_frames, num_feat_channels, src_feat_height, src_feat_width = [
            int(n) for n in src_feats.shape[1:]
        ]
        # change to portrait if we need to.
        if self.matching_height != cur_feats.shape[-2] or self.matching_width != cur_feats.shape[-1]:
            self.matching_width = cur_feats.shape[-1]
            self.matching_height = cur_feats.shape[-2]
            self.initialise_for_projection(device=cur_feats.device)

        uv_scale = torch.tensor(
            [1 / self.matching_width, 1 / self.matching_height],
            dtype=src_poses.dtype,
            device=src_poses.device,
        ).view(1, 1, 1, 2)

        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(cur_feats.size(0), min_depth, max_depth)

        _, _, src_feat_warped_brkfhw, mask_brkhw, _ = self.warp_features(
            src_feats,
            src_extrinsics,
            src_Ks,
            cur_invK,
            depth_planes_bdhw,
            int(num_src_frames),
            int(num_feat_channels),
            uv_scale,
        )

        # Compute the dot product between cur and src features
        cur_feats_b1kfhw = einops.repeat(
            cur_feats, "b f h w -> b 1 k f h w", k=int(depth_planes_bdhw.shape[1])
        )
        dot_product_brkhw = torch.sum(src_feat_warped_brkfhw * cur_feats_b1kfhw, dim=3) * mask_brkhw

        # Sum over the frames
        dot_product_bkhw = dot_product_brkhw.sum(dim=1)

        return dot_product_bkhw, depth_planes_bdhw, None
