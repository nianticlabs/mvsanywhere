import torch
import torch.nn.functional as F
from torch import Tensor

from src.geometryhints.modules.cost_volume import CostVolumeManager
from src.geometryhints.modules.networks import MLP
from src.geometryhints.tools.keyframe_buffer import pose_distance
from src.geometryhints.utils.generic_utils import (
    combine_dims,
    tensor_B_to_bM,
    tensor_bM_to_B,
)
from src.geometryhints.utils.geometry_utils import get_camera_rays


class FeatureHintVolumeManager(CostVolumeManager):

    """
    Class to build a feature volume from extracted features of an input
    reference image and N source images.

    Achieved by backwarping source features onto current features using
    hypothesised depths between min_depth_bin and max_depth_bin, and then
    running an MLP on both visual features and each spatial and depth
    index's metadata. The final tensor is size
    batch_size x num_depths x H x  W tensor.

    """

    def __init__(
        self,
        matching_height,
        matching_width,
        num_depth_bins=64,
        mlp_channels=[202, 128, 128, 1],
        matching_dim_size=16,
        num_source_views=7,
    ):
        """
        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            mlp_channels: number of channels at every input/output of the MLP.
                mlp_channels[-1] defines the output size. mlp_channels[0] will
                be ignored and computed in this initialization function to
                account for all metadata.
            matching_dim_size: number of channels per visual feature.
            num_source_views: number of source views.
        """
        super().__init__(matching_height, matching_width, num_depth_bins)

        # compute dims for visual features and each metadata element
        num_visual_channels = matching_dim_size * (1 + num_source_views)
        num_depth_channels = 1 + num_source_views
        num_ray_channels = 3 * (1 + num_source_views)
        num_ray_angle_channels = num_source_views
        num_mask_channels = num_source_views
        num_num_dot_channels = num_source_views
        num_pose_penalty_channels = 3 * (num_source_views)

        # update mlp channels
        mlp_channels[0] = (
            num_visual_channels
            + num_depth_channels
            + num_ray_channels
            + num_ray_angle_channels
            + num_mask_channels
            + num_num_dot_channels
            + num_pose_penalty_channels
            + 1
        )

        # initialize the MLP
        self.mlp = MLP(channel_list=mlp_channels, disable_final_activation=True)

        # tell the world what's happening here.
        print(f"".center(80, "#"))
        print(f" Using FeatureVolumeManager ".center(80, "#"))
        print(f" Number of source views: ".ljust(30, " ") + f"{num_source_views}  ")
        print(f" Using all metadata.  ")
        print(f" Number of channels: ".ljust(30, " ") + f"{mlp_channels}  ")
        print(f"".center(80, "#"))
        print("")

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
        cv_depth_hint_dict: dict[str, Tensor],
        depth_planes_bdhw: Tensor = None,
        return_mask: bool = False,
    ):
        """
        Build the feature volume. Using hypothesised depths, we backwarp
        src_feats onto cur_feats using known intrinsics and run an MLP on both
        visual features and each pixel and depth plane's metadata.

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

        (
            batch_size,
            num_src_frames,
            num_feat_channels,
            src_feat_height,
            src_feat_width,
        ) = src_feats.shape

        uv_scale = torch.tensor(
            [1 / self.matching_width, 1 / self.matching_height],
            dtype=src_extrinsics.dtype,
            device=src_extrinsics.device,
        ).view(1, 1, 1, 2)

        # construct depth planes if need be.
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, min_depth, max_depth)

        # get poses distances
        frame_pose_dist_B, r_measure_B, t_measure_B = pose_distance(tensor_bM_to_B(src_poses))

        # shape all pose distance tensors.
        frame_pose_dist_bkhw = tensor_B_to_bM(
            frame_pose_dist_B,
            batch_size=batch_size,
            num_views=num_src_frames,
        )[:, :, None, None].expand(
            batch_size,
            num_src_frames,
            src_feat_height,
            src_feat_width,
        )

        r_measure_bkhw = tensor_B_to_bM(
            r_measure_B, batch_size=batch_size, num_views=num_src_frames
        )[:, :, None, None].expand(frame_pose_dist_bkhw.shape)

        t_measure_bkhw = tensor_B_to_bM(
            t_measure_B,
            batch_size=batch_size,
            num_views=num_src_frames,
        )[:, :, None, None].expand(frame_pose_dist_bkhw.shape)

        # init an overall mask if need be
        overall_mask_bhw = None
        if return_mask:
            overall_mask_bhw = torch.zeros(
                (batch_size, self.matching_height, self.matching_width),
                device=src_feats.device,
                dtype=torch.bool,
            )

        prev_hint_depth_b1hw = F.interpolate(
            cv_depth_hint_dict["prev_hint_depth_b1hw"],
            size=depth_planes_bdhw.shape[-2:],
            mode="nearest",
        )

        # building up a map of previous depth hints
        cv_depth_hints_b4N = self.backprojector(
            prev_hint_depth_b1hw,
            cv_depth_hint_dict[f"prev_hint_invK_s{1}_b44"],
        )

        # in the reference frame of the current view
        cv_depth_hints_b4N = (
            cv_depth_hint_dict[f"prev_hint_current_T_prev_b44"] @ cv_depth_hints_b4N
        )

        # TODO use current K
        cv_depth_hints_proj_b3N = self.projector(
            cv_depth_hints_b4N,
            cv_depth_hint_dict[f"prev_hint_K_s{1}_b44"],
            torch.eye(4, device=cv_depth_hints_b4N.device)
            .unsqueeze(0)
            .expand([cv_depth_hints_b4N.shape[0], 4, 4]),
        )

        cv_depth_hints_proj_b3N[0:2] = torch.round(cv_depth_hints_proj_b3N[0:2])

        # multiple points will land on the same pixel, so we'll argsort using
        # min depth w.r.t the current view. This will give us the closest point
        # on the point cloud to the current view when we perform the assignment
        # using multiple points.
        # sample_map = torch.zeros(10,10)
        # one takes precedence over two.
        # sample_map[[5,5],[4,4]] = torch.tensor([2.,1.])
        # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # yields:
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        # need the closest depth at the end, so use descending=True

        def get_sorted_world_points_debug(offset_cv_depth_hints_proj_bN3, depth_cv_hints_proj_bN3):
            batch_size = offset_cv_depth_hints_proj_bN3.shape[0]
            number_of_points = offset_cv_depth_hints_proj_bN3.shape[1]
            num_dims = offset_cv_depth_hints_proj_bN3.shape[2]
            device = offset_cv_depth_hints_proj_bN3.device

            sort_indices_bN = torch.argsort(
                offset_cv_depth_hints_proj_bN3[:, :, 2], dim=1, descending=True
            )
            sort_indices_bN3 = sort_indices_bN.unsqueeze(-1).expand(
                batch_size, number_of_points, num_dims
            )

            batch_indices = (
                torch.arange(batch_size, device=device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(batch_size, number_of_points, num_dims)
            )
            dims_indices = torch.arange(num_dims, device=device).expand(
                batch_size, number_of_points, num_dims
            )

            return (
                offset_cv_depth_hints_proj_bN3[batch_indices, sort_indices_bN3, dims_indices],
                depth_cv_hints_proj_bN3[batch_indices, sort_indices_bN3, dims_indices],
            )

        def get_sorted_world_points(offset_cv_depth_hints_proj_bN3):
            batch_size = offset_cv_depth_hints_proj_bN3.shape[0]
            number_of_points = offset_cv_depth_hints_proj_bN3.shape[1]
            num_dims = offset_cv_depth_hints_proj_bN3.shape[2]
            device = offset_cv_depth_hints_proj_bN3.device

            sort_indices_bN = torch.argsort(
                offset_cv_depth_hints_proj_bN3[:, :, 2], dim=1, descending=True
            )
            sort_indices_bN3 = sort_indices_bN.unsqueeze(-1).expand(
                batch_size, number_of_points, num_dims
            )

            batch_indices = (
                torch.arange(batch_size, device=device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(batch_size, number_of_points, num_dims)
            )
            dims_indices = torch.arange(num_dims, device=device).expand(
                batch_size, number_of_points, num_dims
            )

            return offset_cv_depth_hints_proj_bN3[batch_indices, sort_indices_bN3, dims_indices]

        def index_into_2D_map(offset_cv_depth_hints_proj_bN3, current_hint_map_bhw):
            batch_size = current_hint_map_bhw.shape[0]
            height = current_hint_map_bhw.shape[1]
            width = current_hint_map_bhw.shape[2]
            device = offset_cv_depth_hints_proj_bN3.device
            number_of_points = offset_cv_depth_hints_proj_bN3.shape[1]

            current_hint_map_bhw = torch.nn.functional.pad(
                current_hint_map_bhw, (0, 1, 0, 1, 0, 0), mode="constant", value=-1
            )

            x_indices_bN = offset_cv_depth_hints_proj_bN3[:, :, 0].long()
            x_indices_bN[x_indices_bN < 0] = width
            x_indices_bN[x_indices_bN >= width] = width
            y_indices_bN = offset_cv_depth_hints_proj_bN3[:, :, 1].long()
            y_indices_bN[y_indices_bN < 0] = height
            y_indices_bN[y_indices_bN >= height] = height

            batch_indices = (
                torch.arange(batch_size, device=device)
                .unsqueeze(-1)
                .expand(batch_size, number_of_points)
            )

            current_hint_map_bhw[
                batch_indices, y_indices_bN, x_indices_bN
            ] = offset_cv_depth_hints_proj_bN3[:, :, 2]

            return current_hint_map_bhw[:, 0:-1, 0:-1]

        all_dps = []
        # Intialize the cost volume and the countsx
        # loop through depth planes
        for depth_id in range(self.num_depth_bins):
            # current depth plane
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)

            offset_cv_depth_hints_proj_bN3 = cv_depth_hints_proj_b3N.clone().permute(0, 2, 1)
            # depth_cv_hints_proj_bN3 = offset_cv_depth_hints_proj_bN3.clone()
            offset_cv_depth_hints_proj_bN3[:, :, 2] = torch.abs(
                offset_cv_depth_hints_proj_bN3[:, :, 2] - depth_plane_b1hw[:, 0, 0, 0].unsqueeze(1)
            )
            # offset_cv_depth_hints_proj_bN3, depth_cv_hints_proj_bN3 = get_sorted_world_points_debug(offset_cv_depth_hints_proj_bN3, depth_cv_hints_proj_bN3)
            offset_cv_depth_hints_proj_bN3 = get_sorted_world_points(offset_cv_depth_hints_proj_bN3)
            offset_cv_depth_hints_proj_bN3[torch.isnan(offset_cv_depth_hints_proj_bN3)] = -1

            current_hint_map_bhw = torch.ones_like(depth_plane_b1hw).squeeze(1) * -1
            current_hint_map_b1hw = index_into_2D_map(
                offset_cv_depth_hints_proj_bN3, current_hint_map_bhw
            ).unsqueeze(1)

            # current_hint_map_bhw = torch.ones_like(depth_plane_b1hw).squeeze(1) * -1
            # current_depth_hint_map_b1hw = index_into_2D_map(depth_cv_hints_proj_bN3, current_hint_map_bhw).unsqueeze(1)

            # backproject points at that depth plane to the world, where the
            # world is really the current view.
            world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
            world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, dim=0)

            # project those points down to each source view.
            cam_points_B3N = self.projector(
                world_points_B4N, src_Ks.view(-1, 4, 4), src_extrinsics.view(-1, 4, 4)
            )

            cam_points_B3hw = cam_points_B3N.view(
                -1,
                3,
                self.matching_height,
                self.matching_width,
            )

            # now sample source views at those projected points using
            # grid_sample
            pix_coords_B2hw = cam_points_B3hw[:, :2]
            depths = cam_points_B3hw[:, 2:]

            uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1

            # pad with zeros to bake in bounds protection when matching.
            src_feat_warped = F.grid_sample(
                input=src_feats.view(
                    -1, num_feat_channels, self.matching_height, self.matching_width
                ),
                grid=uv_coords.type_as(src_feats),
                padding_mode="zeros",
                mode="bilinear",
                align_corners=False,
            )

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

            # mask for depth validity for each image. This will be False when
            # a point in world_points_b4N is behind a source view.
            # We don't need to worry about including a pixel bounds mask as part
            # of the mlp since we're padding with zeros in grid_sample.
            mask_b = depths > 0
            mask = mask_b.type_as(src_feat_warped)

            if return_mask:
                # build a mask using depth validity and pixel coordinate
                # validity by checking bounds of source views.
                depth_mask = torch.any(mask_b, dim=1)
                pix_coords_bk2hw = pix_coords_B2hw.view(
                    batch_size,
                    num_src_frames,
                    2,
                    self.matching_height,
                    self.matching_width,
                )
                bounds_mask = torch.any(self.get_mask(pix_coords_bk2hw), dim=1)
                overall_mask_bhw = torch.logical_and(depth_mask, bounds_mask)

            # compute rays to world points for current frame
            cur_points_rays_B3hw = F.normalize(world_points_B4N[:, :3, :], dim=1).view(
                -1,
                3,
                self.matching_height,
                self.matching_width,
            )

            cur_points_rays_bk3hw = tensor_B_to_bM(
                cur_points_rays_B3hw, batch_size=batch_size, num_views=num_src_frames
            )

            # compute rays for world points source frame
            src_poses_B44 = tensor_bM_to_B(src_poses)
            src_points_rays_B3hw = get_camera_rays(
                src_poses_B44, world_points_B4N[:, :3, :], in_camera_frame=False
            ).view(
                -1,
                3,
                self.matching_height,
                self.matching_width,
            )

            src_points_rays_bk3hw = tensor_B_to_bM(
                src_points_rays_B3hw,
                batch_size=batch_size,
                num_views=num_src_frames,
            )

            # combine current and source rays
            all_rays_bchw = combine_dims(
                torch.cat(
                    [cur_points_rays_bk3hw[:, 0, :, :, :][:, None, :, :, :], src_points_rays_bk3hw],
                    dim=1,
                ),
                1,
                3,
            )

            # compute angle difference between rays (dot product)
            ray_angle_bkhw = F.cosine_similarity(
                cur_points_rays_bk3hw, src_points_rays_bk3hw, dim=2, eps=1e-5
            )

            # Compute the dot product between cur and src features
            dot_product_bkhw = (
                torch.sum(
                    src_feat_warped * cur_feats.unsqueeze(1),
                    dim=2,
                )
                * mask
            )

            # combine all visual features from across all images
            combined_visual_features_bchw = combine_dims(
                torch.cat(
                    [src_feat_warped, cur_feats.unsqueeze(1)],
                    dim=1,
                ),
                1,
                3,
            )

            # concat all input visual and metadata features.
            mlp_input_features_bchw = torch.cat(
                [
                    combined_visual_features_bchw,
                    mask,
                    depths,
                    depth_plane_b1hw,
                    dot_product_bkhw,
                    ray_angle_bkhw,
                    all_rays_bchw,
                    frame_pose_dist_bkhw,
                    r_measure_bkhw,
                    t_measure_bkhw,
                    current_hint_map_b1hw,
                ],
                dim=1,
            )

            # run through the MLP!
            mlp_input_features_bhwc = mlp_input_features_bchw.permute(0, 2, 3, 1)
            feature_b1hw = self.mlp(mlp_input_features_bhwc).squeeze(-1).unsqueeze(1)

            # append MLP output to the final cost volume output.
            all_dps.append(feature_b1hw)

        feature_volume = torch.cat(all_dps, dim=1)

        return feature_volume, depth_planes_bdhw, overall_mask_bhw

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
        cv_depth_hint_dict,
        depth_planes_bdhw=None,
        return_mask=False,
    ):
        """Runs the cost volume and gets the lowest cost result"""
        cost_volume, depth_planes_bdhw, overall_mask_bhw = self.build_cost_volume(
            cur_feats=cur_feats,
            src_feats=src_feats,
            src_extrinsics=src_extrinsics,
            src_Ks=src_Ks,
            cur_invK=cur_invK,
            src_poses=src_poses,
            min_depth=min_depth,
            max_depth=max_depth,
            cv_depth_hint_dict=cv_depth_hint_dict,
            depth_planes_bdhw=depth_planes_bdhw,
            return_mask=return_mask,
        )

        # for visualisation - ignore 0s in cost volume for minimum
        with torch.no_grad():
            lowest_cost = self.indices_to_disparity(
                torch.argmax(cost_volume.detach(), 1),
                depth_planes_bdhw,
            )

        return cost_volume, lowest_cost, depth_planes_bdhw, overall_mask_bhw
