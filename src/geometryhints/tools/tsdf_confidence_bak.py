import torch
import torch.nn.functional as TF

from src.geometryhints.tools.tsdf import TSDFFuser


class TSDFConf(TSDFFuser):
    def confidence_to_log_odds(self, confidence: torch.Tensor) -> torch.Tensor:
        """Turn a confidence, expressed as a probability, as log odds"""
        EPS = torch.finfo(confidence.dtype).eps
        return torch.log((confidence + EPS) / (1 - confidence + EPS))

    def log_odds_to_confidence(self, log_odds: torch.Tensor) -> torch.Tensor:
        """Turn a log odds into a probability value"""
        return torch.sigmoid(log_odds)

    def integrate_depth(
        self,
        depth_b1hw,
        cam_T_world_T_b44,
        K_b44,
        depth_mask_b1hw=None,
    ):
        """
        Integrates depth maps into the volume. Supports batching.

        depth_b1hw: tensor with depth map
        cam_T_world_T_b44: camera extrinsics (not pose!).
        K_b44: camera intrinsics.
        cv_confidence_b1hw: confidence values from the cost volume
        depth_mask_b1hw: an optional boolean mask for valid depth points in
            the depth map.
        """
        img_h, img_w = depth_b1hw.shape[2:]
        img_size = torch.tensor([img_w, img_h], dtype=torch.float16).view(1, 1, 1, 2)
        if self.use_gpu:
            depth_b1hw = depth_b1hw.cuda()
            img_size = img_size.cuda()
            self.tsdf.cuda()

        # Project voxel coordinates into images
        cam_points_b3N = self.project_to_camera(cam_T_world_T_b44, K_b44)
        vox_depth_b1N = cam_points_b3N[:, 2:3]
        pixel_coords_b2N = cam_points_b3N[:, :2]

        # Reshape the projected voxel coords to a 2D view of shape Hx(WxD)
        pixel_coords_bhw2 = pixel_coords_b2N.view(
            -1, 2, self.shape[0], self.shape[1] * self.shape[2]
        ).permute(0, 2, 3, 1)
        pixel_coords_bhw2 = 2 * pixel_coords_bhw2 / img_size - 1

        if depth_mask_b1hw is not None:
            depth_b1hw = depth_b1hw.clone()
            depth_b1hw[~depth_mask_b1hw] = -1

        # Sample the depth using grid sample
        sampled_depth_b1hw = TF.grid_sample(
            input=depth_b1hw,
            grid=pixel_coords_bhw2,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_depth_b1N = sampled_depth_b1hw.flatten(start_dim=2)

        # Confidence from InfiniTAM
        tsdf_confidence_b1N = (
            torch.clamp(
                1.0 - (sampled_depth_b1N - self.min_depth) / (self.max_depth - self.min_depth),
                min=0.25,
                max=1.0,
            )
            ** 2
        )

        confidence_b1N = tsdf_confidence_b1N

        # Calculate TSDF values from depth difference by normalizing to [-1, 1]
        dist_b1N = sampled_depth_b1N - vox_depth_b1N
        tsdf_vals_b1N = torch.clamp(dist_b1N / self.truncation, min=-1.0, max=1.0)

        # Get the valid points mask
        valid_points_b1N = (
            (vox_depth_b1N > 0)
            & (dist_b1N > -self.truncation)
            & (sampled_depth_b1N > 0)
            & (vox_depth_b1N > 0)
            # & (vox_depth_b1N < self.max_depth | sampled_depth_b1N > vox_depth_b1N + 0.5)
            & (vox_depth_b1N < self.max_depth)
            & (confidence_b1N > 0)
        )

        # Updating the TSDF has to be sequential so we break out the batch here
        for tsdf_val_1N, valid_points_1N, confidence_1N in zip(
            tsdf_vals_b1N, valid_points_b1N, confidence_b1N
        ):
            # Reshape the valid mask to the TSDF's shape and read the old values
            valid_points_hwd = valid_points_1N.view(self.shape)
            old_tsdf_vals = self.tsdf_values[valid_points_hwd]
            old_weights = self.tsdf_weights[valid_points_hwd]

            # Fetch the new tsdf values and the confidence
            new_tsdf_vals = tsdf_val_1N[valid_points_1N]
            confidence = confidence_1N[valid_points_1N]

            update_rate = 2.5

            # Compute the new weight and the normalization factor
            new_weights = confidence * update_rate / self.maxW
            total_weights = old_weights + new_weights

            # Update the tsdf and the weights
            self.tsdf_values[valid_points_hwd] = (
                old_tsdf_vals * old_weights + new_tsdf_vals * new_weights
            ) / total_weights
            self.tsdf_weights[valid_points_hwd] = torch.clamp(total_weights, max=1.0)
