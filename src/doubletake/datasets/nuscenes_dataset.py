import os
from pathlib import Path

import numpy as np
import torch
# from nuscenes.nuscenes import NuScenes
# from pyquaternion import Quaternion

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset


class NuScenesDataset(GenericMVSDataset):
    """
    MVS nuScenes Dataset class.

    Inherits from GenericMVSDataset and implements methods specific to the nuScenes dataset.
    This dataset class handles the loading of images, depth maps, intrinsics,
    and poses for the nuScenes dataset, including the generated depth maps.
    """

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix="_tuples.txt",
        min_valid_depth=1e-3,
        max_valid_depth=150.0,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
        high_res_image_width=640,
        high_res_image_height=480,
        image_depth_ratio=2,
        shuffle_tuple=False,
        include_full_depth_K=False,
        include_high_res_color=False,
        pass_frame_id=False,
        skip_frames=None,
        skip_to_frame=None,
        verbose_init=False,
        disable_flip=False,
        rotate_images=False,
        matching_scale=0.25,
        prediction_scale=0.5,
        prediction_num_scales=5,
    ):
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            mv_tuple_file_suffix=mv_tuple_file_suffix,
            include_full_res_depth=include_full_res_depth,
            limit_to_scan_id=limit_to_scan_id,
            num_images_in_tuple=num_images_in_tuple,
            tuple_info_file_location=tuple_info_file_location,
            image_height=image_height,
            image_width=image_width,
            high_res_image_width=high_res_image_width,
            high_res_image_height=high_res_image_height,
            image_depth_ratio=image_depth_ratio,
            shuffle_tuple=shuffle_tuple,
            include_full_depth_K=include_full_depth_K,
            include_high_res_color=include_high_res_color,
            pass_frame_id=pass_frame_id,
            skip_frames=skip_frames,
            skip_to_frame=skip_to_frame,
            verbose_init=verbose_init,
            disable_flip=disable_flip,
            matching_scale=matching_scale,
            prediction_scale=prediction_scale,
            prediction_num_scales=prediction_num_scales,
        )

        # Initialize nuScenes dataset
        self.version = "v1.0-trainval"
        self.nusc = NuScenes(version=self.version, dataroot=self.dataset_path, verbose=verbose_init)
        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        split_filename = f"{self.split}{mv_tuple_file_suffix}"
        tuples_file = Path(tuple_info_file_location) / split_filename

        if not tuples_file.exists():
            raise FileNotFoundError(f"Tuples file not found at {tuples_file}")

        with open(tuples_file, "r") as f:
            self.tuples = [line.strip().split() for line in f]

        # Build a mapping from filenames to sample_data_tokens
        self.filename_to_sample_data_token = {
            sd["filename"]: sd["token"] for sd in self.nusc.sample_data
        }

        # Compute matching and depth dimensions
        self.matching_width = int(image_width * matching_scale)
        self.matching_height = int(image_height * matching_scale)
        self.depth_width = int(image_width * prediction_scale)
        self.depth_height = int(image_height * prediction_scale)

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame within the scan."""
        return frame_id

    def get_color_filepath(self, scan_id, frame_id):
        """Returns the filepath for a frame's color file at the dataset's configured RGB resolution."""
        return os.path.join(self.dataset_path, frame_id)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """Returns the filepath for a frame's higher resolution color file."""
        return self.get_color_filepath(scan_id, frame_id)

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """Returns the filepath for a frame's depth file at the native resolution."""
        depth_filename = frame_id.replace(".jpg", ".npy")
        return os.path.join(self.dataset_path, "depth", depth_filename)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics and computes scaled intrinsics matrices for a frame at multiple scales."""
        sample_data_token = self.get_sample_data_token_from_filename(frame_id)
        cam_data = self.nusc.get("sample_data", sample_data_token)
        calib_sensor = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        intrinsics = np.array(calib_sensor["camera_intrinsic"])

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(intrinsics, dtype=torch.float32)

        width_pixels = cam_data["width"]
        height_pixels = cam_data["height"]

        top, left, h, w = self.random_resize_crop.get_params(
            torch.empty((height_pixels, width_pixels)),
            self.random_resize_crop.scale,
            self.random_resize_crop.ratio
        )
        K[0, 2] = K[0, 2] - left
        K[1, 2] = K[1, 2] - top
        width_pixels = w
        height_pixels = h

        if flip:
            K[0, 2] = float(width_pixels) - K[0, 2]

        output_dict = {}

        if self.include_full_depth_K:
            output_dict["K_full_depth_b44"] = K.clone()
            output_dict["invK_full_depth_b44"] = torch.linalg.inv(K)

        # Compute matching intrinsics
        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.linalg.inv(K_matching)

        # Scale intrinsics to the dataset's configured depth resolution
        K[0] *= self.depth_width / float(width_pixels)
        K[1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = torch.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, (left, top, left+width_pixels, top+height_pixels) 

    def _load_depth(self, depth_path, is_float16=True):
        """Loads the depth map from a .npy file."""
        return np.load(depth_path)

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides."""
        full_res_depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        full_res_depth = self._load_depth(full_res_depth_filepath)

        mask_b = torch.tensor(full_res_depth > self.min_valid_depth).bool().unsqueeze(0)
        full_res_depth = torch.tensor(full_res_depth).float().unsqueeze(0)

        # Get the float valid mask
        full_res_mask = mask_b.float()

        # Set invalids to NaN
        full_res_depth[~mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, mask_b

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for."""
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        depth_orig = self._load_depth(depth_filepath)

        if crop:
            depth_orig = depth_orig[crop[1] : crop[3], crop[0] : crop[2]]

        # Use the downscaling method
        depth = self.downscale_depth_preserve_values(
            depth_orig, self.depth_height, self.depth_width
        )

        mask_b = torch.tensor(depth > self.min_valid_depth).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

        mask = mask_b.float()

        # Set invalids to NaN
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose.

        Args:
            scan_id: the scan this file belongs to (not used here).
            frame_id: id for the frame.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera frame to the world frame (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world frame to the camera frame (extrinsics).
        """
        sample_data_token = self.get_sample_data_token_from_filename(frame_id)
        cam_data = self.nusc.get("sample_data", sample_data_token)

        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        # Transformation from ego vehicle frame to world frame (ego_pose)
        ego_rotation = Quaternion(ego_pose["rotation"])
        ego_translation = np.array(ego_pose["translation"]).reshape(3, 1)
        ego_to_world = np.vstack(
            [
                np.hstack((ego_rotation.rotation_matrix, ego_translation)),
                np.array([0, 0, 0, 1]),
            ]
        )

        # Transformation from camera frame to ego vehicle frame
        sensor_sample = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_rotation = Quaternion(sensor_sample["rotation"])
        cam_translation = np.array(sensor_sample["translation"]).reshape(3, 1)
        cam_to_ego = np.vstack(
            [
                np.hstack((cam_rotation.rotation_matrix, cam_translation)),
                np.array([0, 0, 0, 1]),
            ]
        )

        # Transformation from camera frame to world frame
        world_T_cam = ego_to_world @ cam_to_ego

        world_T_cam = world_T_cam.astype(np.float32)

        # Transformation from world frame to camera frame (inverse of world_T_cam)
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def get_sample_data_token_from_filename(self, filename):
        """Extracts the sample_data_token from the filename."""
        sample_data_token = self.filename_to_sample_data_token.get(filename)
        if sample_data_token is None:
            raise ValueError(f"Sample data token not found for filename: {filename}")
        return sample_data_token

    @staticmethod
    def downscale_depth_preserve_values(depth_orig, target_height, target_width):
        """
        Downscale a sparse depth image by mapping non-zero values to the downscaled image
        to avoid losing values.

        Parameters:
        - depth_orig: Original high-resolution depth image (numpy array).
        - target_height: Desired height of the downscaled image.
        - target_width: Desired width of the downscaled image.

        Returns:
        - depth_downscaled: Downscaled depth image with preserved depth values.
        """
        orig_height, orig_width = depth_orig.shape
        depth_downscaled = np.zeros((target_height, target_width), dtype=depth_orig.dtype)

        # Find indices of non-zero depth values
        y_idxs, x_idxs = np.nonzero(depth_orig > 0)
        depth_values = depth_orig[y_idxs, x_idxs]

        x_scale = target_width / orig_width
        y_scale = target_height / orig_height

        # Map coordinates to downscaled image
        x_idxs_down = (x_idxs * x_scale).astype(int)
        y_idxs_down = (y_idxs * y_scale).astype(int)

        x_idxs_down = np.clip(x_idxs_down, 0, target_width - 1)
        y_idxs_down = np.clip(y_idxs_down, 0, target_height - 1)

        # Handle duplicates: Keep the smallest depth value
        for x, y, depth in zip(x_idxs_down, y_idxs_down, depth_values):
            if depth_downscaled[y, x] == 0 or depth < depth_downscaled[y, x]:
                depth_downscaled[y, x] = depth

        return depth_downscaled
