import os
from pathlib import Path


import numpy as np
import torch

from pyquaternion import Quaternion


from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from nuscenes.nuscenes import NuScenes


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
            mv_tuple_file_suffix='_tuples.txt',
            min_valid_depth=1e-3,
            max_valid_depth=1e2,
            include_full_res_depth=False,
            limit_to_scan_id=None,
            num_images_in_tuple=None,
            tuple_info_file_location=None,
            image_height=900,
            image_width=1600,
            high_res_image_width=1600,
            high_res_image_height=900,
            image_depth_ratio=2,
            shuffle_tuple=False,
            include_full_depth_K=False,
            include_high_res_color=False,
            pass_frame_id=False,
            skip_frames=None,
            skip_to_frame=None,
            verbose_init=True,
            disable_flip=True,
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
        self.version = 'v1.0-trainval'
        self.nusc = NuScenes(version=self.version, dataroot=self.dataset_path, verbose=verbose_init)
        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        split_filename = f"{self.split}{mv_tuple_file_suffix}"
        tuples_file = Path(tuple_info_file_location) / split_filename

        if not tuples_file.exists():
            raise FileNotFoundError(f"Tuples file not found at {tuples_file}")

        with open(tuples_file, 'r') as f:
            self.tuples = [line.strip().split() for line in f]

        # Build a mapping from filenames to sample_data_tokens
        self.filename_to_sample_data_token = {
            sd['filename']: sd['token'] for sd in self.nusc.sample_data
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
        depth_filename = frame_id.replace('.jpg', '.npy')
        return os.path.join(self.dataset_path, 'depth', depth_filename)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics and computes scaled intrinsics matrices for a frame at multiple scales."""
        sample_data_token = self.get_sample_data_token_from_filename(frame_id)
        cam_data = self.nusc.get('sample_data', sample_data_token)
        calib_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsics = np.array(calib_sensor['camera_intrinsic'])

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(intrinsics, dtype=torch.float32)

        width_pixels = cam_data['width']
        height_pixels = cam_data['height']

        if flip:
            K[0, 2] = float(width_pixels) - K[0, 2]

        output_dict = {}

        if self.include_full_depth_K:
            output_dict["K_full_depth_b44"] = K.clone()
            output_dict["invK_full_depth_b44"] = torch.inverse(K)

        # Compute matching intrinsics
        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.inverse(K_matching)

        # Scale intrinsics to the dataset's configured depth resolution
        K[0] *= self.depth_width / float(width_pixels)
        K[1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = torch.inverse(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, None

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
            depth_orig = depth_orig[crop[1]:crop[3], crop[0]:crop[2]]

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
        cam_data = self.nusc.get('sample_data', sample_data_token)

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
        sensor_sample = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
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

# import cv2
#
# from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
# from nuscenes.nuscenes import NuScenes
# from pyquaternion import Quaternion
#
#
# class NuScenesDataset(GenericMVSDataset):
#     """
#     MVS nuScenes Dataset class.
#
#     Inherits from GenericMVSDataset and implements methods specific to the nuScenes dataset.
#
#     This dataset class handles the loading of images, depth maps, intrinsics,
#     and poses for the nuScenes dataset, including the generated depth maps.
#     """
#
#     def __init__(
#             self,
#             dataset_path,
#             split,
#             mv_tuple_file_suffix,
#             min_valid_depth=1e-3,
#             max_valid_depth=1e2,
#             include_full_res_depth=False,
#             limit_to_scan_id=None,
#             num_images_in_tuple=None,
#             tuple_info_file_location=None,
#             image_height=900,
#             image_width=1600,
#             high_res_image_width=1600,
#             high_res_image_height=900,
#             image_depth_ratio=2,
#             shuffle_tuple=False,
#             include_full_depth_K=False,
#             include_high_res_color=False,
#             pass_frame_id=False,
#             skip_frames=None,
#             skip_to_frame=None,
#             verbose_init=True,
#             disable_flip=True,
#             rotate_images=False,
#             matching_scale=0.25,
#             prediction_scale=0.5,
#             prediction_num_scales=5,
#     ):
#         super().__init__(
#             dataset_path=dataset_path,
#             split=split,
#             mv_tuple_file_suffix=mv_tuple_file_suffix,
#             include_full_res_depth=include_full_res_depth,
#             limit_to_scan_id=limit_to_scan_id,
#             num_images_in_tuple=num_images_in_tuple,
#             tuple_info_file_location=tuple_info_file_location,
#             image_height=image_height,
#             image_width=image_width,
#             high_res_image_width=high_res_image_width,
#             high_res_image_height=high_res_image_height,
#             image_depth_ratio=image_depth_ratio,
#             shuffle_tuple=shuffle_tuple,
#             include_full_depth_K=include_full_depth_K,
#             include_high_res_color=include_high_res_color,
#             pass_frame_id=pass_frame_id,
#             skip_frames=skip_frames,
#             skip_to_frame=skip_to_frame,
#             verbose_init=verbose_init,
#             disable_flip=disable_flip,
#             matching_scale=matching_scale,
#             prediction_scale=prediction_scale,
#             prediction_num_scales=prediction_num_scales,
#         )
#
#         # Initialize nuScenes dataset
#         self.version = 'v1.0-trainval'
#         self.nusc = NuScenes(version=self.version, dataroot=self.dataset_path, verbose=verbose_init)
#         self.min_valid_depth = min_valid_depth
#         self.max_valid_depth = max_valid_depth
#
#         split_filename = f"{self.split}{mv_tuple_file_suffix}"
#         tuples_file = Path(tuple_info_file_location) / split_filename
#
#         if not tuples_file.exists():
#             raise FileNotFoundError(f"Tuples file not found at {tuples_file}")
#
#         with open(tuples_file, 'r') as f:
#             self.tuples = [line.strip().split() for line in f]
#
#         # Build a mapping from filenames to sample_data_tokens
#         self.filename_to_sample_data_token = {}
#         for sample_data in self.nusc.sample_data:
#             filename_sd = sample_data['filename']
#             self.filename_to_sample_data_token[filename_sd] = sample_data['token']
#
#     def get_frame_id_string(self, frame_id):
#         """Returns an id string for this frame_id that's unique to this frame within the scan."""
#         return frame_id
#
#     def get_color_filepath(self, scan_id, frame_id):
#         """Returns the filepath for a frame's color file at the dataset's configured RGB resolution."""
#         return os.path.join(self.dataset_path, frame_id)
#
#     def get_high_res_color_filepath(self, scan_id, frame_id):
#         """Returns the filepath for a frame's higher resolution color file."""
#         return self.get_color_filepath(scan_id, frame_id)
#
#     def get_full_res_depth_filepath(self, scan_id, frame_id):
#         """Returns the filepath for a frame's depth file at the native resolution."""
#         depth_filename = frame_id.replace('.jpg', '.npy')
#
#         return os.path.join(self.dataset_path, 'depth', depth_filename)
#
#     def load_intrinsics(self, scan_id, frame_id=None, flip=False):
#         """Loads intrinsics and computes scaled intrinsics matrices for a frame at multiple scales."""
#         # Get the sample_data_token from the filename
#         sample_data_token = self.get_sample_data_token_from_filename(frame_id)
#         cam_data = self.nusc.get('sample_data', sample_data_token)
#         calib_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
#         intrinsics = np.array(calib_sensor['camera_intrinsic'])
#
#         K = torch.eye(4, dtype=torch.float32)
#         K[:3, :3] = torch.tensor(intrinsics, dtype=torch.float32)
#
#         width_pixels = cam_data['width']
#         height_pixels = cam_data['height']
#
#         if flip:
#             K[0, 2] = float(width_pixels) - K[0, 2]
#
#         output_dict = {}
#
#         if self.include_full_depth_K:
#             output_dict[f"K_full_depth_b44"] = K.clone()
#             output_dict[f"invK_full_depth_b44"] = torch.inverse(K)
#
#         K_matching = K.clone()
#         K_matching[0] *= self.matching_width / float(width_pixels)
#         K_matching[1] *= self.matching_height / float(height_pixels)
#         output_dict["K_matching_b44"] = K_matching
#         output_dict["invK_matching_b44"] = torch.inverse(K_matching)
#
#         # Scale intrinsics to the dataset's configured depth resolution.
#         K[0] *= self.depth_width / float(width_pixels)
#         K[1] *= self.depth_height / float(height_pixels)
#
#         # Get the intrinsics of all scales at various resolutions.
#         for i in range(self.prediction_num_scales):
#             K_scaled = K.clone()
#             K_scaled[:2] /= 2 ** i
#             invK_scaled = torch.inverse(K_scaled)
#             output_dict[f"K_s{i}_b44"] = K_scaled
#             output_dict[f"invK_s{i}_b44"] = invK_scaled
#
#         return output_dict, None
#
#     def _load_depth(self, depth_path, is_float16=True):
#         """Loads the depth map from a .npy file."""
#         depth = np.load(depth_path)
#         return depth
#
#     def load_full_res_depth_and_mask(self, scan_id, frame_id):
#         """Loads a depth map at the native resolution the dataset provides."""
#         full_res_depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
#         full_res_depth = self._load_depth(full_res_depth_filepath)
#
#         mask_b = torch.tensor(full_res_depth > self.min_valid_depth).bool().unsqueeze(0)
#         full_res_depth = torch.tensor(full_res_depth).float().unsqueeze(0)
#
#         # Get the float valid mask
#         full_res_mask = mask_b.float()
#
#         # Set invalids to NaN
#         full_res_depth[~mask_b] = torch.tensor(np.nan)
#
#         return full_res_depth, full_res_mask, mask_b  # def load_full_res_depth_and_mask(self, scan_id, frame_id):
#         """Loads a depth map at the native resolution the dataset provides."""
#         full_res_depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
#         full_res_depth = self._load_depth(full_res_depth_filepath)
#
#         mask_b = torch.tensor(full_res_depth > self.min_valid_depth).bool().unsqueeze(0)
#         full_res_depth = torch.tensor(full_res_depth).float().unsqueeze(0)
#
#         # Get the float valid mask
#         full_res_mask = mask_b.float()
#
#         # Set invalids to NaN
#         full_res_depth[~mask_b] = torch.tensor(np.nan)
#
#         return full_res_depth, full_res_mask, mask_b
#
#     def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
#         """Loads a depth map at the resolution the dataset is configured for."""
#         depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
#         depth_orig = self._load_depth(depth_filepath)
#
#         if crop:
#             depth_orig = depth_orig[crop[1]:crop[3], crop[0]:crop[2]]
#
#         # Use the new downscaling method
#         depth = self.downscale_depth_preserve_values(
#             depth_orig, self.depth_height, self.depth_width
#         )
#
#         mask_b = torch.tensor(depth > self.min_valid_depth).bool().unsqueeze(0)
#         depth = torch.tensor(depth).float().unsqueeze(0)
#
#         mask = mask_b.float()
#
#         # Set invalids to NaN
#         depth[~mask_b] = torch.tensor(np.nan)
#
#         return depth, mask, mask_b
#
#     def load_pose(self, scan_id, frame_id):
#         """Loads a frame's pose.
#
#         Args:
#             scan_id: the scan this file belongs to (not used here).
#             frame_id: id for the frame.
#
#         Returns:
#             world_T_cam (numpy array): matrix for transforming from the
#                 camera frame to the world frame (pose).
#             cam_T_world (numpy array): matrix for transforming from the
#                 world frame to the camera frame (extrinsics).
#         """
#
#         sample_data_token = self.get_sample_data_token_from_filename(frame_id)
#         cam_data = self.nusc.get('sample_data', sample_data_token)
#
#         ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
#
#         # Transformation from ego vehicle frame to world frame (ego_pose)
#         ego_rotation = Quaternion(ego_pose["rotation"])
#         ego_translation = np.array(ego_pose["translation"]).reshape(3, 1)
#         ego_to_world = np.vstack(
#             [
#                 np.hstack((ego_rotation.rotation_matrix, ego_translation)),
#                 np.array([0, 0, 0, 1]),
#             ]
#         )
#
#         # Transformation from camera frame to ego vehicle frame
#         sensor_sample = self.nusc.get(
#             "calibrated_sensor", cam_data["calibrated_sensor_token"]
#         )
#         cam_rotation = Quaternion(sensor_sample["rotation"])
#         cam_translation = np.array(sensor_sample["translation"]).reshape(3, 1)
#         cam_to_ego = np.vstack(
#             [
#                 np.hstack((cam_rotation.rotation_matrix, cam_translation)),
#                 np.array([0, 0, 0, 1]),
#             ]
#         )
#
#         # Transformation from camera frame to world frame
#         world_T_cam = ego_to_world @ cam_to_ego
#
#         # Transformation from world frame to camera frame (inverse of world_T_cam)
#         cam_T_world = np.linalg.inv(world_T_cam)
#
#         return world_T_cam, cam_T_world
#
#     def get_sample_data_token_from_filename(self, filename):
#         """Extracts the sample_data_token from the filename."""
#         sample_data_token = self.filename_to_sample_data_token.get(filename)
#         if sample_data_token is None:
#             raise ValueError(f"Sample data token not found for filename: {filename}")
#         return sample_data_token
#
#     @staticmethod
#     def downscale_depth_preserve_values(depth_orig, target_height, target_width):
#         """
#         Downscale a sparse depth image by mapping non-zero values to the downscaled image
#         to avoid losing values.
#
#         Parameters:
#         - depth_orig: Original high-resolution depth image (numpy array).
#         - target_height: Desired height of the downscaled image.
#         - target_width: Desired width of the downscaled image.
#
#         Returns:
#         - depth_downscaled: Downscaled depth image with preserved depth values.
#         """
#
#         orig_height, orig_width = depth_orig.shape
#         depth_downscaled = np.zeros((target_height, target_width),
#                                     dtype=depth_orig.dtype)
#
#         # Find indices of non-zero depth values
#         y_idxs, x_idxs = np.nonzero(depth_orig > 0)
#         depth_values = depth_orig[y_idxs, x_idxs]
#
#         x_scale = target_width / orig_width
#         y_scale = target_height / orig_height
#
#         # Map coordinates to downscaled image
#         x_idxs_down = (x_idxs * x_scale).astype(int)
#         y_idxs_down = (y_idxs * y_scale).astype(int)
#
#         x_idxs_down = np.clip(x_idxs_down, 0, target_width - 1)
#         y_idxs_down = np.clip(y_idxs_down, 0, target_height - 1)
#
#         # Handle duplicates: Keep the smallest depth value
#         for x, y, depth in zip(x_idxs_down, y_idxs_down, depth_values):
#             if depth_downscaled[y, x] == 0 or depth < depth_downscaled[y, x]:
#                 depth_downscaled[y, x] = depth
#
#         return depth_downscaled
#
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import torch
# from torch.utils.data import DataLoader
#
# # Import the NuScenesDataset class
# # Ensure this path is correct based on where you saved the class
# from nuscenes_dataset import NuScenesDataset
#
#
# def denormalize_image(image_tensor):
#     """
#     Denormalize an image tensor that was normalized using ImageNet mean and std.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
#     image_denorm = image_tensor * std + mean
#     return image_denorm
#
#
# def load_and_plot_depth_and_image_full_res(npy_file, jpg_file, threshold=0.0):
#     depth_data = np.load(npy_file)
#
#     # Print depth data statistics
#     print(f"Depth data statistics:")
#     print(f"  Data type: {depth_data.dtype}")
#     print(f"  Shape: {depth_data.shape}")
#     print(f"  Min value: {np.nanmin(depth_data)}")
#     print(f"  Max value: {np.nanmax(depth_data)}")
#     print(f"  Mean value: {np.nanmean(depth_data)}")
#     print(f"  Median value: {np.nanmedian(depth_data)}")
#     print(f"  Standard Deviation: {np.nanstd(depth_data)}")
#     print(f"  Number of zeros: {np.sum(depth_data == 0)}")
#     print(f"  Number of NaNs: {np.isnan(depth_data).sum()}")
#     print(f"  Number of Infinities: {np.isinf(depth_data).sum()}")
#     from PIL import Image
#     import numpy.ma as ma
#     image = Image.open(jpg_file)
#
#     masked_depth = ma.masked_where((depth_data <= threshold) | np.isnan(depth_data), depth_data)
#
#     valid_points = np.sum(~masked_depth.mask)
#
#     print(f"Number of valid depth points after masking: {valid_points}")
#     if valid_points == 0:
#         print("No valid depth points above the threshold. Consider lowering the threshold.")
#         return
#
#     vmin = np.nanmin(depth_data[depth_data > threshold])
#     vmax = np.nanmax(depth_data)
#     print(f"Adjusted color scale: vmin = {vmin}, vmax = {vmax}")
#     #
#     # plt.figure(figsize=(8, 4))
#     # plt.hist(depth_data[depth_data > 0].flatten(), bins=100)
#     # plt.title("Histogram of non-zero depth values")
#     # plt.xlabel("Depth")
#     # plt.ylabel("Frequency")
#     # plt.show()
#
#     y_indices, x_indices = np.nonzero(~masked_depth.mask)
#     depth_values = masked_depth[y_indices, x_indices]
#
#     plt.figure(figsize=(8, 6))
#     plt.imshow(image)
#     scatter = plt.scatter(
#         x_indices, y_indices, c=depth_values, cmap="inferno", s=10, vmin=vmin, vmax=vmax
#     )
#     plt.colorbar(scatter, label="Depth")
#     plt.title("Overlaid sparse depth on RGB")
#     plt.axis("off")
#     plt.show()
#
#
# def plot_depth_and_image(depth_data, image_data, threshold=0.0):
#     """
#     Plot the depth data overlaid on the image, including depth statistics.
#     """
#     # Print depth data statistics
#     print(f"Depth data statistics:")
#     print(f"  Data type: {depth_data.dtype}")
#     print(f"  Shape: {depth_data.shape}")
#     print(f"  Min value: {np.nanmin(depth_data)}")
#     print(f"  Max value: {np.nanmax(depth_data)}")
#     print(f"  Mean value: {np.nanmean(depth_data)}")
#     print(f"  Median value: {np.nanmedian(depth_data)}")
#     print(f"  Standard Deviation: {np.nanstd(depth_data)}")
#     print(f"  Number of zeros: {np.sum(depth_data == 0)}")
#     print(f"  Number of NaNs: {np.isnan(depth_data).sum()}")
#     print(f"  Number of Infinities: {np.isinf(depth_data).sum()}")
#
#     masked_depth = np.ma.masked_where((depth_data <= threshold) | np.isnan(depth_data), depth_data)
#
#     valid_points = np.sum(~masked_depth.mask)
#
#     print(f"Number of valid depth points after masking: {valid_points}")
#     if valid_points == 0:
#         print("No valid depth points above the threshold. Consider lowering the threshold.")
#         return
#
#     vmin = np.nanmin(depth_data[depth_data > threshold])
#     vmax = np.nanmax(depth_data)
#     print(f"Adjusted color scale: vmin = {vmin}, vmax = {vmax}")
#
#     y_indices, x_indices = np.nonzero(~masked_depth.mask)
#     depth_values = masked_depth[y_indices, x_indices]
#     from PIL import Image
#
#     plt.figure(figsize=(8, 6))
#     plt.imshow(Image.fromarray(np.uint8(255 * image_data)).resize((depth_data.shape[1], depth_data.shape[0])))
#     scatter = plt.scatter(
#         x_indices, y_indices, c=depth_values, cmap="inferno", s=10, vmin=vmin, vmax=vmax
#     )
#     plt.colorbar(scatter, label="Depth")
#     plt.title("Overlaid sparse depth on RGB")
#     plt.axis("off")
#     plt.show()
#
#
# import numpy as np
# import torch
#
#
# def validate_camera_poses(cur_data, src_data_list):
#     """Validates that the camera poses are correct.
#
#     Args:
#         cur_data: Data dictionary for the reference frame.
#         src_data_list: List of data dictionaries for the source frames.
#
#     Raises:
#         AssertionError: If any validation check fails.
#     """
#
#     # Validate that cam_T_world and world_T_cam are inverses for the reference frame
#     cur_cam_T_world = cur_data['cam_T_world_b44']  # 4x4 numpy array
#     cur_world_T_cam = cur_data['world_T_cam_b44']  # 4x4 numpy array
#
#     # Check if cur_cam_T_world @ cur_world_T_cam is close to identity
#     identity = np.eye(4)
#     product = np.dot(cur_cam_T_world, cur_world_T_cam)
#     if not np.allclose(product, identity, atol=1e-6):
#         raise AssertionError("cam_T_world and world_T_cam are not inverses for the reference frame.")
#
#     # Repeat for each source frame
#     for idx, src_data in enumerate(src_data_list):
#         src_cam_T_world = src_data['cam_T_world_b44']
#         src_world_T_cam = src_data['world_T_cam_b44']
#
#         # Check inverses
#         product = np.dot(src_cam_T_world, src_world_T_cam)
#         if not np.allclose(product, identity, atol=1e-6):
#             raise AssertionError(f"cam_T_world and world_T_cam are not inverses for source frame {idx}.")
#
#         # Compute relative transformation between current and source frame
#         # Transform from source camera frame to reference camera frame
#         cur_cam_T_src_cam = np.dot(cur_cam_T_world, src_world_T_cam)
#
#         # Validate the rotation matrix
#         relative_rotation = cur_cam_T_src_cam[:3, :3]
#         if not np.allclose(np.dot(relative_rotation, relative_rotation.T), np.eye(3), atol=1e-6):
#             raise AssertionError(f"Relative rotation matrix is not valid for source frame {idx}.")
#
#         # Check that determinant is 1 (proper rotation)
#         det = np.linalg.det(relative_rotation)
#         if not np.isclose(det, 1.0, atol=1e-6):
#             raise AssertionError(f"Determinant of rotation matrix is not 1 for source frame {idx}, got {det}.")
#
#         # Optional: Validate translation (e.g., ensure it's within expected bounds)
#         # Here, you can add checks based on your specific dataset requirements.
#
#     print("Camera poses validation passed.")
#
#
# import numpy as np
# import torch
# import cv2
# import matplotlib.pyplot as plt
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2
#
#
# def validate_depth_and_camera_poses(cur_data, src_data):
#     """
#     Validates current and source depth and camera poses by backprojecting.
#
#     Args:
#         cur_data: Data dictionary for the reference frame.
#         src_data: Dict of batched tensors for the source frames.
#     """
#     from PIL import Image
#     # Get current frame data
#     cur_image = cur_data['image_b3hw']  # Tensor of shape [3, H, W]
#     cur_depth = cur_data['depth_b1hw']  # Tensor of shape [1, H, W]
#     cur_intrinsics = cur_data['K_s0_b44']  # Intrinsics at depth resolution
#     cur_world_T_cam = cur_data['world_T_cam_b44']  # 4x4 numpy array
#
#     # Convert tensors to numpy arrays
#     cur_image_np = cur_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
#     cur_depth_np = cur_depth.squeeze(0).cpu().numpy()  # [H, W]
#     cur_intrinsics_np = cur_intrinsics[:3, :3].cpu().numpy()  # [3, 3]
#     cur_world_T_cam_np = cur_world_T_cam  # Already a numpy array
#
#     # Backproject depth map to 3D points in the current camera frame
#     H, W = cur_depth_np.shape
#     u_coord, v_coord = np.meshgrid(np.arange(W), np.arange(H))
#     u_coord = u_coord.reshape(-1)
#     v_coord = v_coord.reshape(-1)
#     depth = cur_depth_np.reshape(-1)
#
#     # Filter out invalid depths
#     valid_mask = np.isfinite(depth) & (depth > 0)
#     u_coord = u_coord[valid_mask]
#     v_coord = v_coord[valid_mask]
#     depth = depth[valid_mask]
#
#     # Optionally sample points to reduce computation
#     max_points = 10000  # Adjust as needed
#     if len(depth) > max_points:
#         indices = np.random.choice(len(depth), max_points, replace=False)
#         u_coord = u_coord[indices]
#         v_coord = v_coord[indices]
#         depth = depth[indices]
#
#     # Camera intrinsics
#     fx = cur_intrinsics_np[0, 0]
#     fy = cur_intrinsics_np[1, 1]
#     cx = cur_intrinsics_np[0, 2]
#     cy = cur_intrinsics_np[1, 2]
#
#     # Backproject to camera frame
#     x = (u_coord - cx) * depth / fx
#     y = (v_coord - cy) * depth / fy
#     z = depth
#
#     ones = np.ones_like(x)
#     points_cam = np.stack([x, y, z, ones], axis=1)  # Shape: (N, 4)
#
#     # Transform points to world frame
#     points_world = (cur_world_T_cam_np @ points_cam.T).T  # Shape: (N, 4)
#
#     # Get the number of source frames
#     num_src_frames = src_data['image_b3hw'].shape[0]
#
#     # For each source frame, reproject points and compare
#     for idx in range(num_src_frames):
#         # Extract data for this source frame
#         src_image = src_data['image_b3hw'][idx]  # Tensor of shape [3, H, W]
#         src_intrinsics_np = src_data['K_s0_b44'][idx][:3, :3]  # Intrinsics at depth resolution
#         src_cam_T_world = src_data['cam_T_world_b44'][idx]  # 4x4 numpy array
#
#         src_image_np = denormalize_image(torch.tensor(src_image)).cpu().numpy()
#         src_image_np = np.transpose(src_image_np, (1, 2, 0))  # H x W x 3
#
#         # Transform world points to source camera frame
#         points_src_cam = (src_cam_T_world @ points_world.T).T  # Shape: (N, 4)
#
#         # Project points onto source image plane
#         x_src = points_src_cam[:, 0] / points_src_cam[:, 2]
#         y_src = points_src_cam[:, 1] / points_src_cam[:, 2]
#
#         u_src = x_src * src_intrinsics_np[0, 0] + src_intrinsics_np[0, 2]
#         v_src = y_src * src_intrinsics_np[1, 1] + src_intrinsics_np[1, 2]
#
#         # Filter points that are within image bounds
#         H_src, W_src = src_image_np.shape[:2]
#         valid_u = (u_src >= 0) & (u_src < W_src)
#         valid_v = (v_src >= 0) & (v_src < H_src)
#         valid_depth = points_src_cam[:, 2] > 0
#         valid_points = valid_u & valid_v & valid_depth
#
#         u_src = u_src[valid_points].astype(int)
#         v_src = v_src[valid_points].astype(int)
#
#         # Visualize the projected points on the source image
#         src_image_np = denormalize_image(torch.tensor(src_image)).cpu().numpy()
#         src_image_np = np.transpose(src_image_np, (1, 2, 0))  # H x W x 3
#         src_image_np = np.uint8(255 * src_image_np)  # Scale to 0-255 and convert to uint8
#         src_image_np = cv2.cvtColor(src_image_np, cv2.COLOR_RGB2BGR)
#         src_image_np = cv2.resize(src_image_np, (cur_depth_np.shape[1], cur_depth_np.shape[0]),
#                                   interpolation=cv2.INTER_LINEAR)  # Scale to 0-255 and convert to uint8
#
#         for u_p, v_p in zip(u_src.astype(int), v_src.astype(int)):
#             cv2.circle(src_image_np, (u_p, v_p), radius=1, color=(0, 255, 0), thickness=-1)
#
#         # Display the image with projected points
#         plt.figure(figsize=(10, 8))
#         plt.imshow(src_image_np)
#         plt.title(f"Reprojected Points on Source Frame {idx}")
#         plt.axis('off')
#         plt.show()
#
#     print("Depth and camera poses validation completed.")
#
#
# def visualize_dataset_samples(dataset_path, split='train', num_samples=5):
#     # Initialize the dataset
#     dataset = NuScenesDataset(
#         dataset_path=dataset_path,
#         split=split,
#         mv_tuple_file_suffix='_tuples.txt',  # Adjust based on your tuple file suffix
#         tuple_info_file_location='/mnt/nas/personal/guillermo/nuscenes/',  # Location of your tuples file
#         image_height=450,
#         image_width=800,
#         high_res_image_width=1600,
#         high_res_image_height=900,
#         min_valid_depth=1e-3,
#         max_valid_depth=1e2,
#         verbose_init=True,
#         pass_frame_id=True,  # Include frame IDs if needed
#     )
#
#     # Initialize the DataLoader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,  # Load one sample at a time
#         shuffle=False,  # Maintain the order of the dataset
#         num_workers=4,  # Use multiple CPU cores for data loading
#         pin_memory=True,  # Speeds up the transfer of data to GPU (if used)
#         collate_fn=lambda x: x  # Since __getitem__ returns a tuple
#     )
#
#     # Iterate over the DataLoader and visualize samples
#     for idx, data in enumerate(dataloader):
#         if idx >= num_samples:
#             break
#
#         # data is a list of length batch_size (which is 1)
#         # Each item in data is a tuple: (cur_data, src_data)
#         cur_data, src_data = data[0]  # Since batch_size=1, data[0] is the first item
#         # validate_camera_poses_from_batch(cur_data, src_data)
#         validate_depth_and_camera_poses(cur_data, src_data)
#         # Extract current frame data
#         color_image = cur_data.get('image_b3hw', None)  # ImageNet normalized image
#         depth_map = cur_data.get('depth_b1hw', None)  # Depth map
#
#         load_and_plot_depth_and_image_full_res(
#             Path('/mnt/nas/shared/datasets/nuscenes/depth/') / cur_data['frame_id_string'].replace('.jpg', '.npy'),
#             Path('/mnt/nas/shared/datasets/nuscenes/') / cur_data['frame_id_string'])
#
#         if color_image is None:
#             print(f"No color image found for sample {idx}.")
#             continue
#
#         if depth_map is None:
#             print(f"No depth map found for sample {idx}.")
#             continue
#
#         # Denormalize the image for visualization
#         color_image = denormalize_image(color_image)
#
#         # Convert to numpy array and permute dimensions for plotting
#         color_image = color_image.permute(1, 2, 0).numpy()  # H x W x C
#         color_image = np.clip(color_image, 0, 1)  # Ensure the values are in [0,1]
#
#         # Convert depth map to numpy array
#         depth_map = depth_map.squeeze(0).numpy()  # Remove batch dimension, shape H x W
#
#         # Plot the depth and image using the provided function
#         plot_depth_and_image(depth_map, color_image, threshold=0.0)
#
#     print('Visualization complete.')
#
#
# if __name__ == "__main__":
#     # Configuration options
#     data_path = '/mnt/nas/shared/datasets/nuscenes/'
#     version = 'v1.0-trainval'
#     split = 'train'
#
#     num_images_in_tuple = 8
#     image_height = 450
#     image_width = 800
#     tuple_info_file_location = '/mnt/nas/personal/guillermo/nuscenes/'  # Location of your tuples file
#
#     # Visualize batches
#     visualize_dataset_samples(data_path, split='train', num_samples=5)
