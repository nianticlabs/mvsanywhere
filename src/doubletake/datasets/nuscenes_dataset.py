
from pathlib import Path

import cv2

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from generic_mvs_dataset import GenericMVSDataset

class NuscDataset(GenericMVSDataset):
    """
    MVS Nuscenes Dataset class.

    Inherits from GenericMVSDataset and implements missing methods for the Nuscenes dataset.

    """

    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_FRONT_RIGHT",
    ]

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
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
        min_valid_depth=1e-3,
        max_valid_depth=1e3,
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

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        version = 'v1.0-trainval'
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=verbose_init)

        # # Load sample tokens
        # if self.tuple_info_file_location is None:
        #     self.tuple_info_file_location = Path(self.dataset_path) / "tuples"
        # tuple_file = self.tuple_info_file_location / f"{self.split}{self.mv_tuple_file_suffix}.txt"
        # with open(tuple_file, 'r') as f:
        #     self.sample_tokens = f.read().splitlines()

    def get_frame_id_string(self, frame_id):
        """Returns a unique frame ID string."""
        return frame_id

    def get_color_filepath(self, scan_id, frame_id):
        """Returns the filepath for the RGB image."""
        return self.get_high_res_color_filepath(scan_id, frame_id)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """Returns the filepath for the high-resolution RGB image."""
        cam_name, sample_token = scan_id.split('_')
        sample_data = self.nusc.get('sample_data', sample_token)
        image_path = os.path.join(self.dataset_path, sample_data['filename'])
        return image_path

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """Returns the filepath for the cached depth image."""
        cam_name, sample_token = scan_id.split('_')
        sample_data = self.nusc.get('sample_data', sample_token)
        depth_filename = sample_data['filename'].replace('samples', 'depth').replace('.jpg', '.png')
        depth_path = os.path.join(self.dataset_path, depth_filename)
        return depth_path

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """Returns the filepath for the full-resolution depth image."""
        return self.get_cached_depth_filepath(scan_id, frame_id)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads camera intrinsics and returns them at multiple scales."""
        cam_name, sample_token = scan_id.split('_')
        sample_data = self.nusc.get('sample_data', sample_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = np.array(calibrated_sensor['camera_intrinsic'])

        width_pixels = sample_data['width']
        height_pixels = sample_data['height']

        K = torch.eye(4)
        K[:3, :3] = torch.tensor(intrinsics)
        K = K.float()

        if flip:
            K[0, 2] = float(width_pixels) - K[0, 2]

        output_dict = {}

        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.inverse(K)

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.inverse(K_matching)

        K_depth = K.clone()
        K_depth[0] *= self.depth_width / float(width_pixels)
        K_depth[1] *= self.depth_height / float(height_pixels)

        for i in range(self.prediction_num_scales):
            K_scaled = K_depth.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = torch.inverse(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, None

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads depth map at the target resolution with validity mask."""
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        depth = self._load_depth(depth_filepath)

        if depth is None:
            # If depth is not available, create an empty depth map
            depth = np.zeros((self.depth_height, self.depth_width), dtype=np.float32)
            mask_b = torch.zeros((1, self.depth_height, self.depth_width), dtype=torch.bool)
            depth = torch.tensor(depth).unsqueeze(0)
            depth[~mask_b] = torch.tensor(np.nan)
            return depth, mask_b.float(), mask_b

        if crop:
            depth = depth[crop[1]:crop[3], crop[0]:crop[2]]

        depth = cv2.resize(depth, (self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST)

        mask_b = torch.tensor((depth > self.min_valid_depth) & (depth < self.max_valid_depth)).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)
        depth[~mask_b] = torch.tensor(np.nan)

        mask = mask_b.float()
        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id, crop=None):
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        depth = self._load_depth(depth_filepath)

        if depth is None:
            # If depth is not available, create an empty depth map
            width, height = self.get_image_size(scan_id, frame_id)
            depth = np.zeros((height, width), dtype=np.float32)
            mask_b = torch.zeros((1, height, width), dtype=torch.bool)
            depth = torch.tensor(depth).unsqueeze(0)
            depth[~mask_b] = torch.tensor(np.nan)
            return depth, mask_b.float(), mask_b

        if crop:
            depth = depth[crop[1]:crop[3], crop[0]:crop[2]]

        mask_b = torch.tensor((depth > self.min_valid_depth) & (depth < self.max_valid_depth)).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)
        depth[~mask_b] = torch.tensor(np.nan)

        mask = mask_b.float()
        return depth, mask, mask_b

    def load_pose(self, scan_id, frame_id):

        cam_name, sample_token = scan_id.split('_')
        sample_data = self.nusc.get('sample_data', sample_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        # Transformation from ego vehicle to camera
        ego_to_cam = self._get_transformation_matrix(calibrated_sensor['translation'], calibrated_sensor['rotation'])

        # Transformation from global to ego vehicle
        sample = self.nusc.get('sample', sample_data['sample_token'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        global_to_ego = self._get_transformation_matrix(ego_pose['translation'], ego_pose['rotation'])

        # The camera pose in global coordinates
        world_T_cam = np.dot(global_to_ego, ego_to_cam)
        cam_T_world = np.linalg.inv(world_T_cam)

        return torch.tensor(world_T_cam).float(), torch.tensor(cam_T_world).float()

    def _load_depth(self, depth_path):
        """Loads the depth image from the given path."""
        if not os.path.exists(depth_path):
            return None
        depth = np.load(depth_path)
        return depth

    def _get_transformation_matrix(self, translation, rotation):
        """Creates a transformation matrix from translation and rotation."""
        tm = np.eye(4)
        tm[:3, :3] = Quaternion(rotation).rotation_matrix
        tm[:3, 3] = translation
        return tm

    def get_image_size(self, scan_id, frame_id):
        """Returns the image width and height."""
        cam_name, sample_token = scan_id.split('_')
        sample_data = self.nusc.get('sample_data', sample_token)
        width = sample_data['width']
        height = sample_data['height']
        return width, height


import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np


# Import your NuscDataset class
# Adjust the import path based on your project structure
# from your_project.datasets.nusc_dataset import NuscDataset

# For demonstration purposes, let's assume NuscDataset is defined in the current scope
# following the implementation we discussed earlier.

def visualize_dataset_samples(dataset_path, split='train', num_samples=5):
    # Initialize the dataset
    dataset = NuscDataset(
        dataset_path=dataset_path,
        split=split,
        mv_tuple_file_suffix=None,
        include_full_res_depth=True,
        image_height=384,
        image_width=512,
        high_res_image_width=1600,
        high_res_image_height=900,
        min_valid_depth=1e-3,
        max_valid_depth=1e2,
        verbose_init=True,
    )

    # Initialize the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Load one sample at a time
        shuffle=False,  # Maintain the order of the dataset
        num_workers=4,  # Use multiple CPU cores for data loading
        pin_memory=True,  # Speeds up the transfer of data to GPU (if used)
    )

    # Iterate over the DataLoader and visualize samples
    for idx, data in enumerate(dataloader):
        if idx >= num_samples:
            break

        # Extract data
        # Adjust the keys based on how your dataset returns data
        color_images = data.get('color_b3hw', None)  # Assuming 'color_b3hw' is the key for RGB images
        depth_maps = data.get('depth_b1hw', None)  # Assuming 'depth_b1hw' is the key for depth maps

        if color_images is None:
            print(f"No color images found for sample {idx}.")
            continue

        # Convert tensors to numpy arrays for visualization
        # Assuming images are in (batch_size, channels, height, width)
        color_image = color_images[0].permute(1, 2, 0).numpy()
        # Normalize images to [0, 1] if necessary
        color_image = np.clip(color_image, 0, 1)

        # Plot the color image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(color_image)
        plt.title(f'Color Image {idx}')
        plt.axis('off')

        # If depth map is available, visualize it
        if depth_maps is not None:
            depth_map = depth_maps[0].squeeze().numpy()

            # Handle invalid depth values
            valid_mask = np.isfinite(depth_map)
            depth_map[~valid_mask] = 0

            # Plot the depth map
            plt.subplot(1, 2, 2)
            plt.imshow(depth_map, cmap='inferno')
            plt.title(f'Depth Map {idx}')
            plt.axis('off')
            plt.colorbar()

        plt.tight_layout()
        plt.show()

    print('Visualization complete.')


# Example usage:
if __name__ == '__main__':
    dataset_path = '/mnt/nas/shared/datasets/nuscenes'  # Replace with the path to your dataset
    visualize_dataset_samples(dataset_path, split='train', num_samples=5)
