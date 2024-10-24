import cv2

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


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
        mv_tuple_file_suffix,
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
        self.filename_to_sample_data_token = {}
        for sample_data in self.nusc.sample_data:
            filename_sd = sample_data['filename']
            self.filename_to_sample_data_token[filename_sd] = sample_data['token']

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
        # Get the sample_data_token from the filename
        sample_data_token = self.get_sample_data_token_from_filename(frame_id)
        cam_data = self.nusc.get('sample_data', sample_data_token)
        calib_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsics = np.array(calib_sensor['camera_intrinsic'])

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(intrinsics, dtype=torch.float32)

        # Image size
        width_pixels = cam_data['width']
        height_pixels = cam_data['height']

        if flip:
            K[0, 2] = float(width_pixels) - K[0, 2]

        output_dict = {}

        # Optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.inverse(K)

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.inverse(K_matching)

        # Scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / float(width_pixels)
        K[1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = torch.inverse(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, None

    def _load_depth(self, depth_path, is_float16=True):
        """Loads the depth map from a .npy file."""
        depth = np.load(depth_path)
        return depth

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for."""
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        depth = self._load_depth(depth_filepath)

        if crop:
            depth = depth[crop[1]:crop[3], crop[0]:crop[2]]

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )

        mask_b = torch.tensor(depth > self.min_valid_depth).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

        mask = mask_b.float()

        # Set invalids to NaN
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

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

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose."""
        sample_data_token = self.get_sample_data_token_from_filename(frame_id)
        cam_data = self.nusc.get('sample_data', sample_data_token)

        # Get the ego pose
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        translation = np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation']).rotation_matrix

        # Transformation from ego to world
        ego_T_world = np.eye(4)
        ego_T_world[:3, :3] = rotation
        ego_T_world[:3, 3] = translation

        # Get sensor (camera) to ego transformation
        calib_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        sensor_translation = np.array(calib_sensor['translation'])
        sensor_rotation = Quaternion(calib_sensor['rotation']).rotation_matrix

        # Transformation from sensor to ego
        sensor_T_ego = np.eye(4)
        sensor_T_ego[:3, :3] = sensor_rotation
        sensor_T_ego[:3, 3] = sensor_translation

        # Compute sensor to world transformation
        sensor_T_world = ego_T_world @ sensor_T_ego

        # Compute world to sensor transformation
        world_T_sensor = np.linalg.inv(sensor_T_world)

        return sensor_T_world, world_T_sensor

    def get_sample_data_token_from_filename(self, filename):
        """Extracts the sample_data_token from the filename."""
        sample_data_token = self.filename_to_sample_data_token.get(filename)
        if sample_data_token is None:
            raise ValueError(f"Sample data token not found for filename: {filename}")
        return sample_data_token


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import the NuScenesDataset class
# Ensure this path is correct based on where you saved the class
from nuscenes_dataset import NuScenesDataset


def visualize_nuscenes_dataloader(dataloader, num_batches=1):
    """
    Visualizes batches from the NuScenesDataset using a DataLoader.

    Args:
        dataloader: A DataLoader wrapping the NuScenesDataset.
        num_batches: Number of batches to visualize.
    """
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count > num_batches:
            break
        print(f"Visualizing batch {batch_count}/{num_batches}")

        # Assuming batch contains:
        # - 'images': list of tensors [batch_size, C, H, W]
        # - 'depth_b1hw': list of tensors [batch_size, 1, H, W]
        # - 'depth_masks_b1hw': list of tensors [batch_size, 1, H, W]
        images = batch['images'][0]  # List of images in the tuple
        depths = batch['depth_b1hw']  # Corresponding depth maps
        masks = batch['depth_masks_b1hw']  # Validity masks

        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i]
            depth = depths[i]
            mask = masks[i]

            # Convert image tensor to numpy array and transpose to (H, W, C)
            image_np = image.permute(1, 2, 0).numpy()

            # Convert depth tensor to numpy array
            depth_np = depth.squeeze(0).numpy()
            mask_np = mask.squeeze(0).numpy()

            # Apply mask to depth map (set invalid depths to NaN for visualization)
            depth_np_masked = np.where(mask_np, depth_np, np.nan)

            # Plot image and depth map side by side
            fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            axs[0].imshow(image_np.astype(np.uint8))
            axs[0].set_title('RGB Image')
            axs[0].axis('off')

            # Use a colormap for depth visualization
            depth_im = axs[1].imshow(depth_np_masked, cmap='plasma', vmin=0, vmax=50)
            axs[1].set_title('Depth Map')
            axs[1].axis('off')
            fig.colorbar(depth_im, ax=axs[1], fraction=0.046, pad=0.04)

            plt.show()


if __name__ == "__main__":
    # Configuration options
    data_path = '/mnt/nas/shared/datasets/nuscenes/'
    version = 'v1.0-trainval'
    split = 'train'
    mv_tuple_file_suffix = ''
    num_images_in_tuple = 8
    image_height = 450
    image_width = 800
    tuple_info_file_location = '/mnt/nas/personal/guillermo/nuscenes/'  # Location of your tuples file

    # Initialize the dataset
    dataset = NuScenesDataset(
        dataset_path=data_path,
        split=split,
        mv_tuple_file_suffix='_tuples.txt',
        image_height=image_height,
        image_width=image_width,
        tuple_info_file_location=tuple_info_file_location,
        num_images_in_tuple=num_images_in_tuple,
        verbose_init=True
    )

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Visualize batches
    visualize_nuscenes_dataloader(dataloader, num_batches=2)
