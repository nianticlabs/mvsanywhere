import os
from pathlib import Path

import numpy as np
import torch

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset


class WaymoDataset(GenericMVSDataset):
    """
    MVS waymo Dataset class.

    Inherits from GenericMVSDataset and implements methods specific to the waymo dataset.
    This dataset class handles the loading of images, depth maps, intrinsics,
    and poses for the Waymo dataset, including the generated depth maps.
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

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        split_filename = f"{self.split}{mv_tuple_file_suffix}"
        tuples_file = Path(tuple_info_file_location) / split_filename

        if not tuples_file.exists():
            raise FileNotFoundError(f"Tuples file not found at {tuples_file}")

        with open(tuples_file, "r") as f:
            self.tuples = [line.strip().split() for line in f]

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
        driving_sequence, camera_id = scan_id.split("^")

        # Load and extract the saved data
        return os.path.join(self.dataset_path, driving_sequence, f"{frame_id}_{camera_id}.jpg")

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """Returns the filepath for a frame's higher resolution color file."""
        return self.get_color_filepath(scan_id, frame_id)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics and computes scaled intrinsics matrices for a frame at multiple scales."""
        driving_sequence, camera_id = scan_id.split("^")
        data = np.load(os.path.join(self.dataset_path, driving_sequence, f"{frame_id}_{camera_id}.npz"))

        intrinsics = np.array(data["intrinsics"])

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(intrinsics, dtype=torch.float32)

        width_pixels = data["width"]
        height_pixels = data["height"]

        # top, left, h, w = self.random_resize_crop.get_params(
        #     torch.empty((height_pixels, width_pixels)),
        #     self.random_resize_crop.scale,
        #     self.random_resize_crop.ratio
        # )
        top, left, h, w = 0, 0, height_pixels, width_pixels
        K[0, 2] = K[0, 2] - left
        K[1, 2] = K[1, 2] - top
        width_pixels = w
        height_pixels = h

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
            K_scaled[:2] /= 2**i
            invK_scaled = torch.inverse(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, (left, top, left+width_pixels, top+height_pixels)

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides."""
        depth = self._load_depth(scan_id=scan_id, frame_id=frame_id)

        mask_b = torch.tensor(depth > self.min_valid_depth).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

        # Set invalids to NaN
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask_b.float(), mask_b

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for."""
        depth = self._load_depth(scan_id=scan_id, frame_id=frame_id, height=self.depth_height, width=self.depth_width, crop=crop)

        mask_b = torch.tensor(depth > self.min_valid_depth).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

        mask = mask_b.float()

        # Set invalids to NaN
        depth[~mask_b] = torch.tensor(np.nan)
        return depth, mask, mask_b

    def _load_depth(self, scan_id, frame_id, height=None, width=None, crop=None):
        """Loads and densifies the sparse depth map to the specified height and width

        If height and width aren't specified, will use the saved height and width
        """
        driving_sequence, camera_id = scan_id.split("^")

        # Load and extract the saved data
        data = np.load(os.path.join(self.dataset_path, driving_sequence, f"{frame_id}_{camera_id}_sparse_depth.npz"))
        x, y = data['xy'].T
        d = data['z']
        original_height = data['height']
        original_width = data['width']

        if height is None:
            assert width is None
            height = original_height
            width = original_width

        # Rescale to the new image size
        x = (x / original_width) * width
        y = (y / original_height) * height

        # (Optionally) crop in the new image space
        if crop is not None:
            crop_left, crop_top, crop_right, crop_bottom = crop
            x -= crop_left
            y -= crop_top

        valid_points = np.logical_and.reduce((x >= 0, y >= 0, x <= width - 1, y <= height -1))

        y = y[valid_points].round().astype(np.int16)
        x = x[valid_points].round().astype(np.int16)
        d = d[valid_points]

        # Build the dense depth map
        # # TODO – deal with collisions. Currently we are just hoping they don't exist.
        depthmap = np.zeros((height, width))
        depthmap[y, x] = d

        return depthmap

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

        driving_sequence, camera_id = scan_id.split("^")
        data = np.load(os.path.join(self.dataset_path, driving_sequence, f"{frame_id}_{camera_id}.npz"))
        world_T_cam = np.array(data['cam2world']).astype(np.float32)  # remember cam2world == world_T_cam?

        # Transformation from world frame to camera frame (inverse of world_T_cam)
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world


if __name__ == "__main__":
    tuple_info_file_location = Path("tmp")
    tuple_info_file_location.mkdir(exist_ok=True)
    with open(tuple_info_file_location / "train_tuples.txt", 'w') as f:
        f.write("individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord^1 00083 00084 00085\n")
        f.write("individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord^2 00083 00084 00085\n")
        f.write("individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord^3 00083 00084 00085\n")
        f.write("individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord^4 00083 00084 00085\n")
        f.write("individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord^5 00083 00084 00085\n")

    dataset = WaymoDataset("/mnt/nas3/shared/datasets/waymo/preprocessed", split='train', tuple_info_file_location=tuple_info_file_location)
    for idx in range(5):
        _, item = dataset[idx]
        # print(item.keys())
        # for key in item:
        #     print(key, item[key].shape, item[key].dtype)
        depth = item['depth_b1hw']

        print(depth[0, 0].shape)
        # assert depth[0, 0, 100:, :].max() > 0

        no_nan_depth = depth[~np.isnan(depth)]
        print(no_nan_depth.min(), no_nan_depth.max(), np.median(no_nan_depth))

        d = depth[0, 0]
        d[np.isnan(d)] = 0.0
        d = d - d.min()
        d /= d.max()
        d = (d * 255).astype(np.uint8)


        import cv2
        image = item['image_b3hw'][0].transpose(1, 2, 0)
        image += 2.117904
        image /= image.max()
        print(image.min(), image.max())
        image = (cv2.resize(image, (256, 192)) * 255).astype(np.uint8)

        image_overlay = image.copy().transpose(2, 0, 1)
        image_overlay[:, d > 0] = d[d > 0]
        print(d.shape, image.shape)

        d = np.dstack((d, d, d))
        combined = np.hstack((image, d, image_overlay.transpose(1,2 ,0)))[:, :, ::-1]

        cv2.imwrite(f"{idx}.png", combined)