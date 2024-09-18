import os
from pathlib import Path
import glob
import json

from doubletake.datasets.change_of_basis import ChangeOfBasis

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
from PIL import Image
import torch
import cv2

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset


class MatrixCityDataset(GenericMVSDataset):
    """
    MVS MatrixCity Dataset class.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.

    NOTE: This dataset will place NaNs where gt depth maps are invalid.

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

        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The 
                tuple filename searched for wil be 
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if 
                None provided, will search in the dataset directory under 
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized 
                to. 
            include_high_res_color: should the dataset pass back higher 
                resolution images.
            high_res_image_height, high_res_image_width: resolution images 
                should be resized if we're passing back higher resolution 
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of 
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the 
                dataset at the highest resolution available.
            shuffle_tuple: by default source images will be ordered according to 
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item 
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing 
                depth.
            verbose_init: if True will let the init print details on the 
                initialization.
            min_valid_depth, max_valid_depth: values to generate a validity mask
                for depth maps.
        
        """

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        # Load all transforms files to save IO time
        self.transforms = {}
        transforms_paths = Path(self.dataset_path) / "big_city/**/train/**/transforms.json"
        for transform_file in glob.glob(str(transforms_paths)):
            with open(transform_file) as f:
                transforms = json.load(f)
            self.transforms["/".join(transform_file.split("/")[-5:-1])] = transforms


        self.tuple_info_file_location = tuple_info_file_location

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        return frame_id
    
    def format_frame_id(self, scan, frame_id):
        if scan in [
            "big_city/aerial/train/big_high_block_1"
            "big_city/aerial/train/big_high_block_2",
            "big_city/aerial/train/big_high_block_3",
            "big_city/aerial/train/big_high_block_4",
            "big_city/aerial/train/big_high_block_5",
            "big_city/aerial/train/big_high_block_6",
        ]:
            return f"{int(frame_id):04d}"
        return frame_id

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""

        scan_dir = (
            Path(self.dataset_path)
            / "valid_frames"
            / self.get_sub_folder_dir(split)
            / scan
        )

        scan_dir.mkdir(parents=True, exist_ok=True)

        return os.path.join(str(scan_dir), "valid_frames.txt")
    
    def _load_transforms_file(self, scan):
        return self.transforms[scan.split(":")[0]]

    def _get_frame_ids(self, split, scan):
        if not hasattr(self, "scans_frames"):
            with open(Path(self.tuple_info_file_location) / "matrix_city_train.json") as f:
                self.scans_frames = json.load(f)
        return self.scans_frames[scan]

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """Either loads or computes the ids of valid frames in the dataset for
        a scan.

        A valid frame is one that has an existing RGB frame, an existing
        depth file, and existing pose file where the pose isn't inf, -inf,
        or nan.

        Args:
            split: the data split (train/val/test)
            scan: the name of the scan
            store_computed: store the valid_frame file where we'd expect to
            see the file in the scan folder. get_valid_frame_path defines
            where this file is expected to be. If the file can't be saved,
            a warning will be printed and the exception reason printed.

        Returns:
            valid_frames: a list of strings with info on valid frames.
            Each string is a concat of the scan_id and the frame_id.
        """
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            # find out which frames have valid poses
            print(f"computing valid frames for scene {scan}.")

            frame_ids = self._get_frame_ids(split, scan)

            valid_frames = []
            dist_to_last_valid_frame = 0
            bad_file_count = 0
            for frame_ind in frame_ids:

                image_path = self.get_color_filepath(scan, frame_ind)
                try:
                    # np.array(Image.open(image_path))
                    assert os.path.isfile(image_path)
                except:
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                depth_filepath = self.get_full_res_depth_filepath(scan, frame_ind)
                try:
                    depth = self._load_depth(depth_filepath)
                    assert (depth != 65504).sum() > 0
                except:
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                world_T_cam_44, _ = self.load_pose(scan, frame_ind)
                if (
                    np.isnan(np.sum(world_T_cam_44))
                    or np.isinf(np.sum(world_T_cam_44))
                    or np.isneginf(np.sum(world_T_cam_44))
                ):
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                valid_frames.append(f"{scan} {frame_ind} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files out of " f"{len(frame_ids)}.")

            # store computed if we're being asked, but wrapped inside a try
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, "w") as f:
                        f.write("\n".join(valid_frames) + "\n")
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, " f"cause:")
                    print(e)

        return valid_frames

    def get_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's color file at the dataset's
        configured RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the size
            required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        frame_id = self.format_frame_id(scan_id.split(":")[0], frame_id)
        cached_resized_path = (
            Path(self.dataset_path)
            / scan_id.split(":")[0]
            / f"{frame_id}.{self.image_width}_{self.image_height}.png"
        )

        if cached_resized_path.exists():
            return str(cached_resized_path)

        path = (
            Path(self.dataset_path)
            / scan_id.split(":")[0]
            / f"{frame_id}.png"
        )

        return str(path)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            The full size RGB frame from the dataset.
        """
        frame_id = self.format_frame_id(scan_id.split(":")[0], frame_id)
        path = (
            Path(self.dataset_path)
            / scan_id.split(":")[0]
            / f"{frame_id}.png"
        )

        # instead return the default image
        return str(path)

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the dataset's
        configured depth resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for a precached depth file at the size
            required.

        """

        # we do not use this method in this dataset
        return ""

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            The full size depth frame from the dataset.

        """
        frame_id = self.format_frame_id(scan_id.split(":")[0], frame_id)
        city, mode, split, block = scan_id.split(":")[0].split("/")
        path = (
            Path(self.dataset_path)
            / f"{city}_depth"
            / mode
            / split
            / f"{block}_depth"
            / f"{frame_id}.exr"
        )

        return str(path)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for Hypersim as images
            share intrinsics across a scene.

        Returns:
            output_dict: A dict with
                - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                (backprojection) where i in [0,1,2,3,4]. i=0 provides
                intrinsics at the scale for depth_b1hw.
                - K_full_depth_b44 and invK_full_depth_b44 provides
                intrinsics for the maximum available depth resolution.
                Only provided when include_full_res_depth is true.

        """
        if "street" in scan_id:
            height_pixels, width_pixels = 1000, 1000
        else:
            height_pixels, width_pixels = 1080, 1920
        transforms = self._load_transforms_file(scan_id)
        angle_x = transforms["camera_angle_x"]

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(.5 * width_pixels / np.tan(.5 * angle_x))
        K[1, 1] = float(.5 * width_pixels / np.tan(.5 * angle_x))
        K[0, 2] = width_pixels / 2
        K[1, 2] = height_pixels / 2

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

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.tensor(np.linalg.inv(K))

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = np.linalg.inv(K_matching)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / float(width_pixels)
        K[1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, (left, top, left+width_pixels, top+height_pixels)
    
    @staticmethod
    def _load_depth(depth_path):
        image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
        return image

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for.

        Internally, if the loaded depth map isn't at the target resolution,
        the depth map will be resized on-the-fly to meet that resolution.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            depth: depth map at the right resolution. Will contain NaNs
                where depth values are invalid.
            mask: a float validity mask for the depth maps. (1.0 where depth
            is valid).
            mask_b: like mask but boolean.
        """
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        
        depth = self._load_depth(depth_filepath)
        if crop:
            depth = depth[
                crop[1]:crop[3],
                crop[0]:crop[2]
            ]

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )

        # mask_b = torch.tensor(depth < 65000).bool().unsqueeze(0)
        mask_b = depth < 65000
        if mask_b.sum() > 0:
            mask_b = torch.tensor(depth < np.quantile(depth[depth < 65000], 0.95)).bool().unsqueeze(0)
        else:
            mask_b = torch.tensor(mask_b).bool().unsqueeze(0)
        depth = torch.tensor(depth / 100).float().unsqueeze(0)

        # # Get the float valid mask
        mask = mask_b.float()

        if mask.sum() == 0:
            print('0')

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            full_res_depth: depth map at the right resolution. Will contain
                NaNs where depth values are invalid.
            full_res_mask: a float validity mask for the depth maps. (1.0
            where depth is valid).
            full_res_mask_b: like mask but boolean.
        """
        full_res_depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        full_res_depth = self._load_depth(full_res_depth_filepath)

        full_res_mask_b = torch.tensor(full_res_depth != 65504).bool().unsqueeze(0)
        full_res_depth = torch.tensor(full_res_depth / 100).float().unsqueeze(0)

        # # Get the float valid mask
        full_res_mask = full_res_mask_b.float()

        # set invalids to nan
        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera to the world (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world to the camera (extrinsics).

        """
        transforms = self._load_transforms_file(scan_id)
        cam_to_world = next(frame["rot_mat"] for frame in transforms["frames"] if frame["frame_index"] == int(frame_id))

        world_T_cam = np.array(cam_to_world)
        world_T_cam[:3, :3] *= 100.0
        # world_T_cam[:3, 3] /= 100.0

        world_T_cam = world_T_cam.astype(np.float32)
        world_T_cam = ChangeOfBasis.convert_matrix_to_vision_convention(world_T_cam).astype(np.float32)
        
        # Other way around?
        cam_T_world = np.linalg.inv(world_T_cam)

        # world_T_cam = np.linalg.inv(world_T_cam)
        # cam_T_world = np.linalg.inv(cam_T_world)

        return world_T_cam, cam_T_world
