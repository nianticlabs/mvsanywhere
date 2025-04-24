import os

import json
from pathlib import Path

import numpy as np
import torch

from mvsanywhere.datasets.generic_mvs_dataset import GenericMVSDataset
from mvsanywhere.datasets.change_of_basis import ChangeOfBasis
from mvsanywhere.utils.geometry_utils import rotx
import cv2

class NeRFDataset(GenericMVSDataset):
    """
    NeRF Dataset class.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.

    NOTE: This dataset will place NaNs where gt depth maps are invalid.

    """

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix="_tuples.txt",
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
        high_res_image_width=None,
        high_res_image_height=None,
        image_depth_ratio=2,
        shuffle_tuple=False,
        include_full_depth_K=False,
        include_high_res_color=False,
        pass_frame_id=False,
        skip_frames=None,
        skip_to_frame=None,
        verbose_init=True,
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


        """


    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        return frame_id.split("/")[-1]

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""
        return Path(self.dataset_path) / scan / "valid_frames.txt"

    def get_valid_frame_ids(self, split, scan, store_computed=False):
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
        scan = scan.rstrip("\n")
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            valid_frames = []
            with open(Path(self.dataset_path) / scan / 'transforms.json') as f:
                transforms = json.load(f)

            for t in transforms["frames"]:
                
                if not self.get_color_filepath(scan, t["file_path"]).is_file():
                    continue

                if not self.get_pose_filepath(scan, t["file_path"]).is_file():
                    continue

                if not np.isfinite(self.load_pose(scan, t["file_path"])[0]).all():
                    continue

                valid_frames.append(f"{scan} {t['file_path']}")

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
        return self.get_high_res_color_filepath(scan_id=scan_id, frame_id=frame_id)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            The full size RGB frame from the dataset.
        """
        path = Path(self.dataset_path) / scan_id / frame_id
        if path.suffix == '':
            path = path.parent / (path.name + '.png')
        return path 

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            The full size depth frame from the dataset.

        """
        return ""

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """
        path = Path(self.dataset_path)
        return path / scan_id / 'transforms.json'

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for MVSSynth as images
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
        output_dict = {}

        camera_data_path = self.get_pose_filepath(scan_id=scan_id, frame_id=frame_id)
        json_data = json.load(open(camera_data_path))
        frame_data = next(t for t in json_data["frames"] if t["file_path"] == frame_id)

        width_pixels = frame_data["w"] if "w" in frame_data else json_data["w"]
        height_pixels = frame_data["h"] if "h" in frame_data else json_data["h"]
        c_x = frame_data["cx"] if "cx" in frame_data else json_data["cx"]
        c_y = frame_data["cy"] if "cy" in frame_data else json_data["cy"]
        f_x = frame_data["fl_x"] if "fl_x" in frame_data else json_data["fl_x"]
        f_y = frame_data["fl_y"] if "fl_y" in frame_data else json_data["fl_y"]
            

        # width_pixels = json_data["w"]
        # height_pixels = json_data["h"]

        # # Read original intrinsics
        # c_x = json_data["cx"]
        # c_y = json_data["cy"]
        # f_x = json_data["fl_x"]
        # f_y = json_data["fl_y"]

        if flip:
            c_x = width_pixels - c_x

        # Construct the intrinsic matrix in pixel coordinates
        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(
            [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=torch.float32
        )

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.linalg.inv(K_matching)

        K_depth = K.clone()
        K_depth[0] *= self.depth_width / float(width_pixels)
        K_depth[1] *= self.depth_height / float(height_pixels)

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K_depth.clone()
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(K_depth)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K_depth.clone()
            K_scaled[0, 0] /= 2**i
            K_scaled[1, 1] /= 2**i
            K_scaled[0, 2] /= 2**i
            K_scaled[1, 2] /= 2**i
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = torch.linalg.inv(K_scaled)

        return output_dict, None

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
        depth = torch.zeros((1, self.depth_height, self.depth_width), dtype=float)

        # # Get the float valid mask
        mask_b = (depth > 0.0)
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b
    
    def load_full_res_depth_and_mask(self, scan_id, frame_id, crop=None):
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
        depth = torch.zeros((1, self.depth_height, self.depth_width), dtype=float)

        # # Get the float valid mask
        mask_b = (depth > 0.0)
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

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
        pose_path = self.get_pose_filepath(scan_id, frame_id)

        pose_data = json.load(open(pose_path))
        pose_data = next(t for t in pose_data["frames"] if t["file_path"] == frame_id)

        world_T_cam = torch.tensor(np.array(pose_data["transform_matrix"]).astype(np.float32))
        # world_T_cam = torch.linalg.inv(world_T_cam)

        gl_to_cv = torch.FloatTensor([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
        world_T_cam *= gl_to_cv
        world_T_cam = world_T_cam.numpy()
        rot_mat = world_T_cam[:3, :3]
        trans = world_T_cam[:3, 3]

        rot_mat = rotx(-np.pi / 2) @ rot_mat
        trans = rotx(-np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans

        world_T_cam = world_T_cam
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam.astype(np.float32), cam_T_world.astype(np.float32)

