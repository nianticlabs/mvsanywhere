import os

import numpy as np
import PIL.Image as pil
import torch

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from doubletake.utils.generic_utils import read_image_file, readlines, read_pfm_file


class DTUDataset(GenericMVSDataset):
    """
    MVS DTU Dataset class.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.


    All resized precached versions of depth and images are nice to have but not
    required. If they don't exist, the full res versions will be loaded, and
    downsampled on the fly.

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
        max_valid_depth=10,
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

    @staticmethod
    def get_sub_folder_dir(split):
        """Where scans are for each split."""
        return "train"

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        return frame_id

    def get_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's color file at the dataset's 
            configured RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """

        return self.get_high_res_color_filepath(scan_id, frame_id)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's higher res color file at the 
            dataset's configured high RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the high res 
                size required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        
        return os.path.join(self.scenes_path, 'Rectified', scan_id, f"rect_{frame_id}_r5000.png")

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for a precached depth file at the size 
                required.

        """
        return self.get_full_res_depth_filepath(scan_id, frame_id)

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the native 
            resolution in the dataset.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth file at the size 
                required, or if that doesn't exist, the full size depth frame 
                from the dataset.

        """

        return os.path.join(self.scenes_path, 'Depths_raw', f'{scan_id}', f"depth_map_{int(frame_id.split('_')[0]) - 1:0>4}.pfm")

    def get_pose_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for pose information.

        """
        return os.path.join(self.scenes_path, 'Cameras', f"{int(frame_id.split('_')[0]) - 1:08d}_cam.txt")

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.
            
            ScanNet intrinsics for color and depth are the same up to scale.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame. Not needed for ScanNet as images 
                share intrinsics across a scene.
                flip: flips intrinsics along x for flipped images.

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

        data = {
            'depthWidth': 1600,
            'depthHeight': 1200,
        }

        intrinsics_filepath = os.path.join(self.scenes_path, 'Cameras', f"{int(frame_id.split('_')[0]) - 1:08d}_cam.txt")
        with open(intrinsics_filepath) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3)))


        if flip:
            K[0, 2] = float(data['depthWidth']) - K[0, 2]

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.tensor(np.linalg.inv(K))

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(data['depthWidth'])
        K_matching[1] *= self.matching_height / float(data["depthHeight"])
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = np.linalg.inv(K_matching)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / float(data['depthWidth'])
        K[1] *= self.depth_height / float(data['depthHeight'])

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the resolution the dataset is configured for.

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
        depth_filepath = self.get_cached_depth_filepath(scan_id, frame_id)

        if not os.path.exists(depth_filepath):
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)

        # Load depth, resize
        depth = read_pfm_file(
            depth_filepath,
            height=self.depth_height,
            width=self.depth_width,
            value_scale_factor=1e-3,
            resampling_mode=pil.NEAREST,
        )
        
        depth_mask = read_image_file(
            os.path.join(self.scenes_path, 'Depths_raw', f'{scan_id}', f"depth_visual_{int(frame_id.split('_')[0]) - 1:0>4}.png"),
            height=self.depth_height,
            width=self.depth_width,
            value_scale_factor=1.0,
            resampling_mode=pil.NEAREST,
        )

        # Get the float valid mask
        mask_b = depth_mask > 0
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)
        
        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the native resolution the dataset provides.

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
        # Load depth
        full_res_depth = read_pfm_file(full_res_depth_filepath, value_scale_factor=1e-3)
        depth_mask = read_image_file(
            os.path.join(self.scenes_path, 'Depths_raw', f'{scan_id}', f"depth_visual_{int(frame_id.split('_')[0]) - 1:0>4}.png"),
            value_scale_factor=1.0
        )       

        # Get the float valid mask
        full_res_mask_b = depth_mask > 0
        full_res_mask = full_res_mask_b.float()

        # set invalids to nan
        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

    def load_pose(self, scan_id, frame_id):
        """ Loads a frame's pose file.

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

        with open(pose_path) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        extrinsics[:3, 3] /= 1000



        cam_T_world = extrinsics
        world_T_cam = np.linalg.inv(cam_T_world)

        return world_T_cam, cam_T_world