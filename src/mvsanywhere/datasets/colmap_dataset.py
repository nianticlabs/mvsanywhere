import functools
import logging
import os
from pathlib import Path

import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

from mvsanywhere.datasets.generic_mvs_dataset import GenericMVSDataset
from mvsanywhere.utils.generic_utils import (
    crop_or_pad,
    fov_to_image_dimension,
    read_image_file,
)
from mvsanywhere.utils.geometry_utils import qvec2rotmat, rotx

import mvsanywhere.datasets.read_write_colmap_model as rwm

logger = logging.getLogger(__name__)


class ColmapDataset(GenericMVSDataset):
    """
    Reads COLMAP undistored images and poses from a text based sparse COLMAP
    reconstruction.

    self.capture_poses is a dictionary indexed with a scan's id and is populated
    with a scan's pose information when a frame is loaded from that scan.

    This class expects each scan's directory to have the following hierarchy:
    
    dataset_path:
        scans.txt (contains list of scans, you can define a different filepath)
        tuples (dir where you store tuples, you can define a different directory)
        scans:
            scan_1:
                images:
                    img1.jpg (undistored image from COLMAP)
                    img2.jpg
                    ...
                    imgN.jpg
                sparse:
                    0:
                        cameras.txt: SIMPLE_PINHOLE camera text file with intrinsics.
                        images.txt: text file output with image poses.
                valid_frames.txt (generated when you run tuple scripts)
                scale.txt a text file with the scale of the scene.
                
                
    The `scale.txt` file should contain a scale factor to go from COLMAP's coords 
    to real metric coords. You can get this by dividing by the real size of an item 
    by the measured size of that item in the COLMAP point cloud. 
    
    This class does not load depth, instead returns dummy data.
    
    Set `modify_to_fov` to True to crop images to an fov of [58.18, 45.12]. 

    Inherits from GenericMVSDataset and implements missing methods.
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
        self.capture_poses = {} # Needed in super init
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


        self.capture_poses = {}
        self.image_resampling_mode = pil.BILINEAR

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        return frame_id

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""

        scan_name, colmap_model = scan.split(':')
        scan_dir = (
            Path(self.dataset_path) / 
            scan_name / 
            "colmap" / 
            "sparse" /
            colmap_model
        )

        return scan_dir / "valid_frames.txt"

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
        scan = scan.rstrip("\n")
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            print(f"Compuiting valid frames for scene {scan}.")
            # find out which frames have valid poses

            # load capture poses for this scan
            self.load_capture_poses(scan)

            bad_file_count = 0
            dist_to_last_valid_frame = 0
            valid_frames = []
            for frame_id in sorted(self.capture_poses[scan]["images"]):
                world_T_cam_44, _ = self.load_pose(scan, frame_id)

                if (
                    np.isnan(np.sum(world_T_cam_44))
                    or np.isinf(np.sum(world_T_cam_44))
                    or np.isneginf(np.sum(world_T_cam_44))
                ):
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                color_file = self.get_color_filepath(scan, frame_id)
                if not os.path.exists(color_file):
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(
                f"Scene {scan} has {bad_file_count} bad frame files out of "
                f"{len(self.capture_poses[scan]['images'])}."
            )

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
        if scan_id not in self.capture_poses:
            self.load_capture_poses(scan_id)

        img = self.capture_poses[scan_id]["images"][int(frame_id)]

        R = qvec2rotmat(-img.qvec)
        t = img.tvec.reshape([3, 1])

        m = np.concatenate([
            np.concatenate([R, t], 1), 
            np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        ], 0)
        world_T_cam = np.linalg.inv(m)
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def load_intrinsics(self, scan_id, frame_id=None, flip=None):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        This function assumes all images have the same intrinsics and
        doesn't handle per image intrinsics from COLMAP

        Images are assumed undistored, so using simple pinhole.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.
            flip: unused

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

        if scan_id not in self.capture_poses:
            self.load_capture_poses(scan_id)

        img = self.capture_poses[scan_id]["images"][int(frame_id)]
        camera = self.capture_poses[scan_id]["cameras"][int(img.camera_id)]

        w = camera.width
        h = camera.height
        fx = camera.params[0]
        fy = camera.params[0]
        cx = w / 2
        cy = h / 2

        if camera.model=='SIMPLE_PINHOLE':
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model=='PINHOLE':
            fy = camera.params[1]
            cx = camera.params[2]
            cy = camera.params[3]
        elif camera.model=='SIMPLE_RADIAL':
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model=='RADIAL':
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model=='OPENCV':
            fy = camera.params[1]
            cx = camera.params[2]
            cy = camera.params[3]
        else:
            print("unknown camera model ", camera.model)

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            K_full = K.clone()
            K_full[0] *= self.depth_width / float(w)
            K_full[1] *= self.depth_height / float(h)
            output_dict[f"K_full_depth_b44"] = K_full
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(K_full)

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(w)
        K_matching[1] *= self.matching_height / float(h)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.linalg.inv(K_matching)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / float(w)
        K[1] *= self.depth_height / float(h)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = torch.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, None

    def load_capture_poses(self, scan_id):
        """Loads in poses for a scan in COLMAP format. Saves these to the
        self.capture_poses dictionary under the key scan_id

        Args:
            scan_id: the id of the scan whose poses will be loaded.
        """

        if scan_id in self.capture_poses:
            return
        
        scan_name, colmap_model = scan_id.split(':')
        scene_path = (
            Path(self.dataset_path) / 
            scan_name / 
            "colmap" /
            "sparse" / 
            colmap_model
        )

        self.capture_poses[scan_id] = {
            'images': rwm.read_images_binary(scene_path / "images.bin"),
            'cameras': rwm.read_cameras_binary(scene_path / "cameras.bin"),
        }


    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for.

        This function is not implemented for COLMAP
        """
        depth = torch.zeros((1, self.depth_height, self.depth_width), dtype=float)

        # # Get the float valid mask
        mask_b = (depth > 0.0)
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b
    
    def load_full_res_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the native resolution the dataset provides.

        This function is not implemented for COLMAP
        """
        depth = torch.zeros((1, self.depth_height, self.depth_width), dtype=float)

        # # Get the float valid mask
        mask_b = (depth > 0.0)
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

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
        return self.get_high_res_color_filepath(scan_id, frame_id)

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the high res
            size required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        if scan_id not in self.capture_poses:
            self.load_capture_poses(scan_id)
    
        color_path = (
            Path(self.dataset_path) / 
            scan_id.split(':')[0] / 
            "images" / 
            self.capture_poses[scan_id]["images"][int(frame_id)].name
        )
        return str(color_path)

