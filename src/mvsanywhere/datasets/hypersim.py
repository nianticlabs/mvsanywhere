import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import h5py
import scipy
import json
import cv2

from mvsanywhere.datasets.generic_mvs_dataset import GenericMVSDataset
from mvsanywhere.utils.geometry_utils import rotx


class HypersimDataset(GenericMVSDataset):
    """
    MVS Hypersim Dataset class.

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

        path = Path(self.dataset_path)
        scan_dir = (
            path.parent.parent 
            / (path.parent.name + "_preprocessed") 
            / "valid_frames"
            / self.get_sub_folder_dir(split)
            / scan
        )

        scan_dir.mkdir(parents=True, exist_ok=True)

        return os.path.join(str(scan_dir), "valid_frames.txt")

    def _get_frame_ids(self, split, scan):

        split_path = Path("data_splits/hypersim/")

        if split == "test":
            split_files_json_path = split_path / "standard_split" / f"{split}_files_all.json"
        else:
            split_files_json_path = split_path / "bd_split" / f"{split}_files_bd.json"

        with open(split_files_json_path, "r") as f:
            frame_ids = json.load(f)[scan]

        return frame_ids

    def _check_hypersim_img_not_anomalous(self, img, threshold=0.3):
        """Looks for bad hypersim images by checking the fraction of pixels in the image
        that are of the same value, if the fraction is higher than the set threshold,
        we consider this image to be anomalous

        Args:
            img: np.array hypersim image.
            threshold: threshold for the fraction of same pixels in an image over which
            we consider an image anomalous.

        Returns:
            False if the image is anomalous, True otherwise.

        """
        mode_count = scipy.stats.mode(img, axis=None, keepdims=True).count[0]
        num_px_total = img.size
        mode_frac = mode_count / num_px_total
        if mode_frac > threshold:
            return False
        return True

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
            # find out which frames have valid poses
            print(f"computing valid frames for scene {scan}.")

            frame_ids = self._get_frame_ids(split, scan)

            valid_frames = []
            dist_to_last_valid_frame = 0
            bad_file_count = 0
            for frame_ind in frame_ids:

                image_path = self.get_color_filepath(scan, frame_ind)
                img = np.array(Image.open(image_path))

                if not self._check_hypersim_img_not_anomalous(img, 0.3):
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                depth_filepath = self.get_full_res_depth_filepath(scan, frame_ind)

                depth_path = Path(depth_filepath)
                depth = np.array(h5py.File(depth_path)["dataset"]).astype(np.float32)

                if not self._check_hypersim_img_not_anomalous(depth, 0.3):
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
        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = scan.parent, scan.name

        cached_resized_path = (
            path
            / scene
            / "images"
            / f"scene_{cam}_final_preview"
            / f"frame.{int(frame_id):04d}.tonemap.{self.image_width}_{self.image_height}.png"
        )

        if cached_resized_path.exists():
            return str(cached_resized_path)

        path = (
            path
            / scene
            / "images"
            / f"scene_{cam}_final_preview"
            / f"frame.{int(frame_id):04d}.tonemap.jpg"
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
        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = scan.parent, scan.name

        path = (
            path
            / scene
            / "images"
            / f"scene_{cam}_final_preview"
            / f"frame.{int(frame_id):04d}.tonemap.jpg"
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
        path = Path(self.dataset_path)
        path = path.parent.parent / (path.parent.name + "_preprocessed")
        scan = Path(scan_id)
        scene, cam = scan.parent, scan.name

        path = (
            path
            / scene
            / "images"
            / f"scene_{cam}_geometry_hdf5"
            / f"frame.{int(frame_id):04d}.depth_meters_planar.hdf5"
        )

        return str(path)

    def get_full_res_distance_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's distance file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            The full size depth (distance) frame from the dataset.

        """
        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = scan.parent, scan.name

        path = (
            path
            / scene
            / "images"
            / f"scene_{cam}_geometry_hdf5"
            / f"frame.{int(frame_id):04d}.depth_meters.hdf5"
        )

        return str(path)

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """

        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = scan.parent, scan.name

        path = path / scene / "_detail" / cam / frame_id

        return path

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
        output_dict = {}

        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = str(scan.parent), scan.name

        metadata_path = (
            path.parent.parent 
            / (path.parent.name + "_preprocessed") 
            / "metadata_camera_parameters.csv"
        )

        df_camera_parameters = pd.read_csv(metadata_path, index_col="scene_name")
        df_ = df_camera_parameters.loc[scene]

        width_pixels = int(df_["settings_output_img_width"])
        height_pixels = int(df_["settings_output_img_height"])

        M_proj = np.array(
            [
                [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
                [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
                [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
                [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
            ]
        )

        # matrix to map to integer screen coordinates from normalized device coordinates
        M_screen_from_ndc = np.matrix(
            [
                [0.5 * (width_pixels - 1), 0, 0, 0.5 * (width_pixels - 1)],
                [0, -0.5 * (height_pixels - 1), 0, 0.5 * (height_pixels - 1)],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1.0],
            ]
        )

        # Extract fx, fy, cx and cy, and build a corresponding matrix
        screen_from_cam = M_screen_from_ndc @ M_proj
        fx = abs(screen_from_cam[0, 0])
        fy = abs(screen_from_cam[1, 1])
        cx = abs(screen_from_cam[0, 2])
        cy = abs(screen_from_cam[1, 2])

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)

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

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(K)

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width_pixels)
        K_matching[1] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.linalg.inv(K_matching)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / float(width_pixels)
        K[1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = torch.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict, (left, top, left+width_pixels, top+height_pixels)

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

        # Load depth, resize
        depth_path = Path(depth_filepath)
        distance = np.array(h5py.File(depth_path)["dataset"]).astype(np.float32)
        if crop:
            distance = distance[
                crop[1]:crop[3],
                crop[0]:crop[2]
            ]

        depth = distance

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )

        depth = torch.tensor(depth).float().unsqueeze(0)

        # # Get the float valid mask
        mask_b = (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
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
        full_res_depth = torch.tensor(
            np.array(h5py.File(full_res_depth_filepath)["dataset"]).astype(np.float32)
        ).unsqueeze(0)

        # # Get the float valid mask
        full_res_mask_b = (full_res_depth > self.min_valid_depth) & (
            full_res_depth < self.max_valid_depth
        )
        full_res_mask = full_res_mask_b.float()

        # set invalids to nan
        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

    def _get_cam_position(self, pose_path, frame):
        """Loads a frame's camera_position.

        Args:
            pose_path: the path to the folder that contains camera positions for a scene.
            frame: id for the frame.

        Returns:
            camera_position (numpy array): matrix for transforming from the
                camera to the world (pose).

        """
        camera_positions_hdf5_file = pose_path / "camera_keyframe_positions.hdf5"
        with h5py.File(camera_positions_hdf5_file, "r") as f:
            camera_position = f["dataset"][frame]
        return camera_position

    def _get_cam_orientation(self, pose_path, frame):
        """Loads a frame's camera_orientation.

        Args:
            pose_path: the path to the folder that contains camera orientations for a scene.
            frame: id for the frame.

        Returns:
            camera_orientation (numpy array): matrix for transforming from the
                camera to the world (pose).

        """
        camera_orientations_hdf5_file = pose_path / "camera_keyframe_orientations.hdf5"
        with h5py.File(camera_orientations_hdf5_file, "r") as f:
            camera_orientation = f["dataset"][frame]
        return camera_orientation

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

        metadata_path, pose_data_path, frame = (
            pose_path.parent.parent,
            pose_path.parent,
            int(pose_path.name),
        )

        scene_metadata = pd.read_csv(metadata_path / "metadata_scene.csv")
        meters_per_asset_unit = scene_metadata[
            scene_metadata.parameter_name == "meters_per_asset_unit"
        ]
        assert len(meters_per_asset_unit) == 1  # Should not be multiply defined
        meters_per_asset_unit = meters_per_asset_unit.parameter_value[0]
        scale_factor = float(meters_per_asset_unit)

        camera_position_world = self._get_cam_position(pose_data_path, frame)
        R_world_from_cam = self._get_cam_orientation(pose_data_path, frame)

        t_world_from_cam = np.array(camera_position_world).T
        R_cam_from_world = np.array(R_world_from_cam).T
        t_cam_from_world = -R_cam_from_world @ t_world_from_cam

        M_cam_from_world = np.zeros((4, 4))
        M_cam_from_world[:3, :3] = R_cam_from_world
        M_cam_from_world[:3, 3] = t_cam_from_world * scale_factor
        M_cam_from_world[3, 3] = 1.0

        M_world_from_cam = np.zeros((4, 4))
        M_world_from_cam[:3, :3] = R_world_from_cam
        M_world_from_cam[:3, 3] = t_world_from_cam * scale_factor
        M_world_from_cam[3, 3] = 1.0

        world_T_cam = M_world_from_cam.astype(np.float32)

        gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])

        world_T_cam *= gl_to_cv

        rot_mat = world_T_cam[:3, :3]
        trans = world_T_cam[:3, 3]

        rot_mat = rotx(-np.pi / 2) @ rot_mat
        trans = rotx(-np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans

        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def _get_M_cam_from_uv(self, scan_id):
        """Loads M_cam_from_uv, which maps points to camera-space from uv-space.

        Args:
            scan_id: the scan this file belongs to.

        Returns:
            M_cam_from_uv

        """

        path = Path(self.dataset_path)
        scan = Path(scan_id)
        scene, cam = str(scan.parent), scan.name

        metadata_path = (
            path.parent.parent 
            / (path.parent.name + "_preprocessed") 
            / "metadata_camera_parameters.csv"
        )

        df_camera_parameters = pd.read_csv(metadata_path, index_col="scene_name")
        df_ = df_camera_parameters.loc[scene]

        M_cam_from_uv = np.matrix(
            [
                [df_["M_cam_from_uv_00"], df_["M_cam_from_uv_01"], df_["M_cam_from_uv_02"]],
                [df_["M_cam_from_uv_10"], df_["M_cam_from_uv_11"], df_["M_cam_from_uv_12"]],
                [df_["M_cam_from_uv_20"], df_["M_cam_from_uv_21"], df_["M_cam_from_uv_22"]],
            ]
        )

        return M_cam_from_uv

    def _get_rays_hypersim_torch(self, intrinsics_inv, H, W):
        """Compute rays from hypersim camera
        Args:
            intrinsics: [B,3x3]
            H, W: int
        Returns:
            rays_d_cam: [H, W, 3]
        """

        device = intrinsics_inv.device

        width_pixels = W
        height_pixels = H

        u_min = -1.0
        u_max = 1.0
        v_min = -1.0
        v_max = 1.0
        half_du = 0.5 * (u_max - u_min) / width_pixels
        half_dv = 0.5 * (v_max - v_min) / height_pixels

        u = torch.linspace(u_min + half_du, u_max - half_du, width_pixels, device=device)
        v = torch.linspace(v_min + half_dv, v_max - half_dv, height_pixels, device=device).flip(
            dims=(-1,)
        )

        u, v = torch.meshgrid(u, v, indexing="ij")
        u = u.T
        v = v.T

        P_uv = torch.stack((u, v, torch.ones_like(u, device=device)), dim=-1)

        rays_d_cam = P_uv @ intrinsics_inv.T
        rays_d_cam = rays_d_cam / torch.linalg.norm(rays_d_cam, dim=-1, keepdim=True)

        return rays_d_cam

    def _get_prependicular_depths(self, scan_id, frame_id):
        """gets prependicular depths from distances.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            depths_perpendicular (numpy array): perpendicular depth.

        """

        distance_filepath = self.get_full_res_distance_filepath(scan_id, frame_id)

        distance_path = Path(distance_filepath)
        distance = np.array(h5py.File(distance_path)["dataset"]).astype(np.float32)

        distance = torch.tensor(distance)
        h, w = distance.shape

        M_cam_from_uv = self._get_M_cam_from_uv(scan_id)
        M_cam_from_uv = torch.tensor(M_cam_from_uv).float()

        rays_d_cam = self._get_rays_hypersim_torch(M_cam_from_uv, h, w)

        depths_perpendicular = -distance * rays_d_cam[..., -1]

        return depths_perpendicular.cpu().numpy()

    def _save_prependicular_depths_to_disk(self, scan_id, frame_id):
        """saves prependicular depths to disk.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        """
        depth = self._get_prependicular_depths(scan_id, frame_id)
        depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
        Path(depth_filepath).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(depth_filepath, "w") as f:
            f.create_dataset(
                "dataset",
                data=depth.astype(np.float16),
                compression="gzip",
                compression_opts=4,
            )