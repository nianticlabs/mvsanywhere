import os
from pathlib import Path

from doubletake.datasets.change_of_basis import ChangeOfBasis

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset


class SAILVOS3DDataset(GenericMVSDataset):
    """
    MVS SAILVOS3D Dataset class.

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

        self.original_width, self.original_height = 1280, 800

        self.tuple_info_file_location = tuple_info_file_location

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

        scan_dir = Path(self.dataset_path) / "valid_frames" / scan

        scan_dir.mkdir(parents=True, exist_ok=True)

        return os.path.join(str(scan_dir), "valid_frames.txt")

    def _get_frame_ids(self, split, scan):
        image_files = (Path(self.dataset_path) / scan / "images").glob("*.bmp")
        return [img.stem for img in image_files]

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
                    np.array(Image.open(image_path))
                    # assert os.path.isfile(image_path)
                except:
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                depth_filepath = self.get_full_res_depth_filepath(scan, frame_ind)
                try:
                    depth = np.load(depth_filepath)
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

        if len(valid_frames) == 1 and valid_frames[0] == "\n":
            return []

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
        cached_resized_path = (
            Path(self.dataset_path)
            / scan_id
            / "images"
            / f"{frame_id}.{self.image_width}_{self.image_height}.bmp"
        )

        if cached_resized_path.exists():
            return str(cached_resized_path)

        path = Path(self.dataset_path) / scan_id / "images" / f"{frame_id}.bmp"

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
        path = Path(self.dataset_path) / scan_id / "images" / f"{frame_id}.bmp"

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
        path = Path(self.dataset_path) / scan_id / "depth" / f"{frame_id}.npy"

        return str(path)

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.
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
        # Use actual image dimensions
        height_pixels, width_pixels = self.image_height, self.image_width

        rage_matrices = self._load_rage_matrices(scan_id, frame_id)

        intrinsics = torch.from_numpy(
            self.compute_intrinsics_from_P(
                rage_matrices["P"], self.original_width, self.original_height
            )
        )

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = intrinsics.clone()

        # Scale intrinsics
        scale_x = width_pixels / self.original_width
        scale_y = height_pixels / self.original_height
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

        if flip:
            K[0, 2] = float(width_pixels) - K[0, 2]

        output_dict = {}

        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.inverse(K)

        K_matching = K.clone()
        K_matching[0, 0] *= self.matching_width / float(width_pixels)
        K_matching[1, 1] *= self.matching_height / float(height_pixels)
        K_matching[0, 2] *= self.matching_width / float(width_pixels)
        K_matching[1, 2] *= self.matching_height / float(height_pixels)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.inverse(K_matching)

        # Scale intrinsics to the dataset's configured depth resolution.
        K_depth = K.clone()
        K_depth[0, 0] *= self.depth_width / float(width_pixels)
        K_depth[1, 1] *= self.depth_height / float(height_pixels)

        # Get the intrinsics of all scales at various resolutions.
        for i in range(self.prediction_num_scales):
            K_scaled = K_depth.clone()
            K_scaled[0, 0] /= 2**i
            K_scaled[1, 1] /= 2**i
            K_scaled[0, 2] /= 2**i
            K_scaled[1, 2] /= 2**i
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = torch.inverse(K_scaled)

        return output_dict, None

    def _load_rage_matrices(self, scan_id, frame_id):
        """
        Loads the rage matrices for a specific scan and frame.

        Parameters:
            scan_id (str): Identifier for the scan.
            frame_id (str): Identifier for the frame within the scan.

        Returns:
            np.lib.npyio.NpzFile: Loaded rage matrices from the .npz file.
        """
        path = Path(self.dataset_path) / scan_id / "rage_matrices" / f"{frame_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Rage matrices file not found: {path}")
        try:
            matrices = np.load(path)
            return matrices
        except Exception as e:
            raise IOError(f"Failed to load rage matrices from {path}: {e}")

    def convert_ndc_depth_to_cam(
        self, depth: np.ndarray, P_inverse: np.ndarray, H: int, W: int
    ) -> np.ndarray:
        """
        Converts a depth map from Normalized Device Coordinates (NDC) to camera space.
        Parameters
        ----------
        depth : np.ndarray
            A 2D numpy array of shape (H, W) representing the depth map in NDC. Depth values are
            assumed to be in a normalized range and will be scaled accordingly.

        P_inverse : np.ndarray
            A 4x4 numpy array representing the inverse of the projection matrix. This matrix is used
            to transform NDC coordinates to camera coordinates.

        H : int
            The height of the depth map in pixels.

        W : int
            The width of the depth map in pixels.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (H, W) containing the depth values in camera space.

        """
        # Apply depth scaling based on dataset specification
        depth_scaled = depth / 6.0 - 4e-5

        # Generate pixel coordinates
        px = np.arange(W)
        py = np.arange(H)
        px_grid, py_grid = np.meshgrid(px, py, sparse=False)
        px_flat = px_grid.reshape(-1)
        py_flat = py_grid.reshape(-1)

        # Retrieve depth values at each pixel
        ndcz = depth_scaled[py_flat, px_flat]  # Depth in NDC

        # Convert pixel coordinates to NDC
        ndcx, ndcy = self.pixels_to_ndcs(px_flat, py_flat, (H, W))

        # Stack NDC coordinates with depth and homogeneous coordinate
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)  # Shape: (N, 4)

        # Transform NDC coordinates to camera space
        camera_coord = ndc_coord @ P_inverse  # Shape: (N, 4)
        camera_coord /= camera_coord[:, -1:]

        # Extract and negate the Z-component to align with camera forward direction
        depth_cam = -camera_coord[:, 2].reshape(H, W)  # Shape: (H, W)

        return depth_cam

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
        rage_matrices = self._load_rage_matrices(scan_id, frame_id)

        depth = np.load(depth_filepath)
        depth = self.convert_ndc_depth_to_cam(
            depth, rage_matrices["P_inv"], depth.shape[0], depth.shape[1]
        )

        if crop:
            depth = depth[crop[1] : crop[3], crop[0] : crop[2]]

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )

        mask_b = depth > 0

        mask_b = torch.tensor(mask_b).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

        # # Get the float valid mask
        mask = mask_b.float()

        if mask.sum() == 0:
            print("0")

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

    @staticmethod
    def pixels_to_ndcs(
        xx: np.ndarray, yy: np.ndarray, size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts pixel coordinates to Normalized Device Coordinates (NDC).

        Parameters
        ----------
        xx : np.ndarray
            A 1D numpy array of x pixel coordinates.

        yy : np.ndarray
            A 1D numpy array of y pixel coordinates.

        size : Tuple[int, int]
            A tuple containing the height (H) and width (W) of the image.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two 1D numpy arrays representing the x and y coordinates in NDC space.
        """
        s_y, s_x = size
        s_x -= 1  # so 1 is being mapped into (n-1)th pixel
        s_y -= 1  # so 1 is being mapped into (n-1)th pixel
        x_ndc = (2.0 / s_x) * xx - 1.0
        y_ndc = (-2.0 / s_y) * yy + 1.0
        return x_ndc, y_ndc

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
        rage_matrices = self._load_rage_matrices(scan_id, frame_id)
        full_res_depth = np.load(full_res_depth_filepath)
        full_res_depth = self.convert_ndc_depth_to_cam(
            full_res_depth, rage_matrices["P_inv"], full_res_depth.shape[0], full_res_depth.shape[1]
        )

        full_res_mask_b = full_res_depth > 0
        full_res_mask_b = torch.tensor(full_res_mask_b).bool().unsqueeze(0)

        full_res_depth = torch.tensor(full_res_depth).float().unsqueeze(0)

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
        camera_file = Path(self.dataset_path) / scan_id / "camera" / f"{frame_id}.yaml"
        if not camera_file.exists():
            return np.full((4, 4), np.nan), np.full((4, 4), np.nan)

        with open(camera_file, "r") as f:
            lines = f.readlines()
        extrinsics = np.array([list(map(float, lines[5 + i][3:-2].split(","))) for i in range(3)])

        # kornia.geometry.conversions.worldtocam_to_camtoworld_Rt
        world_T_cam = np.eye(4)
        world_T_cam[:3, :3] = extrinsics[:3, :3].T
        world_T_cam[:3, -1:] = -extrinsics[:3, :3].T @ extrinsics[:3, -1:]
        world_T_cam = ChangeOfBasis.convert_matrix_to_vision_convention(world_T_cam).astype(
            np.float32
        )

        world_T_cam = world_T_cam.astype(np.float32)

        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    @staticmethod
    def compute_intrinsics_from_P(P: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        """
        Computes camera intrinsics in pixel space from the projection matrix P.

        Args:
            P: 4x4 projection matrix (numpy array)
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            K: 3x3 depth  intrinsic matrix
        """

        P00 = P[0, 0]
        P11 = P[1, 1]

        # Compute field of view angles
        fov_x = 2 * np.arctan(1 / P00)
        fov_y = 2 * np.arctan(1 / P11)

        # Compute focal lengths in pixels
        fx = (image_width / 2) / np.tan(fov_x / 2)
        fy = (image_height / 2) / np.tan(fov_y / 2)

        cx = image_width / 2
        cy = image_height / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return K
