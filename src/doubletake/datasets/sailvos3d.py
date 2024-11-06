import os
from pathlib import Path
import glob
import json

import kornia

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
from doubletake.datasets.change_of_basis import ChangeOfBasis

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
from PIL import Image
import torch
import cv2

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

        # Load all transforms files to save IO time
        # self.transforms = {}
        # transforms_paths = Path(self.dataset_path) / "big_city/**/train/**/transforms.json"
        # for transform_file in glob.glob(str(transforms_paths)):
        #     with open(transform_file) as f:
        #         transforms = json.load(f)
        #     self.transforms["/".join(transform_file.split("/")[-5:-1])] = transforms


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

        scan_dir = (
            Path(self.dataset_path)
            / "valid_frames"
            / scan
        )

        scan_dir.mkdir(parents=True, exist_ok=True)

        return os.path.join(str(scan_dir), "valid_frames.txt")

    def _get_frame_ids(self, split, scan):
        image_files = (
            Path(self.dataset_path)
            / scan
            / "images"
        ).glob("*.bmp")
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

        if len(valid_frames) == 1 and valid_frames[0] == '\n':
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

        path = (
            Path(self.dataset_path)
            / scan_id
            / "images"
            / f"{frame_id}.bmp"
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
        path = (
            Path(self.dataset_path)
            / scan_id
            / "images"
            / f"{frame_id}.bmp"
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
        path = (
            Path(self.dataset_path)
            / scan_id
            / "depth"
            / f"{frame_id}.npy"
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
        height_pixels, width_pixels = 800, 1200

        
        camera_file = (
            Path(self.dataset_path)
            / scan_id
            / "camera"
            / f"{frame_id}.yaml"
        )
        
        with open(camera_file, "r") as f:
            lines = f.readlines()
        intrinsics = torch.tensor(np.array([list(map(float, lines[1+i][3:-2].split(","))) for i in range(3)]))

        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = intrinsics
        K[2, 2] *= -1.0
        K[0, 2] = 1200 / 2
        K[1, 2] = 800 / 2


        focal_length = torch.tensor([K[0, 0], K[1, 1]])
        focal_length = (focal_length * torch.tensor([1200, 800]) / 2) / 400

        principal_point = torch.tensor([0, 0])

        pcam = PerspectiveCameras(focal_length=focal_length[None], principal_point=principal_point[None], R=torch.eye(4)[None], T=torch.zeros(3)[None])
        _, _, KK = opencv_from_cameras_projection(pcam, torch.tensor([800, 1200])[None])
        K[0, 0] = focal_length[0]
        K[1, 1] = focal_length[1]
        # K[3, 3] = 1
        

        # K[0, 2] = 1200
        # K[1, 2] = 800

        # top, left, h, w = self.random_resize_crop.get_params(
        #     torch.empty((height_pixels, width_pixels)),
        #     self.random_resize_crop.scale,
        #     self.random_resize_crop.ratio
        # )
        # K[0, 2] = K[0, 2] - left
        # K[1, 2] = K[1, 2] - top
        # width_pixels = w
        # height_pixels = h

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

        return output_dict, None #(left, top, left+width_pixels, top+height_pixels)
    
    def _load_rage_matrices(self, scan_id, frame_id):
        path = (
            Path(self.dataset_path)
            / scan_id
            / "rage_matrices"
            / f"{frame_id}.npz"
        )
        return np.load(path)
    
    @staticmethod
    def convert_ndc_depth_to_cam(depth, P_inverse, H, W):
        def pixels_to_ndcs(xx, yy, size):
            s_y, s_x = size
            s_x -= 1  # so 1 is being mapped into (n-1)th pixel
            s_y -= 1  # so 1 is being mapped into (n-1)th pixel
            x = (2 / s_x) * xx - 1
            y = (-2 / s_y) * yy + 1
            return x, y
        
        depth = depth/6.0 - 4e-5

        px = np.arange(0, W)
        py = np.arange(0, H)
        px, py = np.meshgrid(px, py, sparse=False)
        px = px.reshape(-1)
        py = py.reshape(-1)

        ndcz = depth[py, px] # get the depth in NDC
        ndcx, ndcy = pixels_to_ndcs(px, py, (H, W))
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)

        camera_coord = ndc_coord @ P_inverse
        camera_coord = camera_coord/camera_coord[:,-1:]

        return -camera_coord[..., 2].reshape(H, W)


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
        depth = self.convert_ndc_depth_to_cam(depth, rage_matrices['P_inv'], 800, 1200)

        if crop:
            depth = depth[
                crop[1]:crop[3],
                crop[0]:crop[2]
            ]

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )


        mask_b = depth > 0
        
        mask_b = torch.tensor(mask_b).bool().unsqueeze(0)
        depth = torch.tensor(depth).float().unsqueeze(0)

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
        rage_matrices = self._load_rage_matrices(scan_id, frame_id)
        full_res_depth = np.load(full_res_depth_filepath)
        full_res_depth = self.convert_ndc_depth_to_cam(full_res_depth, rage_matrices['P_inv'], 800, 1200)

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
        camera_file = (
            Path(self.dataset_path)
            / scan_id
            / "camera"
            / f"{frame_id}.yaml"
        )
        if not camera_file.exists():
            return np.full((4, 4), np.nan), np.full((4, 4), np.nan)

        with open(camera_file, "r") as f:
            lines = f.readlines()
        extrinsics = np.array([list(map(float, lines[5+i][3:-2].split(","))) for i in range(3)])

        # rage_matrices = self._load_rage_matrices(scan_id, frame_id)

        # world_T_cam = np.eye(4)
        # world_T_cam[:3] = pose
        # world_T_cam[:3, 3] /= 6.0


        # Pytorch 3d stuff
        # pose[:3, -1:] /= 6.0
        # pose2 = np.eye(4)
        # pose2[:3, :3] = pose[:3, :3].T
        # pose2[:3, -1:] = -pose[:3, :3].T @ pose[:3, -1:]

        # # pose = np.linalg.inv((rage_matrices["VP"] @ rage_matrices["P_inv"]).T)

        # pcam = PerspectiveCameras(R=pose2[:3, :3][None], T=pose2[:3, 3][None])
        # R, t, _ = opencv_from_cameras_projection(pcam, torch.tensor([384, 512])[None])
        # world_to_cam = np.eye(4)
        # world_to_cam[:3, :3] = R[0]
        # world_to_cam[:3, 3] = t[0]

        # kornia.geometry.conversions.worldtocam_to_camtoworld_Rt
        world_T_cam = np.eye(4)
        world_T_cam[:3, :3] = extrinsics[:3, :3].T
        world_T_cam[:3, -1:] = -extrinsics[:3, :3].T @ extrinsics[:3, -1:]
        world_T_cam = ChangeOfBasis.convert_matrix_to_vision_convention(world_T_cam).astype(np.float32)
        # world_T_cam = kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4(torch.tensor(world_T_cam)[None])[0].numpy()



        # cam_T_world = world_to_cam.astype(np.float32)
        # world_T_cam = np.linalg.inv(cam_T_world)

        world_T_cam = world_T_cam.astype(np.float32)
        # world_T_cam = ChangeOfBasis.convert_matrix_to_vision_convention(world_T_cam).astype(np.float32)
        
        # Other way around?
        cam_T_world = np.linalg.inv(world_T_cam)

        # world_T_cam = np.linalg.inv(world_T_cam)
        # cam_T_world = np.linalg.inv(cam_T_world)

        return world_T_cam, cam_T_world
