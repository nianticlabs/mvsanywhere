import json
import os
from collections import OrderedDict

import numpy as np
import PIL.Image as pil
import torch

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from doubletake.utils.generic_utils import read_image_file, readlines


class ThreeRScanDataset(GenericMVSDataset):
    """
    MVS 3RScanv2 Dataset class.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.

    This dataset expects 3RScanv2 to be in the following format:

    <dataset_path>
        <scanId>
        |-- mesh.refined.v2.obj
            Reconstructed mesh
        |-- mesh.refined.mtl
            Corresponding material file
        |-- mesh.refined_0.png
            Corresponding mesh texture
        |-- sequence.zip
            Calibrated RGB-D sensor stream with color and depth frames, camera poses
        |-- sensor_data
            extracted sensor data from the sequence.zip
            |-- _info.txt
                Metadata file
            |-- frame-000000.color.jpg
                Color image
            |-- frame-000000.depth.pgm
                Depth image
            |-- frame-000000.pose.txt
                Camera pose
        |-- labels.instances.annotated.v2.ply
            Visualization of semantic segmentation
        |-- mesh.refined.0.010000.segs.v2.json
            Over-segmentation of annotation mesh
        |-- semseg.v2.json
            Instance segmentation of the mesh (contains the labels)

    _info.txt contains for example the following information:
        m_versionNumber = 4
        m_sensorName = tangoDevice (calibrated)
        m_colorWidth = 960
        m_colorHeight = 540
        m_depthWidth = 224
        m_depthHeight = 172
        m_depthShift = 1000
        m_calibrationColorIntrinsic = 877.5 0 479.75 0 0 877.5 269.75 0 0 0 1 0 0 0 0 1
        m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        m_calibrationDepthIntrinsic = 204.75 0 111.558 0 0 279.5 85.5793 0 0 0 1 0 0 0 0 1
        m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        m_frames.size = 443

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
        fill_depth_hints=False,
        load_empty_hints=False,
        depth_hint_aug=0.0,
        depth_hint_dir=None,
        disable_flip=False,
        rotate_images=False,
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
            fill_depth_hints=fill_depth_hints,
            load_empty_hints=load_empty_hints,
            depth_hint_dir=depth_hint_dir,
            depth_hint_aug=depth_hint_aug,
            disable_flip=disable_flip,
            rotate_images=rotate_images,
            native_depth_height=192,
            native_depth_width=256,
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
        return ""

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

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(split), scan)

        return os.path.join(scan_dir, "valid_frames.txt")

    @classmethod
    def parse_rescan_transforms(cls, dataset_path: str, scan_list: list[str]):
        """
        Parses the rescan metadata file and returns a dictionary of dictionaries
        where the first key is the reference scan id and the second key is the
        rescan scan id. The value is the transform from the rescan to the
        reference scan.
        example:
        {
            "scene0707_00": {
                "scene0707_01": np.array([[0.999263, -0.010031, 0.037048, -0.038549],
                                        [0.010031, 0.999945, -0.000101, -0.000000],
                                        [-0.037048, 0.000101, 0.999311, -0.000000],
                                        [0.000000, 0.000000, 0.000000, 1.000000]], dtype=np.float32),
                "scene0707_02": np.array([[...], [...], [...], [...]], dtype=np.float32),
                ...
            },
            ...
        }
        """
        scene_metadata_file_path = os.path.join(dataset_path, "3RScan.json")
        forbidden_list_path = "data_splits/3rscan/forbidden_list.txt"
        forbidden_scan_list = readlines(forbidden_list_path)

        scene_metadata = json.load(open(scene_metadata_file_path, "r"))
        rescan_map = {}
        for scene in scene_metadata:
            if scene["reference"] not in scan_list:
                continue
            rescans = OrderedDict()
            for rescan in scene["scans"]:
                if "transform" in rescan:
                    re_to_ref_transform = (
                        np.array([float(x) for x in rescan["transform"]], dtype=np.float32)
                        .reshape(4, 4)
                        .T
                    )
                    if rescan["reference"] in forbidden_scan_list:
                        continue
                    rescans[rescan["reference"]] = re_to_ref_transform
                else:
                    continue
            if len(rescans) > 0:
                rescan_map[scene["reference"]] = rescans
        return rescan_map

    def _parse_metadata(self, metadata_file_path):
        """
        Parses the metadata file for a scan and returns a dictionary of
        the metadata.
        it is assumed that the metadata file is in the format:
        key = value
        where value can be a string, int, float, or np.array.

        m_versionNumber 4
        m_sensorName tangoDevice (calibrated)
        m_colorWidth 960
        m_colorHeight 540
        m_depthWidth 224
        m_depthHeight 172
        m_depthShift 1000
        m_calibrationColorIntrinsic
        [[877.5    0.   479.75   0.  ]
        [  0.   877.5  269.75   0.  ]
        [  0.     0.     1.     0.  ]
        [  0.     0.     0.     1.  ]]
        m_calibrationColorExtrinsic
        [[1. 0. 0. 0.]
        [0. 1. 0. 0.]
        [0. 0. 1. 0.]
        [0. 0. 0. 1.]]
        m_calibrationDepthIntrinsic
        [[204.75     0.     111.558    0.    ]
        [  0.     279.5     85.5793   0.    ]
        [  0.       0.       1.       0.    ]
        [  0.       0.       0.       1.    ]]
        m_calibrationDepthExtrinsic
        [[1. 0. 0. 0.]
        [0. 1. 0. 0.]
        [0. 0. 1. 0.]
        [0. 0. 0. 1.]]
        m_frames.size 443
        """

        meta_data = {}
        with open(metadata_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split(" = ")
                if "calibration" in key:
                    value = np.array(
                        [float(x) for x in value.split(" ")], dtype=np.float32
                    ).reshape(4, 4)
                meta_data[key] = value

        return meta_data

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
            # find out which frames have valid poses

            # get 3RScan directories
            scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(split), scan)
            sensor_data_dir = os.path.join(scan_dir, "sensor_data")
            meta_file_path = os.path.join(sensor_data_dir, "_info.txt")

            # parse metadata
            meta_data = self._parse_metadata(meta_file_path)

            # fetch total number of color files
            color_file_count = int(meta_data["m_frames.size"].strip())

            dist_to_last_valid_frame = 0
            bad_file_count = 0
            valid_frames = []
            for frame_id in range(color_file_count):
                # for a frame to be valid, we need a valid pose and a valid
                # color frame.

                color_filename = os.path.join(sensor_data_dir, f"frame-{frame_id:06d}.color.jpg")
                depth_filename = color_filename.replace(f"color.jpg", f"depth.pgm")
                pose_path = os.path.join(sensor_data_dir, f"frame-{frame_id:06d}.pose.txt")

                # check if an image file exists.
                if not os.path.isfile(color_filename):
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                # check if a depth file exists.
                if not os.path.isfile(depth_filename):
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                world_T_cam_44 = np.genfromtxt(pose_path).astype(np.float32)
                # check if the pose is valid.
                if (
                    np.isnan(np.sum(world_T_cam_44))
                    or np.isinf(np.sum(world_T_cam_44))
                    or np.isneginf(np.sum(world_T_cam_44))
                ):
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                valid_frames.append(f"{scan} {frame_id:06d} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(
                f"Scene {scan} has {bad_file_count} bad frame files out of " f"{color_file_count}."
            )

            # store computed if we're being asked, but wrapped inside a try
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, "w") as f:
                        f.write("\n".join(valid_frames) + "\n")
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, " f"cause:\n", e)

        return valid_frames

    @staticmethod
    def get_gt_mesh_path(dataset_path, split, scan_id):
        """
        Returns a path to a gt mesh reconstruction file.
        """
        gt_path = os.path.join(
            dataset_path,
            ThreeRScanDataset.get_sub_folder_dir(split),
            scan_id,
            "mesh.refined.v2.obj",
        )
        return gt_path

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
        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        cached_resized_path = os.path.join(
            sensor_data_dir, f"frame-{frame_id}.color.{self.image_width}.png"
        )
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path

        # instead return the default image
        return os.path.join(sensor_data_dir, f"frame-{frame_id}.color.jpg")

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

        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        cached_resized_path = os.path.join(
            sensor_data_dir, f"frame-{frame_id}.color.{self.high_res_image_height}.png"
        )
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path

        # instead return the default image
        return os.path.join(sensor_data_dir, f"frame-{frame_id}.color.jpg")

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
        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        cached_resized_path = os.path.join(
            sensor_data_dir, f"frame-{frame_id}.depth.{self.depth_width}.png"
        )

        # instead return the default image
        return cached_resized_path

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached depth file at the size
            required, or if that doesn't exist, the full size depth frame
            from the dataset.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        return os.path.join(sensor_data_dir, f"frame-{frame_id}.depth.pgm")

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """

        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        return os.path.join(sensor_data_dir, f"frame-{frame_id}.pose.txt")

    def get_metadata(self, scan_id):
        """
        Returns the metadata for a scan.
        args:
            scan_id: the scan this file belongs to.
        returns:
            metadata: dictionary of metadata for the scan.
        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        metadata_filename = os.path.join(scene_path, "sensor_data", "_info.txt")
        metadata = self._parse_metadata(metadata_filename)
        return metadata

    def load_color(self, scan_id, frame_id):
        """Loads a frame's RGB file, resizes it to configured RGB size.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at self.image_height and
            self.image_width resolution.

        """

        color_filepath = self.get_color_filepath(scan_id, frame_id)

        image = read_image_file(
            color_filepath,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=self.image_resampling_mode,
            disable_warning=self.disable_resize_warning,
        )

        return image

    def load_high_res_color(self, scan_id, frame_id):
        """Loads a frame's RGB file at a high resolution as configured.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at
            self.high_res_image_height and self.high_res_image_width
            resolution.

        """

        color_high_res_filepath = self.get_high_res_color_filepath(scan_id, frame_id)

        high_res_color = read_image_file(
            color_high_res_filepath,
            height=self.high_res_image_height,
            width=self.high_res_image_width,
            resampling_mode=self.image_resampling_mode,
            disable_warning=self.disable_resize_warning,
        )

        return high_res_color

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        3RScan intrinsics for color and depth are the same up to scale.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for 3RScan as images
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

        # load in basic intrinsics for the full size depth map.
        # parse metadata
        metadata = self.get_metadata(scan_id)

        K_native = torch.tensor(metadata["m_calibrationColorIntrinsic"]).float()
        K = K_native.clone()

        K[0] /= float(metadata["m_colorWidth"])
        K[1] /= float(metadata["m_colorHeight"])

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            full_K = K.clone()

            # scale intrinsics to the dataset's configured depth resolution.
            full_K[0] *= self.native_depth_width
            full_K[1] *= self.native_depth_height

            if self.rotate_images:
                temp = full_K.clone()
                full_K[0, 0] = temp[1, 1]
                full_K[1, 1] = temp[0, 0]
                full_K[1, 2] = temp[0, 2]
                full_K[0, 2] = self.native_depth_height - temp[1, 2]

            output_dict[f"K_full_depth_b44"] = full_K
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(full_K)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width
        K[1] *= self.depth_height

        if self.rotate_images:
            temp = K.clone()
            K[0, 0] = temp[1, 1]
            K[1, 1] = temp[0, 0]
            K[1, 2] = temp[0, 2]
            K[0, 2] = self.depth_height - temp[1, 2]

        # Get the intrinsics of all scales at various resolutions.
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
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
        depth_filepath = self.get_cached_depth_filepath(scan_id, frame_id)

        if not os.path.exists(depth_filepath):
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)

        # Load depth, resize
        depth = read_image_file(
            depth_filepath,
            height=self.depth_height,
            width=self.depth_width,
            value_scale_factor=1e-3,
            resampling_mode=pil.NEAREST,
            disable_warning=True,
        )

        # Get the float valid mask
        mask_b = (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        mask = mask_b.float()

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
        # Load depth

        full_res_depth = read_image_file(full_res_depth_filepath, value_scale_factor=1e-3)

        # Get the float valid mask
        full_res_mask_b = (full_res_depth > self.min_valid_depth) & (
            full_res_depth < self.max_valid_depth
        )
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
        pose_path = self.get_pose_filepath(scan_id, frame_id)

        pose = np.genfromtxt(pose_path).astype(np.float32)

        world_T_cam = pose
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def load_depth_hint(self, scan_id, frame_id, flip=False, mark_all_empty=False):
        """Loads a depth hint for a frame if it exists.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.
            flip: if the hint should be flipped along x.
            mark_all_empty: if the hint should be marked as empty.

        Returns:
            depth_hint_dict: depth hint dict.
        """
        depth_hint_dict = {}

        if mark_all_empty:
            depth_hint_1hw = torch.zeros(1, self.depth_height, self.depth_width)
            depth_hint_1hw[:] = torch.nan
            depth_hint_mask_1hw = torch.zeros_like(depth_hint_1hw)
            depth_hint_mask_b_1hw = torch.zeros_like(depth_hint_1hw).bool()
            sampled_weights_1hw = torch.zeros_like(depth_hint_1hw)
        else:
            partial_hint = torch.rand(1).item() < 0.5 and self.split != "test"

            if partial_hint:
                depth_hint_root = self.depth_hint_dir.replace("/renders", "/partial_renders")
            else:
                depth_hint_root = self.depth_hint_dir

            depth_hint_path = os.path.join(
                depth_hint_root, scan_id, f"rendered_depth_{int(frame_id)}.png"
            )

            depth_hint_1hw = read_image_file(depth_hint_path, value_scale_factor=1 / 256)
            depth_hint_mask_1hw = (depth_hint_1hw > 0).float()
            depth_hint_mask_b_1hw = depth_hint_1hw > 0
            depth_hint_1hw[~depth_hint_mask_b_1hw] = torch.nan

            sampled_weights_path = os.path.join(
                depth_hint_root, scan_id, f"sampled_weights_{int(frame_id)}.png"
            )
            sampled_weights_1hw = read_image_file(sampled_weights_path, value_scale_factor=1 / 256)

            if flip:
                depth_hint_1hw = torch.flip(depth_hint_1hw, (-1,))
                depth_hint_mask_1hw = torch.flip(depth_hint_mask_1hw, (-1,))
                depth_hint_mask_b_1hw = torch.flip(depth_hint_mask_b_1hw, (-1,))
                sampled_weights_1hw = torch.flip(sampled_weights_1hw, (-1,))

            if mark_all_empty:
                depth_hint_mask_1hw = torch.zeros_like(depth_hint_mask_1hw)
                depth_hint_mask_b_1hw = torch.zeros_like(depth_hint_mask_b_1hw).bool()
                depth_hint_1hw[:] = torch.nan
                sampled_weights_1hw = torch.zeros_like(sampled_weights_1hw)

        depth_hint_dict["depth_hint_b1hw"] = depth_hint_1hw
        depth_hint_dict["depth_hint_mask_b1hw"] = depth_hint_mask_1hw
        depth_hint_dict["depth_hint_mask_b_b1hw"] = depth_hint_mask_b_1hw
        depth_hint_dict["sampled_weights_b1hw"] = sampled_weights_1hw

        return depth_hint_dict
