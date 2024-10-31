import os
from pathlib import Path
import pickle
import numpy as np
import PIL.Image as pil
import torch
from doubletake.datasets.change_of_basis import ChangeOfBasis
from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from doubletake.utils.generic_utils import readlines, read_image_file
from doubletake.utils.geometry_utils import rotz
import csv

class ARKitScenesDataset(GenericMVSDataset):
    """
    MVS ScanNetv2 Dataset class for SimpleRecon.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.

    This dataset expects ScanNetv2 to be in the following format:

    dataset_path
        scans_test (test scans)
            scene0707
                scene0707_00_vh_clean_2.ply (gt mesh)
                sensor_data
                    frame-000261.pose.txt
                    frame-000261.color.jpg
                    frame-000261.color.512.png (optional, image at 512x384)
                    frame-000261.color.640.png (optional, image at 640x480)
                    frame-000261.depth.png (full res depth, stored scale *1000)
                    frame-000261.depth.256.png (optional, depth at 256x192 also
                                                scaled)
                scene0707.txt (scan metadata and intrinsics)
            ...
        scans (val and train scans)
            scene0000_00
                (see above)
            scene0000_01
            ....

    In this example scene0707.txt should contain the scan's metadata and
    intrinsics:
        colorHeight = 968
        colorToDepthExtrinsics = 0.999263 -0.010031 0.037048 -0.038549 ........
        colorWidth = 1296
        depthHeight = 480
        depthWidth = 640
        fx_color = 1170.187988
        fx_depth = 570.924255
        fy_color = 1170.187988
        fy_depth = 570.924316
        mx_color = 647.750000
        mx_depth = 319.500000
        my_color = 483.750000
        my_depth = 239.500000
        numColorFrames = 784
        numDepthFrames = 784
        numIMUmeasurements = 1632

    frame-000261.pose.txt should contain pose in the form:
        -0.384739 0.271466 -0.882203 4.98152
        0.921157 0.0521417 -0.385682 1.46821
        -0.0587002 -0.961035 -0.270124 1.51837

    frame-000261.color.512.png is a precached resized version of the original
    image to save load and compute time during training and testing. Similarly
    for frame-000261.color.640.png. frame-000261.depth.256.png is also a
    precached resized version of the depth map.

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
            color_transform: optional color transform that applies when split is
                "train".
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

        file = open("data_splits/arkitscenes/metadata.csv", "r")
        self.scans_metadata = {}
        self.scan_orientation = {}

        # get upsampling scan metadata
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            self.scans_metadata[row[0]] = row
            self.scan_orientation[row[0]] = row[2]

    def handle_spatial_rotation(self, im, scan_id, frame_id=None):
        direction = self.scan_orientation[scan_id]
        if direction == "Up":
            return im
        elif direction == "Left":
            # clockwise
            im = torch.rot90(im, -1, [1, 2])
        elif direction == "Right":
            # anticlockwise
            im = torch.rot90(im, 1, [1, 2])
        elif direction == "Down":
            # 180
            im = torch.rot90(im, 2, [1, 2])
        else:
            raise Exception(f"No such direction (={direction}) rotation")
        return im

    @staticmethod
    def get_sub_folder_dir(split):
        """Where scans are for each split."""
        if split == "val":
            return "raw/Validation"
        else:
            return "raw/Training"

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

        # return os.path.join(scan_dir, "valid_frames.txt")
        return os.path.join(scan_dir, "valid_frames_mixed.txt")

    def get_valid_frame_ids(self, split, scan, store_computed=True, overwrite=False):
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

        if os.path.exists(valid_frame_path) and not overwrite:
            # valid frame file exists, read that to find the ids of frames with
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            # find out which frames have valid poses

            # get scannet directories
            scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(split), scan)

            vga_wide_dir = os.path.join(scan_dir, "vga_wide")
            wide_dir = os.path.join(scan_dir, "wide")
            highres_depth_dir = os.path.join(scan_dir, "highres_depth")
            lowres_depth_dir = os.path.join(scan_dir, "lowres_depth")
            wide_intrinsics_dir = os.path.join(scan_dir, "wide_intrinsics")
            vga_wide_intrinsics_dir = os.path.join(scan_dir, "vga_wide_intrinsics")
            interpolated_pose_dir = os.path.join(scan_dir, "interpolated_wide_poses")

            scan_metadata = self.scans_metadata[scan]

            wide_filenames = os.listdir(wide_dir)
            wide_filenames.sort()

            vga_wide_filenames = os.listdir(vga_wide_dir)
            vga_wide_filenames.sort()

            merged_color_filenames = list(set(wide_filenames + vga_wide_filenames))
            merged_color_filenames.sort()

            color_file_count = len(vga_wide_filenames)
            high_res_count = 0

            dist_to_last_valid_frame = 0
            bad_file_count = 0
            valid_frames = []
            for image_filename in merged_color_filenames:
                # for a frame to be valid, we need a valid pose and a valid
                # color frame.
                frame_id = ".".join(image_filename.split(".")[:-1])

                color_filename = os.path.join(vga_wide_dir, f"{image_filename}")
                high_res_color_filename = os.path.join(wide_dir, f"{image_filename}")
                high_depth_filename = os.path.join(highres_depth_dir, f"{image_filename}")
                low_depth_filename = os.path.join(lowres_depth_dir, f"{image_filename}")
                pose_filepath = os.path.join(
                    interpolated_pose_dir, ".".join(image_filename.split(".")[:-1]) + ".txt"
                )
                wide_intrinsics_path = os.path.join(
                    wide_intrinsics_dir, ".".join(image_filename.split(".")[:-1]) + ".pincam"
                )
                vga_wide_intrinsics_path = os.path.join(
                    vga_wide_intrinsics_dir, ".".join(image_filename.split(".")[:-1]) + ".pincam"
                )

                has_high_res = 0
                # check if high res data exists
                if (
                    os.path.isfile(high_res_color_filename)
                    and os.path.isfile(high_depth_filename)
                    and os.path.isfile(wide_intrinsics_path)
                ):
                    has_high_res = 1
                    high_res_count += 1

                elif (
                    os.path.isfile(color_filename)
                    and os.path.isfile(low_depth_filename)
                    and os.path.isfile(vga_wide_intrinsics_path)
                ):
                    # check if low res data exists
                    pass
                else:
                    # no good data
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                # check if pose exists
                if not os.path.isfile(pose_filepath):
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                world_T_cam_44 = np.genfromtxt(pose_filepath).astype(np.float32)
                # check if the pose is valid.
                if (
                    np.isnan(np.sum(world_T_cam_44))
                    or np.isinf(np.sum(world_T_cam_44))
                    or np.isneginf(np.sum(world_T_cam_44))
                ):
                    dist_to_last_valid_frame += 1
                    bad_file_count += 1
                    continue

                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame} {has_high_res}")
                dist_to_last_valid_frame = 0

            print(
                f"Scene {scan} has {bad_file_count} bad frame files out of "
                f"{color_file_count} {high_res_count}."
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
            ARKitScenesDataset.get_sub_folder_dir(split),
            scan_id,
            f"{scan_id}_vh_clean_2.ply",
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

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        vga_wide_dir = os.path.join(scan_dir, "vga_wide")
        color_filename = os.path.join(vga_wide_dir, f"{frame_id}.png")

        # check if we have cached resized images on disk first
        if os.path.exists(color_filename):
            return color_filename

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        wide_dir = os.path.join(scan_dir, "wide")
        color_filename = os.path.join(wide_dir, f"{frame_id}.png")

        return color_filename

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
        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        vga_wide_dir = os.path.join(scan_dir, "vga_wide")
        color_filename = os.path.join(vga_wide_dir, f"{frame_id}.png")

        # check if we have cached resized images on disk first
        if os.path.exists(color_filename):
            return color_filename

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        wide_dir = os.path.join(scan_dir, "wide")
        color_filename = os.path.join(wide_dir, f"{frame_id}.png")

        return color_filename

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
        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        highres_dir = os.path.join(scan_dir, "highres_depth")
        depth_filename = os.path.join(highres_dir, f"{frame_id}.png")

        if os.path.exists(depth_filename):
            return depth_filename

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        highres_dir = os.path.join(scan_dir, "lowres_depth")
        depth_filename = os.path.join(highres_dir, f"{frame_id}.png")

        return depth_filename

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
        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        highres_dir = os.path.join(scan_dir, "highres_depth")
        depth_filename = os.path.join(highres_dir, f"{frame_id}.png")

        if os.path.exists(depth_filename):
            return depth_filename

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        highres_dir = os.path.join(scan_dir, "lowres_depth")
        depth_filename = os.path.join(highres_dir, f"{frame_id}.png")

        return depth_filename

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)
        interpolated_pose_dir = os.path.join(scan_dir, "interpolated_wide_poses")

        return os.path.join(interpolated_pose_dir, f"{frame_id}.txt")
    
    def load_color(self, scan_id, frame_id, crop=None):
        """Loads a frame's RGB file, resizes it to configured RGB size.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at self.image_height and
            self.image_width resolution.

        """

        color_filepath = self.get_color_filepath(scan_id, frame_id)
        try:
            image = read_image_file(
                color_filepath,
                height=None,
                width=None,
                resampling_mode=self.image_resampling_mode,
                disable_warning=True,
                crop=crop,
            )
        except:
            print("Failed to load: ", scan_id, frame_id)
            image = torch.zeros((3, self.image_height, self.image_width)).float()

        image = self.handle_spatial_rotation(image, scan_id)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), (self.image_height, self.image_width), mode='nearest'
        ).squeeze(0)

        # Remove alpha channel for PNGs
        image = image[:3]

        return image

    def handle_intrinsic_rotation(
        self, intrinsics_44, old_height, old_width, scan_id, frame_id=None
    ):
        old_intrinsics_44 = intrinsics_44.clone()

        direction = self.scan_orientation[scan_id]
        if direction == "Up":
            return intrinsics_44
        elif direction == "Left":
            # clockwise, so x becomes y. new cy is fine. compensate in the new center for x
            # --------------> x       x (old y) <-------------- need to adjust center for new cx
            # |                                               |
            # |                                               |
            # |                                               |
            # |                                               |
            # \/                                              \/
            # y                                           y (old x)
            intrinsics_44[0, 0] = old_intrinsics_44[1, 1]
            intrinsics_44[1, 1] = old_intrinsics_44[0, 0]
            intrinsics_44[1, 2] = old_intrinsics_44[0, 2]
            intrinsics_44[0, 2] = old_height - old_intrinsics_44[1, 2]
        elif direction == "Right":
            # anticlockwise, so y becomes x. new cx is fine. compensate in the new center for y
            # --------------> x                 ^ y (old x) points the other way, so adjusting new cy
            # |                                 |
            # |                                 |
            # |                                 |
            # |                                 |
            # \/                                |
            # y                                 --------------> x (old y)
            intrinsics_44[0, 0] = old_intrinsics_44[1, 1]
            intrinsics_44[1, 1] = old_intrinsics_44[0, 0]
            intrinsics_44[1, 2] = old_width - old_intrinsics_44[0, 2]
            intrinsics_44[0, 2] = old_intrinsics_44[1, 2]
        elif direction == "Down":
            # 180, clockwise twice. both cx and cy need fixing. compensate in the new center for y
            # --------------> x                 ^ y (old y) need to adjust new cy
            # |                                               ^
            # |                                               |
            # |                                               |
            # |                                               |
            # \/                                              |
            # y                                 <-------------- x (old x), nee to adjust new cx
            intrinsics_44[0, 0] = old_intrinsics_44[0, 0]
            intrinsics_44[1, 1] = old_intrinsics_44[1, 1]
            intrinsics_44[0, 2] = old_width - old_intrinsics_44[0, 2]
            intrinsics_44[1, 2] = old_height - old_intrinsics_44[1, 2]
        else:
            raise Exception(f"No such direction (={direction}) rotation")

        return intrinsics_44

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for ScanNet as images
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

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(self.split), scan_id)

        intrinsics_dir = os.path.join(scan_dir, "wide_intrinsics")
        intrinsics_filename = os.path.join(intrinsics_dir, f"{frame_id}.pincam")

        if not os.path.exists(intrinsics_filename):
            intrinsics_dir = os.path.join(scan_dir, "vga_wide_intrinsics")
            intrinsics_filename = os.path.join(intrinsics_dir, f"{frame_id}.pincam")

        file = open(intrinsics_filename, "r")
        intrinsics_line = file.readline()
        intrinsics_data = intrinsics_line.split()
        intrinsics_data = [float(elem) for elem in intrinsics_data]

        # print(intrinsics_data)

        full_color_width = intrinsics_data[0]
        full_color_height = intrinsics_data[1]
        full_color_fx = intrinsics_data[2]
        full_color_fy = intrinsics_data[3]
        full_color_cx = intrinsics_data[4]
        full_color_cy = intrinsics_data[5]

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = full_color_fx
        K[1, 1] = full_color_fy
        K[0, 2] = full_color_cx
        K[1, 2] = full_color_cy

        top, left, h, w = self.random_resize_crop.get_params(
            torch.empty((int(full_color_height), int(full_color_width))),
            self.random_resize_crop.scale,
            self.random_resize_crop.ratio
        )
        K[0, 2] = K[0, 2] - left
        K[1, 2] = K[1, 2] - top
        width_pixels = w
        height_pixels = h

        width_pixels = full_color_width
        height_pixels = full_color_height


        K = self.handle_intrinsic_rotation(
            K,
            old_height=full_color_height,
            old_width=full_color_width,
            scan_id=scan_id,
        )

        if flip:
            K[0, 2] = full_color_width - K[0, 2]

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.tensor(np.linalg.inv(K))

        # scale intrinsics to the dataset's configured depth resolution.
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
        depth_filepath = self.get_cached_depth_filepath(scan_id, frame_id)

        if not os.path.exists(depth_filepath):
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)

        # Load depth, resize
        depth = read_image_file(
            depth_filepath,
            height=None,
            width=None,
            value_scale_factor=1e-3,
            resampling_mode=pil.NEAREST,
            crop=crop,
        )

        depth = self.handle_spatial_rotation(depth, scan_id)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), (self.depth_height, self.depth_width), mode='nearest'
        ).squeeze(0)

        # Get the float valid mask
        mask_b = (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        mask = mask_b.float()
        depth[~mask_b] = torch.tensor(np.nan)

        # set invalids to nan

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

        full_res_depth = self.handle_spatial_rotation(full_res_depth, scan_id)
        full_res_mask = self.handle_spatial_rotation(full_res_mask, scan_id)
        full_res_mask_b = self.handle_spatial_rotation(full_res_mask_b, scan_id)

        return full_res_depth, full_res_mask, full_res_mask_b

    def handle_pose_rotation(
        self,
        world_T_cam,
        scan_id,
        frame_id=None,
    ):
        direction = self.scan_orientation[scan_id]
        if direction == "Up":
            return world_T_cam
        elif direction == "Left":
            # im clockwise rot, z is pointing away from camera. Rotate 90 on z
            T = np.eye(4)
            T[:3, :3] = rotz(-np.pi / 2)
            world_T_cam = world_T_cam @ T
        elif direction == "Right":
            # im anticlockwise rot, z is pointing away from camera. Rotate -90 on z
            T = np.eye(4)
            T[:3, :3] = rotz(np.pi / 2)
            world_T_cam = world_T_cam @ T
        elif direction == "Down":
            # im anticlockwise 180, z is pointing away from camera. Rotate 180 on z
            T = np.eye(4)
            T[:3, :3] = rotz(np.pi)
            world_T_cam = world_T_cam @ T
        else:
            raise Exception(f"No such direction (={direction}) rotation")

        return world_T_cam

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

        world_T_cam = np.genfromtxt(pose_path).astype(np.float32)

        world_T_cam = self.handle_pose_rotation(
            world_T_cam,
            scan_id,
        ).astype(np.float32)

        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world
