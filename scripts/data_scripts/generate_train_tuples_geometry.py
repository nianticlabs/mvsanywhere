"""Script for generating geometry-based multiview lists in the split folder
    indicated. It will export these frame tuples in this format line by line in
    the output file:

    scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    where frame_id_0 is the reference image.

    Run like so for generating a list of train tuples of eight frames (default):

    python ./data_scripts/generate_train_tuples_geometry.py
        --data_config configs/data/scannet/scannet_default_train.yaml
        --num_workers 16

    where scannet_default_train.yaml looks like:
        !!python/object:mvsanywhere.options.Options
        dataset_path: SCANNET_PATH/
        tuple_info_file_location: $tuples_directory$
        dataset_scan_split_file: $train_split_list_location$
        dataset: scannet
        mv_tuple_file_suffix: _eight_view_deepvmvs.txt
        num_images_in_tuple: 8
        frame_tuple_type: default
        split: train

    For val, use configs/data/scannet/scannet_default_val.yaml.

    It will output a tuples file with a tuple list at:
    {tuple_info_file_location}/{split}{mv_split_filesuffix}

"""

import copy
import os
import sys

import random
import string
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from pathlib import Path
import tqdm

import numpy as np
from mvsanywhere.losses import MVDepthLoss
import mvsanywhere.options as options
from mvsanywhere.tools.keyframe_buffer import DVMVS_Config, DVMVS_MatrixCity_Config, DVMVS_TartanAir_Config, is_valid_pair
from mvsanywhere.utils.dataset_utils import get_dataset

import torch


def crawl_subprocess_long(opts_temp_filepath, scan, count, progress):
    """
    Returns a list of DVMVS train tuples according to the options at
    opts_temp_filepath for tuples longer than two frames.

    Args:
        opts_temp_filepath: filepath for an options config file.
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass
                multiprocessing.Manager().Value('i', 0)
            for this.

    Returns:
        item_list: a list of strings where each string is the cocnatenated
            scan id and frame_ids for every tuple.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    """
    scan_item_list = []

    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(
        config_filepaths=opts_temp_filepath,
        ignore_cl_args=True,
    )
    opts = option_handler.options

    # get dataset
    dataset_class, _ = get_dataset(
        opts.datasets[0].dataset,
        opts.datasets[0].dataset_scan_split_file,
        opts.single_debug_scan_id,
        verbose=False,
    )

    ds = dataset_class(
        dataset_path=opts.datasets[0].dataset_path,
        mv_tuple_file_suffix=None,
        split=opts.datasets[0].split,
        tuple_info_file_location=opts.datasets[0].tuple_info_file_location,
        pass_frame_id=True,
        verbose_init=False,
        prediction_scale=0.25,
    )

    loss = MVDepthLoss(ds.depth_height, ds.depth_width).cuda()

    if opts.datasets[0].dataset == "tartanair":
        keyframe_config = DVMVS_TartanAir_Config
    elif opts.datasets[0].dataset == "matrix_city" or opts.datasets[0].dataset == "vkitti":
        keyframe_config = DVMVS_MatrixCity_Config
    else:
        keyframe_config = DVMVS_Config


    valid_frames = ds.get_valid_frame_ids(opts.datasets[0].split, scan)

    if len(valid_frames) == 0:
        print("No valid frames; exiting")
        return []

    frame_ind_to_frame_id = {}
    for frame_ind, frame_line in enumerate(valid_frames):
        frame_ind_to_frame_id[frame_ind] = frame_line.strip().split(" ")[1]

    cam_T_worlds_b44 = []
    world_T_cams_b44 = []
    Ks_b44 = []
    invKs_b44 = []
    depths_b1hw = []
    for frame_ind in range(len(valid_frames)):
        frame_id = frame_ind_to_frame_id[frame_ind]
        world_T_cam_44, cam_T_world_44 = ds.load_pose(scan.rstrip("\n"), frame_id)

        depths_b1hw.append(ds.load_target_size_depth_and_mask(scan.rstrip("\n"), frame_id)[0])
        intrinsics = ds.load_intrinsics(scan.rstrip("\n"), frame_id)[0]
        Ks_b44.append(intrinsics["K_s0_b44"])
        invKs_b44.append(intrinsics["invK_s0_b44"])
        world_T_cams_b44.append(world_T_cam_44)
        cam_T_worlds_b44.append(cam_T_world_44)

    depths_b1hw = torch.stack(depths_b1hw).cuda()
    Ks_b44 = torch.stack(Ks_b44).cuda()
    invKs_b44 = torch.tensor(np.stack(invKs_b44)).cuda()
    cam_T_worlds_b44 = torch.tensor(np.stack(cam_T_worlds_b44)).cuda()
    world_T_cams_b44 = torch.tensor(np.stack(world_T_cams_b44)).cuda()

    subsequence_length = opts.num_images_in_tuple
    sequence_length = len(depths_b1hw)

    used_pairs = set()

    samples = []
    for current_index in range(sequence_length):

        # Grab closer 100 images (that are closer than 100m)
        distance_to_frames = torch.linalg.norm(
            world_T_cams_b44[current_index, :3, 3] - world_T_cams_b44[:, :3, 3], dim=1
        )
        possible_frame_idx = torch.where((0.1 < distance_to_frames) & (distance_to_frames < 100))[0]
        possible_frame_idx = possible_frame_idx[torch.randperm(len(possible_frame_idx))[:100]]
        if len(possible_frame_idx) == 0:
            break

        sample = {"scan": scan, "indices": [current_index]}
        for source_indices in torch.split(possible_frame_idx, 32):

            source_indices_npy = source_indices.cpu().numpy()

            # Pair not previously used
            check_1 = np.array([(current_index, source_index) not in used_pairs for source_index in source_indices_npy])

            # Relative cameras for numerical stability
            src_cam_T_world = cam_T_worlds_b44[source_indices] @ world_T_cams_b44[current_index, None]
            cur_world_T_cam = torch.eye(4).unsqueeze(0).cuda()

            current_depths_b1hw = depths_b1hw[current_index, None]

            # Reproject current geometry into source and count valid pixels
            valid_mask, _ = loss.get_valid_mask(
                current_depths_b1hw,
                depths_b1hw[source_indices],
                invKs_b44[current_index, None],
                Ks_b44[source_indices],
                cur_world_T_cam,
                src_cam_T_world
            )

            # For each source frame, count the fraction of the current points which reprojected
            # into that frame. Only keep those source frames which are above a threshold.
            num_inlier_pixels = valid_mask.sum(dim=(1, 2, 3))
            num_total_pixels = torch.isfinite(current_depths_b1hw).sum(dim=(1, 2, 3))
            check_2 = ((num_inlier_pixels / num_total_pixels) > 0.25).cpu().numpy()

            try:
                check = check_1 & check_2
            except:
                print("What happened bru")
                break

            for source_index in source_indices_npy[check]:
                sample["indices"].append(source_index)
                used_pairs.add((source_index, current_index))
                used_pairs.add((current_index, source_index))

                if len(sample["indices"]) == subsequence_length:
                    break

            if len(sample["indices"]) == subsequence_length:
                samples.append(sample)
                break


    for sample in samples:
        chosen_frame_ids = [frame_ind_to_frame_id[frame_ind] for frame_ind in sample["indices"]]

        cat_ids = " ".join([str(frame_id) for frame_id in chosen_frame_ids])
        scan_item_list.append(f"{scan} {cat_ids}")

    progress.value += 1
    print(f"Completed scan {scan}, {progress.value} of total {count}\r")

    return scan_item_list


def crawl(opts_temp_filepath, opts, scans):
    """
    Multiprocessing helper for crawl_subprocess_long and crawl_subprocess_long.

    Returns a list of train tuples according to the options at
    opts_temp_filepath.

    Args:
        opts_temp_filepath: filepath for an options config file.
        opts: options dataclass.
        scans: scans to multiprocess.

    Returns:
        item_list: a list of strings where each string is the cocnatenated
            scan id and frame_ids for every tuple for every scan in scans.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    """
    pool = Pool(opts.num_workers)
    manager = Manager()

    count = len(scans)
    progress = manager.Value("i", 0)

    item_list = []

    crawler = crawl_subprocess_long

    for scan_item_list in pool.imap_unordered(
        partial(crawler, opts_temp_filepath, count=count, progress=progress),
        scans,
    ):
        item_list.extend(scan_item_list)

    return item_list


if __name__ == "__main__":
    # load options file
    torch.multiprocessing.set_start_method('spawn')
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options

    Path(os.path.join(os.path.expanduser("~"), "tmp/")).mkdir(parents=True, exist_ok=True)

    opts_temp_filepath = os.path.join(
        os.path.expanduser("~"),
        "tmp/",
        "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".yaml",
    )
    option_handler.save_options_as_yaml(opts_temp_filepath, opts)

    np.random.seed(42)
    random.seed(42)

    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    # get dataset
    dataset_class, scan_names = get_dataset(
        opts.datasets[0].dataset, opts.datasets[0].dataset_scan_split_file, opts.single_debug_scan_id, verbose=False
    )

    Path(opts.datasets[0].tuple_info_file_location).mkdir(exist_ok=True, parents=True)
    split_filename = f"{opts.datasets[0].split}{opts.datasets[0].mv_tuple_file_suffix}"
    split_filepath = os.path.join(opts.datasets[0].tuple_info_file_location, split_filename)
    print(f"Saving to {split_filepath}")

    item_list = []
    if opts.single_debug_scan_id is not None:
        crawler = crawl_subprocess_long
        item_list = crawler(
            opts_temp_filepath,
            opts.single_debug_scan_id,
            0,
            Manager().Value("i", 0),
        )
    else:
        item_list = crawl(opts_temp_filepath, opts, scan_names)

    random.shuffle(item_list)

    with open(split_filepath, "w") as f:
        for line in item_list:
            f.write(line + "\n")
    print(f"Saved to {split_filepath}")
