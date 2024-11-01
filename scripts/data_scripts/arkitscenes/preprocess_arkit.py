import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import roma
import scipy.spatial.transform
import torch
from PIL import Image
import csv
from tqdm import tqdm
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from pathlib import Path

def readlines(filepath):
    """ Reads in a text file and returns lines in a list. """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines

def traj_string_to_extrinsics(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts, time_stamp
        quat_world_T_cam, quaternian
        t_world_T_cam, quaternian
    """
    tokens = traj_str.split()
    assert len(tokens) == 7
    
    # timestamp
    ts = float(tokens[0])
    
    # Rotation
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
    
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    
    # get pose
    Rt = torch.tensor(np.linalg.inv(extrinsics))
    
    # quat from rot
    quat_cam_T_world=roma.rotmat_to_unitquat(Rt[:3,:3])
    
    t_world_T_cam = Rt[:3,-1]
    
    return [ts, quat_cam_T_world, t_world_T_cam]

def get_poses(filepath):
    pose_lines = readlines(filepath)
    
    poses = []
    for pose_string in pose_lines:
        poses.append(traj_string_to_extrinsics(pose_string))
        
    return poses

def extract_scan_poses_high_res(scan_dir, count=0, progress=None):
    pose_infos = get_poses(os.path.join(scan_dir, "lowres_wide.traj"))
    rgb_kept_after_cull = []
    wide_dir = os.path.join(scan_dir, "wide")
    highres_dir = os.path.join(scan_dir, "highres_depth")
    intrinsics_dir = os.path.join(scan_dir, "wide_intrinsics")
    
    interpolated_pose_dir = os.path.join(scan_dir, "interpolated_wide_poses")
    Path(interpolated_pose_dir).mkdir(exist_ok=True)

    wide_filenames = os.listdir(wide_dir)
    wide_filenames.sort()

    bad_count = 0
    for image_filename in wide_filenames:
        # print(image_filename)
        try:
            filepath=os.path.join(highres_dir, f"{image_filename}")
            test_depth = Image.open(filepath)
        except:
            bad_count+=1
            continue
        try:
            filepath=os.path.join(intrinsics_dir, ".".join(image_filename.split(".")[:-1])+".pincam")
            # print(filepath)
        
            open(filepath, "r")
        except:
            bad_count+=1
            continue

        pose_filename = ".".join(image_filename.split(".")[:-1])+".txt"
        pose_filepath = os.path.join(interpolated_pose_dir, pose_filename)

        timestamp = float(".".join(image_filename.split("_")[1].split(".")[:-1]))
        
        # check for valid start and end poses 
        start_pose_ind = -1
        least_time = 1
        for pose_index, pose_info in enumerate(pose_infos):
            if timestamp > pose_info[0] and (timestamp - pose_info[0]) < least_time:
                if pose_index + 1 < len(pose_infos):
                    least_time = timestamp - pose_info[0]
                    start_pose_ind = pose_index
        
        if start_pose_ind == -1:
            bad_count+=1
            continue
        
        dist_from_first = ((timestamp - pose_infos[start_pose_ind][0])/
                                (pose_infos[start_pose_ind+1][0]-pose_infos[start_pose_ind][0]))
        
        rgb_kept_after_cull.append(image_filename)
        interped_quat = roma.utils.unitquat_slerp(pose_infos[start_pose_ind][1], pose_infos[start_pose_ind+1][1], torch.tensor(dist_from_first))
        interped_center = pose_infos[start_pose_ind][2] * (1-dist_from_first) + pose_infos[start_pose_ind+1][2] * dist_from_first

        interped_rotmat = roma.unitquat_to_rotmat(interped_quat)
        
        
        pose_mat = np.eye(4, 4)
        pose_mat[:3, :3] = interped_rotmat.numpy()
        pose_mat[:3, -1] = interped_center.numpy()
        
        # print(pose_mat)
        np.savetxt(pose_filepath, pose_mat)
    
    print(f"Completed scan {scan_dir.split('/')[-1]}, bad count {bad_count}, total count {len(wide_filenames)}")
    
def extract_scan_poses_low_res(scan_dir, count=0, progress=None):
    try:
        pose_infos = get_poses(os.path.join(scan_dir, "lowres_wide.traj"))
        rgb_kept_after_cull = []
        vga_wide_dir = os.path.join(scan_dir, "vga_wide")
        wide_dir = os.path.join(scan_dir, "wide")
        depth_dir = os.path.join(scan_dir, "lowres_depth")
        intrinsics_dir = os.path.join(scan_dir, "vga_wide_intrinsics")
        
        interpolated_pose_dir = os.path.join(scan_dir, "interpolated_wide_poses")
        Path(interpolated_pose_dir).mkdir(exist_ok=True)

        wide_filenames = os.listdir(wide_dir)
        wide_filenames.sort()
        
        vga_wide_filenames = os.listdir(vga_wide_dir)
        vga_wide_filenames.sort()
        
        
        merged_color_filenames = list(set(wide_filenames + vga_wide_filenames))
        merged_color_filenames.sort()

        bad_count = 0
        for image_filename in merged_color_filenames:
            
            # try:
            #     filepath=os.path.join(depth_dir, f"{image_filename}")
            #     test_depth = Image.open(filepath)
            # except:
            #     bad_count+=1
            #     continue
            # try:
            #     filepath=os.path.join(intrinsics_dir, ".".join(image_filename.split(".")[:-1])+".pincam")
            #     # print(filepath)
            
            #     open(filepath, "r")
            # except:
            #     bad_count+=1
            #     continue

            pose_filename = ".".join(image_filename.split(".")[:-1])+".txt"
            pose_filepath = os.path.join(interpolated_pose_dir, pose_filename)

            timestamp = float(".".join(image_filename.split("_")[1].split(".")[:-1]))
            
            # check for valid start and end poses 
            start_pose_ind = -1
            least_time = 1
            for pose_index, pose_info in enumerate(pose_infos):
                if timestamp > pose_info[0] and (timestamp - pose_info[0]) < least_time:
                    if pose_index + 1 < len(pose_infos):
                        least_time = timestamp - pose_info[0]
                        start_pose_ind = pose_index
            
            if start_pose_ind == -1:
                bad_count+=1
                continue
            
            dist_from_first = ((timestamp - pose_infos[start_pose_ind][0])/
                                    (pose_infos[start_pose_ind+1][0]-pose_infos[start_pose_ind][0]))
            
            rgb_kept_after_cull.append(image_filename)
            interped_quat = roma.utils.unitquat_slerp(pose_infos[start_pose_ind][1], pose_infos[start_pose_ind+1][1], torch.tensor(dist_from_first))
            interped_center = pose_infos[start_pose_ind][2] * (1-dist_from_first) + pose_infos[start_pose_ind+1][2] * dist_from_first

            interped_rotmat = roma.unitquat_to_rotmat(interped_quat)
            
            
            pose_mat = np.eye(4, 4)
            pose_mat[:3, :3] = interped_rotmat.numpy()
            pose_mat[:3, -1] = interped_center.numpy()
            
            # print(pose_mat)
            np.savetxt(pose_filepath, pose_mat)
        
        print(f"{progress.value}/{count} Completed scan {scan_dir.split('/')[-1]}, bad count {bad_count}, total count {len(vga_wide_filenames)}")
    except Exception as e:
        print(f"{progress.value}/{count} Unrecoverable errors {scan_dir}. {e}")
        
    if progress is not None:
        progress.value += 1
    return []

def multi_process_scans(scan_dirs):
    """
    Multiprocessing helper for crawl_subprocess_long and crawl_subprocess_long.

    Precomputes a scan's valid frames by calling the dataset's appropriate 
    function.

    Args:
        opts_temp_filepath: filepath for an options config file.
        opts: options dataclass.
        scans: scans to multiprocess.
    """
    pool = Pool(48)
    manager = Manager()

    count = len(scan_dirs)
    progress = manager.Value('i', 0)

    item_list = []

    for scan_item_list in pool.imap_unordered(
                                    partial(
                                        extract_scan_poses_low_res,
                                        count=count,
                                        progress=progress
                                    ),
                                    scan_dirs,
                                ):
        item_list.extend(scan_item_list)

    return item_list



def main():
    dataset_path = "/mnt/disks/arkitscenes/"
    # split="Training"
    split="Validation"
    # orientation = "portrait"
    orientation = "landscape"
    
    # extract_scan_poses_low_res(os.path.join(dataset_path, "raw", split, "47333890"), progress=Manager().Value('i', 0))

    scans = readlines(f"/mnt/disks/arkitscenes/{split}_{orientation}.txt")
    # scans = readlines(f"/home/mohameds_nianticlabs_com/code/binary-depth/data_splits/arkitscenes/Validation_portrait.txt")
    # scans = ["48018733"]
    scan_dirs = []
    for scan in scans:
        scan_dirs.append(os.path.join(dataset_path, "raw", split, scan))

    item_list = multi_process_scans(scan_dirs)

    print(f"Complete")

if __name__ == '__main__':
    main()
    
        
    