from copy import deepcopy
import pickle
import kornia
import torch

import numpy as np
import cv2
from pathlib import Path
import tqdm


def main():
    with open('/mnt/nas3/shared/projects/fmvs/fmvs/src/rmvd/data/sample_lists/kitti.robustmvd.mvd.pickle', 'rb') as f:
        kitti_data = pickle.load(f)

    sample_data = kitti_data[22]

    poses = np.loadtxt('/mnt/nas3/shared/projects/fmvs/fmvs/tnt_data/kitti_seq_odo_5.txt')

    samples = []
    for idx in list(range(2400, 2600, 1)):

        sample = deepcopy(sample_data)
        
        # Change images
        sample.data["images"][0].path = f'raw_data/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/{idx:010d}.png'

        for src_i, src in enumerate([i for i in range(-10, 11) if i != 0]):
            sample.data["images"][src_i+1].path = f'raw_data/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/{idx+src:010d}.png'

        # Change poses
        sample.data["poses"][0] = np.linalg.inv(np.vstack([poses[idx].reshape((3, 4)), [0, 0, 0, 1]]))
        for src_i, src in enumerate([i for i in range(-10, 11) if i != 0]):
            sample.data["poses"][src_i+1] = np.linalg.inv(np.vstack([poses[idx+src].reshape((3, 4)), [0, 0, 0, 1]]))

        sample.data["keyview_idx"] = 0

        # Change depthmap
        sample.data["depth"].path = f'depth_completion_prediction/train/2011_09_30_drive_0018_sync/proj_depth/groundtruth/image_02/{idx:010d}.png'
        samples.append(sample)

    with open('kitti_sequence.robustmvd.mvd.pickle', 'wb') as f:
        pickle.dump(samples, f)


            


main()