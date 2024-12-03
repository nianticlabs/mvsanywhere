from copy import deepcopy
import pickle
import kornia
import torch

import numpy as np
import cv2
from pathlib import Path
import tqdm

def read_trajectory(filename):

    barn_T = np.array(
        [-3.024223114080324848e+00, 1.337200447181571550e-01, 2.822018693855588012e+00, -8.033102841327895760e+00,
        -2.824574281070231230e+00, -2.290640822308912528e-01, -3.016107720144580728e+00, 6.856971949289079049e-01,
        5.874257013764642293e-02, -4.130041718728730160e+00, 2.586517244893801193e-01, -2.187313805091164554e+01,
        0.000000000000000000e+00 ,0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ).reshape((4, 4))

    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4), dtype=np.float64)
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=np.float64, sep=" \t")
            # mat[:4, -1:] = barn_T @ mat[:4, -1:]
            traj.append(np.linalg.inv(mat))
            metastr = f.readline()
    return traj

def main():
    with open('/mnt/nas3/shared/projects/fmvs/fmvs/src/rmvd/data/sample_lists/tanks_and_temples.robustmvd.mvd.pickle', 'rb') as f:
        tnt_data = pickle.load(f)

    sample_data = tnt_data[0]

    barn = read_trajectory("tnt_data/Barn_COLMAP_SfM.log")

    samples = []
    for idx in list(range(5, len(barn)-5, 1)):

        sample = deepcopy(sample_data)
        
        # Change images
        sample.data["images"][0].path = f"images/{idx:08d}.jpg"

        for src_i, src in enumerate([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]):
            sample.data["images"][src_i+1].path = f"images/{idx+src:08d}.jpg"

        # Change poses
        sample.data["poses"][0] = barn[idx]
        for src_i, src in enumerate([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]):
            sample.data["poses"][src_i+1] = barn[idx+src]

        sample.data["keyview_idx"] = 0

        # Change depthmap
        sample.data["depth"].path = f"depth/{idx+1:06d}.npz"

        samples.append(sample)

    with open('tnt_sequence.robustmvd.mvd.pickle', 'wb') as f:
        pickle.dump(samples, f)


            


main()