import pickle
import kornia
import torch

import numpy as np
import cv2
from pathlib import Path
import tqdm

PATH_TO_SCANNET = '/mnt/nas3/shared/datasets/scannet/'


def main():

    with open('src/rmvd/data/sample_lists/scannet.robustmvd.mvd.pickle', 'rb') as f:
        scannet_data = pickle.load(f)

    better_tuples = np.loadtxt('data_splits/ScanNetv2/rmvd_split/train_eight_view_rmvd_better_tuples.txt', dtype=object)


    for sample in tqdm.tqdm(scannet_data):

        # Find the corresponding tuple in the list
        improved_tuple = [
            row for row in better_tuples if (sample.base.split('/')[-1] == row[0]) and row[1] == sample.data["images"][sample.data["keyview_idx"]].path.split("/")[1].split(".")[0].zfill(6)
        ][0]

        for idx, image in enumerate(sample.data["images"]):

            if idx == sample.data["keyview_idx"]:
                # Shouldn't be neccesary but this forces the ref images to stay absolutely the same
                continue

            # Change src image RGB
            sample.data["images"][idx].path = f"color/{int(improved_tuple[idx+1])}"
            
            # Change src pose
            pose_file = str(
                Path(PATH_TO_SCANNET)
                / "scans"
                / sample.base.split("/")[-1]
                / "sensor_data"
                / f"frame-{improved_tuple[idx+1]}.pose.txt"
            )
            sample.data["poses"][idx] = np.linalg.inv(np.genfromtxt(pose_file).astype(np.float32))

    with open('src/rmvd/data/sample_lists/scannet_better_tuples.robustmvd.mvd.pickle', 'wb') as f:
        pickle.dump(scannet_data, f)


if __name__ == "__main__":
    main()
