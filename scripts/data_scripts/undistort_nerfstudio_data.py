import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tyro
import json
import tqdm
import shutil


def undistort_nerfstudio_data(
    data_dir: Path,
    output_dir: Path,
):
    """
    Undistort images in a directory using OpenCV.

    Code heavily inspired by Nerfstudio

    Args:
        data_dir (Path): The directory containing the images to undistort.
        output_dir (Path): The directory to save the undistorted images to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    with open(data_dir / "transforms.json", "r") as f:
        capture_data = json.load(f)
    
    frame_data = capture_data["frames"]
    uses_per_frame_K = "cx" in frame_data[0]

    for frame in tqdm.tqdm(frame_data):

        file_path = data_dir / frame["file_path"]
        image = np.array(Image.open(file_path))

        K_data = frame if uses_per_frame_K else capture_data
        cx = K_data["cx"]
        cy = K_data["cy"]
        fx = K_data["fl_x"]
        fy = K_data["fl_y"]

        K = np.eye(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5

        camera_model = K_data.get("camera_model", "OPENCV")
        if camera_model == 'OPENCV_FISHEYE':
            distortion_params = np.array([K_data['k1'], K_data['k2'], K_data['k3'], K_data['k4']])
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
            )
            # and then remap:
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            newK[0, 2] = newK[0, 2] + 0.5
            newK[1, 2] = newK[1, 2] + 0.5

        elif camera_model == 'OPENCV':
            distortion_params = np.array(
            [
                K_data['k1'],
                K_data['k2'],
                K_data['p1'],
                K_data['p2'],
                K_data.get('k3', 0),
                0,
                0,
                0,
            ]
            )
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore

            # crop the image and update the intrinsics accordingly
            x, y, w, h = roi
            image = image[y : y + h, x : x + w]
            newK[0, 2] -= x
            newK[1, 2] -= y

        K = newK
        height, width = image.shape[:2]

        if uses_per_frame_K:
            K_data["fl_x"] = K[0, 0]
            K_data["fl_y"] = K[1, 1]
            K_data["cx"] = K[0, 2]
            K_data["cy"] = K[1, 2]

            K_data["camera_model"] = "OPENCV"
            K_data["k1"] = 0
            K_data["k2"] = 0
            K_data["p1"] = 0
            K_data["p2"] = 0
            K_data["k3"] = 0
            K_data["k4"] = 0

            K_data['w'] = width
            K_data['h'] = height

        # save the undistorted image
        image = Image.fromarray(image)
        image.save(output_dir / frame["file_path"])

    if not uses_per_frame_K:
        # just save once at the end for all frames
        K_data["fl_x"] = K[0, 0]
        K_data["fl_y"] = K[1, 1]
        K_data["cx"] = K[0, 2]
        K_data["cy"] = K[1, 2]

        K_data["camera_model"] = "OPENCV"
        K_data["k1"] = 0
        K_data["k2"] = 0
        K_data["p1"] = 0
        K_data["p2"] = 0
        K_data["k3"] = 0
        K_data["k4"] = 0

        K_data['w'] = width
        K_data['h'] = height

    # save the updated transforms
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(capture_data, f)

    # also copy over the ply if it exists
    ply_path = capture_data.get("ply_file_path", None)
    if ply_path is not None:
        shutil.copy(data_dir / ply_path, output_dir / ply_path)

    colmap_path = data_dir / 'colmap'
    if colmap_path.exists():
        shutil.copytree(colmap_path, output_dir / 'colmap')
        

if __name__ == '__main__':
    tyro.cli(undistort_nerfstudio_data)
