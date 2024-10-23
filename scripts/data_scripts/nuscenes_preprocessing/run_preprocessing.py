from pathlib import Path
from typing import List
import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes  # pip install nuscenes-devkit
from pyquaternion import Quaternion
import concurrent.futures as futures


class DepthGenerator:
    def __init__(self, data_path: str, version: str, save_path: str, split: str = 'val') -> None:
        """
        Initialize the DepthGenerator with nuscenes dataset, file paths, and split.

        Args:
            data_path (str): Path to the nuscenes dataset.
            version (str): The dataset version to use, e.g., 'v1.0-trainval'.
            save_path (str): Path to save the generated depth maps.
            split (str): Dataset split to use, either 'val' or 'train'. Default is 'val'.
        """
        self.data_path = Path(data_path)
        self.version = version
        self.save_path = Path(save_path)
        self.split = split

        # Initialize NuScenes dataset
        self.nusc = NuScenes(version=self.version, dataroot=str(self.data_path), verbose=False)

        # Load the appropriate split (val.txt or train.txt)
        split_file = Path(f'{self.split}.txt')
        if not split_file.exists():
            raise FileNotFoundError(f"{self.split}.txt file not found in the working directory.")

        with split_file.open('r') as f:
            self.data: List[str] = f.readlines()

        # Setup output directories for depth maps
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                             'CAM_FRONT_RIGHT']

        for camera_name in self.camera_names:
            (self.save_path / 'samples' / camera_name).mkdir(parents=True, exist_ok=True)

    def __call__(self, num_workers: int = 8) -> None:
        print(f'Generating nuscene depth maps from LiDAR projections using {self.split}.txt')

        def process_one_sample(index: int) -> None:
            """Process one sample and generate the depth map for each camera."""
            index_t = self.data[index].strip()  # Get sample token
            rec = self.nusc.get('sample', index_t)  # Get sample record

            # Get LiDAR sample data and ego pose
            lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
            lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])

            # Create LiDAR transformation matrix (lidar_to_world)
            lidar_rotation = Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            # Get LiDAR points from the file
            lidar_file = self.data_path / lidar_sample['filename']
            lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)[:, :4]

            # Transform LiDAR points to ego frame
            sensor_sample = self.nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_lidar_rot = Quaternion(sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_lidar_trans = np.array(sensor_sample['translation']).reshape(1, 3)
            ego_lidar_points = np.dot(lidar_points[:, :3], lidar_to_ego_lidar_rot.T) + lidar_to_ego_lidar_trans
            homo_ego_lidar_points = np.concatenate((ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()

            for cam in self.camera_names:
                # Get camera sample data and ego pose
                camera_sample = self.nusc.get('sample_data', rec['data'][cam])
                car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])

                # Create transformations (world_to_car_egopose and car_egopose_to_sensor)
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = -np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack(
                        (egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])

                # From ego pose to sensor frame
                sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
                intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
                sensor_rotation = Quaternion(sensor_sample['rotation'])
                sensor_translation = np.array(sensor_sample['translation'])[:, None]
                car_egopose_to_sensor = np.vstack([
                    np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                    np.array([0, 0, 0, 1])
                ])
                car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

                # Combine transformations to get lidar_to_sensor matrix
                lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
                lidar_to_sensor = torch.from_numpy(lidar_to_sensor).float()

                # Load camera image
                image_filename = self.data_path / camera_sample['filename']
                img = Image.open(image_filename)
                img = np.array(img)
                sparse_depth = torch.zeros((img.shape[:2]))

                # Transform LiDAR points to camera frame
                camera_points = torch.mm(homo_ego_lidar_points, lidar_to_sensor.t())
                depth_mask = camera_points[:, 2] > 0
                camera_points = camera_points[depth_mask]

                # Project camera points to 2D pixel coordinates
                viewpad = torch.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                pixel_points = torch.mm(camera_points, viewpad.t())[:, :3]
                pixel_points[:, :2] = pixel_points[:, :2] / pixel_points[:, 2:3]

                # Filter valid pixel coordinates
                pixel_uv = pixel_points[:, :2].round().long()
                height, width = sparse_depth.shape
                valid_mask = (pixel_uv[:, 0] >= 0) & (pixel_uv[:, 0] < width) & \
                             (pixel_uv[:, 1] >= 0) & (pixel_uv[:, 1] < height)
                valid_pixel_uv = pixel_uv[valid_mask]
                valid_depth = camera_points[:, 2][valid_mask]

                # Store valid depth values in sparse depth map
                sparse_depth[valid_pixel_uv[:, 1], valid_pixel_uv[:, 0]] = valid_depth
                sparse_depth = sparse_depth.numpy()

                # Save the sparse depth map as a .npy file
                output_path = self.save_path / camera_sample['filename'].replace('.jpg', '.npy')
                np.save(output_path, sparse_depth)

            print(f'Finished processing index = {index:06d}')

        # Process samples in parallel
        sample_id_list = list(range(len(self.data)))
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_one_sample, sample_id_list)


if __name__ == "__main__":
    data_path = '/mnt/nas/shared/datasets/nuscenes'
    version = 'v1.0-trainval'
    save_path = '/mnt/nas/shared/datasets/nuscenes/depth'
    split = 'val'

    model = DepthGenerator(data_path=data_path, version=version, save_path=save_path, split=split)
    model()
