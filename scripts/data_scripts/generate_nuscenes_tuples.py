import random
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    data_path = '/mnt/nas/shared/datasets/nuscenes/'
    version = 'v1.0-trainval'
    split = 'train'

    num_images_in_tuple = 8

    output_dir = Path(data_path) / 'tuples'
    camera_names = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]  # List of all camera names in NuScenes

    # Initialize NuScenes dataset
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    splits = create_splits_scenes()
    split_scenes = splits[split]

    frame_tuples = []

    # Iterate over each camera separately
    for cam_name in camera_names:
        # Iterate over all scenes in the specified split
        for scene in nusc.scene:
            scene_name = scene['name']
            if scene_name not in split_scenes:
                continue  # Skip scenes not in the specified split

            # Get all sample data tokens for the camera in the scene, ordered by timestamp
            cam_data_tokens = []
            sample_token = scene['first_sample_token']
            while sample_token != '':
                sample = nusc.get('sample', sample_token)
                cam_token = sample['data'][cam_name]
                cam_data = nusc.get('sample_data', cam_token)
                cam_data_tokens.append(cam_data)
                sample_token = sample['next']

            # Generate frame tuples for the camera in the scene
            for i in range(len(cam_data_tokens)):
                # Ensure we have enough previous samples to form a complete tuple
                if i < num_images_in_tuple - 1:
                    continue  # Skip if not enough previous samples

                # Collect the current and previous sample data
                current_cam_data = cam_data_tokens[i]
                src_cam_data = cam_data_tokens[i - num_images_in_tuple + 1: i]

                filenames = []
                filenames.append(current_cam_data['filename'])

                for data in src_cam_data:
                    filenames.append(data['filename'])

                frame_tuple = scene_name + ' ' + cam_name + ' ' + ' '.join(filenames)
                frame_tuples.append(frame_tuple)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"nuscenes_{split}_tuples.txt"

    print(f"Saving to {output_file}")
    with open(output_file, 'w') as f:
        for line in frame_tuples:
            f.write(line + '\n')
    print(f"Saved to {output_file}")