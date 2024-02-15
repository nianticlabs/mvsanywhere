import os
import shutil
from tqdm import tqdm

def copy_directories(source_dir, destination_dir, names_file):
    with open(names_file, 'r') as file:
        directory_names = file.read().splitlines()
        print(f"Copying {len(directory_names)} directories from '{source_dir}' to '{destination_dir}'.")
    for name in tqdm(directory_names):
        source_path = os.path.join(source_dir, name)
        destination_path = os.path.join(destination_dir, name)
        if os.path.exists(source_path):
            print(source_path, destination_path)
            shutil.copytree(source_path, destination_path)
            print(f"Directory '{name}' copied successfully.")
        else:
            print(f"Directory '{name}' does not exist in the source directory.")

source_directory = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/hero_model/scannet/default/meshes/0.04_3.0_ours/partial_renders/"
destination_directory = '/home/mohameds/geometryhints/data/noisy_renders/partial_renders_noise'
names_file = "data_splits/ScanNetv2/standard_split/scannetv2_val.txt"


copy_directories(source_directory, destination_directory, names_file)
