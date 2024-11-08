import os
import json
from pathlib import Path

from doubletake.datasets.arkitscenes_dataset import ARKitScenesDataset
from doubletake.datasets.blendedmvg import BlendedMVGDataset
from doubletake.datasets.dynamic_replica import DynamicReplicaDataset
from doubletake.datasets.kitti import KITTIDataset
from doubletake.datasets.matrix_city import MatrixCityDataset
from doubletake.datasets.scannet_dataset import ScannetDataset
from doubletake.datasets.seven_scenes_dataset import SevenScenesDataset
from doubletake.datasets.threer_scan_dataset import ThreeRScanDataset
from doubletake.datasets.vdr_dataset import VDRDataset
from doubletake.datasets.dtu_dataset import DTUDataset
from doubletake.datasets.hypersim import HypersimDataset
from doubletake.datasets.tartanair import TartanAirDataset
from doubletake.datasets.vkitti import VirtualKITTIDataset
from doubletake.datasets.nuscenes_dataset import NuScenesDataset
from doubletake.datasets.sailvos3d import SAILVOS3DDataset
from doubletake.datasets.waymo_dataset import WaymoDataset
from doubletake.datasets.mvssynth import MVSSynthDataset


def get_dataset(dataset_name, split_filepath, single_debug_scan_id=None, verbose=True):
    """Helper function for passing back the right dataset class, and helps with
    itentifying the scans in a split file.

    dataset_name: a string pointing to the right dataset name, allowed names
        are:
            - scannet
            - arkit: arkit format as obtained and processed by NeuralRecon
            - vdr
            - scanniverse
            - colmap: colmap text format.
            - 7scenes: processed and undistorted seven scenes.
    split_filepath: a path to a text file that contains a list of scans that
        will be passed back as a list called scans.
    single_debug_scan_id: if not None will override the split file and will
        be passed back in scans as the only item.
    verbose: if True will print the dataset name and number of scans.

    Returns:
        dataset_class: A handle to the right dataset class for use in
            creating objects of that class.
        scans: a lit of scans in the split file.
    """
    split_filepath = Path(os.environ["PWD"]) / split_filepath
    if dataset_name == "scannet":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScannetDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ScanNet Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "arkit":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ARKitDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ARKit Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "vdr":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = VDRDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" VDR Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "scanniverse":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScanniverseDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" Scanniverse Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "colmap":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ColmapDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" Colmap Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "7scenes":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = SevenScenesDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" 7Scenes Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "3rscan":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ThreeRScanDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" 3RScan Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "dtu":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = DTUDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" DTU Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "hypersim":
        with open(split_filepath, "r") as file:
            scans = json.load(file)
            scans = [scan for scan in scans.keys()]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = HypersimDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" Hypersim Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "tartanair":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = TartanAirDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" TartanAir Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "blendedmvg":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = BlendedMVGDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" BlendedMVG Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "dynamic_replica":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = DynamicReplicaDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" DynamicReplica Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "matrix_city":
        with open(split_filepath) as file:
            scans = json.load(file)
            scans = scans.keys()

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = MatrixCityDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" MatrixCity Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "vkitti":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = VirtualKITTIDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" VirtualKITTI Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "kitti":
        scans = KITTIDataset.SEQUENCES
        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = KITTIDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" KITTI Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "nuscenes":
        dataset_class = NuScenesDataset
        scans = None

    elif dataset_name == "waymo":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        dataset_class = WaymoDataset
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

    elif dataset_name == "arkitscenes":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ARKitScenesDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ARKitScenes Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "mvssynth":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = MVSSynthDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" MVSSynth Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")
    elif dataset_name == "sailvos3d":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = SAILVOS3DDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" SAILVOS3D Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")




    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")

    return dataset_class, scans
