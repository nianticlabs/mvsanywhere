import json
import os
from pathlib import Path

from mvsanywhere.datasets.blendedmvg import BlendedMVGDataset
from mvsanywhere.datasets.colmap_dataset import ColmapDataset
from mvsanywhere.datasets.dynamic_replica import DynamicReplicaDataset
from mvsanywhere.datasets.hypersim import HypersimDataset
from mvsanywhere.datasets.matrix_city import MatrixCityDataset
from mvsanywhere.datasets.mvssynth import MVSSynthDataset
from mvsanywhere.datasets.nerf_dataset import NeRFDataset
from mvsanywhere.datasets.nerfstudio_dataset import NerfStudioDataset
from mvsanywhere.datasets.sailvos3d import SAILVOS3DDataset
from mvsanywhere.datasets.scannet_dataset import ScannetDataset
from mvsanywhere.datasets.tartanair import TartanAirDataset
from mvsanywhere.datasets.vdr_dataset import VDRDataset
from mvsanywhere.datasets.vkitti import VirtualKITTIDataset


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

    elif dataset_name == "nerf":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = NeRFDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" NeRF Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "nerfstudio":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = NerfStudioDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" NeRF Studio Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")
  

    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")

    return dataset_class, scans
