"""Script for generating planar depth maps for the Hypersim Dataset

    Run like so for generating/saving depth maps to dataset_path defined in the yaml:
    
    python ./data_scripts/generate_hypersim_planar_depths.py 
        --data_config configs/data/hypersim_default_train.yaml
        --num_workers 8 

    where hypersim_default_train.yaml looks like: 
        !!python/object:options.Options
        dataset_path: HYPERSIM_PATH/
        tuple_info_file_location: $tuples_directory$
        dataset_scan_split_file: $train_split_list_location$
        dataset: hypersim
        mv_tuple_file_suffix: _eight_view_deepvmvs.txt
        num_images_in_tuple: 8
        frame_tuple_type: default
        split: train

    For val, use configs/data/hypersim_default_val.yaml.

    This script will save the planar depth maps in the following pattern:
    {dataset_preprocessed_path}_preprocessed
                / {scene}
                / "images"
                / f"scene_{cam}_geometry_hdf5"
                / f"frame.{int(frame_id):04d}.depth_meters_planar.hdf5"

"""
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

from mvsanywhere import options
from mvsanywhere.utils.dataset_utils import get_dataset


def crawl_subprocess(opts, scan, count, progress):
    """
    Generates and saves planar depth maps to disk for a given scene

    Args:
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass
                multiprocessing.Manager().Value('i', 0)
            for this.

    """

    print(f"Generating planar depths for scene {scan}")

    # get dataset
    dataset_class, _ = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id, verbose=False
    )

    ds = dataset_class(
        dataset_path=opts.dataset_path,
        mv_tuple_file_suffix=None,
        split=opts.split,
        tuple_info_file_location=opts.tuple_info_file_location,
        pass_frame_id=True,
        verbose_init=False,
    )

    frame_ids = ds._get_frame_ids(opts.split, scan)
    for frame_ind in frame_ids:
        ds._save_prependicular_depths_to_disk(scan, frame_ind)

    progress.value += 1
    print(f"Completed scan {scan}, {progress.value} of total {count}\r")


def crawl(opts, scans):
    """
    Multiprocessing helper for crawl_subprocess

    Args:
        opts: options dataclass.
        scans: scans to multiprocess.

    """
    pool = Pool(opts.num_workers)
    manager = Manager()

    count = len(scans)
    progress = manager.Value("i", 0)

    crawler = crawl_subprocess

    pool.imap_unordered(
        partial(crawler, opts, count=count, progress=progress),
        scans,
    )
    pool.close()
    # wait for all issued task to complete
    pool.join()


if __name__ == "__main__":
    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options

    # get dataset
    dataset_class, scan_names = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id, verbose=False
    )

    if opts.single_debug_scan_id is not None:
        crawler = crawl_subprocess
        crawler(
            opts,
            opts.single_debug_scan_id,
            0,
            Manager().Value("i", 0),
        )
    else:
        crawl(opts, scan_names)