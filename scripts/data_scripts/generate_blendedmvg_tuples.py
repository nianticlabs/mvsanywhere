import sys
import random
import numpy as np
from pathlib import Path
from mvsanywhere import options
from mvsanywhere.utils.dataset_utils import get_dataset

if __name__ == "__main__":

    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options

    np.random.seed(42)
    random.seed(42)

    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    # get dataset
    dataset_class, scan_names = get_dataset(
        opts.datasets[0].dataset, 
        opts.datasets[0].dataset_scan_split_file, 
        opts.single_debug_scan_id, verbose=False
    )
    
    ds = dataset_class(
        dataset_path=opts.datasets[0].dataset_path, 
        mv_tuple_file_suffix=None,
        split=opts.datasets[0].split,
        tuple_info_file_location=opts.datasets[0].tuple_info_file_location,
        pass_frame_id=True,
        verbose_init=False,
    )

    frame_tuples = []
    for scan in scan_names:
        try:
            with open(Path(ds.scenes_path) / scan / "cams" / "pair.txt") as f:
                num_viewpoints = int(f.readline())
                for view_idx in range(num_viewpoints):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    if len(src_views) >= opts.num_images_in_tuple - 1:
                        frame_tuples.append(scan + " " + " ".join([f"{ref_view:08d}"] + [f"{sv:08d}" for sv in src_views[:opts.num_images_in_tuple - 1]]))
        except:
            print(scan)
            continue


    split_filename = f"{opts.datasets[0].split}{opts.datasets[0].mv_tuple_file_suffix}"
    split_filepath = Path(opts.datasets[0].tuple_info_file_location) / split_filename
    print(f"Saving to {split_filepath}")
    with open(split_filepath, "w") as f:
        for line in frame_tuples:
            f.write(line + "\n")
    print(f"Saved to {split_filepath}")
