import sys
import random
import numpy as np
from pathlib import Path
from doubletake import options
from doubletake.utils.dataset_utils import get_dataset

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
        images = (
            Path(ds.dataset_path)
            / "val"
            / scan
            / "proj_depth"
            / "groundtruth"
            / "image_02"
        ).glob("*.png")
        for filename in sorted(images):
            cur_image = int(filename.stem)
            src_images = list(map(lambda x: max(0, x), range(cur_image-opts.num_images_in_tuple+1, cur_image)))
            frame_tuples.append(scan + " " + " ".join([f"{cur_image:010d}"] + [f"{si:010d}" for si in src_images]))


    split_filename = f"{opts.datasets[0].split}{opts.datasets[0].mv_tuple_file_suffix}"
    split_filepath = Path(opts.datasets[0].tuple_info_file_location) / split_filename
    print(f"Saving to {split_filepath}")
    with open(split_filepath, "w") as f:
        for line in frame_tuples:
            f.write(line + "\n")
    print(f"Saved to {split_filepath}")
