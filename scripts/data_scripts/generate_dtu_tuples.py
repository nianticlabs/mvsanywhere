import sys
import random
import numpy as np
from pathlib import Path
sys.path.append("/".join(sys.path[0].split("/")[:-1]))
import options
from utils.dataset_utils import get_dataset

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
        opts.dataset, 
        opts.dataset_scan_split_file, 
        opts.single_debug_scan_id, verbose=False
    )
    
    ds = dataset_class(
        dataset_path=opts.dataset_path, 
        mv_tuple_file_suffix=None,
        split=opts.split,
        tuple_info_file_location=opts.tuple_info_file_location,
        pass_frame_id=True,
        verbose_init=False,
    )

    if opts.split == 'train' or opts.split == 'val':
        frame_sets = []
        with open(Path(ds.scenes_path) / "Cameras" / "pair.txt") as f:
            num_viewpoints = int(f.readline())
            for view_idx in range(num_viewpoints):
                ref_view = int(f.readline().rstrip()) + 1
                src_views = [int(x) + 1 for x in f.readline().rstrip().split()[1::2]]

                lights = np.arange(7)
                for light_idx in lights:
                    frame_sets.append(" ".join([f"{ref_view:03d}_{light_idx}"] + [f"{sv:03d}_{light_idx}" for sv in np.random.choice(src_views, opts.num_images_in_tuple - 1, replace=False)]))

        frame_tuples = []
        for scan in scan_names:
            for fs in frame_sets:
                frame_tuples.append(scan + " " + fs)

    elif opts.split == 'test':
        frame_tuples = []
        for scan in scan_names:
            with open(Path(ds.scenes_path) / scan / "pair.txt") as f:
                num_viewpoints = int(f.readline())
                for view_idx in range(num_viewpoints):
                    ref_view = int(f.readline().rstrip()) + 1
                    src_views = [int(x) + 1 for x in f.readline().rstrip().split()[1::2]]

                    frame_tuples.append(scan + " " + " ".join([f"{ref_view:08d}"] + [f"{sv:08d}" for sv in src_views[:opts.num_images_in_tuple - 1]]))



    else:
        raise ValueError(f'{opts.split} must be either train, val or test')

    split_filename = f"{opts.split}{opts.mv_tuple_file_suffix}"
    split_filepath = Path(opts.tuple_info_file_location) / split_filename
    print(f"Saving to {split_filepath}")
    with open(split_filepath, "w") as f:
        for line in frame_tuples:
            f.write(line + "\n")
    print(f"Saved to {split_filepath}")
