import glob
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from doubletake import options
from doubletake.utils.dataset_utils import get_dataset

if __name__ == "__main__":

    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options

    np.random.seed(42)

    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    transforms_dict = {}
    transforms_paths = Path(opts.datasets[0].dataset_path) / "big_city/**/train/**/transforms.json"
    for transform_file in glob.glob(str(transforms_paths)):
        with open(transform_file) as f:
            transforms = json.load(f)
        transforms_dict["/".join(transform_file.split("/")[-5:-1])] = transforms["frames"]


    scans = {}
    for scan_name, frames in transforms_dict.items():

        locations = np.stack([np.array(f["rot_mat"])[:3, 3] for f in frames])
        idx = np.arange(len(locations))

        subscan_idx = 0
        while len(locations) > 50:
            print(scan_name, subscan_idx, len(locations))
            sample_loc = locations[np.random.choice(len(locations))]
            d = np.argsort(np.linalg.norm(locations - sample_loc, axis=1))

            scans[f"{scan_name}:{subscan_idx}"] = idx[d[:500]].tolist()
            locations = locations[d[500:]]
            idx = idx[d[500:]]
            subscan_idx += 1

    
    with open(Path(opts.datasets[0].tuple_info_file_location) / "matrix_city_train.json", "w") as f:
        json.dump(scans, f, indent=4)




        



