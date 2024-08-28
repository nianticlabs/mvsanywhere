import numpy as np

TUPLES_FILE = "data_splits/hypersim/bd_split/{}_eight_view_deepvmvs_bd.txt"
FILTER_FILE = "data_splits/hypersim/bd_split/{}_files_mean_10_m_no_bad_scenes.txt"

OUTPUT_FILE = "data_splits/hypersim/bd_split/{}_clean_eight_view_deepvmvs_bd.txt"

if __name__ == "__main__":
    # load options file
    for phase in ("val", "train"):

        tuples_list = np.loadtxt(TUPLES_FILE.format(phase), dtype=object)
        filter_list = np.loadtxt(FILTER_FILE.format(phase), dtype=object)

        tuples_keys = np.array(["_".join(row) for row in tuples_list[:, :2]])
        filter_keys = np.array(["_".join(row) for row in filter_list])

        clean_idx = np.array([idx for idx, row in enumerate(tuples_keys) if row in filter_keys])
        np.savetxt(OUTPUT_FILE.format(phase), tuples_list[clean_idx], fmt="%s")