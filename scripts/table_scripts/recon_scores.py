import argparse
import json
import pandas as pd
import numpy as np


def distance_formatter(val):
    return f"{val*100:.2f}"


def perc_formatter(val):
    return f"{val:.3f}".lstrip("0")


def itentity_formatter(val):
    return val


def order_formatter(elem, index):
    if index == 0:
        return "\cellcolor{firstcolor}" + elem
    elif index == 1:
        return "\cellcolor{secondcolor}" + elem
    elif index == 2:
        return "\cellcolor{thirdcolor}" + elem
    else:
        return elem


def print_table(mask_type="tf", scores_table="online", format_ordering=True):
    assert mask_type in ["tf", "ours"]
    assert scores_table in ["online", "offline"]

    all_scores_dict = {}

    used_metrics = ["acc", "compl", "chamfer", "prc", "rec", "f1_score"]
    sort_direction = [False, False, False, False, True, True, True]
    number_formatters = [
        itentity_formatter,
        distance_formatter,
        distance_formatter,
        distance_formatter,
        perc_formatter,
        perc_formatter,
        perc_formatter,
    ]

    if mask_type == "tf":
        scores_json_name = "scores.json"
    else:
        scores_json_name = "scores_our_masks.json"

    metrics = ["acc", "compl", "chamfer", "prc", "rec", "f1_score"]

    if scores_table == "offline":
        if mask_type == "tf":
            all_scores_dict["COLMAP~\cite{schonberger2016pixelwise,schoenberger2016sfm} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [10.22 / 100, 11.88 / 100, 11.05 / 100, 0.509, 0.474, 0.489]
                )
            }
            all_scores_dict["ATLAS~\cite{murez2020atlas} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [7.16 / 100, 7.61 / 100, 7.38 / 100, 0.675, 0.605, 0.636]
                )
            }
            all_scores_dict["3DVNet~\cite{rich20213dvnet} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [6.73 / 100, 7.72 / 100, 7.22 / 100, 0.655, 0.596, 0.621]
                )
            }

        if mask_type == "tf":
            all_scores_dict["TransformerFusion~\cite{bozic2021transformerfusion} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [5.5194 / 100, 8.2680 / 100, 6.8937 / 100, 0.7285, 0.5999, 0.6554]
                )
            }
        else:
            all_scores_dict["TransformerFusion~\cite{bozic2021transformerfusion} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [4.6822 / 100, 8.2680 / 100, 6.4751 / 100, 0.6984, 0.5999, 0.6442]
                )
            }

        if mask_type == "tf":
            all_scores_dict["VoRTX~\cite{stier2021vortx} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [4.3138 / 100, 7.2266 / 100, 5.7702 / 100, 0.7675, 0.6508, 0.7030]
                )
            }
        else:
            all_scores_dict["VoRTX~\cite{stier2021vortx} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [4.3794 / 100, 7.2266 / 100, 5.8030 / 100, 0.7257, 0.6508, 0.6851]
                )
            }

        all_scores_dict[
            "SimpleRecon~\cite{sayed2022simplerecon} (offline) (2cm)  & No"
        ] = json.load(
            open(
                f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores_offline/scannet/default/meshes/0.02_3.5_ours_neg_trunc/{scores_json_name}"
            )
        )[
            "overall"
        ]

        if mask_type == "tf":
            all_scores_dict["FineRecon~\cite{Stier_2023_ICCV} (1cm) & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [5.2537 / 100, 5.0599 / 100, 5.1568 / 100, 0.7794, 0.7373, 0.7560]
                )
            }
        else:
            all_scores_dict["FineRecon~\cite{Stier_2023_ICCV} (1cm) & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [4.9243 / 100, 5.0599 / 100, 4.9921 / 100, 0.6869, 0.7373, 0.7098]
                )
            }

        all_scores_dict["\\textbf{Ours} (two-pass) (2cm)& No"] = json.load(
            open(
                f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_trunc_trick/scannet/default/meshes/0.02_3.5_ours/{scores_json_name}"
            )
        )["overall"]

    elif scores_table == "online":
        if mask_type == "tf":
            all_scores_dict["RevisitingSI~\cite{hu2019revisiting} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [14.29 / 100, 16.19 / 100, 15.24 / 100, 0.346, 0.293, 0.314]
                )
            }
            all_scores_dict["MVDepthNet~\cite{wang2018mvdepthnet} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [12.94 / 100, 8.34 / 100, 10.64 / 100, 0.443, 0.487, 0.460]
                )
            }
            all_scores_dict["GPMVS~\cite{hou2019multi} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [12.90 / 100, 8.02 / 100, 10.46 / 100, 0.453, 0.510, 0.477]
                )
            }
            all_scores_dict["ESTDepth~\cite{long2021multi} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [12.71 / 100, 7.54 / 100, 10.12 / 100, 0.456, 0.542, 0.491]
                )
            }
            all_scores_dict["DPSNet~\cite{im2019dpsnet} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [11.94 / 100, 7.58 / 100, 9.77 / 100, 0.474, 0.519, 0.492]
                )
            }
            all_scores_dict["DELTAS~\cite{sinha2020deltas} & No"] = {
                metric: value
                for metric, value in zip(
                    metrics, [11.95 / 100, 7.46 / 100, 9.71 / 100, 0.478, 0.533, 0.501]
                )
            }
            # all_scores_dict["DeepVideoMVS~\cite{duzceker2021deepvideomvs} & No"] = {
            #     metric: value
            #     for metric, value in zip(
            #         metrics, [10.68 / 100, 6.90 / 100, 8.79 / 100, 0.541, 0.592, 0.563]
            #     )
            # }
            all_scores_dict["NeuralRecon~\cite{sun2021neuralrecon} & Yes"] = {
                metric: value
                for metric, value in zip(
                    metrics, [5.09 / 100, 9.13 / 100, 7.11 / 100, 0.630, 0.612, 0.619]
                )
            }

        all_scores_dict["DeepVideoMVS~\cite{duzceker2021deepvideomvs} & No"] = json.load(
            open(f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/dvmvs_4cm/{scores_json_name}")
        )["overall"]

        all_scores_dict["SimpleRecon~\cite{sayed2022simplerecon} (online) (4cm)  & No"] = json.load(
            open(
                f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/scannet/default/meshes/0.04_3.0_ours/{scores_json_name}"
            )
        )["overall"]
        all_scores_dict["\\textbf{Ours} (online) (4cm)& No"] = json.load(
            open(
                f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/scannet/default/meshes/0.04_3.0_ours/{scores_json_name}"
            )
        )["overall"]
        # all_scores_dict["\\textbf{Ours} (online) (2cm)& No"] = json.load(open(f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental_2cm/scannet/default/meshes/0.02_3.0_ours/{scores_json_name}"))["overall"]

    # Define the scores dictionary
    scores = [
        ["&" + key] + [all_scores_dict[key][metric] for metric in used_metrics]
        for key in all_scores_dict.keys()
    ]

    df = pd.DataFrame(scores)

    for col_ind, col in enumerate(df.columns):
        if col_ind == 0:
            continue
        ordered = list(np.argsort(df[col_ind]))

        # reverse ordered
        if sort_direction[col_ind]:
            ordered = ordered[::-1]

        for order_ind, row_ind in enumerate(ordered):
            rounded_str_elem = number_formatters[col_ind](df[col_ind][row_ind])
            if format_ordering:
                df.loc[row_ind, col] = order_formatter(rounded_str_elem, order_ind)
            else:
                df.loc[row_ind, col] = rounded_str_elem

    print(df.to_latex(header=False, index=False))

    print()
    print("mask_type = ", mask_type)
    print("scores_table = ", scores_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_type", type=str, default="tf")
    parser.add_argument("--scores_table", type=str, default="online")
    parser.add_argument("--format_ordering", action="store_true")
    # parse
    args = parser.parse_args()

    print_table(
        mask_type=args.mask_type,
        scores_table=args.scores_table,
        format_ordering=args.format_ordering,
    )


# legacy table
# scores = [
#    ["RevisitingSI~\cite{hu2019revisiting} & No",14.29, 16.19, 15.24, 0.346, 0.293, 0.314],
#    ["MVDepthNet~\cite{wang2018mvdepthnet} & No",12.94,8.34,10.64,0.443,0.487,0.460],
#    ["GPMVS~\cite{hou2019multi} & No",12.90,8.02,10.46,0.453,0.510,0.477],
#    ["ESTDepth~\cite{long2021multi} & No",12.71,7.54,10.12,0.456,0.542,0.491],
#    ["DPSNet~\cite{im2019dpsnet} & No",11.94,7.58,9.77,0.474,0.519,0.492 ],
#    ["DELTAS~\cite{sinha2020deltas} & No",11.95,7.46,9.71,0.478,0.533,0.501],
#    ["DeepVideoMVS~\cite{duzceker2021deepvideomvs} & No",10.68,6.90,8.79,0.541,0.592,0.563],
#    ["COLMAP~\cite{schonberger2016pixelwise,schoenberger2016sfm} & No",10.22,11.88,11.05,0.509,0.474,0.489],
#    ["ATLAS~\cite{murez2020atlas} & Yes",7.16,7.61,7.38,0.675,0.605,0.636 ],
#    ["NeuralRecon~\cite{sun2021neuralrecon} & Yes",5.09,9.13,7.11,0.630,0.612,0.619],
#    ["3DVNet~\cite{rich20213dvnet} & Yes",6.73,7.72,7.22,0.655,0.596,0.621 ],
#    ["TransformerFusion~\cite{bozic2021transformerfusion} & Yes",5.52,8.27, 6.89,0.728,0.600,0.655],
#    ["VoRTX~\cite{stier2021vortx} & Yes",4.31,7.23,5.77,0.767,0.651,0.703],
#    ["SimpleRecon~\cite{sayed2022simplerecon} (4cm)  & No",4.56,5.91,5.24,0.729,0.668,0.696],
#    ["FineRecon~\cite{Stier_2023_ICCV} (1cm) & Yes",5.25,5.06, 5.16,0.779,0.737,0.756],
#    ["\\textbf{Ours} (online) (4cm)& No",4.40,5.84,5.12,0.742,0.672,0.704],
#    ["\\textbf{Ours} (two-pass) (2cm)& No",4.50 ,4.68,4.59,0.761,0.740,0.755],
# ]
