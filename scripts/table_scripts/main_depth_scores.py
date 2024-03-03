import pandas as pd
import numpy as np
import json


def distance_formatter(val):
    return f"{val:.4f}".lstrip("0")


def a_formatter(val):
    return f"{val:.2f}"


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


scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/7scenes/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_online_7scenes = json.load(f)["scores"]


scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_online_scannet = json.load(f)["scores"]


scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/7scenes/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    sr_7scenes = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    sr_scannet = json.load(f)["scores"]


used_metrics = ["abs_diff", "abs_rel", "sq_rel", "a5", "a25"]

scores = [
    [
        "DPSNet ~\cite{im2019dpsnet}",
        0.1552,
        0.0795,
        0.0299,
        49.36,
        93.27,
        0.1966,
        0.1147,
        0.0550,
        38.81,
        87.07,
    ],
    [
        "MVDepthNet ~\cite{wang2018mvdepthnet}",
        0.1648,
        0.0848,
        0.0343,
        46.71,
        92.77,
        0.2009,
        0.1161,
        0.0623,
        38.81,
        87.70,
    ],
    [
        "DELTAS~\cite{sinha2020deltas}",
        0.1497,
        0.0786,
        0.0276,
        48.64,
        93.78,
        0.1915,
        0.1140,
        0.0490,
        36.36,
        88.13,
    ],
    [
        "GPMVS ~\cite{hou2019multi}",
        0.1494,
        0.0757,
        0.0292,
        51.04,
        93.96,
        0.1739,
        0.1003,
        0.0462,
        42.71,
        90.32,
    ],
    [
        "DeepVideoMVS, fusion~\cite{duzceker2021deepvideomvs}*",
        0.1186,
        0.0583,
        0.0190,
        60.20,
        96.76,
        0.1448,
        0.0828,
        0.0335,
        47.96,
        93.79,
    ],
    # ["SimpleRecon \cite{sayed2022simplerecon}", 0.0873, 0.0430, 0.0128,  74.11,  98.05, 0.1045, 0.0575, 0.0156, 60.12, 97.33],
    ["SimpleRecon \cite{sayed2022simplerecon}"]
    + [sr_scannet[metric_name] for metric_name in used_metrics]
    + [sr_7scenes[metric_name] for metric_name in used_metrics],
    # ["\\textbf{Ours} (online)", 0.0768, 0.0371, 0.0111,  79.77,  98.36, 0.0951, 0.0520, 0.0149, 65.98, 97.34],
    ["\\textbf{Ours} (online)"]
    + [ours_online_scannet[metric_name] for metric_name in used_metrics]
    + [ours_online_7scenes[metric_name] for metric_name in used_metrics],
]


sort_direction = [False, False, False, False, True, True, False, False, False, True, True]
number_formatters = [
    itentity_formatter,
    distance_formatter,
    distance_formatter,
    distance_formatter,
    a_formatter,
    a_formatter,
    distance_formatter,
    distance_formatter,
    distance_formatter,
    a_formatter,
    a_formatter,
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
        df.loc[row_ind, col] = order_formatter(rounded_str_elem, order_ind)


# df.style.highlight_max(axis=0, props="textbf:--rwrap;")
print(df.to_latex(header=False, index=False))

# %%
