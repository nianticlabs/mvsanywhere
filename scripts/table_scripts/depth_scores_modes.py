import json
import pandas as pd
import numpy as np


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


scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/7scenes/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    sr_7scenes = json.load(f)["scores"]
scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    sr_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/7scenes/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_online_7scenes = json.load(f)["scores"]
scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_online_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_no_hint/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_no_hint_scannet = json.load(f)["scores"]
scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_no_hint/7scenes/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_no_hint_7scenes = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_two_pass/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_two_pass_scannet = json.load(f)["scores"]
scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_two_pass/7scenes/offline/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_two_pass_7scenes = json.load(f)["scores"]

show_bold = False

used_metrics = ["abs_diff", "abs_rel", "sq_rel", "a5", "a25"]

scores = [
    ["SimpleRecon \cite{sayed2022simplerecon}"]
    + [sr_scannet[metric_name] for metric_name in used_metrics]
    + [sr_7scenes[metric_name] for metric_name in used_metrics],
    ["\\textbf{Ours} (no hint)"]
    + [ours_no_hint_scannet[metric_name] for metric_name in used_metrics]
    + [ours_no_hint_7scenes[metric_name] for metric_name in used_metrics],
    ["\\textbf{Ours} (online)"]
    + [ours_online_scannet[metric_name] for metric_name in used_metrics]
    + [ours_online_7scenes[metric_name] for metric_name in used_metrics],
    ["\\textbf{Ours} (Two Pass)"]
    + [ours_two_pass_scannet[metric_name] for metric_name in used_metrics]
    + [ours_two_pass_7scenes[metric_name] for metric_name in used_metrics],
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
        if show_bold:
            df.loc[row_ind, col] = order_formatter(rounded_str_elem, order_ind)
        else:
            df.loc[row_ind, col] = rounded_str_elem


# df.style.highlight_max(axis=0, props="textbf:--rwrap;")
print(df.to_latex(header=False, index=False))

# %%
