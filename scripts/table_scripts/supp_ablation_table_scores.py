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


metrics = [
    "abs_diff",
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "a5",
    "a10",
    "a25",
    "a0",
    "a1",
    "a2",
    "a3",
]
null_elements = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
null_elements = {metric: value for metric, value in zip(metrics, null_elements)}

metrics = [
    "abs_diff",
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "a5",
    "a10",
    "a25",
    "a0",
    "a1",
    "a2",
    "a3",
]
tocd_sr = [
    0.0880,
    0.0437,
    0.0134,
    0.1505,
    0.0702,
    74.1887,
    90.4090,
    97.8279,
    90.4090,
    97.8279,
    99.4579,
    99.8047,
]
tocd_sr = {metric: value for metric, value in zip(metrics, tocd_sr)}

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    sr_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/no_weight_ablation_incremental_no_clip/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    no_confidence_model = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/warped_depth_ablation/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    warped_depth_ablation = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental_2cm/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_online_scannet = json.load(f)["scores"]


scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/mesh_model_fast/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_fast_online_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_no_hint/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_no_hint_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_two_pass/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_two_pass_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_two_pass/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    ours_two_pass_scannet = json.load(f)["scores"]

scores_path = "/mnt/nas3/personal/mohameds/geometry_hints/outputs/ablations_single_mlp_incremental/scannet/default/scores/all_frame_avg_metrics_test.json"
with open(scores_path, "r") as f:
    single_mlp_ablation = json.load(f)["scores"]


scores_path = "scripts/table_scripts/simple_mapping.json"
with open(scores_path, "r") as f:
    simple_mapping_scannet = json.load(f)["scores"]

scores_path = "scripts/table_scripts/guidedMVS.json"
with open(scores_path, "r") as f:
    guided_mvs_scannet = json.load(f)["scores"]


scores_path = "scripts/table_scripts/cost_vol_encoder_hint.json"
with open(scores_path, "r") as f:
    cost_vol_encoder_hint = json.load(f)["scores"]

show_bold = False

used_metrics = ["abs_diff", "abs_rel", "sq_rel", "rmse", "rmse_log", "a5", "a25"]

scores = [
    ["\\row{row:ours_no_hint_mlp} & \\textbf{Ours} without Hint MLP (as in SimpleRecon~\cite{sayed2022simplerecon})"]
    + [sr_scannet[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_no_confidence} & \\textbf{Ours} w/ online hint, without confidence"]
    + [no_confidence_model[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_hint_in_cv_enc} & \\textbf{Ours} w/ hint \& confidence to cost volume encoder"]
    + [cost_vol_encoder_hint[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_warped_depth} & \\textbf{Ours} w/ warped depth as hint"]
    + [warped_depth_ablation[metric_name] for metric_name in used_metrics],
    # ["\\row{row:ours_sdf_conf} & \\textbf{Ours} w/ sampled SDF confidences \& hint MLP"]
    # + [null_elements[metric_name] for metric_name in used_metrics],
    # ["\\row{row:ours_guided_cv} & \\textbf{Ours} w/ guided cost volume \cite{poggi2022multi}"]
    # + [guided_mvs_scannet[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_variable_depth_planes} & \\textbf{Ours} w/ variable depth planes \cite{Xin2023ISMAR}"]
    + [simple_mapping_scannet[metric_name] for metric_name in used_metrics],
    # ["\\row{row:ours_pred_depth_twice} & \\textbf{Ours} w/ current predicted depth run twice"]
    # + [null_elements[metric_name] for metric_name in used_metrics],
    ["\\row{row:tocd} & SimpleRecon~\cite{sayed2022simplerecon} w/ TOCD \cite{khan2023temporally}"]
    + [tocd_sr[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_single_mlp} & \\textbf{Ours} w/ single MLP for matching and hints"]
    + [single_mlp_ablation[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_no_hint} & \\textbf{Ours} (no hint)"]
    + [ours_no_hint_scannet[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours_fast} & \textbf{Ours} (\incremental, fast)"]
    + [ours_fast_online_scannet[metric_name] for metric_name in used_metrics],
    ["\\row{row:ours} & \textbf{Ours} (\incremental)"]
    + [ours_online_scannet[metric_name] for metric_name in used_metrics],
]


sort_direction = [False, False, False, False, True, True, False, False, False, False, True, True]
number_formatters = [
    itentity_formatter,
    distance_formatter,
    distance_formatter,
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
