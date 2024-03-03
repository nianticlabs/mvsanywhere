
import json
import pandas as pd
import numpy as np


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

def distance_formatter(val):
    return f"{val*100:.2f}"

def perc_formatter(val):
    return f"{val:.3f}".lstrip('0')

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

all_scores_dict = {}

used_metrics = ["acc", "compl", "chamfer", "prc", "rec", "f1_score"]
sort_direction = [False, False, False, False, True, True, True]
number_formatters = [itentity_formatter, distance_formatter, distance_formatter, distance_formatter, perc_formatter, perc_formatter, perc_formatter]
show_bold = True

mask_type = "tf"

if mask_type == "tf":
    scores_json_name = "scores.json"
else:
    scores_json_name = "scores_our_masks.json"


metrics = ["acc", "compl", "chamfer", "prc", "rec", "f1_score"]
if mask_type == "tf":
    all_scores_dict["RevisitingSI~\cite{hu2019revisiting} & No"] = {metric: value for metric, value in zip(metrics, [14.29/100, 16.19/100, 15.24/100, 0.346, 0.293, 0.314])}
    all_scores_dict["MVDepthNet~\cite{wang2018mvdepthnet} & No"] = {metric: value for metric, value in zip(metrics, [12.94/100,8.34/100,10.64/100,0.443,0.487,0.460])}
    all_scores_dict["GPMVS~\cite{hou2019multi} & No"] = {metric: value for metric, value in zip(metrics, [12.90/100,8.02/100,10.46/100,0.453,0.510,0.477])}
    all_scores_dict["ESTDepth~\cite{long2021multi} & No"] = {metric: value for metric, value in zip(metrics, [12.71/100,7.54/100,10.12/100,0.456,0.542,0.491])}
    all_scores_dict["DPSNet~\cite{im2019dpsnet} & No"] = {metric: value for metric, value in zip(metrics, [11.94/100,7.58/100,9.77/100,0.474,0.519,0.492])}
    all_scores_dict["DELTAS~\cite{sinha2020deltas} & No"] = {metric: value for metric, value in zip(metrics, [11.95/100,7.46/100,9.71/100,0.478,0.533,0.501])}
    all_scores_dict["DeepVideoMVS~\cite{duzceker2021deepvideomvs} & No"] = {metric: value for metric, value in zip(metrics, [10.68/100,6.90/100,8.79/100,0.541,0.592,0.563])}
    all_scores_dict["NeuralRecon~\cite{sun2021neuralrecon} & Yes"] = {metric: value for metric, value in zip(metrics, [5.09/100,9.13/100,7.11/100,0.630,0.612,0.619])}


all_scores_dict["SimpleRecon~\cite{sayed2022simplerecon} (online) (4cm)  & No"] = json.load(open(f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/sr_new_scores/scannet/default/meshes/0.04_3.0_ours/{scores_json_name}"))["overall"]
all_scores_dict["\\textbf{Ours} (online) (4cm)& No"] = json.load(open(f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental/scannet/default/meshes/0.04_3.0_ours/{scores_json_name}"))["overall"]
# all_scores_dict["\\textbf{Ours} (online) (2cm)& No"] = json.load(open(f"/mnt/nas3/personal/mohameds/geometry_hints/outputs/final_model_new_renders_incremental_2cm/scannet/default/meshes/0.02_3.0_ours/{scores_json_name}"))["overall"]

# Define the scores dictionary
scores = [["&" + key] + [all_scores_dict[key][metric] for metric in used_metrics] for key in all_scores_dict.keys()]




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
