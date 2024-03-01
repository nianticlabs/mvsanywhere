
import pandas as pd
import numpy as np


def distance_formatter(val):
    return f"{val:.4f}".lstrip('0')

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

ours_twopass_7scenes = {
    "abs_diff": 0.0985,
    "abs_rel" : 0.0534,
    "sq_rel"  : 0.0156,
    "rmse"   : 0.1600,
    "rmse_log": 0.0838,
    "a5"      : 64.7567,
    "a10"     : 87.6873,
    "a25"     : 97.0115,
    "a0"      : 87.6873,
    "a1"      : 97.0115,
    "a2"      : 99.1971,
    "a3"      : 99.8179,
}

ours_online_7scenes = {
    "abs_diff": 0.0985,
    "abs_rel" : 0.0534,
    "sq_rel"  : 0.0156,
    "rmse"   : 0.1600,
    "rmse_log": 0.0838,
    "a5"      : 64.7567,
    "a10"     : 87.6873,
    "a25"     : 97.0115,
    "a0"      : 87.6873,
    "a1"      : 97.0115,
    "a2"      : 99.1971,
    "a3"      : 99.8179,
}

ours_no_hint_7scenes = {
    "abs_diff": 0.0985,
    "abs_rel" : 0.0534,
    "sq_rel"  : 0.0156,
    "rmse"   : 0.1600,
    "rmse_log": 0.0838,
    "a5"      : 64.7567,
    "a10"     : 87.6873,
    "a25"     : 97.0115,
    "a0"      : 87.6873,
    "a1"      : 97.0115,
    "a2"      : 99.1971,
    "a3"      : 99.8179,
}


ours_online_scannet = {
    "abs_diff": 0.0768139511346817,
    "abs_rel": 0.037004340440034866,
    "sq_rel": 0.011169915087521076,
    "rmse": 0.13763342797756195,
    "rmse_log": 0.06290297210216522,
    "a5": 79.81596374511719,
    "a10": 93.19886779785156,
    "a25": 98.34837341308594,
    "a0": 93.19886779785156,
    "a1": 98.34837341308594,
    "a2": 99.5193099975586,
    "a3": 99.81543731689453,
}

ours_nohint_scannet = {
    "abs_diff": 0.0768139511346817,
    "abs_rel": 0.037004340440034866,
    "sq_rel": 0.011169915087521076,
    "rmse": 0.13763342797756195,
    "rmse_log": 0.06290297210216522,
    "a5": 79.81596374511719,
    "a10": 93.19886779785156,
    "a25": 98.34837341308594,
    "a0": 93.19886779785156,
    "a1": 98.34837341308594,
    "a2": 99.5193099975586,
    "a3": 99.81543731689453,
}

ours_twopass_scannet = {
    "abs_diff": 0.0768139511346817,
    "abs_rel": 0.037004340440034866,
    "sq_rel": 0.011169915087521076,
    "rmse": 0.13763342797756195,
    "rmse_log": 0.06290297210216522,
    "a5": 79.81596374511719,
    "a10": 93.19886779785156,
    "a25": 98.34837341308594,
    "a0": 93.19886779785156,
    "a1": 98.34837341308594,
    "a2": 99.5193099975586,
    "a3": 99.81543731689453,
}

used_metrics = ["abs_diff", "abs_rel", "sq_rel", "a5", "a25"]

scores = [
    ["DPSNet ~\cite{im2019dpsnet}",0.1552, 0.0795, 0.0299, 49.36, 93.27, 0.1966, 0.1147, 0.0550, 38.81, 87.07],
    ["MVDepthNet ~\cite{wang2018mvdepthnet}", 0.1648, 0.0848, 0.0343, 46.71, 92.77, 0.2009, 0.1161, 0.0623, 38.81, 87.70],
    ["DELTAS~\cite{sinha2020deltas}", 0.1497, 0.0786, 0.0276, 48.64, 93.78, 0.1915, 0.1140, 0.0490, 36.36, 88.13],
    ["GPMVS ~\cite{hou2019multi}", 0.1494, 0.0757, 0.0292, 51.04, 93.96, 0.1739, 0.1003, 0.0462, 42.71, 90.32],
    ["DeepVideoMVS, fusion~\cite{duzceker2021deepvideomvs}*", 0.1186, 0.0583, 0.0190, 60.20, 96.76, 0.1448, 0.0828, 0.0335,  47.96, 93.79],
    ["SimpleRecon \cite{sayed2022simplerecon}", 0.0873, 0.0430, 0.0128,  74.11,  98.05, 0.1045, 0.0575, 0.0156, 60.12, 97.33],
    # ["\\textbf{Ours} (online)", 0.0768, 0.0371, 0.0111,  79.77,  98.36, 0.0951, 0.0520, 0.0149, 65.98, 97.34],
    ["\\textbf{Ours} (online)"] + [ours_online_scannet[metric_name] for metric_name in used_metrics] + [ours_online_7scenes[metric_name] for metric_name in used_metrics],
]


sort_direction = [False, False, False, False, True, True, False, False, False, True, True]
number_formatters = [itentity_formatter, distance_formatter, distance_formatter, distance_formatter, a_formatter, a_formatter, distance_formatter, distance_formatter, distance_formatter, a_formatter, a_formatter]


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
