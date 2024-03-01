
import pandas as pd
import numpy as np


def distance_formatter(val):
    return f"{val:.2f}"

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

# Define the scores dictionary
scores = [
#    ["COLMAP~\cite{schonberger2016pixelwise,schoenberger2016sfm} & No",10.22,11.88,11.05,0.509,0.474,0.489],
#    ["ATLAS~\cite{murez2020atlas} & Yes",7.16,7.61,7.38,0.675,0.605,0.636 ],
#    ["NeuralRecon~\cite{sun2021neuralrecon} & Yes",5.09,9.13,7.11,0.630,0.612,0.619],
#    ["3DVNet~\cite{rich20213dvnet} & Yes",6.73,7.72,7.22,0.655,0.596,0.621 ],
   ["TransformerFusion~\cite{bozic2021transformerfusion} & Yes",4.6822, 8.2680, 6.4751, 0.6984, 0.5999, 0.6442],
   ["VoRTX~\cite{stier2021vortx} & Yes",4.3794, 7.2266, 5.8030, 0.7257, 0.6508, 0.6851],
   ["SimpleRecon~\cite{sayed2022simplerecon} (4cm)  & No",5.0456, 5.9082, 5.4769, 0.6736, 0.6681, 0.6692],
   ["FineRecon~\cite{Stier_2023_ICCV} (1cm) & Yes",4.9243, 5.0599, 4.9921, 0.6869, 0.7373, 0.7098],
   ["\\textbf{Ours} (online) (4cm)& No",4.6769, 5.8353, 5.2561, 0.7006, 0.6721, 0.6849],
   ["\\textbf{Ours} (two-pass) (2cm)& No",3.8893, 5.1732, 4.5313, 0.7574, 0.7233, 0.7390],
]

sort_direction = [False, False, False, False, True, True, True]
number_formatters = [itentity_formatter, distance_formatter, distance_formatter, distance_formatter, perc_formatter, perc_formatter, perc_formatter]


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
