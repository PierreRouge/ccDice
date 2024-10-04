#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:35:44 2023

@author: rouge
"""

from glob import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Directory of csv files with results
dir_res = 'res/xp_runtime_ccDicev1/runtime.csv'

data = pd.read_csv(dir_res)

dataframe= data.melt("n", var_name="Metric", value_name="Runtime")

palette = ['#9336FD', '#F77976', '#FDF148', '#A0E426', '#33A8C7', '#FFAB00']


# Default theme
sns.set_theme(style="white")

# Using the font_scale parameter
sns.set(font_scale=3.)

ax1 = sns.lineplot(data=dataframe, x='n', y='Runtime', hue='Metric', hue_order=['Dice_runtime', 'ccDice_runtime', 'clDice_runtime','B0Error_runtime', 'BettiMatchingError_runtime'], marker='o',  errorbar=None, linewidth = 2.5, markersize=12, palette=palette )

# plt.scatter(600000, 40, marker='o', s=100)




