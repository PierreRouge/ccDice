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
dir_res = 'res/xp3/full_res.csv'

data = pd.read_csv(dir_res)

dataframe= data.melt("nb_disconnections", var_name="Metric", value_name="Score")


df1 = dataframe.loc[dataframe['Metric'] == 'Dice']
df2 = dataframe.loc[dataframe['Metric'] == 'ccDice']
df3 =  dataframe.loc[dataframe['Metric'] == 'clDice']

df_1 = pd.concat((df1, df2, df3))

df1 = dataframe.loc[dataframe['Metric'] == 'B0Error']
df2 = dataframe.loc[dataframe['Metric'] == 'BettiMatchingError']

df_2 = pd.concat((df1, df2))

# Default theme
sns.set_theme(style="white")

# Using the font_scale parameter
sns.set(font_scale=3.)

palette1 = ['#9336FD', '#F77976', '#FDF148']
palette2 = ['#A0E426', '#33A8C7', '#FFAB00']

# ax1 = sns.lineplot(data=df_1, x="nb_disconnections", y="Score", hue="Metric", errorbar='sd', err_style='band', estimator='mean', marker='o', linewidth = 1, palette=palette1, legend=True)
ax1 = sns.lineplot(data=df_1, x="nb_disconnections", y="Score", hue="Metric", marker='o', errorbar=None, linewidth = 2.5, markersize=12, palette=palette1, legend=False)
# ax1 = sns.scatterplot(data=df_1, x="nb_disconnections", y="Score", hue="Metric")

# ax1 = sns.boxplot(data=df_1, x="nb_disconnections", y="Score", hue="Metric")

ax2 = plt.twinx()
ax2 = sns.lineplot(data=df_2, x="nb_disconnections", y="Score", hue="Metric", marker='o', errorbar=None, linewidth = 2.5, markersize=12, palette=palette2, legend=False)

plt.rcParams['text.usetex'] = False
# plt.xlim((1, 10))
ax1.set_xlabel('NbrDisconnections')
ax1.set_ylabel('Dice/clDice/ccDice')
ax2.set_ylabel('b0error/bmatchingerror')
ax2.invert_yaxis()
# ax2.set_ylim(1,16)
ax2.grid(None)





