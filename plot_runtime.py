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
dir_res = 'res/exp_runtime/runtime.csv'

data = pd.read_csv(dir_res)

# Default theme
sns.set_theme(style="white")

# Using the font_scale parameter
sns.set(font_scale=3.)

ax1 = sns.lineplot(data=data, x='n', y='ccDice_runtime')






