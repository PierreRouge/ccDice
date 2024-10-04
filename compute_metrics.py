#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:48:07 2024

@author: rouge
"""

import os
import csv
import time
import numpy as np
from glob import glob
from skimage import io

from utils.utils_measure import dice_numpy, b0_error_numpy, cldice_numpy
from ccDice import ccDice_v2
from utils.BettiMatching import BettiMatching

input_dir_gt = 'data/CHASE/GT/*'
input_dir_seg = 'data/CHASE/decoV1/'
res_dir = 'res/'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

nb_disconnections = 20

full_res = open(res_dir + '/full_res.csv', 'w')
fieldnames = ['nb_disconnections', 'Dice', 'clDice', 'B0Error', 'BettiMatchingError', 'ccDice']
writer_full_res = csv.DictWriter(full_res, fieldnames=fieldnames)
writer_full_res.writeheader()

res = open(res_dir + '/res.csv', 'w')
fieldnames = ['nb_disconnections', 'Dice', 'clDice', 'B0Error', 'BettiMatchingError', 'ccDice']
writer_res = csv.DictWriter(res, fieldnames=fieldnames)
writer_res.writeheader()

runtime = open(res_dir + '/runtime.csv', 'w')
fieldnames = ['nb_disconnections', 'Dice_runtime', 'clDice_runtime', 'B0Error_runtime', 'BettiMatchingError_runtime', 'ccDice_runtime']
writer_runtime = csv.DictWriter(runtime, fieldnames=fieldnames)
writer_runtime.writeheader()


for j in range(1, nb_disconnections + 1):
    
    # input_dir_deco1 = os.path.join(input_dir_gt, f'nb_disconnections={j}/*')
    input_dir_deco2 = os.path.join(input_dir_seg, f'nb_disconnections={j}')
    
    dice_list = []
    cldice_list = []
    b0_error_list = []
    ccdice_list = []
    bettimatching_list = []
    time_dice_list = []
    time_cldice_list = []
    time_b0error_list = []
    time_ccdice_list = []
    time_bettimatching_list = []
    for file in glob(input_dir_gt):
        
        filename = file.split('/')[-1]
        
        image = io.imread(file).astype(bool)
        image_disconnected = io.imread(os.path.join(input_dir_deco2, filename)).astype(bool)
        
        start = time.time()
        ccdice = ccDice_v2(image_disconnected, image, alpha=0.5)
        stop = time.time()
        time_ccdice = stop - start
        print("Time ccDice")
        print(time_ccdice)
        print(ccdice)
            
        start = time.time()
        dice = dice_numpy(y_pred=image_disconnected, y_true=image)
        stop = time.time()
        time_dice = stop - start
        print("Time Dice")
        print(time_dice)
        print(dice)
              
        start = time.time()
        b0_error, _, _ = b0_error_numpy(y_pred=image_disconnected, y_true=image, method='difference')
        stop = time.time()
        time_b0error = stop - start
        print("Time b0error")
        print(time_b0error)
        print(b0_error)
        
        start = time.time()
        cldice = cldice_numpy(y_pred=image_disconnected, y_true=image)
        stop = time.time()
        time_cldice = stop - start
        print("Time clDice")
        print(time_cldice)
        print(cldice)
        
        # start = time.time()
        # bm = BettiMatching(image_disconnected, image)
        # BettiMatchingError = bm.loss()
        # stop = time.time()
        # time_bm = stop - start
        # print('Time BettiMatchingError')
        # print(time_bm)
        # print(BettiMatchingError)
        
        dice_list.append(dice)
        cldice_list.append(cldice)
        b0_error_list.append(b0_error)
        # bettimatching_list.append(BettiMatchingError)
        ccdice_list.append(ccdice)
        time_ccdice_list.append(time_ccdice)
        time_dice_list.append(time_dice)
        time_b0error_list.append(time_b0error)
        # time_bettimatching_list.append(time_bm)
        time_cldice_list.append(time_cldice)
        
        dict_csv = {
            'nb_disconnections': str(j),
            'Dice': dice,
            'clDice': cldice,
            'B0Error': b0_error,
            # 'BettiMatchingError': BettiMatchingError,
            'ccDice': ccdice,
        }
        writer_full_res.writerow(dict_csv)
        
    dict_csv = {
        'nb_disconnections': str(j),
        'Dice': np.mean(dice_list),
        'clDice': np.mean(cldice_list),
        'B0Error': np.mean(b0_error_list),
        'BettiMatchingError': np.mean(bettimatching_list),
        'ccDice': np.mean(ccdice_list),
    }
    writer_res.writerow(dict_csv)
    
    dict_runtime = {
        'nb_disconnections': str(j),
        'Dice_runtime': np.mean(time_dice_list),
        'clDice_runtime': np.mean(time_cldice_list),
        'B0Error_runtime': np.mean(time_b0error_list),
        'BettiMatchingError_runtime': np.mean(time_bettimatching_list),
        'ccDice_runtime': np.mean(time_ccdice_list)
    }
    writer_runtime.writerow(dict_runtime)
    
full_res.close()
res.close()
runtime.close()
