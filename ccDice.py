#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:45:30 2024

@author: rouge
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.draw import ellipse


def S(y1, y2):
    return np.sum(y1 * y2) / np.sum(y1)


def ccDice(y_pred, y_true, alpha=0.5):
    
    y_pred_label, cc_pred = label(y_pred, return_num=True)
    y_true_label, cc_true = label(y_true, return_num=True)
    
    y_true_label = y_true_label + cc_pred

    list_s = []
    indices_cc = []
    m = np.zeros((cc_pred + cc_true, cc_pred + cc_true))
    list_coor_m = []
    for a in range(1, cc_pred + 1):
        for b in range(cc_pred + 1, cc_pred + cc_true + 1):
            
            y1 = np.zeros(y_pred_label.shape)
            y1[y_pred_label == a] = 1
            
            y2 = np.zeros(y_true_label.shape)
            y2[y_true_label == b] = 1
            
            s_ab = S(y1, y2)
            s_ba = S(y2, y1)
            
            list_s.append(s_ab)
            list_s.append(s_ba)
            
            m[a - 1, b - 1] = s_ab
            m[b - 1, a - 1] = s_ba
            
            list_coor_m.append((a - 1, b - 1))
            list_coor_m.append((b - 1, a - 1))
            
            indices_cc.append((a, b))
            indices_cc.append((b, a))
            
    # Sort the list
    list_s = np.array(list_s)
    indices = np.argsort(-list_s)
    indices_cc = np.array(indices_cc)
    list_coor_m = np.array(list_coor_m)
    
    list_s = np.array(list_s)
    list_s = list_s[indices]
    indices_cc = indices_cc[indices]
    list_coor_m = list_coor_m[indices]
    
    left_list = []
    right_list = []
    tp = 0
    tp_soft = 0.0
    i = 0
    s = 1.0
    while s >= alpha and i < len(list_s):
        s = list_s[i]
        coor = indices_cc[i]
        coor_m = list_coor_m[i]
        if (coor[0] not in left_list) and (coor[1] not in right_list):
        
            left_list.append(coor[0])
            right_list.append(coor[1])
            tp += 1
            tp_soft += m[coor_m[0], coor_m[1]]
            
        i += 1
        if i < len(list_s):
            s = list_s[i]
            coor = indices_cc[i]
            coor_m = list_coor_m[i]
               
    ccdice = tp / (cc_pred + cc_true)
    weighted_ccdice = tp_soft / (tp_soft + cc_pred + cc_true - tp)
    
    return ccdice, weighted_ccdice

if __name__ == '__main__':
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))


    y_pred = np.zeros((500, 500), dtype=np.double)
    y_true = np.zeros((500, 500), dtype=np.double)
    
    # fill ellipse
    rr, cc = ellipse(300, 300, 30, 30, y_pred.shape)
    y_pred[rr, cc] = 1
    
    # rr, cc = ellipse(330, 300, 15, 15, y_pred.shape)
    # y_pred[rr, cc] = 1
    
    # rr, cc = ellipse(360, 300, 15, 15, y_pred.shape)
    # y_pred[rr, cc] = 1
    
    # rr, cc = ellipse(400, 400, 20, 20, y_pred.shape)
    # y_pred[rr, cc] = 1
    
    rr, cc = ellipse(300, 300, 50, 50, y_pred.shape)
    y_true[rr, cc] = 1
    
    # rr, cc = ellipse(150, 105, 50, 50, y_pred.shape)
    # y_true[rr, cc] = 1
    
    # rr, cc = ellipse(50, 50, 15, 15, y_pred.shape)
    # y_true[rr, cc] = 1
    
    ax1.imshow(y_pred)
    ax2.imshow(y_true)
    
    res = ccDice(y_pred, y_true)
    print(res)
    
     
