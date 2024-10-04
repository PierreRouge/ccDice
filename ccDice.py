#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:45:30 2024

@author: rouge
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.draw import ellipse

from utils.BettiMatching import BettiMatching


def S(y1, y2):
    return np.sum(y1 * y2) / np.sum(y1)


def ccDice(y_pred, y_true, alpha=0.6):
    
    y_pred_label, cc_pred = label(y_pred, return_num=True)
    y_true_label, cc_true = label(y_true, return_num=True)
    
    y_true_label[y_true_label != 0] = y_true_label[y_true_label != 0] + cc_pred

    list_s = []
    indices_cc = []
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
            
            indices_cc.append((a, b))
            indices_cc.append((b, a))
        
    if alpha <= 0.5:
        # Sort the list
        list_s = np.array(list_s)
        indices = np.argsort(-list_s)
        indices_cc = np.array(indices_cc)
        
        list_s = np.array(list_s)
        list_s = list_s[indices]
        indices_cc = indices_cc[indices]
    
    left_list = []
    right_list = []
    tp = 0
    i = 0
    s = list_s[0]
    coor = indices_cc[0]
    while s >= alpha and i < len(list_s):
        
        if (coor[0] not in left_list) and (coor[1] not in right_list):
        
            left_list.append(coor[0])
            right_list.append(coor[1])
            tp += 1
            
        i += 1
        if i < len(list_s):
            s = list_s[i]
            coor = indices_cc[i]
          
    ccdice = tp / (cc_pred + cc_true)
    
    return ccdice


def compute_matching(C_A, C_B, u, t, alpha=0.6):
    n, m = C_A.shape
    L = [{} for i in range(u)]
    size_cc = np.zeros((u, 1))
    for i in range(n):
        for j in range(m):
            if (C_A[i][j] > 0 and C_B[i][j] > 0):
                
                if f"{C_B[i][j]}" not in L[C_A[i][j] - 1]:
                    
                    L[C_A[i][j] - 1][f"{C_B[i][j]}"] = 1
                    
                else:
                    
                    L[C_A[i][j] - 1][f"{C_B[i][j]}"] += 1
                    
            if C_A[i][j] > 0:
                
                size_cc[C_A[i][j] - 1] += 1
      
    indices_A = []
    indices_B = []
    embedding_scores = []
    for i in range(len(L)):
        for key in L[i].keys():
            embedding_scores.append(L[i][key] / size_cc[i].item())
            indices_A.append(i + 1)
            indices_B.append(int(key))
    
    # Sort the list
    if alpha <= 0.5:
        embedding_scores = np.array(embedding_scores)
        indices_A = np.array(indices_A)
        indices_B = np.array(indices_B)
        indices = np.argsort(-embedding_scores)
        embedding_scores = embedding_scores[indices]
        indices_A = indices_A[indices]
        indices_B = indices_B[indices]
    
    mu_ab = 0
    
    if len(embedding_scores) != 0:
        i = 0
        s = embedding_scores[0]
        coor_a = indices_A[0]
        coor_b = indices_B[0]
        left_list = []
        right_list = []
        while s >= alpha and i < len(embedding_scores):
            if (coor_a not in left_list) and (coor_b not in right_list):
                
                left_list.append(coor_a)
                right_list.append(coor_b)
                mu_ab += 1
            
            i += 1
            if i < len(embedding_scores):
                s = embedding_scores[i]
                coor_a = indices_A[i]
                coor_b = indices_B[i]
    return mu_ab
    

if __name__ == '__main__':
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))

    y_pred = np.zeros((500, 500), dtype=np.double)
    y_true = np.zeros((500, 500), dtype=np.double)
    
    # fill ellipse
    # rr, cc = ellipse(300, 300, 40, 40, y_pred.shape)
    # y_pred[rr, cc] = 1
    
    rr, cc = ellipse(330, 300, 15, 15, y_pred.shape)
    y_pred[rr, cc] = 1
    
    rr, cc = ellipse(360, 300, 15, 15, y_pred.shape)
    y_pred[rr, cc] = 1
    
    rr, cc = ellipse(390, 300, 15, 15, y_pred.shape)
    y_pred[rr, cc] = 1
    
    rr, cc = ellipse(360, 300, 70, 70, y_pred.shape)
    y_true[rr, cc] = 1
    
    # rr, cc = ellipse(150, 105, 50, 50, y_pred.shape)
    # y_true[rr, cc] = 1
    
    # rr, cc = ellipse(50, 50, 15, 15, y_pred.shape)
    # y_true[rr, cc] = 1
    
    ax1.imshow(y_pred)
    ax2.imshow(y_true)
    
    alpha=0.5
    
    start = time.time()
    ccdice_v1 = ccDice(y_pred, y_true, alpha=alpha)
    stop = time.time()
    print(f"time v1 {stop - start}")
    
