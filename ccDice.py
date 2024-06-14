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


def ccDice(y_pred, y_true, alpha=0.5):
    
    y_pred_label, cc_pred = label(y_pred, return_num=True)
    y_true_label, cc_true = label(y_true, return_num=True)
    
    y_true_label = y_true_label + cc_pred

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


def compute_matching(A, B, C_A, C_B, alpha=0.5):
    n, m = A.shape
    L = []
    indices_A = []
    indices_B = []
    for i in range(n):
        for j in range(m):
            if (C_A[i][j]>0 and C_B[i][j]>0) and ((C_A[i][j],C_B[i][j]) not in list_couples):
                
                a = np.zeros(A.shape)
                a[C_A==C_A[i][j]] = 1
                
                b = np.zeros(B.shape)
                b[C_B == C_B[i][j]] = 1
                
                s_ab = S(a, b)
                
                L.append(s_ab)
                indices_A.append(C_A[i][j])
                indices_B.append(C_B[i][j])
                list_couples.append((C_A[i][j],C_B[i][j]))
    
    #Sort the list
    L = np.array(L)
    indices_A = np.array(indices_A)
    indices_B = np.array(indices_B)
    indices = np.argsort(-L)
    L = L[indices]
    indices_A = indices_A[indices]
    indices_B = indices_B[indices]
    
    mu_ab = 0
    i = 0
    s = L[0]
    coor_a =  indices_A[0]
    coor_b = indices_B[0]
    left_list = []
    right_list = []
    while s >= alpha and i < len(L):
        if (coor_a not in left_list) and (coor_b not in right_list):
            
            left_list.append(coor_a)
            right_list.append(coor_b)
            mu_ab += 1
        
        i += 1
        if i < len(L):
            s = L[i]
            coor_a = indices_A[i]
            coor_b = indices_B[i]
    return mu_ab
    
    
    
def ccDice_v2(y_pred, y_true, alpha=0.5):
    
    y_pred_label, cc_pred = label(y_pred, return_num=True)
    y_true_label, cc_true = label(y_true, return_num=True)
    
    y_true_label = y_true_label + cc_pred
    
    mu_ab = compute_matching(y_pred, y_true, y_pred_label, y_true_label, alpha)
    mu_ba = compute_matching(y_true, y_pred, y_true_label, y_pred_label, alpha)
    
    cc_dice = (mu_ab + mu_ba) / (cc_pred + cc_true)
    
    return cc_dice
    

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
    
    start = time.time()
    ccdice_v1 = ccDice(y_pred, y_true)
    stop = time.time()
    print(f"time v1 {stop - start}")
    
    start = time.time()
    ccdice_v2 = ccDice_v2(y_pred, y_true)
    stop = time.time()
    print(f"time v2 {stop - start}")
    # bm = BettiMatching(y_pred, y_true)
    # BettiMatchingError = bm.loss()
    print(ccdice_v1)
    print(ccdice_v2)
    # print(BettiMatchingError)
    
     
