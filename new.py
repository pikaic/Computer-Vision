# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:55:16 2022

@author: PIKAI
"""
import argparse
import json
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================

test_img=cv2.imread('data/test_img.jpg',0)

th, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY_INV)
height,width=test_img.shape

img=test_img

for i in range(0, height):
    for j in range(0, width):
        if img[i,j]==255:
            img[i,j]=1
        elif img[i,j]==0:
            img[i,j]=0



#first pass

def ccl(img):
    current_label=1
    img1=np.array(img)
    labels=np.array(img)
    
    
    label_conv = []
    label_conv.append([])
    label_conv.append([])
    count = 0
    for i in range(1, len(img1)):
        for j in range(1, len(img1[0])):
            if img1[i][j] > 0:
                label_x = labels[i][j - 1]
                label_y = labels[i - 1][j]
                
                
                if label_x>0:
                    if label_y>0:
                        if not label_x==label_y:
                            labels[i][j]==min(label_x,label_y)
                            if max(label_x, label_y) not in label_conv[0]:
                                label_conv[0].append(max(label_x, label_y))
                                label_conv[1].append(min(label_x, label_y))
                            elif max(label_x, label_y) in label_conv[0]:
                                ind = label_conv[0].index(max(label_x, label_y))
                                if label_conv[1][ind] > min(label_x, label_y):
                                    l = label_conv[1][ind]
                                    label_conv[1][ind] = min(label_x, label_y)
                                    while l in label_conv[0] and count < 100:
                                        count += 1
                                        ind = label_conv[0].index(l)
                                        l = label_conv[1][ind]
                                        label_conv[1][ind] = min(label_x, label_y)
                                    label_conv[0].append(l)
                                    label_conv[1].append(min(label_x, label_y))
                        
                        else:
                            labels[i][j] = label_y
                    else:
                        labels[i][j] = label_x
                        
                elif label_y > 0:
                    labels[i][j] = label_y
                else:
                    labels[i][j] = current_label
                    current_label += 1
                    
                    
    count = 1
    for idx, val in enumerate(label_conv[0]):
        if label_conv[1][idx] in label_conv[0] and count < 100:
            count += 1
            ind = label_conv[0].index(label_conv[1][idx])
            label_conv[1][idx] = label_conv[1][ind]
    for i in range(1, len(labels)):
        for j in range(1, len(labels[0])):
            if labels[i][j] in label_conv[0]:
                ind = label_conv[0].index(labels[i][j])
                labels[i][j] = label_conv[1][ind]
                
    return labels



img = ccl(img)