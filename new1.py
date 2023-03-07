# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:47:36 2022

@author: PIKAI
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


test_img = cv2.imread('data/test_img.jpg', 0)

th, test_img = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Image",test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
