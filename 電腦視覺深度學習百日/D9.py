# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:46:10 2020

@author: user
"""

import numpy as np
import cv2

img = cv2.imread('lena.png')

# 轉灰階圖片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 建立 SIFT 物件
SIFT_detector = cv2.xfeatures2d.SIFT_create()

# 取得 SIFT 關鍵點位置
keypoints = SIFT_detector.detect(img_gray, None)

#　畫圖 + 顯示圖片
img_show = cv2.drawKeypoints(img_gray, keypoints, img)
while True:
    cv2.imshow('SIFT', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break