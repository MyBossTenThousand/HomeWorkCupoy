# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:35:36 2020

@author: user
"""

import cv2
import time
import numpy as np

img_path = 'lena.png'
img = cv2.imread(img_path)

# 水平翻轉 (horizontal)
img_hflip = img[::, ::-1, :]

# 垂直翻轉 (vertical)
img_vflip = img[::-1, :, :]

# 水平 + 垂直翻轉
img_hvflip = img[::-1, ::-1, :]

# 組合 + 顯示圖片
hflip = np.hstack((img, img_hflip))
vflip = np.hstack((img_vflip, img_hvflip))
img_flip = np.vstack((hflip, vflip))
while True:
    cv2.imshow('flip image', img_flip)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break


# 將圖片縮小成原本的 20%
img_test = cv2.resize(img, None, fx=0.2, fy=0.2)

# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
fx, fy = 8, 8

# 鄰近差值 scale + 計算花費時間
start_time = time.time()
img_area_scale = cv2.resize(img_test, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
print('INTER_NEAREST zoom cost {}'.format(time.time() - start_time))

# 雙立方差補 scale + 計算花費時間
start_time = time.time()
img_cubic_scale = cv2.resize(img_test, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
print('INTER_CUBIC zoom cost {}'.format(time.time() - start_time))


# 組合 + 顯示圖片
img_zoom = np.hstack((img_area_scale, img_cubic_scale))
while True:
    cv2.imshow('zoom image', img_zoom)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break

# 設定 translation transformation matrix
# x 平移 50 pixel; y 平移 100 pixel
M = np.array([[1, 0, 50],
              [0, 1, 100]], dtype=np.float32)
shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# 組合 + 顯示圖片
img_shift = np.hstack((img, shift_img))
while True:
    cv2.imshow('shift image', img_shift)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
