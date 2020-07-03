# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:22:00 2020

@author: user
"""

import cv2

img_path = 'lena.png'

# 以彩色圖片的方式載入
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# 改變不同的 color space
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# 為了要不斷顯示圖片，所以使用一個迴圈
while True:
    cv2.imshow('bgr', img)
    cv2.imshow('hls', img_hls)

    # 直到按下 ESC 鍵才會自動關閉視窗結束程式
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break