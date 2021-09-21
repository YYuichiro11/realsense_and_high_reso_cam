#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:16:44 2021

@author: yuichiro

ここでは，sonyのα7cをpython上で制御し，1秒に一枚程度の間隔で撮影を行います
下記に従ってやってみる
https://github.com/Bloodevil/sony_camera_api

これらは、動画からフレームを切り取るだけ
https://www.yuseiphotos.work/entry/sonywebcam
https://yusei-roadstar.hatenablog.com/entry/2019/11/19/193312

"""
import sys
import cv2
import time
import datetime


# カメラが使えるかどうかテストします

# https://qiita.com/opto-line/items/7ade854c26a50a485159

# VideoCapture オブジェクトを取得します
cc = cv2.VideoCapture(0)

while True:
    # ret, frame = cc.read()
    # # resize the window
    # 

    # 
    
    rr, img = cc.read()
    
    windowsize = (3000,2000)
    img = cv2.resize(img, windowsize)
    # cv2.imshow('title',img)
    
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cv2.imwrite( now + '.jpg',img)
    time.sleep(0.8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cc.release()
cv2.destroyAllWindows()






# カメラの情報を読み込みます

# def check_camera_connection():
#     """
#     Check the connection between any camera and the PC.

#     """

#     print('[', datetime.datetime.now(), ']', 'searching any camera...')
#     true_camera_is = []  # 空の配列を用意

#     # カメラ番号を0～9まで変えて、COM_PORTに認識されているカメラを探す
#     for camera_number in range(0, 10):
#         cap = cv2.VideoCapture(camera_number)
#         ret, frame = cap.read()

#         if ret is True:
#             true_camera_is.append(camera_number)
#             print("camera_number", camera_number, "Find!")

#         else:
#             print("camera_number", camera_number, "None")
#     print("接続されているカメラは", len(true_camera_is), "台です。")


# if __name__ == "__main__":
#     check_camera_connection()

