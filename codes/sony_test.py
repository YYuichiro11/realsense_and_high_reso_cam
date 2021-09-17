#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:16:44 2021

@author: yuichiro

ここでは，sonyのα7cをpython上で制御し，1秒に一枚程度の間隔で撮影を行います
https://www.yuseiphotos.work/entry/sonywebcam
https://yusei-roadstar.hatenablog.com/entry/2019/11/19/193312

"""


# import cv2

# ##############  Camera capture ################################
# cap = cv2.VideoCapture(1)
# cols = 1920
# rows = 1080
# framerate = 60



# cap.set(cv2.CAP_PROP_FPS, framerate)      #カメラFPSを60FPSに設定
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, cols)   #カメラ画像の横幅を1920設定
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)  #カメラ画像の縦幅を1080に設定


# print (cap.get(cv2.CAP_PROP_FPS))
# print (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()


#     cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
#     cv2.imshow('frame',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#     cv2.imshow("result", frame)



# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


# https://qiita.com/opto-line/items/7ade854c26a50a485159

import cv2

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(2)

while(True):
    ret, frame = capture.read()
    # resize the window
    windowsize = (800, 600)
    frame = cv2.resize(frame, windowsize)

    cv2.imshow('title',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


# import cv2
# import datetime


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

