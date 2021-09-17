#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:40:07 2021

@author: yuichiro

このコードでは，奥平くんのコードを参考に，実際にどのような処理を行っていたのかを検証します
参考にするコードはfinal.pyです
"""

import pyrealsense2 as rs
import numpy as np
import datetime
import cv2
import math
import subcode

#ディレクトリの準備（dataファイルの中にcamera1~6のファイルを作る)
directry = '/Users/yuichiro/Research/realsense/realsense_and_high_reso_cam/okudaira'

#cameraformation(distance,angle)   
camera1formation = (0.19,0)

#カメラ位置の登録
fx1,fy1 = subcode.cameraformation(camera1formation[0],camera1formation[1])

#下準備
array_datas1 = np.array((1,0,0,0,0,0,0,0,0))
positions = np.array((0,0,0,0))
position = np.array((0,0,0))
position1_old = np.array((0,0,0))
t_err = 1
i = 0

#ビデオの出力パラメータ
fps = 10
w = 300
h = 300

#特徴点検出用パラメータ
feature_params = dict(maxCorners=0,qualityLevel=0.3,minDistance=7,
                      blockSize=5,useHarrisDetector=False)

#ステレオ視の準備
cameradistance = 6.7
pix_to_cm = 0.0008
window_size = 5
min_disp = 0
# must be divisible by 16
num_disp = 112 - min_disp
max_disp = min_disp + num_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,
                               blockSize = 16,P1 = 8*3*window_size**2,
                               P2 = 32*3*window_size**2,disp12MaxDiff = 1,
                               uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32)
stereo_fov_rad = 90 * (math.pi/180)
stereo_height_px = 300
stereo_focal_px = stereo_height_px/2 / math.tan(stereo_fov_rad/2)
stereo_width_px = stereo_height_px + max_disp
stereo_size = (stereo_width_px, stereo_height_px)
stereo_cx = (stereo_height_px - 1)/2 + max_disp
stereo_cy = (stereo_height_px - 1)/2
P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                   [0, stereo_focal_px, stereo_cy, 0],
                   [0,               0,         1, 0]])
P_right = P_left.copy()
m1type = cv2.CV_32FC1
kernel = np.ones((3,3),np.uint8)

#ファイル名を出力
dt_now = datetime.datetime.now()
FILE_OUT= 'out_T265_{0:test.avi}'.format(dt_now)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1 = cv2.VideoWriter(directry+'\\camera1\\camera1'+FILE_OUT, fourcc,fps,(w,h),True)
# 自己位置姿勢データを取得する設定
# カメラ1・・・
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("2322110136")
config_1.enable_stream(rs.stream.pose)
config_1.enable_stream(rs.stream.fisheye,1)
config_1.enable_stream(rs.stream.fisheye,2)

# 全てのカメラのストリーミングの開始
pipeline_1.start(config_1)
print('press q to quit this program')

while True:
    # カメラ1
    # 自己位置姿勢データおよび深度推定 
    frames_1 = pipeline_1.wait_for_frames()
    profiles_1 = pipeline_1.get_active_profile()
    pose_frame_1 = frames_1.get_pose_frame()
    streams_1 = {"left"  : profiles_1.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                 "right" : profiles_1.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics_1 = {"left"  : streams_1["left"].get_intrinsics(),
                    "right" : streams_1["right"].get_intrinsics()}
    K1_left  = subcode.camera_matrix(intrinsics_1["left"])
    D1_left  = subcode.fisheye_distortion(intrinsics_1["left"])
    K1_right = subcode.camera_matrix(intrinsics_1["right"])
    D1_right = subcode.fisheye_distortion(intrinsics_1["right"])
    (width, height) = (intrinsics_1["left"].width, intrinsics_1["left"].height)
    (R1, T1) = subcode.get_extrinsics(streams_1["left"], streams_1["right"])
    R1_left = np.eye(3)
    R1_right = R1
    P_right[0][3] = T1[0]*stereo_focal_px
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T1[0], 0]])
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K1_left, D1_left, R1_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K1_right, D1_right, R1_right, P_right, stereo_size, m1type)
    undistort_rectify1 = {"left"  : (lm1, lm2),
                          "right" : (rm1, rm2)}
    fisheye_frame1_r = frames_1.get_fisheye_frame(1)
    fisheye_image1_r= np.asanyarray(fisheye_frame1_r.get_data())
    fisheye_frame1_l = frames_1.get_fisheye_frame(2)
    fisheye_image1_l = np.asanyarray(fisheye_frame1_l.get_data())
    center_undistorted1 = {"left" : cv2.remap(src = fisheye_image1_r,
                                    map1 = undistort_rectify1["left"][0],
                                    map2 = undistort_rectify1["left"][1],
                                    interpolation = cv2.INTER_LINEAR),
                           "right": cv2.remap(src = fisheye_image1_l,
                                    map1 = undistort_rectify1["right"][0],
                                    map2 = undistort_rectify1["right"][1],
                                    interpolation = cv2.INTER_LINEAR)}
    disparity1 = stereo.compute(center_undistorted1["left"], center_undistorted1["right"]).astype(np.float32) / 16.0
    disparity1 = disparity1[:,max_disp:]
    ret1,disparity1BW = cv2.threshold(disparity1,0,255,cv2.THRESH_BINARY)
    disparity1BW = cv2.morphologyEx(disparity1BW,cv2.MORPH_OPEN,kernel)
    disparity1MAX = np.max(disparity1[disparity1BW == 255])
    depth1 = cameradistance*stereo_focal_px*16*pix_to_cm/disparity1MAX
    color_image1 = cv2.cvtColor(center_undistorted1["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)
    
    #必要なデータの取得する
    if pose_frame_1:
        data1 = pose_frame_1.get_pose_data()
        print("Camera1Frame #{}".format(pose_frame_1.frame_number))
        print("Camera1Position: {}".format(data1.translation))
        print("Camera1Velocity: {}".format(data1.velocity))
        print("Camera1Acceleration: {}\n".format(data1.acceleration))
        x1 = data1.translation.x
        z1 = data1.translation.y
        y1 = -data1.translation.z
        x1,y1 = subcode.coordinatetransformation(x1,y1,camera1formation[1])
        omg1 = data1.velocity.x
        phi1 = data1.velocity.y
        kpp1 = data1.velocity.z
        array_data1 = [1,pose_frame_1.frame_number,x1+fx1,y1+fy1,z1,omg1,phi1,kpp1,depth1]
        array_data1 = np.array(array_data1)
        array_datas1 = np.vstack((array_datas1,array_data1))
            
    
    
    #ビデオにフレーム数を印字
    cv2.putText(color_image1,str(pose_frame_1.frame_number),(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
    
    #ビデオを保存する
    video1.write(color_image1)
    
    #ビデオを出力する
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense',color_image1 )
    
    i = i+1
    t1 = np.array((x1,y1,z1)) - position1_old
    
    if abs(t1[0]) < t_err and abs(t1[1]) < t_err and abs(t1[2]) < t_err:
        position = position + t1
        id_position = np.insert(position,0,i)
        positions = np.vstack((positions,id_position))
    
            
    position1_old = np.array((x1,y1,z1)) 
    
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        #ウィンドウなどを解放する
        video1.release()
        cv2.destroyAllWindows()
        
        #オドメトリデータを出力する
        dt_now = datetime.datetime.now()
        FILE_OUT= 'out_T265_{0:test.csv}'.format(dt_now)
        np.savetxt(directry + '\\camera1\\camera1'  +FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
        np.savetxt(directry + '\\tracking\\tracking'+FILE_OUT,positions,delimiter=',',fmt='%.3f')
        #ストリーミングを終わる
        pipeline_1.stop()
        break



