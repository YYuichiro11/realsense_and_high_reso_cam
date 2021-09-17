#必要なモジュールを保存(fromでmathを使うので警告無視)
import pyrealsense2 as rs
import numpy as np
import datetime
import cv2
import math
from subcode import coordinatetransformation
from subcode import cameraformation

#シリアルコード 2322110136 929122111473 947122110536 952322110642
#ディレクトリの準備（dataファイルの中にcamera1~6のファイルを作る)
directry = 'C:\graduation invastigation\data'

#カメラ位置及ぶ角度  
camera1formation = (0.19,90)
camera2formation = (0.19,270)
camera3formation = (0.11,270)

#カメラ原点の調整
fx1,fy1 = cameraformation(camera1formation[0],camera1formation[1])
fx2,fy2 = cameraformation(camera2formation[0],camera2formation[1])
fy3,fz3 = cameraformation(camera3formation[0],camera3formation[1])

#作業前準備(ループ処理の下準備・保存用動画ファイル作り)
array_datas1 = np.array((1,0,0,0,0,0,0,0))
array_datas2 = np.array((2,0,0,0,0,0,0,0))
array_datas3 = np.array((3,0,0,0,0,0,0,0))
dt_now = datetime.datetime.now()
FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.avi}'.format(dt_now)
fps = 30
w = 848
h = 800
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1_r = cv2.VideoWriter(directry+'\camera1\camera1_r'+FILE_OUT, fourcc,fps,(w,h),False)
video2_r = cv2.VideoWriter(directry+'\camera2\camera2_r'+FILE_OUT, fourcc,fps,(w,h),False)
video3_r = cv2.VideoWriter(directry+'\camera3\camera3_r'+FILE_OUT, fourcc,fps,(w,h),False)

# 自己位置姿勢データを取得する設定
# カメラ1・・・
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("947122110536")
config_1.enable_stream(rs.stream.pose)
config_1.enable_stream(rs.stream.fisheye,1)
config_1.enable_stream(rs.stream.fisheye,2)

# カメラ2・・・
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("952322110642")
config_2.enable_stream(rs.stream.pose)
config_2.enable_stream(rs.stream.fisheye,1)
config_2.enable_stream(rs.stream.fisheye,2)

# カメラ3・・・
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device("2322110136")
config_3.enable_stream(rs.stream.pose)
config_3.enable_stream(rs.stream.fisheye,1)
config_3.enable_stream(rs.stream.fisheye,2)

#全てのカメラを起動する
pipeline_1.start(config_1)
pipeline_2.start(config_2)
pipeline_3.start(config_3)

while True:
    # カメラ1
    # フレームの読み込み:自己位置姿勢データ取得
    frames_1 = pipeline_1.wait_for_frames()
    pose_frame_1 = frames_1.get_pose_frame()
    fisheye_frame1_r = frames_1.get_fisheye_frame(1)
    fisheye_image1_r= np.asanyarray(fisheye_frame1_r.get_data())
    fisheye_frame1_l = frames_1.get_fisheye_frame(2)
    fisheye_image1_l = np.asanyarray(fisheye_frame1_l.get_data())
    if pose_frame_1:
        data1 = pose_frame_1.get_pose_data()
        print("Camera1Frame #{}".format(pose_frame_1.frame_number))
        print("Camera1Position: {}".format(data1.translation))
        print("Camera1Velocity: {}".format(data1.velocity))
        print("Camera1Acceleration: {}\n".format(data1.acceleration))
        x1 = data1.translation.x
        z1 = data1.translation.y
        y1 = data1.translation.z
        x1,y1 = coordinatetransformation(x1, y1, camera1formation[1])
        omg1 = data1.velocity.x
        phi1 = data1.velocity.y
        kpp1 = data1.velocity.z
        array_data1 = [1,pose_frame_1.frame_number,x1+fx1,y1+fy1,z1,omg1,phi1,kpp1]
        array_data1 = np.array(array_data1)
        array_datas1 = np.vstack((array_datas1,array_data1))
            
    # カメラ2
    # フレームの読み込み:自己位置姿勢データ取得
    frames_2 = pipeline_2.wait_for_frames()
    pose_frame_2 = frames_2.get_pose_frame()
    fisheye_frame2_r = frames_2.get_fisheye_frame(1)
    fisheye_image2_r = np.asanyarray(fisheye_frame2_r.get_data())
    fisheye_frame2_l = frames_2.get_fisheye_frame(2)
    fisheye_image2_l = np.asanyarray(fisheye_frame2_l.get_data())
    if pose_frame_2:
        data2 = pose_frame_2.get_pose_data()
        print("Camera2Frame #{}".format(pose_frame_2.frame_number))
        print("Camera2Position: {}".format(data2.translation))
        print("Camera2Velocity: {}".format(data2.velocity))
        print("Camera2Acceleration: {}\n".format(data2.acceleration))
        x2 = data2.translation.x
        z2 = data2.translation.y
        y2 = data2.translation.z
        x2,y2 = coordinatetransformation(x2, y2, camera2formation[1])
        omg2 = data2.velocity.x
        phi2 = data2.velocity.y
        kpp2 = data2.velocity.z
        array_data2 = [2,pose_frame_2.frame_number,x2+fx2,y2+fy2,z2,omg2,phi2,kpp2]
        array_data2 = np.array(array_data2)
        array_datas2 = np.vstack((array_datas2,array_data2))
        
    frames_3 = pipeline_3.wait_for_frames()
    pose_frame_3 = frames_3.get_pose_frame()
    fisheye_frame3_r = frames_2.get_fisheye_frame(1)
    fisheye_image3_r = np.asanyarray(fisheye_frame3_r.get_data())
    fisheye_frame3_l = frames_2.get_fisheye_frame(2)
    fisheye_image3_l = np.asanyarray(fisheye_frame3_l.get_data())
    if pose_frame_3:
        data3 = pose_frame_3.get_pose_data()
        print("Camera3Frame #{}".format(pose_frame_3.frame_number))
        print("Camera3Position: {}".format(data3.translation))
        print("Camera3Velocity: {}".format(data3.velocity))
        print("Camera3Acceleration: {}\n".format(data3.acceleration))
        x3 = data3.translation.x
        z3 = data3.translation.y
        y3 = data3.translation.z
        y3,z3 = coordinatetransformation(y3, z3, camera3formation[1])
        omg3 = data2.velocity.x
        phi3 = data2.velocity.y
        kpp3 = data2.velocity.z
        array_data3 = [3,pose_frame_3.frame_number,x3,y3+fy3,z3+fz3,omg3,phi3,kpp3]
        array_data3 = np.array(array_data3)
        array_datas3 = np.vstack((array_datas3,array_data3))
    
    #複数カメラの静止画を保存
    cv2.putText(fisheye_image1_r,pose_frame_1.frame_number,(0,0),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(fisheye_image2_r,pose_frame_2.frame_number,(0,0),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(fisheye_image3_r,pose_frame_3.frame_number,(0,0),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA) 
    video1_r.write(fisheye_image1_r)
    video2_r.write(fisheye_image2_r)
    video3_r.write(fisheye_image3_r)
    
    #カメラ1の映像を再生
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense',fisheye_image1_r )
    
    #qキーでシステムが終了する
    if cv2.waitKey(1) & 0xff == ord('q'):
        video1_r.release()
        video2_r.release()
        video3_r.release()
        cv2.destroyAllWindows()
        #これまでの自己位置姿勢推定結果をcsvファイルで取得
        dt_now = datetime.datetime.now()
        FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
        np.savetxt(directry+'\camera1\camera1'+FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera2\camera2'+FILE_OUT,array_datas2,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera3\camera3'+FILE_OUT,array_datas2,delimiter=',',fmt='%.3f')
        # 全てのカメラの再生を終了
        pipeline_1.stop()
        pipeline_2.stop()
        pipeline_3.stop()
        break