import pyrealsense2 as rs
import numpy as np
import datetime
import cv2
import math
import subcode

#ディレクトリの準備（dataファイルの中にcamera1~6のファイルを作る)
directry = 'C:\graduation invastigation\data'

#cameraformation(distance,angle)   
camera1formation = (0.19,0)
camera2formation = (0.19,90)
camera3formation = (0.19,270)

#カメラ位置の登録
fx1,fy1 = subcode.cameraformation(camera1formation[0],camera1formation[1])
fx2,fy2 = subcode.cameraformation(camera2formation[0],camera2formation[1])
fx3,fy3 = subcode.cameraformation(camera3formation[0],camera3formation[1])

#下準備
array_datas1 = np.array((1,0,0,0,0,0,0,0,0))
array_datas2 = np.array((2,0,0,0,0,0,0,0,0))
array_datas3 = np.array((3,0,0,0,0,0,0,0,0))

#カメラのパラメータ(T265の規格値)
fps = 10
w = 300
h = 300

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
FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.avi}'.format(dt_now)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1 = cv2.VideoWriter(directry+'\camera1\camera1'+FILE_OUT, fourcc,fps,(w,h),True)
video2 = cv2.VideoWriter(directry+'\camera2\camera2'+FILE_OUT, fourcc,fps,(w,h),True)
video3 = cv2.VideoWriter(directry+'\camera3\camera3'+FILE_OUT, fourcc,fps,(w,h),True)

# 自己位置姿勢データを取得する設定
# カメラ1・・・
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("2322110136")
config_1.enable_stream(rs.stream.pose)
config_1.enable_stream(rs.stream.fisheye,1)
config_1.enable_stream(rs.stream.fisheye,2)

# カメラ2・・・
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("947122110536")
config_2.enable_stream(rs.stream.pose)
config_2.enable_stream(rs.stream.fisheye,1)
config_2.enable_stream(rs.stream.fisheye,2)

#　カメラ3・・・
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device("952322110642")
config_3.enable_stream(rs.stream.pose)
config_3.enable_stream(rs.stream.fisheye,1)
config_3.enable_stream(rs.stream.fisheye,2)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)
pipeline_3.start(config_3)

while True:
    # Camera 1
    # Wait for a coherent pair of frames:pose 
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
    if pose_frame_1:
        data1 = pose_frame_1.get_pose_data()
        print("Camera1Frame #{}".format(pose_frame_1.frame_number))
        print("Camera1Position: {}".format(data1.translation))
        print("Camera1Velocity: {}".format(data1.velocity))
        print("Camera1Acceleration: {}\n".format(data1.acceleration))
        x1 = data1.translation.x
        z1 = data1.translation.y
        y1 = data1.translation.z
        x1,y1 = subcode.coordinatetransformation(x1,y1,camera1formation[1])
        omg1 = data1.velocity.x
        phi1 = data1.velocity.y
        kpp1 = data1.velocity.z
        array_data1 = [1,pose_frame_1.frame_number,x1,y1,z1,omg1,phi1,kpp1,depth1]
        array_data1 = np.array(array_data1)
        array_datas1 = np.vstack((array_datas1,array_data1))
            
    # Camera 2
    # Wait for a coherent pair of frames: pose
    frames_2 = pipeline_2.wait_for_frames()
    profiles_2 = pipeline_2.get_active_profile()
    pose_frame_2 = frames_2.get_pose_frame()
    streams_2 = {"left"  : profiles_2.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                 "right" : profiles_2.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics_2 = {"left"  : streams_2["left"].get_intrinsics(),
                    "right" : streams_2["right"].get_intrinsics()}
    K2_left  = subcode.camera_matrix(intrinsics_2["left"])
    D2_left  = subcode.fisheye_distortion(intrinsics_2["left"])
    K2_right = subcode.camera_matrix(intrinsics_2["right"])
    D2_right = subcode.fisheye_distortion(intrinsics_2["right"])
    (width, height) = (intrinsics_2["left"].width, intrinsics_2["left"].height)
    (R2, T2) = subcode.get_extrinsics(streams_2["left"], streams_2["right"])
    R2_left = np.eye(3)
    R2_right = R2
    P_right[0][3] = T2[0]*stereo_focal_px
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T1[0], 0]])
    (lm3, lm4) = cv2.fisheye.initUndistortRectifyMap(K2_left, D2_left, R2_left, P_left, stereo_size, m1type)
    (rm3, rm4) = cv2.fisheye.initUndistortRectifyMap(K2_right, D2_right, R2_right, P_right, stereo_size, m1type)
    undistort_rectify2 = {"left"  : (lm3, lm4),
                         "right" : (rm3, rm4)}
    fisheye_frame2_r = frames_2.get_fisheye_frame(1)
    fisheye_image2_r = np.asanyarray(fisheye_frame2_r.get_data())
    fisheye_frame2_l = frames_2.get_fisheye_frame(2)
    fisheye_image2_l = np.asanyarray(fisheye_frame2_l.get_data())
    center_undistorted2 = {"left" : cv2.remap(src = fisheye_image1_r,
                                    map1 = undistort_rectify2["left"][0],
                                    map2 = undistort_rectify2["left"][1],
                                    interpolation = cv2.INTER_LINEAR),
                           "right": cv2.remap(src = fisheye_image1_l,
                                    map1 = undistort_rectify2["right"][0],
                                    map2 = undistort_rectify2["right"][1],
                                    interpolation = cv2.INTER_LINEAR)}
    disparity2 = stereo.compute(center_undistorted2["left"], center_undistorted2["right"]).astype(np.float32) / 16.0
    disparity2 = disparity2[:,max_disp:]
    ret2,disparity2BW = cv2.threshold(disparity2,0,255,cv2.THRESH_BINARY)
    disparity2BW = cv2.morphologyEx(disparity2BW,cv2.MORPH_OPEN,kernel)
    disparity2MAX = np.max(disparity2[disparity2BW == 255])
    depth2 = cameradistance*stereo_focal_px*16*pix_to_cm/disparity2MAX
    color_image2 = cv2.cvtColor(center_undistorted2["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)
    if pose_frame_2:
        data2 = pose_frame_2.get_pose_data()
        print("Camera2Frame #{}".format(pose_frame_2.frame_number))
        print("Camera2Position: {}".format(data2.translation))
        print("Camera2Velocity: {}".format(data2.velocity))
        print("Camera2Acceleration: {}\n".format(data2.acceleration))
        x2 = data2.translation.x
        z2 = data2.translation.y
        y2 = data2.translation.z
        x2,y2 = subcode.coordinatetransformation(x2,y2,camera2formation[1])
        omg2 = data2.velocity.x
        phi2 = data2.velocity.y
        kpp2 = data2.velocity.z
        array_data2 = [2,pose_frame_2.frame_number,x2,y2,z2,omg2,phi2,kpp2,depth2]
        array_data2 = np.array(array_data2)
        array_datas2 = np.vstack((array_datas2,array_data2))
        
    # Camera 3
    # Wait for a coherent pair of frames: pose
    frames_3 = pipeline_3.wait_for_frames()
    profiles_3 = pipeline_3.get_active_profile()
    pose_frame_3 = frames_3.get_pose_frame()
    streams_3 = {"left"  : profiles_3.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                 "right" : profiles_3.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics_3 = {"left"  : streams_3["left"].get_intrinsics(),
                    "right" : streams_3["right"].get_intrinsics()}
    K3_left  = subcode.camera_matrix(intrinsics_3["left"])
    D3_left  = subcode.fisheye_distortion(intrinsics_3["left"])
    K3_right = subcode.camera_matrix(intrinsics_3["right"])
    D3_right = subcode.fisheye_distortion(intrinsics_3["right"])
    (width, height) = (intrinsics_3["left"].width, intrinsics_3["left"].height)
    (R3, T3) = subcode.get_extrinsics(streams_3["left"], streams_3["right"])
    R3_left = np.eye(3)
    R3_right = R3
    P_right[0][3] = T3[0]*stereo_focal_px
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T3[0], 0]])
    (lm5, lm6) = cv2.fisheye.initUndistortRectifyMap(K3_left, D3_left, R3_left, P_left, stereo_size, m1type)
    (rm5, rm6) = cv2.fisheye.initUndistortRectifyMap(K3_right, D3_right, R3_right, P_right, stereo_size, m1type)
    undistort_rectify3 = {"left"  : (lm5, lm6),
                         "right" : (rm5, rm6)}
    fisheye_frame3_r = frames_3.get_fisheye_frame(1)
    fisheye_image3_r = np.asanyarray(fisheye_frame3_r.get_data())
    fisheye_frame3_l = frames_3.get_fisheye_frame(2)
    fisheye_image3_l = np.asanyarray(fisheye_frame3_l.get_data())
    center_undistorted3 = {"left" : cv2.remap(src = fisheye_image3_r,
                                    map1 = undistort_rectify3["left"][0],
                                    map2 = undistort_rectify3["left"][1],
                                    interpolation = cv2.INTER_LINEAR),
                           "right": cv2.remap(src = fisheye_image3_l,
                                    map1 = undistort_rectify3["right"][0],
                                    map2 = undistort_rectify3["right"][1],
                                    interpolation = cv2.INTER_LINEAR)}
    disparity3 = stereo.compute(center_undistorted3["left"], center_undistorted3["right"]).astype(np.float32) / 16.0
    disparity3 = disparity3[:,max_disp:]
    ret3,disparity3BW = cv2.threshold(disparity3,0,255,cv2.THRESH_BINARY)
    disparity3BW = cv2.morphologyEx(disparity2BW,cv2.MORPH_OPEN,kernel)
    disparity3MAX = np.max(disparity3[disparity2BW == 255])
    depth3 = cameradistance*stereo_focal_px*16*pix_to_cm/disparity3MAX
    color_image3 = cv2.cvtColor(center_undistorted3["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)
    if pose_frame_3:
        data3 = pose_frame_3.get_pose_data()
        print("Camera3Frame #{}".format(pose_frame_3.frame_number))
        print("Camera3Position: {}".format(data3.translation))
        print("Camera3Velocity: {}".format(data3.velocity))
        print("Camera3Acceleration: {}\n".format(data3.acceleration))
        x3 = data3.translation.x
        z3 = data3.translation.y
        y3 = data3.translation.z
        x3,y3 = subcode.coordinatetransformation(x3,y3,camera3formation[1])
        omg3 = data3.velocity.x
        phi3 = data3.velocity.y
        kpp3 = data3.velocity.z
        array_data3 = [3,pose_frame_3.frame_number,x3,y3,z3,omg3,phi3,kpp3,depth3]
        array_data3 = np.array(array_data3)
        array_datas3 = np.vstack((array_datas3,array_data3))
    
    cv2.putText(color_image1,str(pose_frame_1.frame_number),(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(color_image2,str(pose_frame_2.frame_number),(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(color_image3,str(pose_frame_3.frame_number),(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
    #ビデオを保存する
    video1.write(color_image1)
    video2.write(color_image2) 
    video3.write(color_image3)
    
    #ビデオを出力する
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense',color_image1 )

    if cv2.waitKey(1) & 0xff == ord('q'):
        #ウィンドウなどを解放する
        video1.release()
        video2.release()
        video3.release()
        cv2.destroyAllWindows()
        #オドメトリデータを出力する
        dt_now = datetime.datetime.now()
        FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
        np.savetxt(directry+'\camera1\camera1'+FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera2\camera2'+FILE_OUT,array_datas2,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera3\camera3'+FILE_OUT,array_datas3,delimiter=',',fmt='%.3f')
        # Stop streaming
        pipeline_1.stop()
        pipeline_2.stop()
        break