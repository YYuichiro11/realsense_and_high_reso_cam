import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import math
import subcode

directry = 'C:\graduation invastigation\data'
array_datas1 = np.array((1,0,0))
dt_now = datetime.datetime.now()
FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.avi}'.format(dt_now)
fps = 1
w = 300
h = 300
dot_r = 2
cameradistance = 6.7
pix_to_cm = 0.0008
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1_r = cv2.VideoWriter(directry+'\camera1\camera1_r'+FILE_OUT, fourcc,fps,(w,h),True)
window_size = 5
min_disp = 0
# must be divisible by 16
num_disp = 112 - min_disp
max_disp = min_disp + num_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                               numDisparities = num_disp,
                               blockSize = 16,
                               P1 = 8*3*window_size**2,
                               P2 = 32*3*window_size**2,
                               disp12MaxDiff = 1,
                               uniquenessRatio = 10,
                               speckleWindowSize = 100,
                               speckleRange = 32)
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

# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('2322110136')
config_1.enable_stream(rs.stream.pose)
config_1.enable_stream(rs.stream.fisheye,1)
config_1.enable_stream(rs.stream.fisheye,2)

# Start streaming from both cameras
pipeline_1.start(config_1)

while True:
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
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
        if frames_1:
            center_undistorted1 = {"left" : cv2.remap(src = fisheye_image1_r,
                                          map1 = undistort_rectify1["left"][0],
                                          map2 = undistort_rectify1["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = fisheye_image1_l,
                                          map1 = undistort_rectify1["right"][0],
                                          map2 = undistort_rectify1["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}
            disparity1 = stereo.compute(center_undistorted1["left"], center_undistorted1["right"]).astype(np.float32) / 16.0
            disparity1 = disparity1[:,max_disp:]
            disparity1_m = np.mean(disparity1[149:151,149:151])
            depth1 = cameradistance*stereo_focal_px*16*pix_to_cm/disparity1_m
            color_image1 = cv2.cvtColor(center_undistorted1["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)
            print("Depth: {}".format(depth1))
            array_data1 = [1,pose_frame_1.frame_number,depth1]
            array_data1 = np.array(array_data1)
            array_datas1 = np.vstack((array_datas1,array_data1))
            
        cv2.putText(color_image1,str(pose_frame_1.frame_number),(0,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(color_image1,(150,150),dot_r,(255,0,0),-1,cv2.LINE_AA)
        video1_r.write(color_image1)
        
        #カメラ1の映像を再生
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense',color_image1 )
        
        #qキーでシステムが終了する
        if cv2.waitKey(1) & 0xff == ord('q'):
            video1_r.release()
            cv2.destroyAllWindows()
            #これまでの自己位置姿勢推定結果をcsvファイルで取得
            FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
            np.savetxt(directry+'\camera1\camera1'+FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
            # 全てのカメラの再生を終了
            pipeline_1.stop()
            break

