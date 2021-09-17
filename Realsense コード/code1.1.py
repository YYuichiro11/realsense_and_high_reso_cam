import pyrealsense2 as rs
import numpy as np
import datetime
import cv2

array_datas1 = np.array((1,0,0,0,0,0,0,0))
array_datas2 = np.array((2,0,0,0,0,0,0,0))
dt_now = datetime.datetime.now()
FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.avi}'.format(dt_now)
fps = 30
w = 848
h = 800
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1_r = cv2.VideoWriter('C:\graduation invastigation\data\camera1\camera1_r'+FILE_OUT, fourcc,fps,(w,h),False)
video1_l = cv2.VideoWriter('C:\graduation invastigation\data\camera1\camera1_l'+FILE_OUT, fourcc,fps,(w,h),False)
video2_r = cv2.VideoWriter('C:\graduation invastigation\data\camera2\camera2_r'+FILE_OUT, fourcc,fps,(w,h),False)
video2_l = cv2.VideoWriter('C:\graduation invastigation\data\camera2\camera2_l'+FILE_OUT, fourcc,fps,(w,h),False)
# Configure posedata...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("929122111473")
config_1.enable_stream(rs.stream.pose)
config_1.enable_stream(rs.stream.fisheye,1)
config_1.enable_stream(rs.stream.fisheye,2)


# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("947122110536")
config_2.enable_stream(rs.stream.pose)
config_2.enable_stream(rs.stream.fisheye,1)
config_2.enable_stream(rs.stream.fisheye,2)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

while True:
    # Camera 1
    # Wait for a coherent pair of frames:pose 
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
        y1 = data1.translation.y
        z1 = data1.translation.z
        omg1 = data1.velocity.x
        phi1 = data1.velocity.y
        kpp1 = data1.velocity.z
        array_data1 = [1,pose_frame_1.frame_number,x1,y1,z1,omg1,phi1,kpp1]
        array_data1 = np.array(array_data1)
        array_datas1 = np.vstack((array_datas1,array_data1))
            
    # Camera 2
    # Wait for a coherent pair of frames: pose
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
        y2 = data2.translation.y
        z2 = data2.translation.z
        omg2 = data2.velocity.x
        phi2 = data2.velocity.y
        kpp2 = data2.velocity.z
        array_data2 = [2,pose_frame_2.frame_number,x2,y2,z2,omg2,phi2,kpp2]
        array_data2 = np.array(array_data2)
        array_datas2 = np.vstack((array_datas2,array_data2))
    
    video1_r.write(fisheye_image1_r)
    video1_l.write(fisheye_image1_l) 
    video2_r.write(fisheye_image2_r)
    video2_l.write(fisheye_image2_l)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense',fisheye_image1_r )

    if cv2.waitKey(1) & 0xff == ord('q'):
        video1_r.release()
        video1_l.release()
        video2_r.release()
        video2_l.release()
        cv2.destroyAllWindows()
        dt_now = datetime.datetime.now()
        FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
        np.savetxt('C:\graduation invastigation\data\camera1\camera1'+FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
        np.savetxt('C:\graduation invastigation\data\camera2\camera2'+FILE_OUT,array_datas2,delimiter=',',fmt='%.3f')
        # Stop streaming
        pipeline_1.stop()
        pipeline_2.stop()
        break