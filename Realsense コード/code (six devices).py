import pyrealsense2 as rs
import numpy as np
import datetime
import cv2
import math

#ディレクトリの準備（dataファイルの中にcamera1~6のファイルを作る)
directry = 'C:\graduation invastigation\data'

#cameraformation(distance,angle)   
camera1formation = (1,0)
camera2formation = (1,90)
camera3formation = (1,180)
camera4formation = (1,270)
camera5formation = (1,90)
camera6formation = (1,270)

def coordinatetransformation(x,y,θ):
    x = x*math.cos(math.radians(θ))-y*math.sin(math.radians(θ))
    y = x*math.sin(math.radians(θ))+y*math.cos(math.radians(θ))
    return np.array(x,y)
    
def cameraformation (l,θ):
    x = l*math.cos(math.radians(θ))
    y = l*math.sin(math.radians(θ))
    return np.array(x,y)

#カメラ位置の登録
fx1,fy1 = cameraformation(camera1formation)
fx2,fy2 = cameraformation(camera2formation)
fx3,fy3 = cameraformation(camera3formation)
fx4,fy4 = cameraformation(camera4formation)
fx5,fy5 = cameraformation(camera5formation)
fx6,fx6 = cameraformation(camera6formation)

#下準備
array_datas1 = np.array((1,0,0,0,0,0,0,0))
array_datas2 = np.array((2,0,0,0,0,0,0,0))
array_datas3 = np.array((3,0,0,0,0,0,0,0))
array_datas4 = np.array((4,0,0,0,0,0,0,0))
array_datas5 = np.array((5,0,0,0,0,0,0,0))
array_datas6 = np.array((6,0,0,0,0,0,0,0))

#カメラのパラメータ(T265の規格値)
fps = 30
w = 848
h = 800

#ファイル名を出力
dt_now = datetime.datetime.now()
FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.avi}'.format(dt_now)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video1_r = cv2.VideoWriter(directry+'camera1\camera1_r'+FILE_OUT, fourcc,fps,(w,h),False)
video1_l = cv2.VideoWriter(directry+'camera1\camera1_l'+FILE_OUT, fourcc,fps,(w,h),False)
video2_r = cv2.VideoWriter(directry+'camera2\camera2_r'+FILE_OUT, fourcc,fps,(w,h),False)
video2_l = cv2.VideoWriter(directry+'camera2\camera2_l'+FILE_OUT, fourcc,fps,(w,h),False)
video3_r = cv2.VideoWriter(directry+'camera3\camera3_r'+FILE_OUT, fourcc,fps,(w,h),False)
video3_l = cv2.VideoWriter(directry+'camera3\camera3_l'+FILE_OUT, fourcc,fps,(w,h),False)
video4_r = cv2.VideoWriter(directry+'camera4\camera4_r'+FILE_OUT, fourcc,fps,(w,h),False)
video4_l = cv2.VideoWriter(directry+'camera4\camera4_l'+FILE_OUT, fourcc,fps,(w,h),False)
video5_r = cv2.VideoWriter(directry+'camera5\camera5_r'+FILE_OUT, fourcc,fps,(w,h),False)
video5_l = cv2.VideoWriter(directry+'camera5\camera5_l'+FILE_OUT, fourcc,fps,(w,h),False)
video6_r = cv2.VideoWriter(directry+'camera6\camera6_r'+FILE_OUT, fourcc,fps,(w,h),False)
video6_l = cv2.VideoWriter(directry+'camera6\camera6_l'+FILE_OUT, fourcc,fps,(w,h),False)

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

# ...from Camera 3
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device()
config_3.enable_stream(rs.stream.pose)
config_3.enable_stream(rs.stream.fisheye,1)
config_3.enable_stream(rs.stream.fisheye,2)

# ...from Camera 4
pipeline_4= rs.pipeline()
config_4 = rs.config()
config_4.enable_device()
config_4.enable_stream(rs.stream.pose)
config_4.enable_stream(rs.stream.fisheye,1)
config_4.enable_stream(rs.stream.fisheye,2)

# ...from Camera 5
pipeline_5 = rs.pipeline()
config_5 = rs.config()
config_5.enable_device()
config_5.enable_stream(rs.stream.pose)
config_5.enable_stream(rs.stream.fisheye,1)
config_5.enable_stream(rs.stream.fisheye,2)

# ...from Camera 6
pipeline_6 = rs.pipeline()
config_6 = rs.config()
config_6.enable_device()
config_6.enable_stream(rs.stream.pose)
config_6.enable_stream(rs.stream.fisheye,1)
config_6.enable_stream(rs.stream.fisheye,2)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)
pipeline_3.start(config_3)
pipeline_4.start(config_4)
pipeline_5.start(config_5)
pipeline_6.start(config_6)

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
        z1 = data1.translation.y
        y1 = data1.translation.z
        x1,y1 = coordinatetransformation(x1,y1,camera1formation[1])
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
        z2 = data2.translation.y
        y2 = data2.translation.z
        x2,y2 = coordinatetransformation(x2,y2,camera2formation[1])
        omg2 = data2.velocity.x
        phi2 = data2.velocity.y
        kpp2 = data2.velocity.z
        array_data2 = [2,pose_frame_2.frame_number,x2,y2,z2,omg2,phi2,kpp2]
        array_data2 = np.array(array_data2)
        array_datas2 = np.vstack((array_datas2,array_data2))
        
    # Camera 3
    # Wait for a coherent pair of frames: pose
    frames_3 = pipeline_3.wait_for_frames()
    pose_frame_3 = frames_3.get_pose_frame()
    fisheye_frame3_r = frames_3.get_fisheye_frame(1)
    fisheye_image3_r = np.asanyarray(fisheye_frame3_r.get_data())
    fisheye_frame3_l = frames_3.get_fisheye_frame(2)
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
        x3,y3 = coordinatetransformation(x3,y3,camera3formation[1])
        omg3 = data3.velocity.x
        phi3 = data3.velocity.y
        kpp3 = data3.velocity.z
        array_data3 = [3,pose_frame_3.frame_number,x3,y3,z3,omg3,phi3,kpp3]
        array_data3 = np.array(array_data3)
        array_datas3 = np.vstack((array_datas3,array_data3))
        
    # Camera 4
    # Wait for a coherent pair of frames: pose
    frames_4 = pipeline_4.wait_for_frames()
    pose_frame_4 = frames_4.get_pose_frame()
    fisheye_frame4_r = frames_4.get_fisheye_frame(1)
    fisheye_image4_r = np.asanyarray(fisheye_frame4_r.get_data())
    fisheye_frame4_l = frames_4.get_fisheye_frame(2)
    fisheye_image4_l = np.asanyarray(fisheye_frame4_l.get_data())
    if pose_frame_4:
        data4 = pose_frame_4.get_pose_data()
        print("Camera4Frame #{}".format(pose_frame_4.frame_number))
        print("Camera4Position: {}".format(data4.translation))
        print("Camera4Velocity: {}".format(data4.velocity))
        print("Camera4Acceleration: {}\n".format(data4.acceleration))
        x4 = data4.translation.x
        z4 = data4.translation.y
        y4 = data4.translation.z
        x4,y4 = coordinatetransformation(x4,y4,camera4formation[1])
        omg4 = data4.velocity.x
        phi4 = data4.velocity.y
        kpp4 = data4.velocity.z
        array_data4 = [4,pose_frame_4.frame_number,x4,y4,z4,omg4,phi4,kpp4]
        array_data4 = np.array(array_data4)
        array_datas4 = np.vstack((array_datas4,array_data4))
        
    # Camera 5
    # Wait for a coherent pair of frames: pose
    frames_5 = pipeline_5.wait_for_frames()
    pose_frame_5 = frames_5.get_pose_frame()
    fisheye_frame5_r = frames_5.get_fisheye_frame(1)
    fisheye_image5_r = np.asanyarray(fisheye_frame5_r.get_data())
    fisheye_frame5_l = frames_5.get_fisheye_frame(2)
    fisheye_image5_l = np.asanyarray(fisheye_frame5_l.get_data())
    if pose_frame_5:
        data5 = pose_frame_5.get_pose_data()
        print("Camera5Frame #{}".format(pose_frame_5.frame_number))
        print("Camera5Position: {}".format(data5.translation))
        print("Camera5Velocity: {}".format(data5.velocity))
        print("Camera5Acceleration: {}\n".format(data5.acceleration))
        x5 = data5.translation.x
        z5 = data4.translation.y
        y5 = data5.translation.z
        x5,y5 = coordinatetransformation(x5,y5,camera5formation[1])
        omg5 = data5.velocity.x
        phi5 = data5.velocity.z
        kpp5 = data5.velocity.y
        array_data5 = [5,pose_frame_5.frame_number,x5,y5,z5,omg5,phi5,kpp5]
        array_data5 = np.array(array_data5)
        array_datas5 = np.vstack((array_datas5,array_data5))
        
    # Camera 6
    # Wait for a coherent pair of frames: pose
    frames_6 = pipeline_6.wait_for_frames()
    pose_frame_6 = frames_6.get_pose_frame()
    fisheye_frame6_r = frames_6.get_fisheye_frame(1)
    fisheye_image6_r = np.asanyarray(fisheye_frame6_r.get_data())
    fisheye_frame6_l = frames_6.get_fisheye_frame(2)
    fisheye_image6_l = np.asanyarray(fisheye_frame6_l.get_data())
    if pose_frame_6:
        data6 = pose_frame_6.get_pose_data()
        print("Camera6Frame #{}".format(pose_frame_6.frame_number))
        print("Camera6Position: {}".format(data6.translation))
        print("Camera6Velocity: {}".format(data6.velocity))
        print("Camera6Acceleration: {}\n".format(data6.acceleration))
        x6 = data6.translation.x
        z6 = data6.translation.y
        y6 = data6.translation.z
        x6,y6 = coordinatetransformation(x6,y6,camera6formation[1])
        omg6 = data6.velocity.x
        phi6 = data6.velocity.z
        kpp6 = data6.velocity.y
        array_data6 = [6,pose_frame_6.frame_number,x6,y6,z6,omg6,phi6,kpp6]
        array_data6 = np.array(array_data6)
        array_datas6 = np.vstack((array_datas6,array_data6))
    
    #ビデオを保存する
    video1_r.write(fisheye_image1_r)
    video1_l.write(fisheye_image1_l) 
    video2_r.write(fisheye_image2_r)
    video2_l.write(fisheye_image2_l)
    video3_r.write(fisheye_image3_r)
    video3_l.write(fisheye_image3_l) 
    video4_r.write(fisheye_image4_r)
    video4_l.write(fisheye_image4_l)
    video5_r.write(fisheye_image5_r)
    video5_l.write(fisheye_image5_l) 
    video6_r.write(fisheye_image6_r)
    video6_l.write(fisheye_image6_l)
    
    #ビデオを出力する
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense',fisheye_image1_r )

    if cv2.waitKey(1) & 0xff == ord('q'):
        #ウィンドウなどを解放する
        video1_r.release()
        video1_l.release()
        video2_r.release()
        video2_l.release()
        video3_r.release()
        video3_l.release()
        video4_r.release()
        video4_l.release()
        video5_r.release()
        video5_l.release()
        video6_r.release()
        video6_l.release()
        cv2.destroyAllWindows()
        #オドメトリデータを出力する
        dt_now = datetime.datetime.now()
        FILE_OUT= 'out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
        np.savetxt(directry+'\camera1\camera1'+FILE_OUT,array_datas1,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera2\camera2'+FILE_OUT,array_datas2,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera3\camera3'+FILE_OUT,array_datas3,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera4\camera4'+FILE_OUT,array_datas4,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera5\camera5'+FILE_OUT,array_datas5,delimiter=',',fmt='%.3f')
        np.savetxt(directry+'\camera6\camera6'+FILE_OUT,array_datas6,delimiter=',',fmt='%.3f')
        # Stop streaming
        pipeline_1.stop()
        pipeline_2.stop()
        break