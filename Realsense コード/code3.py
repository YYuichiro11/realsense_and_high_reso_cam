import pyrealsense2 as rs
import numpy as np
import sys
import datetime

# Configure posedata...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device("929122111473")
config_1.enable_stream(rs.stream.pose)

# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("947122110536")
config_2.enable_stream(rs.stream.pose)


# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

try:
    while True:
        # Camera 1
        # Wait for a coherent pair of frames: pose
        frames_1 = pipeline_1.wait_for_frames()
        pose_frame_1 = frames_1.get_pose_frame()
        color_frame_1 = frames_1.get_color_frame()
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
        
        # Camera 2
        # Wait for a coherent pair of frames: pose
        frames_2 = pipeline_2.wait_for_frames()
        pose_frame_2 = frames_2.get_pose_frame()
        time_frame_2 = frames_2.get_timestamp()
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
            
        array_data = np.vstack((array_data1,array_data2))
        dt_now = datetime.datetime.now()
        FILE_OUT= 'C:\graduation invastigation\camera\camera\out_T265_{0:%Y%m%d%H%M%S.csv}'.format(dt_now)
        np.savetxt(FILE_OUT,array_data,delimiter=',',fmt='%.3f')

#system exit ctrl+c
except KeyboardInterrupt:
    sys.exit()

#fixing…
finally:
    pipeline_1.stop()
    pipeline_2.stop()