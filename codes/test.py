# 視差計算を行うコード　https://asukiaaa.blogspot.com/2019/04/ubuntuintel-realsense-t265.html

# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2

# setup opencv stereo
minDisp = 0
numDisp = 64 - minDisp
windowSize = 5
stereo = cv2.StereoSGBM_create(
    minDisparity = minDisp,
    numDisparities = numDisp,
    blockSize = 16,
    P1 = 8*3*windowSize**2,
    P2 = 32*3*windowSize**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()

cfg.enable_stream(rs.stream.pose)
cfg.enable_stream(rs.stream.fisheye, 1)
cfg.enable_stream(rs.stream.fisheye, 2)

# データを格納してみますqqqqqqqqqqqqqqqqqqqqqq
frame = []
position = []
velocity = []
accelerationation = []

# Start streaming with requested config
pipe.start(cfg)
print('press q to quit this program')

try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        
        pose = frames.get_pose_frame()
        # Fetch pose frame
        if pose:
            data = pose.get_pose_data()
            frame.append(format(pose.frame_number))
            # 位置情報の保存
            p = format(data.translation)
            p = p.split()
            x = p[1].split(',')
            y = p[3].split(',')
            position.append([float(x[0]),float(y[0]),float(p[5])])
            # 速度情報の保存
            v = format(data.velocity)
            v = v.split()
            x = v[1].split(',')
            y = v[3].split(',')
            velocity.append([float(x[0]),float(y[0]),float(v[5])])
            # 加速度情報の保存
            a = format(data.acceleration)
            a = a.split()
            x = a[1].split(',')
            y = a[3].split(',')
            accelerationation.append([float(x[0]),float(y[0]),float(a[5])])

        # Get images
        # fisheye 1: left, 2: right
        fisheye_left_frame = frames.get_fisheye_frame(1)
        fisheye_right_frame = frames.get_fisheye_frame(2)
        fisheye_left_image = np.asanyarray(fisheye_left_frame.get_data())
        fisheye_right_image = np.asanyarray(fisheye_right_frame.get_data())

        # Calculate disparity
        width = fisheye_left_frame.get_width()
        height = fisheye_left_frame.get_height()
        x1 = int(width/3 - numDisp / 2)
        x2 = int(width*2/3 + numDisp / 2)
        y1 = int(height/3)
        y2 = int(height*2/3)
        rect_left_image = fisheye_left_image[y1:y2, x1:x2]
        rect_right_image = fisheye_right_image[y1:y2, x1:x2]
        disparity = stereo.compute(rect_left_image, rect_right_image).astype(np.float32)/16
        disparity = (disparity - minDisp) / numDisp

        # Display images
        cv2.rectangle(fisheye_left_image, (x1, y1), (x2, y2), (255,255,255), 5)
        cv2.rectangle(fisheye_right_image, (x1, y1), (x2, y2), (255,255,255), 5)
        cv2.imshow('fisheye target', np.hstack((fisheye_left_image, fisheye_right_image)))
        cv2.imshow('disparity', disparity)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    pipe.stop()