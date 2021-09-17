# 視差計算を行うコード　https://asukiaaa.blogspot.com/2019/04/ubuntuintel-realsense-t265.html
# 位置情報と姿勢を読み込みます

# First import the library
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math as m
import time
import pathlib
import csv

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

# データを格納してみます
rs_outs = []
xs = []
ys = []
zs = []

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
            
            # 位置データ
            x = data.translation.x
            y = data.translation.y
            z = data.translation.z
            xs.append(x)
            ys.append(y)
            zs.append(z)
            
            # 姿勢データ？
            omg = data.velocity.x
            phi = data.velocity.y
            kpp = data.velocity.z
            
            # 回転行列の計算を行います
            R = np.array([[m.cos(phi)*m.cos(kpp), m.cos(omg)*m.sin(kpp)+m.sin(omg)*m.sin(phi)*m.cos(kpp), m.sin(omg)*m.sin(kpp)-m.cos(omg)*m.sin(phi)*m.cos(kpp)],
                          [-m.cos(phi)*m.sin(kpp), m.cos(omg)*m.cos(kpp)-m.sin(omg)*m.sin(phi)*m.sin(kpp),m.sin(omg)*m.cos(kpp)+m.cos(omg)*m.sin(phi)*m.sin(kpp)],
                          [m.sin(phi), -m.sin(omg)*m.cos(phi), m.cos(omg)*m.cos(phi)]])
            
            w = data.rotation.w
            x = -data.rotation.z
            y = data.rotation.x
            z = -data.rotation.y
            
            pitch =  -m.asin(2.0 * (x*z - w*y)) * 180.0 / m.pi;
            roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / m.pi;
            yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / m.pi;
            
            # 回転行列の計算を行います
            R = np.array([[m.cos(m.radians(roll))*m.cos(m.radians(yaw))-m.sin(m.radians(roll))*m.sin(m.radians(pitch))*m.sin(m.radians(yaw)), -m.cos(m.radians(roll))*m.sin(m.radians(yaw))-m.sin(m.radians(roll))*m.sin(m.radians(pitch))*m.cos(m.radians(yaw)), m.cos(m.radians(pitch))*m.sin(m.radians(roll))],
                          [m.sin(m.radians(roll))*m.cos(m.radians(yaw))+m.cos(m.radians(roll))*m.sin(m.radians(pitch))*m.sin(m.radians(yaw)), -m.sin(m.radians(roll))*m.sin(m.radians(yaw))+m.cos(m.radians(roll))*m.sin(m.radians(pitch))*m.cos(m.radians(yaw)), -m.cos(m.radians(pitch))*m.cos(m.radians(roll))],
                          [m.cos(m.radians(pitch))*m.sin(m.radians(yaw)), m.cos(m.radians(pitch))*m.cos(m.radians(yaw)), m.sin(m.radians(pitch))]])
            
            # 時刻を取得します
            t = float(time.time())
            
            # 全情報を一行に格納します
            rs_out = [x,y,z,omg,phi,kpp,roll,pitch,yaw,t]
            rs_outs.append(rs_out)
        
        
        # ここから，現在のフレームの可視化と視差の計算を行います
        # 処理が重くなるので一旦無視します
        
        # Get images
        # fisheye 1: left, 2: right
        fisheye_left_frame = frames.get_fisheye_frame(1)
        fisheye_right_frame = frames.get_fisheye_frame(2)
        fisheye_left_image = np.asanyarray(fisheye_left_frame.get_data())
        fisheye_right_image = np.asanyarray(fisheye_right_frame.get_data())

        # # Calculate disparity
        # width = fisheye_left_frame.get_width()
        # height = fisheye_left_frame.get_height()
        # x1 = int(width/3 - numDisp / 2)
        # x2 = int(width*2/3 + numDisp / 2)
        # y1 = int(height/3)
        # y2 = int(height*2/3)
        # rect_left_image = fisheye_left_image[y1:y2, x1:x2]
        # rect_right_image = fisheye_right_image[y1:y2, x1:x2]
        # disparity = stereo.compute(rect_left_image, rect_right_image).astype(np.float32)/16
        # disparity = (disparity - minDisp) / numDisp

        # Display images
        # cv2.rectangle(fisheye_left_image, (x1, y1), (x2, y2), (255,255,255), 5)
        # cv2.rectangle(fisheye_right_image, (x1, y1), (x2, y2), (255,255,255), 5)
        cv2.imshow('fisheye target', np.hstack((fisheye_left_image, fisheye_right_image)))
        # cv2.imshow('disparity', disparity)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            
            # csvファイルへ出力
            p_new = pathlib.Path('/Users/yuichiro/Research/realsense/realsense_and_high_reso_cam/results/rs_out.csv')
            with p_new.open(mode='w') as f:
                f.write('')
                
            header = ['x','y','z','omg','phi','kpp','Roll','Pitch','Yaw','UNIXtime']
            with open('/Users/yuichiro/Research/realsense/realsense_and_high_reso_cam/results/rs_out.csv', 'w', encoding='cp932') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rs_outs)
                
            break;
            


finally:
    pipe.stop()
    cv2.destroyAllWindows()

# 位置の可視化
# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

# 曲線を描画
ax.scatter(xs, ys, zs, s = 40, c = "blue")

plt.show()


