# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:57:07 2021

@author: y

コマンドラインから camera remote SDKを起動します
"""

import subprocess


# a = os.listdir()
# print(a)

# result = subprocess.run(['start', 'dir', '/w'], shell=True)
# print(result)

path = r"C:\Users\jamis\CrSDK_v1.04.00_20210511a_Win64\build\Debug\RemoteCli.exe"

result = subprocess.run([path])
print(result)

# command = ["python", r"C:\Users\jamis\Documents\GitHub\realsense_and_high_reso_cam\codes\sony\send1.py"]
 
# subprocess.Popen(command)
    

# result = subprocess.run(['remoteCli.exe'])
# print(result)

# result = subprocess.run(['cd', '../'], shell=True)
# print(result)
# result = subprocess.run(['cd', '../'], shell=True)
# print(result)
# result = subprocess.run(['cd', '../'], shell=True)
# print(result)
# result = subprocess.run(['cd', '../'], shell=True)
# print(result)