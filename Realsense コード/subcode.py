#補助コード
import math 
import numpy as np

def coordinatetransformation(a1,b1,θ):
    a = a1*math.cos(math.radians(θ))-b1*math.sin(math.radians(θ))
    b = a1*math.sin(math.radians(θ))+b1*math.cos(math.radians(θ))
    return a,b
    
def cameraformation (l,θ):
    a = l*math.cos(math.radians(θ-90))
    b = -l*math.sin(math.radians(θ-90))
    return a,b

def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)