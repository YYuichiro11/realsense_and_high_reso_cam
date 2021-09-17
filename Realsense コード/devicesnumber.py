import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print (devices[0])
print (devices[1])
print (devices[2])