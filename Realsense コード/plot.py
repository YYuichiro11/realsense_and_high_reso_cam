import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

a = np.loadtxt('C:/graduation invastigation/data/camera1/camera1out_T265_20201224180845.csv',delimiter=',')
a = a[:,4:7]

fig = plt.figure(facecolor = 'white')

ax = Axes3D(fig)
x = a[:,0]
y = a[:,1]
z = a[:,2]
ax.plot(x,y,z)

plt.show()


