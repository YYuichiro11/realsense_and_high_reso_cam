import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

a = np.loadtxt('C:/graduation invastigation/data/camera1/camera1out_T265_20210120181156.csv',delimiter=',')
a = a[:,4:7]
b = np.loadtxt('C:/graduation invastigation/data/camera2/camera2out_T265_20210120181156.csv',delimiter=',')
b = b[:,4:7]
c = np.loadtxt('C:/graduation invastigation/data/camera3/camera3out_T265_20210120181156.csv',delimiter=',')
c = c[:,4:7]
d = np.loadtxt('C:/graduation invastigation/data/tracking/trackingout_T265_20210120181156.csv',delimiter=',')
d = d[:,1:4]

x1 = a[:,0]
y1 = a[:,1]
x2 = b[:,0]
y2 = b[:,1]
x3 = c[:,0]
y3 = c[:,1]
x4 = d[:,0]
y4 = d[:,1]

rec1 = patches.Rectangle((0,0),width=6,height=18,ec='r',fill=False,label='Actual route')
rec2 = patches.Rectangle((0,0),width=6,height=18,ec='r',fill=False)
rec3 = patches.Rectangle((0,0),width=6,height=18,ec='r',fill=False,label='Actual route')
rec4 = patches.Rectangle((0,0),width=6,height=18,ec='r',fill=False)

fig1 = plt.figure(facecolor="w")
fig2 = plt.figure(facecolor="w")

ax1 = fig1.add_subplot(121, aspect="equal")
ax2 = fig1.add_subplot(122, aspect="equal")
ax3 = fig2.add_subplot(121, aspect="equal")
ax4 = fig2.add_subplot(122, aspect="equal")

ax1.scatter(x1, y1,label='Estimated data',c='k',s=2)
ax1.add_patch(rec1)
ax2.scatter(x2, y2,c='k',s=2)
ax2.add_patch(rec2)
ax3.scatter(x3, y3,label='Estimated data',c='k',s=2)
ax3.add_patch(rec3)
ax4.scatter(x4, y4,c='k',s=2)
ax4.add_patch(rec4)

ax1.set_title('Upward camera')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(bbox_to_anchor=(5,0.5))

ax2.set_title('Front camera')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

ax3.set_title('Leftward camera')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend(bbox_to_anchor=(5,0.5))

ax4.set_title('Rectified data')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

fig1.show()
fig2.show()