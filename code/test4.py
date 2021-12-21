import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

file=tf.TiffSequence("AF647_npc.tif").asarray().reshape(10001,256,256)

def sum_(file):
    file_=np.zeros((10001,2,256))
    for i in range(file_.shape[0]):
        file_[i,0]=np.sum(file[i],axis=0)
        file_[i,1]=np.sum(file[i],axis=1)
    return file_
file_=sum_(file)

image=1000

fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(4,4),constrained_layout=True)

ax1.plot(file_[image,0],label='X')
ax1.plot(file_[image,1],label='Y')
ax1.set_xlim(0,250)
ax1.legend()

ax2.imshow(file[image])

plt.show()
