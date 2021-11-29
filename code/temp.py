from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf
from numba import jit

image=0
file=tf.TiffSequence("Perfect Spots r8.00/spot014.tif").asarray()

@jit(nopython=True)
def triangle(centre,half_base,height):
    x=np.arange((centre-half_base),(centre+half_base),1)
    y=np.linspace(0,height,half_base)
    y=np.append(y,y[::-1])
    return x,y
@jit(nopython=True)
def res(file,centre,half_base,height,image):
    tri_x,tri_y=triangle(centre,half_base,height)
    area=np.zeros_like(file[image,24,:],dtype=np.float64)
    for i in range((centre-half_base),(centre+half_base)):
        area[i]=tri_y[i-(centre-half_base)]
    return np.absolute(file[image,24,:]-area)
@jit(nopython=True)
def opt(file,image):
    output=np.zeros((10,8,26000))
    for i in range(20,30):
        for j in range(3,11):
            for k in range(40000,66000):
                output[i-20,j-3,k-40000]=np.sqrt(np.mean(res(file,i,j,k,image)**2))
        print(i)
    return output,(np.where(output==np.min(output)))

#output,min_vals=opt(file,image)
#print(min_vals)

centre,half_base,height=24,7,61748
tri_x,tri_y=triangle(centre,half_base,height)
res=res(file,centre,half_base,height,image)

'''
fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(8,4),constrained_layout=True)

ax1.plot(tri_x,tri_y)
ax1.plot(file[image,24,:])
ax1.set_xlim(0,50)
ax1.set_ylim(0,65000)

x2=np.arange(0,50,1)
ax2.bar(x2,file[image,24,:])
ax2.bar(x2,res)
ax2.set_ylim(0,65000)
ax2.set_xlim(0,50)
plt.show()
'''
