from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf
from numba import jit
import time

start=time.perf_counter()

image=0
file=tf.TiffSequence("Perfect Spots r8.00/spot014.tif").asarray()

@jit(nopython=True)
def triangle(centre,half_base,height,magnif):
    x=np.arange((centre-half_base),(centre+half_base),1)
    y=np.arange(0,height,(height/(len(x)/2)))
    y=np.append(y,y[::-1])
    return x,y
@jit(nopython=True)
def res(file,centre,half_base,height,image,magnif):
    tri_x,tri_y=triangle(centre,half_base,height,magnif)
    area=np.zeros(len(file[image,24,:])*magnif,dtype=np.float64)
    for i in range((centre-half_base),(centre+half_base)):
        area[i]=tri_y[i-(centre-half_base)]
    X = np.arange(0, magnif*len(file[image,24,:]), magnif)
    X_new = np.arange(magnif*len(file[image,24,:]))
    Y_new = np.interp(X_new, X, file[image,24,:])
    return np.absolute(Y_new-area),Y_new,area
@jit(nopython=True)
def opt(file,image,magnif):
    output=np.zeros((10*magnif,7*magnif,26000))
    for i in range(20*magnif,30*magnif):
        for j in range(3*magnif,10*magnif):
            for k in range(40000,66000):
                res1,Y_new,area=res(file,i,j,k,image,magnif)
                output[i-20*magnif,j-3*magnif,int((k-40000))]=np.sqrt(np.mean(res1**2))
        print(i)
    return output,(np.where(output==np.min(output)))

magnif=4
#output,min_vals=opt(file,image,magnif)
#print(min_vals)

centre,half_base,height=96,25,65992
adj_c=centre/magnif
print(adj_c)
stop=time.perf_counter()
print(stop-start)

tri_x,tri_y=triangle(centre,half_base,height,magnif)
res,Y_new,area=res(file,centre,half_base,height,image,magnif)

fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(8,4),constrained_layout=True)

ax1.plot(tri_x,tri_y)
ax1.plot(Y_new)
ax1.set_xlim(0,50*magnif)
ax1.set_ylim(0,65000)


x2=np.arange(50*magnif)
ax2.plot(x2,Y_new)
ax2.plot(x2,area)
ax2.bar(x2,res)
ax2.set_ylim(0,65000)
ax2.set_xlim(0,50*magnif)
plt.show()