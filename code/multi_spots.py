import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit

file=tf.TiffSequence("ideal_spots.tif")
file=file.asarray()
file=file.reshape(100,50,50)

start1=time.perf_counter()
def centroid(file):
    loc=np.empty(shape=(1,2))
    for i in range(file.shape[0]):
        sum_int=0
        x_pos=0
        y_pos=0
        for j in range(file.shape[1]):
            for k in range(file.shape[2]):
                sum_int=sum_int+file[i,j,k]
                x_pos=x_pos+(file[i,j,k]*j)
                y_pos=y_pos+(file[i,j,k]*k)
        x_=(x_pos/sum_int)
        y_=(y_pos/sum_int)
        loc=np.vstack((loc,[x_,y_]))
    return loc
stop1=time.perf_counter()
start2=time.perf_counter()

def local_max(file):
    results=np.empty(shape=(1,2))
    iter_=2
    loop=int(file.shape[1]/iter_)
    for i in range(file.shape[0]):
        filemax_thresh=np.amax(file[i])-1
        for j in range(loop-1):
            for k in range(loop-1):
                max_=np.amax(file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
                if max_>filemax_thresh:
                    x=np.flip(np.argwhere(file[i]==max_))
                    results=np.vstack((results,x))
                else:
                    continue
    return results
#@jit(nopython=True)
def centriod2(file,loc_max,box_size):
    loc=np.zeros_like(loc_max)
    index=round((box_size-1)/2)
    for i in range(loc_max.shape[0]):
        sum_int=0
        x_pos=0
        y_pos=0
        x,y=int(loc_max[i,0]),int(loc_max[i,1])
        for j in range(box_size-1):
            for k in range(box_size-1):
                sum_int+=file[i,(x-index)+j,(y-index)+k]
                x_pos+=file[i,(x-index)+j,(y-index)+k]*(j+(x-index))
                y_pos+=file[i,(x-index)+j,(y-index)+k]*(k+(y-index))
        x_=(x_pos/sum_int)
        y_=(y_pos/sum_int)
        loc[i]=[x_,y_]
    return loc
stop2=time.perf_counter()


picture=0
box_size=7
loc_max=np.delete((local_max(file)),0,0)
loc_centroid=np.delete(centroid(file),0,0)
loc_centroid2=np.delete(centriod2(file,loc_max,box_size),0,0)
finish1=stop1-start1
finish2=stop2-start2
print('local max(blue): ',np.round(loc_max[picture],2),'\ncentriod with guess(black) and box size of ', box_size, ':',np.round(loc_centroid2[picture],2), ' and took ',finish2, 's', '\ncentroid of whole image(red):',np.round(loc_centroid[picture],2),' and took ',finish1,'s')
plt.imshow(file[picture],cmap='binary_r')
plt.plot(loc_centroid[picture,0],loc_centroid[picture,1],'x',color='r')
plt.plot(loc_centroid2[picture,0],loc_centroid2[picture,1],'x',color='k')
plt.plot(loc_max[picture,0],loc_max[picture,1],'x',color='b')
plt.colorbar()
plt.show()