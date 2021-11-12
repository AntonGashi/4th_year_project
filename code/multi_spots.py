import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit
import pandas as pd

file=tf.TiffSequence("ideal_spots.tif")
file=file.asarray()
file=file.reshape(100,50,50)

ground_truth=pd.DataFrame.to_numpy(pd.read_csv('groundtruth.csv'))
points=np.add(np.full((ground_truth.shape),24.5),ground_truth)

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
loc_max=np.delete((local_max(file)),0,0)
#@jit(nopython=True)
def centriod(file,loc_max,box_size):
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

def display(num_of_boxes,picture):
    start=time.perf_counter()

    box_size=np.zeros(num_of_boxes,dtype=int)
    loc_centroid=np.zeros([num_of_boxes,100,2])
    error=np.zeros([num_of_boxes,100,2])
    avg_diff=np.zeros([num_of_boxes,2])
    for i in range(num_of_boxes):
        box_size[i]=int((2*(i+1))+1)
    for j in range(num_of_boxes):
        loc_centroid[j]=centriod(file,loc_max,box_size[j])
    for k in range(num_of_boxes):
        error[k]=np.absolute(np.subtract(loc_centroid[k],points))
        avg_diff[k,0],avg_diff[k,1]=np.average(error[k,0:,0]),np.average(error[k,0:,1])
    
    fig_one_box=np.argmin(avg_diff[:,0])
    #fig_one_box=int(np.argmin(avg_diff[0]))
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4),constrained_layout=True)
    ax1.title.set_text(('Plot of Predicted Spot For Picture {} of {}'.format(picture,file.shape[0])))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.imshow(file[picture],cmap='binary_r')
    ax1.plot(points[picture,0],points[picture,1],'x',color='r',label='Ground Truth')
    ax1.plot(loc_centroid[fig_one_box,0,0],loc_centroid[fig_one_box,0,1],'x',color='k',label=('Centroid Box Size = {}'.format(box_size[fig_one_box])))
    ax1.legend(shadow=True,fancybox=True)
    ax2.title.set_text('Average Difference in Position')
    ax2.set_xlabel('Box Size')
    ax2.set_ylabel('Differance From Ground Truth (absolute value)')
    ax2.plot(avg_diff[0:,0],label='X Values',linestyle='-')
    ax2.plot(avg_diff[0:,1],label='Y Values',linestyle='--')
    ax2.legend(shadow=True,fancybox=True)
    stop=time.perf_counter()
    print('Total Time Taken',np.round((stop-start),2))
    plt.show()
    pass

display(16,45)