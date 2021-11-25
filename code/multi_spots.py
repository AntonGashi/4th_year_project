from os import path
import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit
import pandas as pd
import scipy.optimize as sco

def input(path_to_tiff,path_to_groundtruth):
    file=tf.TiffSequence("{}".format(path_to_tiff)).asarray()
    if len(file.shape)>4:
        print('ERROR')
    elif len(file.shape)==4:
        file=file.reshape(file.shape[1],file.shape[2],file.shape[3])
    ground_truth=pd.DataFrame.to_numpy(pd.read_csv("{}".format(path_to_groundtruth)))
    return file,ground_truth


@jit(nopython=True)
def local_max(file):
    results=np.zeros(shape=(1,2))
    iter_=2
    loop=int(file.shape[1]/iter_)
    for i in range(file.shape[0]):
        filemax_thresh=(np.amax(file[i]))-1
        for j in range(loop-1):
            for k in range(loop-1):
                max_=np.amax(file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
                if max_>filemax_thresh:
                    x=np.flip(np.argwhere(file[i]==max_))
                    results=np.vstack((results,x))
                else:
                    continue
    return results

@jit(nopython=True)
def centriod(file,loc_max,box_size):
    loc=np.zeros_like(loc_max)
    index=round((box_size-1)/2)
    for i in range(loc_max.shape[0]):
        sum_int,x_pos,y_pos=0,0,0
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

    box_size=np.arange(3,num_of_boxes+1,2,dtype=int)
    loc_centroid=np.zeros([len(box_size),loc_max.shape[0],2])
    error=np.zeros([len(box_size),loc_max.shape[0],2])
    avg_diff=np.zeros([len(box_size),2])
    for j in range(len(box_size)):
        loc_centroid[j]=centriod(file,loc_max,box_size[j])
    for k in range(len(box_size)):
        error[k]=np.absolute(np.subtract(loc_centroid[k],points))
        avg_diff[k,0],avg_diff[k,1]=np.average(error[k,0:,0]),np.average(error[k,0:,1])
    fig_one_box=np.argmin(avg_diff[:,0])
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4),constrained_layout=True)
    ax1.title.set_text(('Plot of Predicted Spot For Picture {} of {}'.format(picture+1,file.shape[0])))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.imshow(file[picture],cmap='binary_r')

    ax1.plot(points[picture,0],points[picture,1],'x',color='r',label='Ground Truth')
    ax1.plot(loc_centroid[fig_one_box,0,0],loc_centroid[fig_one_box,0,1],'x',color='k',label=('Centroid Box Size = {}'.format(box_size[fig_one_box])))
    ax1.legend(shadow=True,fancybox=True)
    ax2.title.set_text('Average Difference in Position')
    ax2.set_xlabel('Box Size (in odd increments)')
    ax2.set_ylabel('Differance From Ground Truth (absolute value)')
    ax2.plot(box_size,avg_diff[0:,0],label='X Values',linestyle='-')
    ax2.plot(box_size,avg_diff[0:,1],label='Y Values',linestyle='--')
    ax2.set_ylim(0)
    ax2.set_xlim(box_size[0],box_size[-1])
    ax2.hlines(avg_diff[fig_one_box,0],box_size[0],box_size[-1],linestyle=':',color='k',label='Minimum at {}'.format(np.round((avg_diff[fig_one_box,0]),4)))
    ax2.legend(shadow=True,fancybox=True)
    stop=time.perf_counter()
    print('Total Time Taken {}s'.format(np.round((stop-start),2)))
    plt.show()
    pass

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)
def triangle(x,m,a):
    return m*(x-a)*np.sign((a-x))+m*a+(m*(x/2-a)*np.sign((a-x/2))+m*a)



#enter like: folder="r1.00 r1.41 r2.00 r2.83 r4.00 r5.66 r8.00"
folder="r8.00"
file,ground_truth=input("Perfect Spots {}/Perfect Spots {}.tif".format(folder,folder),"Perfect Spots {}/groundtruth.csv".format(folder))
#file,ground_truth=input("Multiple Spots/AF647_npc_1frame.tif")
loc_max=np.delete((local_max(file)),0,0)
points=np.add(np.full((ground_truth.shape),24.5),ground_truth)

#display(num_of_boxes(max is 49 as picture size is 50x50),picture(max is 99))
display(49,0)