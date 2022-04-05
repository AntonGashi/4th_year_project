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


#@jit(nopython=True)
def local_max(file):
    results=np.zeros(shape=(1,2))
    iter_=2
    loop=int(file.shape[1]/iter_)
    for i in range(file.shape[0]):

        filemax_thresh=(np.amax(file[i]))-0.5
        for j in range(loop-1):
            for k in range(loop-1):
                max_=np.amax(file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
                if max_>filemax_thresh:
                    x=np.flip(np.argwhere(file[i]==max_))
                    results=np.vstack((results,x))
                else:
                    continue
        if len(results)>=101:
            break ##temp fix 
    return results

#@jit(nopython=True,parallel=True)
def centriod(file,loc_max,box_size):
    loc=np.zeros_like(loc_max)
    index=int((box_size-1)/2)
    for i in range(len(file)-1):
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


def display(num_of_boxes,picture,file_):
    start=time.perf_counter()

    groundtruth_r8=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.00/groundtruth.csv"))+24.5
    box_size=np.arange(3,num_of_boxes+1,2,dtype=int)
    loc_centroid=np.zeros([len(box_size),loc_max.shape[0],2])
    error=np.zeros([len(box_size),loc_max.shape[0],2])
    avg_diff=np.zeros([len(box_size),2])
    abs_error=np.zeros(len(box_size))
    for j in range(len(box_size)):
        loc_centroid[j]=centriod(file_,loc_max,box_size[j])
    for k in range(len(box_size)):
        error[k]=np.absolute(np.subtract(loc_centroid[k],groundtruth_r8))
        avg_diff[k,0],avg_diff[k,1]=np.average(error[k,0:,0]),np.average(error[k,0:,1])
        abs_error[k]=np.linalg.norm(error[k,:])
    fig_one_box=np.argmin(avg_diff[:,0])
    
    plt.xlabel('Box Size (in odd increments)')
    plt.ylabel('Differance From Ground Truth Average (absolute value)')
    plt.plot(box_size,avg_diff[0:,1],linestyle='-')
    #plt.plot(box_size,avg_diff[0:,1],label='Y Values',linestyle='--')
    plt.ylim(0)
    plt.xlim(box_size[0],box_size[-1])
    plt.hlines(avg_diff[fig_one_box,0],box_size[0],box_size[-1],linestyle=':',color='k',label='Minimum at {}'.format(np.round((avg_diff[fig_one_box,0]),4)))
    #plt.hlines(abs_error[fig_one_box],box_size[0],box_size[-1])
    plt.legend(shadow=True,fancybox=True)
    stop=time.perf_counter()
    print('Total Time Taken {}s'.format(np.round((stop-start),2)))
    plt.tight_layout()
    #plt.show()
    plt.savefig('box_size_var_r2_noise.png',dpi=300)
    pass

def gauss(x,avg,var):
    return 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(x-avg)**2/var) 

#enter like: folder="r1.00 r1.41 r2.00 r2.83 r4.00 r5.66 r8.00"
folder="r4.00"

#file,ground_truth=input("Perfect Spots {}/Perfect Spots {}.tif".format(folder,folder),"Perfect Spots {}/groundtruth.csv".format(folder))
#file,ground_truth=input("Multiple Spots/AF647_npc_1frame.tif")

#points=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))+24.5
#display(num_of_boxes(max is 49 as picture size is 50x50),picture(max is 99))

    
file_=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)
#file_=tf.TiffSequence("Perfect Spots r2.00/Perfect Spots r2.00.tif").asarray().reshape(100,50,50)
loc_max=np.delete((local_max(file_)),0,0)
display(15,1,file_)

