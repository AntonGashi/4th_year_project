from os import path
import re
import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit
import pandas as pd
import cv2 as cv

def input(path_to_tiff,path_to_groundtruth=0):
    file=tf.TiffSequence("{}".format(path_to_tiff)).asarray()
    if len(file.shape)>4:
        print('ERROR')
    elif len(file.shape)==4:
        file=file.reshape(file.shape[1],file.shape[2],file.shape[3])
    if path_to_groundtruth==0:
        return file
    else:
        ground_truth=pd.DataFrame.to_numpy(pd.read_csv("{}".format(path_to_groundtruth)))
        return file,ground_truth

def local_max(file):

    file2=cv.medianBlur(file,5)
    alt_file=np.zeros_like(file)
    results=np.zeros((file.shape[0],100,2))
    iter_=13
    loop=int(file.shape[1]/iter_)
    for i in range(file.shape[0]):
        for j in range(loop):
            for k in range(loop):
                alt_file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)]=np.amax(file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
    for i in range(file.shape[0]):
        count=0
        for j in range(file.shape[1]):
            for k in range(file.shape[2]):
                if file[i,j,k]==alt_file[i,j,k] and (np.max(file2[i])*0.4)<file[i,j,k]:
                    
                    results[i]=[count,[j,k]]
                    count+=1
    return results,alt_file


file=input("Multiple Spots/AF647_npc_1frame.tif")
loc_max,alt_file=local_max(file)


fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4),constrained_layout=True)

ax1.imshow(file[0])
ax2.imshow(alt_file[0])
ax1.plot(loc_max[:,1],loc_max[:,0],'x',color='k')
ax2.plot(loc_max[:,1],loc_max[:,0],'x',color='k')
plt.show()