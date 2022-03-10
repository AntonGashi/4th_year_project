import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit,njit
import time
from scipy.spatial import distance

file_=tf.TiffSequence("Perfect Spots r1.00/Perfect Spots r1.00.tif").asarray().reshape(100,50,50)
#file_=tf.TiffSequence("Noisy Spots/Noisy Spots r8.00/Noisy Spots r8.00.tif").asarray().reshape(100,50,50)
groundtruth=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.00/groundtruth.csv"))+25

def sum_(file):
    file_x=np.zeros((100,50))
    file_y=np.zeros((100,50))
    for i in range(file_.shape[0]):
        file_x[i]=np.sum(file[i],axis=0)
        file_y[i]=np.sum(file[i],axis=1)
    return file_x,file_y

file_sum_x,file_sum_y=sum_(file_)


def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]


def area(tri_in,file):
    grad=(tri_in[1][1]-tri_in[1][0])/(tri_in[0][1]-tri_in[0][0])
    start,stop=math.floor(tri_in[0][0]),math.ceil(tri_in[0][2])
    area=np.zeros_like(file,dtype=float)#need to impliment a lower bound on the noise
    for i in range(start,stop):
        if i==start:
            height=((i+1)-tri_in[0][0])*grad
            area[i]=0.5*((i+1)-tri_in[0][0])*(height)
            print(0.5*((i+1)-tri_in[0][0])*(height))
        elif i<math.floor(tri_in[0][1]):
            area[i]=height+(0.5*grad)
            height=height+grad
        elif i==math.floor(tri_in[0][1]):
            area1=((tri_in[0][1]-i)*height)+(0.5*(tri_in[0][1]-i)*(tri_in[1][1]-height))
            area2=(((i+1)-tri_in[0][1])*height)+(0.5*((i+1)-tri_in[0][1])*(tri_in[1][1]-height))
            area[i]=area1+area2
        elif stop-1>i>math.floor(tri_in[0][1]):
            height=height-grad
            area[i]=height+(0.5*grad)
        elif i==stop:
            height=((i-1)-tri_in[0][2])*grad
            area[i-1]=0.5*((i-1)-tri_in[0][2])*height
    return area

tri_var=triangle(24,5,100)
tri_var_2=triangle(24.4,5,100)
area_var=area(tri_var,file_sum_x)
area_var_2=area(tri_var_2,file_sum_x)

#print(area_var)
#print(area_var_2)
#plt.plot(area_var)
#plt.plot(area_var_2)
#plt.show()
