import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit,njit
import time
from scipy.spatial import distance
from matplotlib.ticker import (MultipleLocator,FormatStrFormatter, AutoMinorLocator)

file_=tf.TiffSequence("Perfect Spots r8.00/Perfect Spots r8.00.tif").asarray().reshape(100,50,50)
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
    area=np.zeros_like(file[0],dtype=float)#need to impliment a lower bound on the noise
    for i in range(start,stop):
        if i==start:
            height=((i+1)-tri_in[0][0])*grad
            area[i]=0.5*((i+1)-tri_in[0][0])*(height)
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

def res(file_in,area_,tri_in,image):
    res=np.sum(file_in[image])-np.sum(area_)
    display=abs(file_in[image]-area_)
    #res=file_in[image]-area_
    #### write about how changing this helped
    return res**2,display


tri_var=triangle(25,5,5e5)

print(groundtruth[0])

area_var=area(tri_var,file_sum_x)
x=np.linspace(0,50,50)

res_var=abs(file_sum_x[0]-area_var)

fig, (ax1,ax2)=plt.subplots(2,1,sharex=True,gridspec_kw={'hspace': 0, 'wspace': 0})

fig.supxlabel('Pixels (summed x-axis)')
fig.supylabel('Pixel Intensities')
ax1.plot(tri_var[0],tri_var[1],color='r',label='Triangle Function')
ax1.bar(x,file_sum_x[0],label='Input Spot')
#ax1.bar(x,area_var,label='Calculated Area')
ax1.yaxis.set_major_formatter(FormatStrFormatter('% .0e'))
ax1.legend()

#ax2.bar(x,res_var,color='k',label='Residual Area')
ax2.set_ylim(0,np.max(area_var))
ax2.invert_yaxis()
ax2.yaxis.set_major_formatter(FormatStrFormatter('% .0e'))

plt.legend()
plt.tight_layout()
plt.savefig('visual_test.png',dpi=400)
