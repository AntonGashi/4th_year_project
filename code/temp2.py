from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf
from numba import jit
import time
import pandas as pd
import math


file=tf.TiffSequence("Perfect Spots r8.00/Perfect Spots r8.00.tif").asarray().reshape(100,50,50)

groundtruth=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))
def sum_(file):
    file_=np.zeros((100,50))
    for i in range(file_.shape[0]):
        file_[i]=np.sum(file[i],axis=1)
    return file_

file_=sum_(file)

start=time.perf_counter()
def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]

#triangle(7.333,3.2,10)

def area(tri):
    grad=(tri[1][1]-tri[1][0])/(tri[0][1]-tri[0][0])
    start,stop=math.floor(tri[0][0]),math.ceil(tri[0][2])
    area=np.zeros(stop-start)
    for i in range(start,stop+1):
        if i==start:
            height=((i+1)-tri[0][0])*grad
            area[i-start]=0.5*((i+1)-tri[0][0])*(height)
        elif i<math.floor(tri[0][1]):
            area[i-start]=height+(0.5*grad)
            height=height+grad
        elif i==math.floor(tri[0][1]):
            area1=((tri[0][1]-i)*height)+(0.5*(tri[0][1]-i)*(tri[1][1]-height))
            area2=(((i+1)-tri[0][1])*height)+(0.5*((i+1)-tri[0][1])*(tri[1][1]-height))
            area[i-start]=area1+area2
        elif stop-1>i>math.floor(tri[0][1]):
            height=height-grad
            area[i-start]=height+(0.5*grad)
        elif i==stop:
            height=((i-1)-tri[0][2])*grad
            area[i-start-1]=0.5*((i-1)-tri[0][2])*height
    return area

def res(file,area_,tri,image):
    res=np.zeros_like(file)
    res=res+file[image]
    for i in range(len(area_)):
        res[image,(i+math.floor(tri[0][0]))]=res[image,(i+math.floor(tri[0][0]))]-area_[i]
    return np.absolute(res)

tri=triangle(23.333,6.5,5e5)
area_=area(tri)
res_=res(file_,area_,tri,0)
stop=time.perf_counter()
print(stop-start)

x=np.arange(50)
#plt.bar(x,file_[0])
plt.bar(x,res_[0])
plt.plot(tri[0],tri[1],'r')
plt.show()