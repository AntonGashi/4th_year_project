from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf
from numba import jit
import time
import pandas as pd
import math


file=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)

groundtruth=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r4.00/groundtruth.csv"))
def sum_(file):
    file_=np.zeros((100,50))
    for i in range(file_.shape[0]):
        file_[i]=np.sum(file[i],axis=0)
    return file_

file_=sum_(file)
#file_=file_[:,19:31]
start=time.perf_counter()

plt.plot(file_[0])
plt.show()

#@jit(nopython=True)
def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]

#triangle(7.333,3.2,10)
#@jit(nopython=True)
def area(tri,file):
    grad=(tri[1][1]-tri[1][0])/(tri[0][1]-tri[0][0])
    start,stop=math.floor(tri[0][0]),math.ceil(tri[0][2])
    area=np.zeros(file.shape[1])
    for i in range(start,stop+1):
        if i==start:
            height=((i+1)-tri[0][0])*grad
            area[i]=0.5*((i+1)-tri[0][0])*(height)
        elif i<math.floor(tri[0][1]):
            area[i]=height+(0.5*grad)
            height=height+grad
        elif i==math.floor(tri[0][1]):
            area1=((tri[0][1]-i)*height)+(0.5*(tri[0][1]-i)*(tri[1][1]-height))
            area2=(((i+1)-tri[0][1])*height)+(0.5*((i+1)-tri[0][1])*(tri[1][1]-height))
            area[i]=area1+area2
        elif stop-1>i>math.floor(tri[0][1]):
            height=height-grad
            area[i]=height+(0.5*grad)
        elif i==stop:
            height=((i-1)-tri[0][2])*grad
            area[i-1]=0.5*((i-1)-tri[0][2])*height
    return area

#@jit(nopython=True)
def res(file,area_,tri,image):
    res=np.zeros_like(file[image])
    res=file[image]
    res=res-area_
    return np.sum(res**2)

#@jit(nopython=True)
def grid_serch(file,image):
    j=0
    init_search=(np.random.randn(50,1))+25
    hyper=np.zeros_like(init_search)
    for i in range(len(init_search)):
        tri=triangle(init_search[i], 6.5, 4.7e5)
        area_=area(tri,file)
        hyper[j]=res(file,area_,tri,image)
        j+=1
    first_search=init_search[np.argmin(hyper)]
    
    search=np.arange(3,9,0.1)
    hyper=np.zeros_like(search)
    n=0
    for m in range(len(search)):
        tri=triangle(first_search, search[m], 4.7e5)
        area_=area(tri,file)
        hyper[n]=res(file, area_, tri, image)
        n+=1
    opt_hbase=search[np.argmin(hyper)]

    sec_search=(0.05*np.random.randn(100,1))+first_search
    hyper=np.zeros_like(sec_search)
    l=0
    for k in range(len(sec_search)):
        tri=triangle(sec_search[k], opt_hbase,4.7e5)
        area_=area(tri,file)
        hyper[l]=res(file,area_,tri,image)
        l+=1
    opt_cen=sec_search[np.argmin(hyper)]
    
    return opt_cen,opt_hbase

#@jit(nopython=True)
def all_file(file):
    all_res_=np.zeros((len(file),2))
    for i in range(len(file)):
        opt=grid_serch2(file, i)
        all_res_[i,0]=opt
        #all_res_[i,1]=opt[1]
    return all_res_

def grid_serch2(file,image):
    first_pass_centre=np.linspace(23,27,10)
    first_pass_height=np.linspace(4.8e5,6e5,10)
    first_pass_res=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            tri_=triangle(first_pass_centre[i], 6, first_pass_height[j])
            area_=area(tri_, file)
            first_pass_res[i,j]=res(file, area_, tri_, image)
    first_sol=np.argmin(first_pass_res)
    c,h=first_pass_centre[math.floor(first_sol/10)],first_pass_height[first_sol-math.floor(first_sol/10)*10]
    second_pass_centre=np.linspace(c-2,c+2,20)
    second_pass_hb=np.linspace(6-2,6+2,20)
    second_pass_res=np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            tri_=triangle(second_pass_centre[i],second_pass_hb[j], h)
            area_=area(tri_, file)
            second_pass_res[i,j]=res(file, area_, tri_, image)
    second_sol=np.argmin(second_pass_res)
    c,hb=second_pass_centre[math.floor(second_sol/20)],second_pass_hb[second_sol-math.floor(second_sol/20)*20]
    return c

def grid_serch3(file,image):
    first_pass_centre=np.linspace(23,26,5)
    first_pass_hb=np.linspace(3,7,5)
    first_pass_height=np.linspace(2e5,6e5,5)
    first_pass_res=np.zeros((5,5,5))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                tri_=triangle(first_pass_centre[i],first_pass_hb[j],first_pass_height[k])
                area_=area(tri_, file)
                first_pass_res[i,j,k]=res(file, area_, tri_, image)
    loc_1=np.where(first_pass_res==np.min(first_pass_res))
    c,hb,h=first_pass_centre[loc_1[0]],first_pass_hb[loc_1[1]],first_pass_height[loc_1[2]]
    second_pass_centre=np.linspace(c-0.5,c+0.5,10)
    second_pass_hb=np.linspace(hb-0.5,hb+0.5,10)
    second_pass_height=np.linspace(h-1e5,h+1e5,10)
    second_pass_res=np.zeros((10,10,10))
    for i in range(10):
        for j in range(10):
            for k in range(10):
                tri_=triangle(second_pass_centre[i],second_pass_hb[j],second_pass_height[k])
                area_=area(tri_, file)
                second_pass_res[i,j,k]=res(file, area_, tri_, image)
    loc_1=np.where(second_pass_res==np.min(second_pass_res))
    c,hb,h=second_pass_centre[loc_1[0]],second_pass_hb[loc_1[1]],second_pass_height[loc_1[2]]
    return c,hb,h
'''
pic=9
c,hb,h=grid_serch3(file_,pic)
tri_=triangle(float(c),float(hb),float(h))
all_res=all_file(file_)
error2=np.absolute((groundtruth[:,1]+25)-all_res[:,0])
plt.plot(error2)
plt.show()
#plt.plot(tri_[0],tri_[1])
#plt.plot(file_[pic,:])
#plt.show()
'''
'''
pic=9
new_s=grid_serch2(file_,pic)
gt=groundtruth[pic,1]+25
#gt2=groundtruth[pic,0]+25
plt.plot(file_[pic,:])
plt.vlines(gt, 0, 5e5)
#plt.vlines(gt2, 0, 5e5,'r')
plt.vlines(new_s, 0, 5e5,'k')
tri_=triangle(new_s,6, 4.67e5)
plt.plot(tri_[0],tri_[1])
plt.show()

all_res=all_file(file_)
#error_=np.absolute((groundtruth[:,0]+25)-all_res[:,0])
error2=np.absolute((groundtruth[:,1]+25)-all_res[:,0])
#stop=time.perf_counter()
#print('time taken : {} seconds'.format(stop-start))
#plt.plot(error_)
plt.plot(error2)
plt.hlines((np.mean(error2)),0,100,color='r')
plt.show()


tri=triangle(24.7, 12,5.7e5)
area_=area(tri, file_)
print(res(file_, area_, tri, 0))
x=np.arange(19,31,1)
plt.bar(x,file_[0])
plt.plot(tri[0],tri[1],'r')
plt.show()
'''
