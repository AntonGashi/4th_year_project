import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit,njit
import time
from scipy.spatial import distance
import scipy.optimize as spy
from scipy.stats import norm
import matplotlib.mlab as mlab



def sum_(file_):
    file_x=np.zeros((100,50))
    file_y=np.zeros((100,50))
    for i in range(file_.shape[0]):
        file_x[i]=np.sum(file_[i],axis=0)
        file_y[i]=np.sum(file_[i],axis=1)
    return file_x,file_y


def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]


def area(tri_in,file):
    grad=(tri_in[1][1]-tri_in[1][0])/(tri_in[0][1]-tri_in[0][0])
    start,stop=math.floor(tri_in[0][0]),math.ceil(tri_in[0][2])
    area=np.full(file.shape[1],0,dtype=float)#need to impliment a lower bound on the noise
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
    res=(file_in[image]-area_)**2
    #res=file_in[image]-area_
    #### write about how changing this helped
    return np.sum(res)

### dont need test function

def test(file_):
    N=10000
    array_of_res=np.zeros(N)
    array_of_centre=np.linspace(22,27,N)
    
    for i in range(N):
        tri_var=triangle(array_of_centre[i],5.6,1.7e5)
        area_=area(tri_var,file_)
        array_of_res[i]=res(file_,area_,tri_var,0)
    
    return array_of_res,array_of_centre


def search(file_in,image):
    N=50
    first_search=np.zeros((N,N,N))
    first_cen_guess=np.linspace(23, 27,N)
    first_hb_guess=np.linspace(4.5,7.5,N)
    first_height_guess=np.linspace(1.3e5,2.5e5,N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                tri_var=triangle(first_cen_guess[i],first_hb_guess[j],first_height_guess[k])
                area_var=area(tri_var,file_in)
                first_search[i,j,k]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(first_search)==first_search)
    return first_cen_guess[result[0]]


def search2(file_in,image):
    N=5
    first_search=np.zeros((N,N,N))
    first_cen_guess=np.linspace(23, 27,N)
    first_hb_guess=np.linspace(3,8,N)
    first_height_guess=np.linspace(1e5,7e5,N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                tri_var=triangle(first_cen_guess[i],first_hb_guess[j],first_height_guess[k])
                area_var=area(tri_var,file_in)
                first_search[i,j,k]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(first_search)==first_search)
    second_search=np.zeros((10,5,5))
    second_cen_guess=np.linspace((first_cen_guess[result[0]]-1), (first_cen_guess[result[0]]+1),10)
    second_hb_guess=np.linspace((first_hb_guess[result[1]]-1),(first_hb_guess[result[1]]+1),5)
    second_height_guess=np.linspace((first_height_guess[result[2]]-1e3),(first_height_guess[result[2]]+1e3),5)
    for i in range(10):
        for j in range(5):
            for k in range(5):
                tri_var=triangle(second_cen_guess[i],second_hb_guess[j],second_height_guess[k])
                area_var=area(tri_var,file_in)
                second_search[i,j,k]=res(file_in,area_var,tri_var,image)
    
    result=np.where(np.min(second_search)==second_search)
    return second_cen_guess[result[0]]

#first_cen_guess[result[0]],first_hb_guess[result[1]],first_height_guess[result[2]]

def search3(file_in,image):
    global sweep_res_array
    N=100
    M=5
    sweep_res_array=np.zeros((N,M))
    sweep_cen_values=np.linspace(22,28,N)
    sweep_hb_values=np.linspace(2,8,M)
    max=7e5
    for i in range(N):
        for j in range(M):
            tri_var=triangle(sweep_cen_values[i],sweep_hb_values[j],max)
            area_var=area(tri_var,file_in)
            sweep_res_array[i,j]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(sweep_res_array)==sweep_res_array)
    return sweep_cen_values[result[0]]

def search4(file_in,image):
    N=100
    center_values=np.linspace(22,28,N)
    cen_sweep=np.zeros(N)
    for i in range(N):
        tri_var=triangle(center_values[i],4,6e5)
        area_var=area(tri_var,file_in)
        cen_sweep[i]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(cen_sweep)==cen_sweep)
    return center_values[result]

def full_search(file_in_x,file_in_y):
    array_of_result_x=np.zeros(file_in_x.shape[0])
    array_of_result_y=np.zeros(file_in_y.shape[0])
    for i in range(file_in_x.shape[0]):
        result_x=search4(file_in_x,i)
        result_y=search4(file_in_y,i)
        if result_x.shape==1:
            array_of_result_x[i]=result_x
            array_of_result_y[i]=result_y
        else:
            array_of_result_x[i]=np.average(result_x)
            array_of_result_y[i]=np.average(result_y)
    return array_of_result_x,array_of_result_y

def gauss(x,avg,var):
    return 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(x-avg)**2/var) 

