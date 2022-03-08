import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit,njit


file_=tf.TiffSequence("Perfect Spots r2.00/Perfect Spots r2.00.tif").asarray().reshape(100,50,50)
#file_=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)
groundtruth=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r2.00/groundtruth.csv"))


def sum_(file):
    file_=np.zeros((100,50))
    for i in range(file_.shape[0]):
        file_[i]=np.sum(file[i],axis=0)
    return file_

file_sum=sum_(file_)


def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]


def area(tri_in,file):
    grad=(tri_in[1][1]-tri_in[1][0])/(tri_in[0][1]-tri_in[0][0])
    start,stop=math.floor(tri_in[0][0]),math.ceil(tri_in[0][2])
    print(start,stop)
    area=np.zeros(file.shape[1])
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


def res(file,area_,tri_in,image):
    res=file[image]-area_
    return np.sum(res**2),abs(res)

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

''' ### proof that the method works but the opt function isn't yet
x_res,x_cen=test(file_sum)

cen_pos=np.where(np.min(x_res)==x_res)
print("guess= ",x_cen[cen_pos])

plt.plot(x_cen,x_res)
plt.xlabel("center values")
plt.ylabel("residual")
plt.show()
'''


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
#first_cen_guess[result[0]],first_hb_guess[result[1]],first_height_guess[result[2]]


def search2(file_in,image):
    N=10
    first_search=np.zeros((N,N,N))
    first_cen_guess=np.linspace(23, 27,N)
    first_hb_guess=np.linspace(3,8,N)
    first_height_guess=np.linspace(1e5,6e5,N)
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
    N=100
    M=10
    sweep_res_array=np.zeros((N,M))
    sweep_cen_values=np.linspace(24,26,N)
    sweep_hb_values=np.linspace(4,6,M)
    max=1.5e5
    for i in range(N):
        for j in range(M):
            tri_var=triangle(sweep_cen_values[i],sweep_hb_values[j],max)
            area_var=area(tri_var,file_in)
            sweep_res_array[i,j]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(sweep_res_array)==sweep_res_array)
    return sweep_cen_values[result[0]], sweep_res_array

def full_search(file_in):
    array_of_result=np.zeros(file_in.shape[0])
    for i in range(file_in.shape[0]):
        array_of_result[i]=search2(file_in,i)
    return array_of_result

#full_search_result=full_search(file_sum)
#error=abs(full_search_result-(groundtruth[:,1]+25))


#x=np.linspace(0,100,100)
#plt.scatter(x,error)
#plt.xlabel("index of images")
#plt.ylabel("pixel error")
#plt.hlines((np.average(error)),0,100,color='r',linestyle=':',label='average')
#plt.hlines((np.median(error)),0,100,color='k',linestyle=':',label='median')
#plt.legend()



#guess,array=search3(file_sum,22)
#print(guess,groundtruth[22,1]+25)



#plt.plot(array)
tri_test=triangle(25,6,50)
area_array=area(tri_test,file_sum)
res_num,res_array=res(file_sum,area_array,tri_test,22)
print(format(res_num,".2e"))
plt.plot(area_array,'k')
plt.plot(tri_test[0],tri_test[1])
#plt.plot(file_sum[22])
#plt.plot(res_array)
plt.show()
