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
    area=np.full(file.shape[1],0)#need to impliment a lower bound on the noise
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
    #res=np.sum(file_in[image])-np.sum(area_)
    res=file_in[image]-area_
    return np.sum(res**2)

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
    N=10
    M=10
    sweep_res_array=np.zeros((N,M))
    sweep_cen_values=np.linspace(24,26,N)
    sweep_hb_values=np.linspace(0.5,5,M)
    max=7e5
    for i in range(N):
        for j in range(M):
            tri_var=triangle(sweep_cen_values[i],sweep_hb_values[j],max)
            area_var=area(tri_var,file_in)
            sweep_res_array[i,j]=res(file_in,area_var,tri_var,image)
    result=np.where(np.min(sweep_res_array)==sweep_res_array)
    return sweep_cen_values[result[0]]

def full_search(file_in_x,file_in_y):
    array_of_result_x=np.zeros(file_in_x.shape[0])
    array_of_result_y=np.zeros(file_in_y.shape[0])
    for i in range(file_in_x.shape[0]):
        result_x=search3(file_in_x,i)
        result_y=search3(file_in_y,i)
        if result_x.shape==1:
            array_of_result_x[i]=result_x
            array_of_result_y[i]=result_y
        else:
            array_of_result_x[i]=np.average(result_x)
            array_of_result_y[i]=np.average(result_y)
    return array_of_result_x,array_of_result_y

start=time.perf_counter()
full_search_result_x,full_search_result_y=full_search(file_sum_x,file_sum_y)
stop=time.perf_counter()

timetaken=stop-start


###
#All for display#
###
error_x=abs(full_search_result_x-(groundtruth[:,1]))
error_y=abs(full_search_result_y-(groundtruth[:,0]))

calculated_points=np.stack((full_search_result_x,full_search_result_y),axis=1)
sub_of_cal_ground=groundtruth.reshape(2,100)-calculated_points.reshape(2,100)
abs_error=np.zeros(100)

for i in range(len(groundtruth)):
    abs_error[i]=np.linalg.norm(sub_of_cal_ground[:,i])

fig, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,constrained_layout=False)

fig.suptitle('Spot-finding for Perfect Spots of Radius 1 out of 100 Images was Done in, {}s'.format(format(timetaken,".2")))

x=np.linspace(0,100,100)

ax1.set_title('X-axis Error From X Part of Groundtruth',pad=10)
ax1.scatter(x,error_x,marker='x')
ax1.set_xlabel("Index of 100 Images")
ax1.set_ylabel("Error")
avg_x=np.average(error_x)
med_x=np.median(error_x)
ax1.axhline(avg_x,0,100,color='r',linestyle=':',label='Average {}'.format(format(avg_x,".2")))
ax1.axhline(med_x,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_x,".2")))
ax1.legend()

ax2.set_title('Y-axis Error From Y Part of Groundtruth')
ax2.scatter(x,error_y,marker='x')
ax2.set_xlabel("Index of 100 Images")
ax2.set_ylabel("Error")
avg_y=np.average(error_y)
med_y=np.median(error_y)
ax2.axhline(avg_y,0,100,color='r',linestyle=':',label='Average {}'.format(format(avg_y,".2")))
ax2.axhline(med_y,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_y,".2")))
ax2.legend()


ax3.set_title('Euclidean Distance From Groundtruth')
ax3.scatter(x,abs_error,marker='x')
ax3.set_xlabel("Index of 100 Images")
ax3.set_ylabel("Error")
avg_abs=np.average(abs_error)
med_abs=np.median(abs_error)
ax3.axhline(avg_abs,0,100,color='r',linestyle=':',label='Average {}'.format(format(avg_abs,".2")))
ax3.axhline(med_abs,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_abs,".2")))
ax3.legend()

plt.show()
###
#All for display#
###


#guess=search3(file_sum,0)
#print(guess,groundtruth[0,1]+25)
#
#tri_var=triangle(24.1,3,1.6e5)
#area_var=area(tri_var,file_sum)
#plt.plot(area_var)
#plt.plot(file_sum[0,:])
#plt.show()
