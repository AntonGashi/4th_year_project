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
    res=np.sum(file_in[image])-np.sum(area_)
    #res=file_in[image]-area_
    #### write about how changing this helped
    return res**2

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

### file variable define ###


file_r1=tf.TiffSequence("Perfect Spots r1.00/Perfect Spots r1.00.tif").asarray().reshape(100,50,50)
groundtruth_r1=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.00/groundtruth.csv"))+25
file_sum_x_r1,file_sum_y_r1=sum_(file_r1)

file_r141=tf.TiffSequence("Perfect Spots r1.41/Perfect Spots r1.41.tif").asarray().reshape(100,50,50)
groundtruth_r141=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.41/groundtruth.csv"))+25
file_sum_x_r141,file_sum_y_r141=sum_(file_r141)

file_r2=tf.TiffSequence("Perfect Spots r2.00/Perfect Spots r2.00.tif").asarray().reshape(100,50,50)
groundtruth_r2=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r2.00/groundtruth.csv"))+25
file_r2_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)
file_sum_x_r2,file_sum_y_r2=sum_(file_r2)
file_sum_x_r2_noise,file_sum_y_r2_noise=sum_(file_r2_noise)


file_r283=tf.TiffSequence("Perfect Spots r2.83/Perfect Spots r2.83.tif").asarray().reshape(100,50,50)
groundtruth_r283=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r2.83/groundtruth.csv"))+25
file_r283_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.83/Noisy Spots r2.83.tif").asarray().reshape(100,50,50)
file_sum_x_r283,file_sum_y_r283=sum_(file_r283)
file_sum_x_r283_noise,file_sum_y_r283_noise=sum_(file_r283_noise)


file_r4=tf.TiffSequence("Perfect Spots r4.00/Perfect Spots r4.00.tif").asarray().reshape(100,50,50)
groundtruth_r4=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r4.00/groundtruth.csv"))+25
file_r4_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r4.00/Noisy Spots r4.00.tif").asarray().reshape(100,50,50)
file_sum_x_r4,file_sum_y_r4=sum_(file_r4)
file_sum_x_r4_noise,file_sum_y_r4_noise=sum_(file_r4_noise)


file_r566=tf.TiffSequence("Perfect Spots r5.66/Perfect Spots r5.66.tif").asarray().reshape(100,50,50)
groundtruth_r566=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r5.66/groundtruth.csv"))+25
file_r566_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r5.66/Noisy Spots r5.66.tif").asarray().reshape(100,50,50)
file_sum_x_r566,file_sum_y_r566=sum_(file_r566)
file_sum_x_r566_noise,file_sum_y_r566_noise=sum_(file_r566_noise)


file_r8=tf.TiffSequence("Perfect Spots r8.00/Perfect Spots r8.00.tif").asarray().reshape(100,50,50)
groundtruth_r8=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))+25
file_r8_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r8.00/Noisy Spots r8.00.tif").asarray().reshape(100,50,50)
file_sum_x_r8,file_sum_y_r8=sum_(file_r8)
file_sum_x_r8_noise,file_sum_y_r8_noise=sum_(file_r8_noise)

### file variable define end ###

start=time.perf_counter()
full_search_result_x_r2,full_search_result_y_r2=full_search(file_sum_x_r2_noise,file_sum_y_r2_noise)
full_search_result_x_r283,full_search_result_y_r283=full_search(file_sum_x_r283_noise,file_sum_y_r283_noise)
full_search_result_x_r4,full_search_result_y_r4=full_search(file_sum_x_r4_noise,file_sum_y_r4_noise)
full_search_result_x_r566,full_search_result_y_r566=full_search(file_sum_x_r566_noise,file_sum_y_r566_noise)
full_search_result_x_r8,full_search_result_y_r8=full_search(file_sum_x_r8_noise,file_sum_y_r8_noise)
stop=time.perf_counter()

timetaken=stop-start

'''
start=time.perf_counter()
full_search_result_x_r1,full_search_result_y_r1=full_search(file_sum_x_r1,file_sum_y_r1)
full_search_result_x_r141,full_search_result_y_r141=full_search(file_sum_x_r141,file_sum_y_r141)
full_search_result_x_r2,full_search_result_y_r2=full_search(file_sum_x_r2,file_sum_y_r2)
full_search_result_x_r283,full_search_result_y_r283=full_search(file_sum_x_r283,file_sum_y_r283)
full_search_result_x_r4,full_search_result_y_r4=full_search(file_sum_x_r4,file_sum_y_r4)
full_search_result_x_r566,full_search_result_y_r566=full_search(file_sum_x_r566,file_sum_y_r566)
full_search_result_x_r8,full_search_result_y_r8=full_search(file_sum_x_r8,file_sum_y_r8)
stop=time.perf_counter()
'''
###
#All for display#
###
'''
error_x_r1=abs(full_search_result_x_r1-(groundtruth_r1[:,1]))
error_y_r1=abs(full_search_result_y_r1-(groundtruth_r1[:,0]))

error_x_r141=abs(full_search_result_x_r141-(groundtruth_r141[:,1]))
error_y_r141=abs(full_search_result_y_r141-(groundtruth_r141[:,0]))
'''
error_x_r2=abs(full_search_result_x_r2-(groundtruth_r2[:,1]))
error_y_r2=abs(full_search_result_y_r2-(groundtruth_r2[:,0]))

error_x_r283=abs(full_search_result_x_r283-(groundtruth_r283[:,1]))
error_y_r283=abs(full_search_result_y_r283-(groundtruth_r283[:,0]))

error_x_r4=abs(full_search_result_x_r4-(groundtruth_r4[:,1]))
error_y_r4=abs(full_search_result_y_r4-(groundtruth_r4[:,0]))

error_x_r566=abs(full_search_result_x_r566-(groundtruth_r566[:,1]))
error_y_r566=abs(full_search_result_y_r566-(groundtruth_r566[:,0]))

error_x_r8=abs(full_search_result_x_r8-(groundtruth_r8[:,1]))
error_y_r8=abs(full_search_result_y_r8-(groundtruth_r8[:,0]))


'''
calculated_points_r1=np.stack((full_search_result_x_r1,full_search_result_y_r1),axis=1)
sub_of_cal_ground_r1=groundtruth_r1.reshape(2,100)-calculated_points_r1.reshape(2,100)
abs_error_r1=np.zeros(100)

calculated_points_r141=np.stack((full_search_result_x_r141,full_search_result_y_r141),axis=1)
sub_of_cal_ground_r141=groundtruth_r141.reshape(2,100)-calculated_points_r141.reshape(2,100)
abs_error_r141=np.zeros(100)
'''
calculated_points_r2=np.stack((full_search_result_x_r2,full_search_result_y_r2),axis=1)
sub_of_cal_ground_r2=groundtruth_r2.reshape(2,100)-calculated_points_r2.reshape(2,100)
abs_error_r2=np.zeros(100)

calculated_points_r283=np.stack((full_search_result_x_r283,full_search_result_y_r283),axis=1)
sub_of_cal_ground_r283=groundtruth_r283.reshape(2,100)-calculated_points_r283.reshape(2,100)
abs_error_r283=np.zeros(100)

calculated_points_r4=np.stack((full_search_result_x_r4,full_search_result_y_r4),axis=1)
sub_of_cal_ground_r4=groundtruth_r4.reshape(2,100)-calculated_points_r4.reshape(2,100)
abs_error_r4=np.zeros(100)

calculated_points_r566=np.stack((full_search_result_x_r566,full_search_result_y_r566),axis=1)
sub_of_cal_ground_r566=groundtruth_r566.reshape(2,100)-calculated_points_r566.reshape(2,100)
abs_error_r566=np.zeros(100)

calculated_points_r8=np.stack((full_search_result_x_r8,full_search_result_y_r8),axis=1)
sub_of_cal_ground_r8=groundtruth_r8.reshape(2,100)-calculated_points_r8.reshape(2,100)
abs_error_r8=np.zeros(100)



for i in range(len(groundtruth_r1)):
    #abs_error_r1[i]=np.linalg.norm(sub_of_cal_ground_r1[:,i])
    #abs_error_r141[i]=np.linalg.norm(sub_of_cal_ground_r141[:,i])
    abs_error_r2[i]=np.linalg.norm(sub_of_cal_ground_r2[:,i])
    abs_error_r283[i]=np.linalg.norm(sub_of_cal_ground_r283[:,i])
    abs_error_r4[i]=np.linalg.norm(sub_of_cal_ground_r4[:,i])
    abs_error_r566[i]=np.linalg.norm(sub_of_cal_ground_r566[:,i])
    abs_error_r8[i]=np.linalg.norm(sub_of_cal_ground_r8[:,i])




x=np.linspace(0,100,100)

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,sharey=True,sharex=False,tight_layout=True)

print(format(timetaken,".2"))



fig.supxlabel('Index Of Images')
fig.supylabel('Absolute Error (pixels)')

#fig.supxlabel('Absolute Error (Pixels)')
#fig.supylabel('Absolute Amount')

ax1.scatter(x,abs_error_r2,marker='x')
avg_abs=np.average(abs_error_r2)
med_abs=np.median(abs_error_r2)
var_abs=np.var(abs_error_r2)
#hist,bins,patches=ax1.hist(abs_error_r2,30)
#pdf_y=gauss(bins,avg_abs,var_abs)
#ax1.plot(bins,pdf_y,label='Fitted Gaussian')
#ax1.vlines(avg_abs,0,8,color='r',label='Average')
ax1.axhline(avg_abs,100,color='r',linestyle=':',label='Average')
ax1.axhline(med_abs,0,100,color='k',linestyle=':',label='Median')
ax1.set_title('R2.00')
#print(avg_abs,var_abs)

ax2.set_title('R2.83')
ax2.scatter(x,abs_error_r283,marker='x')
avg_abs=np.average(abs_error_r283)
med_abs=np.median(abs_error_r283)
var_abs=np.var(abs_error_r283)
#hist2,bins2,patches2=ax2.hist(abs_error_r283,30)
#pdf_y2=gauss(bins2,avg_abs,var_abs)
#ax2.plot(bins2,pdf_y2)
#ax2.vlines(avg_abs,0,8,color='r')
ax2.axhline(avg_abs,0,100,color='r',linestyle=':')
ax2.axhline(med_abs,0,100,color='k',linestyle=':')

ax3.set_title('R4.00')
ax3.scatter(x,abs_error_r4,marker='x')
avg_abs=np.average(abs_error_r4)
med_abs=np.median(abs_error_r4)
var_abs=np.var(abs_error_r4)
#hist,bins,patches=ax3.hist(abs_error_r4,30)
#pdf_y=gauss(bins,avg_abs,var_abs)
#ax3.plot(bins,pdf_y)
#ax3.vlines(avg_abs,0,8,color='r')
ax3.axhline(avg_abs,0,100,color='r',linestyle=':')
ax3.axhline(med_abs,0,100,color='k',linestyle=':')

ax4.set_title('R5.66')
ax4.scatter(x,abs_error_r566,marker='x')
aavg_abs=np.average(abs_error_r566)
med_abs=np.median(abs_error_r566)
var_abs=np.var(abs_error_r566)
#hist,bins,patches=ax4.hist(abs_error_r566,30)
#pdf_y=gauss(bins,avg_abs,var_abs)
#ax4.plot(bins,pdf_y)
#ax4.vlines(avg_abs,0,8,color='r')
ax4.axhline(avg_abs,0,100,color='r',linestyle=':')
ax4.axhline(med_abs,0,100,color='k',linestyle=':')

ax5.set_title('R8.00')
ax5.scatter(x,abs_error_r8,marker='x')
avg_abs=np.average(abs_error_r8)
med_abs=np.median(abs_error_r8)
var_abs=np.var(abs_error_r8)
#hist,bins,patches=ax5.hist(abs_error_r8,30)
#pdf_y=gauss(bins,avg_abs,var_abs)
#ax5.plot(bins,pdf_y)
#ax5.vlines(avg_abs,0,8,color='r')
ax5.axhline(avg_abs,0,100,color='r',linestyle=':')
ax5.axhline(med_abs,0,100,color='k',linestyle=':')


plt.delaxes(ax6)
fig.legend(loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig('noise_scatter.png',dpi=400)





'''
x=np.linspace(0,100,100)

fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8))=plt.subplots(2,4,sharey=True,sharex=False,tight_layout=True)

print(format(timetaken,".2"))



fig.supxlabel('Index Of Images')
fig.supylabel('Absolute Error (pixels)')


#ax1.scatter(x,abs_error_r1,marker='x')
avg_abs=np.average(abs_error_r1)
med_abs=np.median(abs_error_r1)
var_abs=np.var(abs_error_r1)
hist,bins,patches=ax1.hist(abs_error_r1,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax1.plot(bins,pdf_y,label='Fitted Gaussian')
ax1.vlines(avg_abs,0,8,color='r',label='Average')
#ax1.axhline(avg_abs,100,color='r',linestyle=':',label='Average')
#ax1.axhline(med_abs,0,100,color='k',linestyle=':',label='Median')
ax1.set_title('R1.00')
print(avg_abs,var_abs)

ax2.set_title('R1.41')
#ax2.scatter(x,abs_error_r141,marker='x')
avg_abs=np.average(abs_error_r141)
med_abs=np.median(abs_error_r141)
var_abs=np.var(abs_error_r141)
hist2,bins2,patches2=ax2.hist(abs_error_r141,30)
pdf_y2=gauss(bins2,avg_abs,var_abs)
ax2.plot(bins2,pdf_y2)
ax2.vlines(avg_abs,0,8,color='r')
#ax2.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax2.axhline(med_abs,0,100,color='k',linestyle=':')

ax3.set_title('R2.00')
#ax3.scatter(x,abs_error_r2,marker='x')
avg_abs=np.average(abs_error_r2)
med_abs=np.median(abs_error_r2)
var_abs=np.var(abs_error_r2)
hist,bins,patches=ax3.hist(abs_error_r2,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax3.plot(bins,pdf_y)
ax3.vlines(avg_abs,0,8,color='r')
#ax3.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax3.axhline(med_abs,0,100,color='k',linestyle=':')

ax4.set_title('R2.38')
#ax4.scatter(x,abs_error_r283,marker='x')
aavg_abs=np.average(abs_error_r283)
med_abs=np.median(abs_error_r283)
var_abs=np.var(abs_error_r283)
hist,bins,patches=ax4.hist(abs_error_r283,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax4.plot(bins,pdf_y)
ax4.vlines(avg_abs,0,8,color='r')
#ax4.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax4.axhline(med_abs,0,100,color='k',linestyle=':')

ax5.set_title('R4.00')
#ax5.scatter(x,abs_error_r4,marker='x')
avg_abs=np.average(abs_error_r4)
med_abs=np.median(abs_error_r4)
var_abs=np.var(abs_error_r4)
hist,bins,patches=ax5.hist(abs_error_r4,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax5.plot(bins,pdf_y)
ax5.vlines(avg_abs,0,8,color='r')
#ax5.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax5.axhline(med_abs,0,100,color='k',linestyle=':')

ax6.set_title('R5.66')
#ax6.scatter(x,abs_error_r566,marker='x')
avg_abs=np.average(abs_error_r566)
med_abs=np.median(abs_error_r566)
var_abs=np.var(abs_error_r566)
hist,bins,patches=ax6.hist(abs_error_r566,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax6.plot(bins,pdf_y)
ax6.vlines(avg_abs,0,8,color='r')
#ax6.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax6.axhline(med_abs,0,100,color='k',linestyle=':')

ax7.set_title('R8.00')
#ax7.scatter(x,abs_error_r8,marker='x')
avg_abs=np.average(abs_error_r8)
med_abs=np.median(abs_error_r8)
var_abs=np.var(abs_error_r8)
hist,bins,patches=ax7.hist(abs_error_r8,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax7.plot(bins,pdf_y)
ax7.vlines(avg_abs,0,8,color='r')
#ax7.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax7.axhline(med_abs,0,100,color='k',linestyle=':')
print(avg_abs,var_abs)

plt.delaxes(ax8)
fig.legend(loc='lower right')
plt.tight_layout()
plt.show()
#plt.savefig('distro.png',dpi=400)
## maybe put a min and max hline on the graphs
###
#All for display#
###
'''

'''
### extra that i may or may not need

fig, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,sharex=False,tight_layout=True)

ax1.set_title('A',pad=10)
ax1.scatter(x,error_x_r1,marker='x')
avg_x=np.average(error_x_r1)
med_x=np.median(error_x_r1)
ax1.axhline(avg_x,0,100,color='r',linestyle=':',label='Average {}'.format(format(avg_x,".2")))
ax1.axhline(med_x,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_x,".2")))
ax1.legend()

ax2.set_title('B')
ax2.scatter(x,error_y_r1,marker='x')
avg_y=np.average(error_y_r1)
med_y=np.median(error_y_r1)
ax2.axhline(avg_y,0,100,color='r',linestyle=':',label='Average {}'.format(format(avg_y,".2")))
ax2.axhline(med_y,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_y,".2")))
ax2.legend()

ax3.set_title('C')
ax3.scatter(x,abs_error_r1,marker='x')
avg_abs=np.average(abs_error_r1)
med_abs=np.median(abs_error_r1)
#ax1.hist(abs_error_r1,30)
#ax1.vlines(avg_abs,0,8,color='r',label='Average')
ax3.axhline(avg_abs,100,color='r',linestyle=':',label='Average {}'.format(format(avg_abs,".2")))
ax3.axhline(med_abs,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_abs,".2")))
ax3.legend()

fig.supxlabel('Index of 100 Images')
fig.supylabel('Error (pixels)')
plt.savefig('single_test.png',dpi=400)
'''
'''
fig, (ax1,ax2)=plt.subplots(1,2,sharey=True,sharex=False,tight_layout=True)

ax1.set_title('A')
ax1.scatter(x,abs_error_r1,marker='x')
avg_abs=np.average(abs_error_r1)
med_abs=np.median(abs_error_r1)
#ax1.hist(abs_error_r1,30)
#ax1.vlines(avg_abs,0,8,color='r',label='Average')
ax1.axhline(avg_abs,100,color='r',linestyle=':',label='Average {}'.format(format(avg_abs,".2")))
ax1.axhline(med_abs,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_abs,".2")))
ax1.set_xlabel('Image Stack')
ax1.legend()

ax2.set_title('B')
ax2.hist(abs_error_r1,bins=30,orientation='horizontal')
ax2.axhline(avg_abs,100,color='r',linestyle=':',label='Average {}'.format(format(avg_abs,".2")))
ax2.axhline(med_abs,0,100,color='k',linestyle=':',label='Median {}'.format(format(med_abs,".2")))
ax2.set_xlabel('Count')

fig.supylabel('Absolute Error (pixels)')
plt.savefig('single_histo.png',dpi=400)
'''
