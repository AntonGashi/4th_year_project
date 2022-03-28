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


@jit(nopython=True)
def local_max(file):
    results=np.zeros(shape=(1,2))
    iter_=2
    loop=int(file.shape[1]/iter_)
    for i in range(file.shape[0]):
        filemax_thresh=(np.amax(file[i]))-1
        for j in range(loop-1):
            for k in range(loop-1):
                max_=np.amax(file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
                if max_>filemax_thresh:
                    x=np.flip(np.argwhere(file[i]==max_))
                    results=np.vstack((results,x))
                else:
                    continue
    return results

@jit(nopython=True)
def centriod(file,loc_max,box_size):
    loc=np.zeros_like(loc_max)
    index=round((box_size-1)/2)
    for i in range(loc_max.shape[0]):
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

    groundtruth_r8=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r4.00/groundtruth.csv"))+24.5
    box_size=np.arange(3,num_of_boxes+1,2,dtype=int)
    loc_centroid=np.zeros([len(box_size),loc_max.shape[0],2])
    error=np.zeros([len(box_size),loc_max.shape[0],2])
    avg_diff=np.zeros([len(box_size),2])
    for j in range(len(box_size)):
        loc_centroid[j]=centriod(file_,loc_max,box_size[j])
    for k in range(len(box_size)):
        error[k]=np.absolute(np.subtract(loc_centroid[k],groundtruth_r8))
        avg_diff[k,0],avg_diff[k,1]=np.average(error[k,0:,0]),np.average(error[k,0:,1])
    fig_one_box=np.argmin(avg_diff[:,0])

    plt.xlabel('Box Size (in odd increments)')
    plt.ylabel('Differance From Ground Truth (absolute value)')
    plt.plot(box_size,avg_diff[0:,0],label='X Values',linestyle='-')
    plt.plot(box_size,avg_diff[0:,1],label='Y Values',linestyle='--')
    plt.ylim(0)
    plt.xlim(box_size[0],box_size[-1])
    plt.hlines(avg_diff[fig_one_box,0],box_size[0],box_size[-1],linestyle=':',color='k',label='Minimum at {}'.format(np.round((avg_diff[fig_one_box,0]),4)))
    plt.legend(shadow=True,fancybox=True)
    stop=time.perf_counter()
    print('Total Time Taken {}s'.format(np.round((stop-start),2)))
    plt.show()
    pass

def gauss(x,avg,var):
    return 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(x-avg)**2/var) 
#enter like: folder="r1.00 r1.41 r2.00 r2.83 r4.00 r5.66 r8.00"
folder="r4.00"

#file,ground_truth=input("Perfect Spots {}/Perfect Spots {}.tif".format(folder,folder),"Perfect Spots {}/groundtruth.csv".format(folder))
#file,ground_truth=input("Multiple Spots/AF647_npc_1frame.tif")

#points=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))+24.5
#display(num_of_boxes(max is 49 as picture size is 50x50),picture(max is 99))

    
file_=tf.TiffSequence("Noisy Spots/Noisy Spots r8.00/Noisy Spots r8.00.tif").asarray().reshape(100,50,50)

loc_max=np.delete((local_max(file_)),0,0)
display(49,0,file_)

###
#temp
###


file_r2_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)
file_r283_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.83/Noisy Spots r2.83.tif").asarray().reshape(100,50,50)
file_r4_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r4.00/Noisy Spots r4.00.tif").asarray().reshape(100,50,50)
file_r566_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r5.66/Noisy Spots r5.66.tif").asarray().reshape(100,50,50)
file_r8_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r8.00/Noisy Spots r8.00.tif").asarray().reshape(100,50,50)

file_r1=tf.TiffSequence("Perfect Spots r1.00/Perfect Spots r1.00.tif").asarray().reshape(100,50,50)
groundtruth_r1=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.00/groundtruth.csv"))+24.5


file_r141=tf.TiffSequence("Perfect Spots r1.41/Perfect Spots r1.41.tif").asarray().reshape(100,50,50)
groundtruth_r141=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r1.41/groundtruth.csv"))+24.5

file_r2=tf.TiffSequence("Perfect Spots r2.00/Perfect Spots r2.00.tif").asarray().reshape(100,50,50)
groundtruth_r2=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r2.00/groundtruth.csv"))+24.5
file_r2_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.00/Noisy Spots r2.00.tif").asarray().reshape(100,50,50)


file_r283=tf.TiffSequence("Perfect Spots r2.83/Perfect Spots r2.83.tif").asarray().reshape(100,50,50)
groundtruth_r283=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r2.83/groundtruth.csv"))+24.5
file_r283_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r2.83/Noisy Spots r2.83.tif").asarray().reshape(100,50,50)


file_r4=tf.TiffSequence("Perfect Spots r4.00/Perfect Spots r4.00.tif").asarray().reshape(100,50,50)
groundtruth_r4=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r4.00/groundtruth.csv"))+24.5
file_r4_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r4.00/Noisy Spots r4.00.tif").asarray().reshape(100,50,50)


file_r566=tf.TiffSequence("Perfect Spots r5.66/Perfect Spots r5.66.tif").asarray().reshape(100,50,50)
groundtruth_r566=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r5.66/groundtruth.csv"))+24.5
file_r566_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r5.66/Noisy Spots r5.66.tif").asarray().reshape(100,50,50)


file_r8=tf.TiffSequence("Perfect Spots r8.00/Perfect Spots r8.00.tif").asarray().reshape(100,50,50)
groundtruth_r8=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))+24.5
file_r8_noise=tf.TiffSequence("Noisy Spots/Noisy Spots r8.00/Noisy Spots r8.00.tif").asarray().reshape(100,50,50)


start=time.perf_counter()

no_of_boxes=11

calculated_points_r1=centriod(file_r1,loc_max,no_of_boxes)
sub_of_cal_ground_r1=groundtruth_r1.reshape(2,100)-calculated_points_r1.reshape(2,100)
abs_error_r1=np.zeros(100)


calculated_points_r141=centriod(file_r141,loc_max,no_of_boxes)
sub_of_cal_ground_r141=groundtruth_r141.reshape(2,100)-calculated_points_r141.reshape(2,100)
abs_error_r141=np.zeros(100)


calculated_points_r2=centriod(file_r2_noise,loc_max,no_of_boxes)
sub_of_cal_ground_r2=groundtruth_r2.reshape(2,100)-calculated_points_r2.reshape(2,100)
abs_error_r2=np.zeros(100)

calculated_points_r283=centriod(file_r283_noise,loc_max,no_of_boxes)
sub_of_cal_ground_r283=groundtruth_r283.reshape(2,100)-calculated_points_r283.reshape(2,100)
abs_error_r283=np.zeros(100)

calculated_points_r4=centriod(file_r4_noise,loc_max,no_of_boxes)
sub_of_cal_ground_r4=groundtruth_r4.reshape(2,100)-calculated_points_r4.reshape(2,100)
abs_error_r4=np.zeros(100)

calculated_points_r566=centriod(file_r566_noise,loc_max,no_of_boxes)
sub_of_cal_ground_r566=groundtruth_r566.reshape(2,100)-calculated_points_r566.reshape(2,100)
abs_error_r566=np.zeros(100)

calculated_points_r8=centriod(file_r8_noise,loc_max,no_of_boxes)
sub_of_cal_ground_r8=groundtruth_r8.reshape(2,100)-calculated_points_r8.reshape(2,100)
abs_error_r8=np.zeros(100)
stop=time.perf_counter()

timetaken=stop-start

for i in range(len(groundtruth_r1)):
    abs_error_r1[i]=np.linalg.norm(sub_of_cal_ground_r1[:,i])
    abs_error_r141[i]=np.linalg.norm(sub_of_cal_ground_r141[:,i])
    abs_error_r2[i]=np.linalg.norm(sub_of_cal_ground_r2[:,i])
    abs_error_r283[i]=np.linalg.norm(sub_of_cal_ground_r283[:,i])
    abs_error_r4[i]=np.linalg.norm(sub_of_cal_ground_r4[:,i])
    abs_error_r566[i]=np.linalg.norm(sub_of_cal_ground_r566[:,i])
    abs_error_r8[i]=np.linalg.norm(sub_of_cal_ground_r8[:,i])




'''
x=np.linspace(0,100,100)

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,sharey=True,sharex=False,tight_layout=True)

print(format(timetaken,".2"))



#fig.supxlabel('Index Of Images')
#fig.supylabel('Absolute Error (pixels)')

fig.supxlabel('Absolute Error (Pixels)')
fig.supylabel('Absolute Amount')

#ax1.scatter(x,abs_error_r2,marker='x')
avg_abs=np.average(abs_error_r2)
med_abs=np.median(abs_error_r2)
var_abs=np.var(abs_error_r2)
hist,bins,patches=ax1.hist(abs_error_r2,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax1.plot(bins,pdf_y,label='Fitted Gaussian')
ax1.vlines(avg_abs,0,8,color='r',label='Average')
#ax1.axhline(avg_abs,100,color='r',linestyle=':',label='Average')
#ax1.axhline(med_abs,0,100,color='k',linestyle=':',label='Median')
ax1.set_title('R2.00')
#print(avg_abs,var_abs)

ax2.set_title('R2.83')
#ax2.scatter(x,abs_error_r283,marker='x')
avg_abs=np.average(abs_error_r283)
med_abs=np.median(abs_error_r283)
var_abs=np.var(abs_error_r283)
hist2,bins2,patches2=ax2.hist(abs_error_r283,30)
pdf_y2=gauss(bins2,avg_abs,var_abs)
ax2.plot(bins2,pdf_y2)
ax2.vlines(avg_abs,0,8,color='r')
#ax2.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax2.axhline(med_abs,0,100,color='k',linestyle=':')

ax3.set_title('R4.00')
#ax3.scatter(x,abs_error_r4,marker='x')
avg_abs=np.average(abs_error_r4)
med_abs=np.median(abs_error_r4)
var_abs=np.var(abs_error_r4)
hist,bins,patches=ax3.hist(abs_error_r4,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax3.plot(bins,pdf_y)
ax3.vlines(avg_abs,0,8,color='r')
#ax3.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax3.axhline(med_abs,0,100,color='k',linestyle=':')

ax4.set_title('R5.66')
#ax4.scatter(x,abs_error_r566,marker='x')
avg_abs=np.average(abs_error_r566)
med_abs=np.median(abs_error_r566)
var_abs=np.var(abs_error_r566)
hist,bins,patches=ax4.hist(abs_error_r566,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax4.plot(bins,pdf_y)
ax4.vlines(avg_abs,0,8,color='r')
#ax4.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax4.axhline(med_abs,0,100,color='k',linestyle=':')

ax5.set_title('R8.00')
#ax5.scatter(x,abs_error_r8,marker='x')
avg_abs=np.average(abs_error_r8)
med_abs=np.median(abs_error_r8)
var_abs=np.var(abs_error_r8)
hist,bins,patches=ax5.hist(abs_error_r8,30)
pdf_y=gauss(bins,avg_abs,var_abs)
ax5.plot(bins,pdf_y)
ax5.vlines(avg_abs,0,8,color='r')
#ax5.axhline(avg_abs,0,100,color='r',linestyle=':')
#ax5.axhline(med_abs,0,100,color='k',linestyle=':')


plt.delaxes(ax6)
fig.legend(loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig('noise_cen_scatter_11.png',dpi=400)

'''

'''
x=np.linspace(0,100,100)

fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8))=plt.subplots(2,4,sharey=True,sharex=False,tight_layout=True)

print(format(timetaken,".2"))


#fig.supxlabel('Index Of Images')
#fig.supylabel('Absolute Error (pixels)')

fig.supxlabel('Absolute Error (Pixels)')
fig.supylabel('Absolute Amount')



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
#plt.show()
plt.savefig('distro_centriod_5.png',dpi=400)
'''
