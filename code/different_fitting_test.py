from os import path
import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit
import pandas as pd
import scipy.optimize as sco

start_centroid=time.perf_counter()

def input(path_to_tiff,path_to_groundtruth):
    file=tf.TiffSequence("{}".format(path_to_tiff)).asarray()
    if len(file.shape)>4:
        print('ERROR')

    ground_truth=pd.DataFrame.to_numpy(pd.read_csv("{}".format(path_to_groundtruth)))
    return file,ground_truth


#@jit(nopython=True)
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

#@jit(nopython=True)
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


#enter like: folder="r1.00 r1.41 r2.00 r2.83 r4.00 r5.66 r8.00"
folder="r8.00"
file,ground_truth=input("Perfect Spots {}/spot014.tif".format(folder),"Perfect Spots {}/groundtruth.csv".format(folder))
#file,ground_truth=input("Perfect Spots {}/Perfect Spots {}.tif".format(folder,folder),"Perfect Spots {}/groundtruth.csv".format(folder))
#file,ground_truth=input("Multiple Spots/AF647_npc_1frame.tif")
loc_max=np.delete((local_max(file)),0,0)
points=np.add(np.full((ground_truth.shape),24.5),ground_truth)

end_centroid=time.perf_counter()
finish=end_centroid-start_centroid


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)
def triangle(x,m,a):
    return (m*(x-a)*np.sign((a-x))+m*a)
    
def triangle2(x,m,a):
    fit=np.zeros_like(x)
    for i in range(len(x)):
        if np.absolute(x[i])>=1:
            fit[i]=0
        else:
            fit[i]=1-np.absolute(x[i])
    return fit

image=0

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(8,4),constrained_layout=True)

start_gauss=time.perf_counter()

###1-d gaussian
location=centriod(file,loc_max,9)
x=np.arange(0,50,1)
popt, pcov = sco.curve_fit(gaussian, x,file[image,24,:])
ym=gaussian(x,popt[0],popt[1],popt[2])
end_gauss=time.perf_counter()
finish_gauss=end_gauss-start_gauss
ax1.plot(file[image,24,:],':')
ax1.plot(x,ym)

###1-d triangle
start_tri=time.perf_counter()
x2=np.arange(0,50,1)
popt2, pcov2 = sco.curve_fit(triangle, x2,file[image,24,:],method='lm')
ym=triangle(x2,popt2[0],popt2[1])
end_tri=time.perf_counter()
finish_tri=end_tri-start_tri
ax2.plot(file[image,24,:],':')
ax2.plot(x,ym)
print('gaussian fit guess [24 {}] and took {}s'.format(popt[1],finish_gauss),'\ngroundtruth :{}'.format(points[image]),'\ncentroid guess :{} and took {}s'.format(location[0],finish),'\ntriangle guess :{} and took {}s'.format(popt2[1],finish_tri))

gauss_diff=np.absolute(points[0,0]-popt[1])
centroid_diff=np.absolute(points[0,0]-location[0,0])
triangle_diff=np.absolute(points[0,0]-popt2[1])


ax3.plot(finish_gauss,gauss_diff,'x',label='gaussain difference')
ax3.plot(finish,centroid_diff,'x',label='centroid difference')
ax3.plot(finish_tri,triangle_diff,'x',label='triangle difference')
ax3.legend(shadow=True,fancybox=True)
plt.show()