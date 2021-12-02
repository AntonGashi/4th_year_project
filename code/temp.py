from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf
from numba import jit
import time
import pandas as pd

start=time.perf_counter()

image=0
file=tf.TiffSequence("Perfect Spots r8.00/spot014.tif").asarray()
groundtruth=pd.DataFrame.to_numpy(pd.read_csv("Perfect Spots r8.00/groundtruth.csv"))
print('Ground Truth: ',groundtruth[14,0]+24.5)

@jit(nopython=True)
def triangle(centre,half_base,height,magnif):
    x=np.arange((centre-half_base),(centre+half_base),1)
    y=np.arange(0,height,(height/(len(x)/2)))
    y=np.append(y,y[::-1])
    return x,y
@jit(nopython=True)
def res(file,centre,half_base,height,image,magnif):
    tri_x,tri_y=triangle(centre,half_base,height,magnif)
    area=np.zeros(len(file[image,24,:])*magnif,dtype=np.float64)
    for i in range((centre-half_base),(centre+half_base)):
        area[i]=tri_y[i-(centre-half_base)]
    X = np.arange(0, magnif*len(file[image,24,:]), magnif)
    X_new = np.arange(magnif*len(file[image,24,:]))
    Y_new = np.interp(X_new, X, file[image,24,:])
    return np.absolute(Y_new-area),Y_new,area
@jit(nopython=True)
def opt(file,image,magnif):
    output=np.zeros((10*magnif,7*magnif,26000))
    for i in range(20*magnif,30*magnif):
        for j in range(3*magnif,10*magnif):
            for k in range(40000,66000):
                res1,Y_new,area=res(file,i,j,k,image,magnif)
                output[i-20*magnif,j-3*magnif,int((k-40000))]=np.sqrt(np.mean(res1**2))
        print(i)
    return output,(np.where(output==np.min(output)))
@jit(nopython=True)
def opt2(file,image,magnif):
    height=66000
    sample=30
    output,test=np.zeros(sample),np.random.randint((20*magnif),(40*magnif),sample)
    for i in range(sample):
        res1,Y_new,area=res(file,test[i],6*magnif,height,image,magnif)
        output[i]=np.sqrt(np.mean(res1**2))
    opt_cen=test[np.argmin(output)]
    output2=np.zeros((10*magnif-3*magnif))
    for j in range(3*magnif,10*magnif):
        res1,Y_new,area=res(file,opt_cen,j,height,image,magnif)
        output2[j-3*magnif]=np.sqrt(np.mean(res1**2))
    opt_base=(np.argmin(output2)+3*magnif)
    test2=np.random.randint((opt_cen-(0.5*magnif)),(opt_cen+(0.5*magnif)),sample)
    for i in range(sample):
        res1,Y_new,area=res(file,test2[i],opt_base,height,image,magnif)
        output[i]=np.sqrt(np.mean(res1**2))
    opt_cen=test2[np.argmin(output)]
    return opt_cen,(np.argmin(output2)+3*magnif),height

#output,min_vals=opt(file,image,magnif)
#print(min_vals)

magnif=6
values=opt2(file,image,magnif)

centre,half_base,height=values[0],values[1],values[2]
adj_c=centre/magnif
print('Estimated Centre: ',adj_c)
stop=time.perf_counter()
print('Time Taken: ',stop-start)

tri_x,tri_y=triangle(centre,half_base,height,magnif)
res,Y_new,area=res(file,centre,half_base,height,image,magnif)

fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(8,4),constrained_layout=True)

ax1.plot(tri_x,tri_y)
ax1.plot(Y_new)
ax1.set_xlim(0,50*magnif)
ax1.set_ylim(0,65000)

x2=np.arange(50*magnif)
ax2.bar(x2,Y_new)
ax2.bar(x2,area,alpha=0.5)
ax2.bar(x2,res)
ax2.set_ylim(0,65000)
ax2.set_xlim(0,50*magnif)
plt.show()