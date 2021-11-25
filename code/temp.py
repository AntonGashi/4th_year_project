from matplotlib import widgets
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf

image=0
file=tf.TiffSequence("Perfect Spots r8.00/spot014.tif").asarray()

box_size=11

def triangle(centre,half_base,height,size):
    x=np.linspace((centre-half_base),(centre),size)
    y=np.linspace(0,height,size)
    return x,y

def tri_area(x,y):
    area=np.zeros(len(x)-1)
    base=np.absolute(x[0]-x[1])
    for i in range(len(x)-1):
        box_heigth=y[i]
        box_area=base*box_heigth
        tri_area=0.5*base*(y[i+1]-y[i])
        area[i]=box_area+tri_area
    return area


tri_x,tri_y=triangle(24,7,60000,box_size)
total_image_area=np.sum(file[image,24,20:30])
tri_area=tri_area(tri_x,tri_y)
res=np.absolute(tri_area-file[image,24,20:30])


fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(8,4),constrained_layout=True)

ax1.plot(tri_x,tri_y)
ax1.plot(file[image,24,:])
ax1.set_xlim(0,50)

x=np.arange(20,30,1)
ax2.bar(x,res)
ax2.set_xlim(0,50)
plt.show()