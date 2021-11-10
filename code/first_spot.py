import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

file=tf.tifffile.imread(files="spot.tif")

def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def centroid(file):
    sum_int=0
    x_pos=0
    y_pos=0
    for i in range(len(file)):
        for j in range(len(file)):
            sum_int=sum_int+file[i,j]
            x_pos=x_pos+(file[i,j]*i)
            y_pos=y_pos+(file[i,j]*j)
    loc=np.array(((x_pos/sum_int),(y_pos/sum_int)))
    return loc

spot=centroid(file)
plt.plot(spot[0],spot[1],'x',color='red')
plt.imshow(file,cmap='binary_r')
plt.colorbar()
plt.show()