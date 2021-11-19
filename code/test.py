from os import path
import numpy as np
from numpy.core.fromnumeric import reshape
import tifffile as tf
import matplotlib.pyplot as plt
import time
from numba import jit
import pandas as pd

class spot_finder:
    def __init__(self,folder,number_of_boxes,picture):
        self.folder=folder
        self.number_of_boxes=number_of_boxes
        self.picture=picture
        self.file,self.ground_truth=input(self)
        self.points=np.add(np.full((self.ground_truth.shape),24.5),self.ground_truth)
        self.loc_max=np.delete((spot_finder.local_max(self.file)),0,0)
        self.display=spot_finder.display(self)

    def input(self):
        path_to_tiff,path_to_groundtruth=("Perfect Spots {}/Perfect Spots {}.tif".format(self.folder,self.folder),"Perfect Spots {}/groundtruth.csv".format(self.folder))
        file=tf.TiffSequence("{}".format(path_to_tiff)).asarray()
        ground_truth=pd.DataFrame.to_numpy(pd.read_csv("{}".format(path_to_groundtruth)))
        if len(file.shape)>4:
            print('ERROR')
        elif len(file.shape)==4:
            file=file.reshape(file.shape[1],file.shape[2],file.shape[3])
            return file,ground_truth
        return file,ground_truth

    @jit(nopython=True)
    def local_max(self):
        results=np.empty(shape=(1,2))
        iter_=2
        loop=int(self.file.shape[1]/iter_)
        for i in range(self.file.shape[0]):
            filemax_thresh=np.amax(self.file[i])-1
            for j in range(loop-1):
                for k in range(loop-1):
                    max_=np.amax(self.file[i,(j*iter_):((j+1)*iter_),(k*iter_):((k+1)*iter_)])
                    if max_>filemax_thresh:
                        x=np.flip(np.argwhere(self.file[i]==max_))
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

    def display(self):
        start=time.perf_counter()

        box_size=np.zeros(self.num_of_boxes,dtype=int)
        loc_centroid=np.zeros([self.num_of_boxes,100,2])
        error=np.zeros([self.num_of_boxes,100,2])
        avg_diff=np.zeros([self.num_of_boxes,2])

        for i in range(self.num_of_boxes):
            box_size[i]=int((2*(i+1))+1)
        for j in range(self.num_of_boxes):
            loc_centroid[j]=spot_finder.centriod(self.file,self.loc_max,box_size[j])
        for k in range(self.num_of_boxes):
            error[k]=np.absolute(np.subtract(loc_centroid[k],self.points))
            avg_diff[k,0],avg_diff[k,1]=np.average(error[k,0:,0]),np.average(error[k,0:,1])
        fig_one_box=np.argmin(avg_diff[:,0])
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4),constrained_layout=True)
        ax1.title.set_text(('Plot of Predicted Spot For self.picture {} of {}'.format(self.picture,self.file.shape[0])))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.imshow(self.file[self.picture],cmap='binary_r')
        ax1.plot(self.points[self.picture,0],self.points[self.picture,1],'x',color='r',label='Ground Truth')
        ax1.plot(loc_centroid[fig_one_box,0,0],loc_centroid[fig_one_box,0,1],'x',color='k',label=('Centroid Box Size = {}'.format(box_size[fig_one_box])))
        ax1.legend(shadow=True,fancybox=True)
        ax2.title.set_text('Average Difference in Position')
        ax2.set_xlabel('Box Size (in odd increments)')
        ax2.set_ylabel('Differance From Ground Truth (absolute value)')
        ax2.plot(avg_diff[0:,0],label='X Values',linestyle='-')
        ax2.plot(avg_diff[0:,1],label='Y Values',linestyle='--')
        ax2.set_ylim(0)
        ax2.legend(shadow=True,fancybox=True)
        stop=time.perf_counter()
        print('Total Time Taken {}s'.format(np.round((stop-start),2)))
        plt.show()
        pass

    #enter like: folder="r1.00 r1.41 r2.00 r2.83 r4.00 r5.66 r8.00"
    #display(num_of_boxes(max is 24),picture)

first=spot_finder("r1.00",24,0)
