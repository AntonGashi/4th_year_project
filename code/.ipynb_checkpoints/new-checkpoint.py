import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import math


file_=tf.TiffSequence("Triangle_01.tif").asarray().reshape(100,50,50)
file_=np.sum(file_,1)
print(file_.shape)


def triangle(centre,halfbase,height):
    return [centre-halfbase,centre,centre+halfbase],[0,height,0]


def area(tri,file):
    grad=(tri[1][1]-tri[1][0])/(tri[0][1]-tri[0][0])
    start,stop=math.floor(tri[0][0]),math.ceil(tri[0][2])
    area=np.zeros(file.shape[1])
    for i in range(start,stop+1):
        if i==start:
            height=((i+1)-tri[0][0])*grad
            area[i]=0.5*((i+1)-tri[0][0])*(height)
        elif i<math.floor(tri[0][1]):
            area[i]=height+(0.5*grad)
            height=height+grad
        elif i==math.floor(tri[0][1]):
            area1=((tri[0][1]-i)*height)+(0.5*(tri[0][1]-i)*(tri[1][1]-height))
            area2=(((i+1)-tri[0][1])*height)+(0.5*((i+1)-tri[0][1])*(tri[1][1]-height))
            area[i]=area1+area2
        elif stop-1>i>math.floor(tri[0][1]):
            height=height-grad
            area[i]=height+(0.5*grad)
        elif i==stop:
            height=((i-1)-tri[0][2])*grad
            area[i-1]=0.5*((i-1)-tri[0][2])*height
    return area


def res(file,area_,tri,image):
    res=np.zeros_like(file[image])
    res=file[image]
    res=res-area_
    return np.sum(res**2)

def test(file_):
    array_of_res=np.zeros(100)
    array_of_centre=np.linspace(22,28,100)
    for i in range(100):
        print(i)
        #tri_=triangle(array_of_centre[i],5,8.5e4)
        #area_=area(tri_,file_)
        #array_of_res[i]=res(file_,area_,tri_,0)
    return array_of_centre 

print(test(file_))

#plt.plot(tri_[0],tri_[1])
#plt.plot(file_[0])
#plt.show()
