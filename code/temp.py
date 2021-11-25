import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tifffile as tf

image=0
file=tf.TiffSequence("Perfect Spots r8.00/spot014.tif").asarray()

def triangle3(x,max_,b):
    fit=np.zeros_like(x)
    for i in range(len(x)):
        if np.absolute(x[i])>=1:
            fit[i]=0
        else:
            fit[i]=1-np.absolute(x[i])
    return (fit-b)*max_

def triangle4(x,cen,max_,b1,b2):
    fit=np.zeros_like(x)
    inc1_const=max_/(cen-b1)
    inc2_const=max_/(b2-cen)
    inc1,inc2=inc1_const,max_
    for i in range(len(fit)):
        if cen>=i>=b1:
            fit[i]=inc1
            inc1+=inc1_const
        elif b2>=i>cen:
            fit[i]=inc2
            inc2-=inc2_const

    return fit


x2=np.linspace(0,51,50)
ym=triangle4(x2,23,55000,15,35)
print(ym)

#popt2, pcov2 = sco.curve_fit(triangle4, x2,file[image,24,:],method='lm')
#print(popt2)
#ym=triangle4(x2,popt2[0],popt2[1],popt2[2],popt2[3])

plt.plot(x2,ym)
plt.plot(file[image,24,:],':')
plt.show()