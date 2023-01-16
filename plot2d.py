import numpy as np 
import matplotlib.pyplot as plt 


def myhist2d(ax,x,y,Nbins=25,range=None):
    h,xed,yed=np.histogram2d(x,y,bins=Nbins,range=range)
    h=np.transpose(h)
    hmask = np.ma.masked_where( 10 > h, h)
    ax.imshow(hmask,origin='lower',extent=[xed[0],xed[-1],yed[0],yed[-1]])


x=np.random.normal(size=5000)
y=np.random.normal(size=5000)

f,axs=plt.subplots(nrows=1,ncols=2)
axs[0].hist2d(x,y,bins=25,range=[[-2,2],[-2,2]],cmin=10)
myhist2d(axs[1],x,y,range=[[-2,2],[-2,2]])

plt.show()