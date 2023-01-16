import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon

import observations as obs

def triangles(ax,x,y,height,width,color='green',**kwargs):
    '''draw a triangle on an axis'''
    lower_left=[x-0.5*width,y-0.5*height]
    lower_right=[x+0.5*width,y-0.5*height]        
    middle=[x,y-(0.5*height+width)]
    points=[lower_left,lower_right,middle]            
    ax.add_patch(Polygon(points,closed=True,
                edgecolor = color,fill=True,lw=1))

def boxes(ax,x,y,height,width,color='green',**kwargs):
    '''draws a series of open boxes on an axis'''
    for i in range(len(x)):
        lower_left=[x[i]-0.5*width[i],y[i]-0.5*height[i]]
        ax.add_patch(Rectangle(lower_left, width[i], height[i]))

def boxes_with_uplim(ax,x,y,height,width,color='green',uplim=False):
    '''draws a series of open boxes on an axis where uplim
        is true instead draws a polygon'''
    N=len(x)
    if type(uplim)==np.bool:
        uplim=N*[uplim]
    for i in range(N):
        lower_left=[x[i]-0.5*width[i],y[i]-0.5*height[i]]
        upper_left=[x[i]-0.5*width[i],y[i]+0.5*height[i]]
        upper_right=[x[i]+0.5*width[i],y[i]+0.5*height[i]]
        lower_right=[x[i]+0.5*width[i],y[i]-0.5*height[i]]
        middle=[x[i],y[i]-(0.5*height[i]+width[i])]
        points=[lower_left,upper_left,upper_right,lower_right]
        ax.add_patch(Polygon(points,closed=True,
            edgecolor = color,fill=False,lw=1))        
        if uplim[i]==True:
            points=[lower_left,lower_right,middle]            
            ax.add_patch(Polygon(points,closed=True,
                edgecolor = color,fill=True,lw=1))

def myboxplot(xarray,ydistribution,axis,upper_limits=False,color='green'):
    '''a version of a box plot which can show upper/lower limits'''
    #TODO: add uplims, probably need to pass uplim distribution
    N=len(xarray)
    if isinstance(ydistribution,np.ndarray):
        ydistribution=np.transpose(ydistribution)
    y=np.zeros(N)
    percents=100*stats.norm().cdf((-2,-1,0,1,2)) #need percentage
    one_sigma=np.zeros((2,N))
    two_sigma=np.zeros((2,N))   
    percentile=np.zeros((N,5))
    for i,x in enumerate(xarray):
        yvals=ydistribution[i]
        levels=np.percentile(yvals,percents)
        two_sigma[0,i]=levels[2]-levels[0] #lower value
        one_sigma[0,i]=levels[2]-levels[1]
        y[i]=levels[2]
        one_sigma[1,i]=levels[3]-levels[2] 
        two_sigma[1,i]=levels[4]-levels[2] #upper value
    
    boxes(axis,xarray,y,N*[0.02],N*[0.04],fill=True) 
    box_y=y+0.5*(one_sigma[1]-one_sigma[0])
    box_h=one_sigma[0]+one_sigma[1]
    uplimbar,uplimbox=False,False
    if upper_limits:
        uplimbar=np.zeros(len(upper_limits))
        uplimbox=np.zeros(len(upper_limits))
        for i,up in enumerate(upper_limits):
            if up > 0.5:
                two_sigma[0][i]=0
                triangles(axis,xarray[i],y[i],0.02,0.04)
                box_y[i]=y[i]+0.5*(one_sigma[1][i])
                box_h[i]=one_sigma[1][i]
            elif up > 0.16:
                two_sigma[0][i]=one_sigma[0][i]
                uplimbox[i]=1
            elif up > 0.05:
                y[i]=y[i]+two_sigma[1][i]
                two_sigma[0][i]=two_sigma[0][i]+two_sigma[1][i]
                uplimbar[i]=1

    boxes_with_uplim(axis,xarray,box_y,box_h,N*[0.04],uplim=uplimbox,
        color='green')
    axis.errorbar(xarray,y,yerr=two_sigma,fmt='none',ecolor=color,
            capsize=3.0,capthick=2,uplims=uplimbar)

def test(boxplot=False):
    Nbins=6
    x=np.random.random(size=500)
    y=np.random.random(size=500)
    up = y < 0.66*x 
    #make arrays for each bin
    xedges=np.linspace(0,1,Nbins+1)
    x_cent=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0
    result,prob_uplim=[],[]
    for i in range(Nbins):
        bin=bin_number==i
        result.append(y[bin_number==i])
        prob_uplim.append(np.sum(up[bin])/np.sum(bin))
    res_cent=result
    f,axis=plt.subplots()
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    axis.scatter(x,y,marker='.',s=5, alpha=0.3)
    axis.scatter(x[up],y[up],marker='v',s=5,alpha=0.3)
#    res_cent,x_cent,prob_uplim=obs.bootstrap_resample(x,y,xrange=[0,1],Nbins=4,
#        Nsamples=100,upper_limits=up) 
    if boxplot:
        axis.boxplot(res_cent,positions=x_cent,widths=0.06,whis=(2.5,97.5),
            showfliers=True,manage_ticks=False,patch_artist=False,
            boxprops=dict(color='g'),whiskerprops=dict(color='g'),capprops=dict(color='g'))  
    myboxplot(x_cent,res_cent,axis,upper_limits=prob_uplim)
    print("Note: boxplot box is 25-75 while myboxplot is 16-84")
    plt.savefig('test2.pdf')
    plt.show()

test()