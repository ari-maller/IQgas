import numpy as np 
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats
import IQ 

def check_finite(array):
    bad=(np.invert(np.isfinite(array))).sum()
    if bad > 0:
        print(f"This array has {bad} non finite values")
        return True
    else:
        return False

def perpdis(x1,y1,m,b):
    #returns the perpinduclar distance between the points x1,y1 and the line given by y=mx+b
    dis=(-m*x1+y1-b)/np.sqrt(m*m+1)
    return dis

def fitmedian(x,y,xrange=None,axis=None,**kwargs):
    if np.any(np.isnan(y)):
        print(f"contains {(np.isnan(y)).sum()} NaN in y")
    N=10
    if xrange==None:
        xsort=np.sort(x)
        xrange=[xsort[int(0.1*x.size)],xsort[int(0.9*x.size)]]
#        print("Taking x range to be {:.2f} to {:.2f}".format(xrange[0],xrange[1]))
    xedges=np.linspace(xrange[0],xrange[1],N)
    xmids=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0 
    med=np.zeros(N-1)
    for i in range(N-1):
        med[i]=np.median(y[bin_number==i]) #gives nan if empty set   
    if check_finite(med):
        print("A bin had no members, try chaning the range or number of bins")
        sys.exit(1)
    p=np.polyfit(xmids,med,1)
    if axis:
        axis.plot(xmids,p[0]*xmids+p[1],**kwargs)
    return p

def plotdistribution(axis,x,y,**kwargs):
    if len(x) < 1500:
        axis.scatter(x,y,marker='o',s=3)
    else:
        IQ.hist2dplot(axis,x,y,**kwargs)

def field_intex(field):
    #return a label for a given field
    labels={'logMstar':r"M_{*}",
            'logMHI': r"M_{HI}",
            'logMH2':r"M_{H_2}",
            'logMgas':r"M_{gas}",
            'logMhalo':r"M_{halo}",
            'log_SFR':r"SFR",
            'r_disk': r"R_{50}",
            'logRstar': r"R_{50}"}
    try:
        ans=labels[field]
    except:
        print(f"field {field} not known")
    return ans

def diff_labels(xfield,yfield):
    xlabel=field_intex(xfield)
    ylabel=IQ.field_labels(yfield)
    ylabel=ylabel[0:ylabel.find('\,')]
    difflabel=r'$\Delta \overline{'+field_intex(yfield)+ \
                '}('+field_intex(xfield)+')$'
    return difflabel

#plots
def two_diff(data,xfield,yfield1,yfield2,sim='',xfield2='',
        cent=False,sat=False):
    # if cent and sat false plot all, if one true plot only that one
    # if both true then plot each one separetely 
    if not xfield2:
        xfield2=xfield
    if cent:
        cent=(data['central']==True).to_numpy()
        data=data[cent]
    if sat:
        keep=(data['central']==False).to_numpy()
    if sim=='xCOLDGASS':
        uplim=(data['uplim']==True).to_numpy()
        detect=np.invert(uplim)
        data=data[detect]
    else:
        keep=np.logical_and(data['log_SFR'] > -2,data['logMH2'] > 7.5)
        data=data[keep]
    data.reset_index(inplace=True)

    xtit1=IQ.field_labels(xfield)
    xtit2=IQ.field_labels(xfield2)
    ytit1=IQ.field_labels(yfield1)
    ytit2=IQ.field_labels(yfield2)
    xvals1=data[xfield]
    xvals2=data[xfield2]
    yvals1=data[yfield1]
    yvals2=data[yfield2]

    #create axes for the figure
    f=plt.figure(figsize=(6,4))
    ax1=f.add_subplot(221)
    ax2=f.add_subplot(223)
    ax3=f.add_axes([0.6,0.25,0.35,0.5])

    #first plot scatter and median line for yfield 1
    plotdistribution(ax1,xvals1,yvals1)
#    ax1.scatter(data[xfield],data[yfield1],marker='o',s=3)
    if xfield=='logMstar' and yfield1=='log_SFR':
        p1=IQ.SFS(ax1,sim=sim,params=True,Plot=True,color='orange')
    else:
        p1=fitmedian(xvals1,yvals1,axis=ax1,color='orange')
    diff1=yvals1-(p1[0]*xvals1+p1[1])
    ax2.set_xlabel(xtit1)    
    ax1.set_ylabel(ytit1)

    #second plot scatter and median line for yfield 2
    plotdistribution(ax2,xvals2,yvals2)
#    ax2.scatter(data[xfield],data[yfield2],marker='o',s=3)
    p2=fitmedian(xvals2,yvals2,axis=ax2,color='orange')
    diff2=yvals2-(p2[0]*xvals2+p2[1])
    ax2.set_xlabel(xtit2)
    ax2.set_ylabel(ytit2)

    #third plot diff vrs diff
    plotdistribution(ax3,diff1,diff2,bins=30)
    ax3.set_xlabel(diff_labels(xfield,yfield1))
    ax3.set_ylabel(diff_labels(xfield2,yfield2))
    ax3.set_xlim([-2,2])
    ax3.set_ylim([-2,2])
    slope,intercept,r_val,p_val,std_err = stats.linregress(diff1,diff2)
    ax3.annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
        ,xycoords='axes fraction',fontsize='small')
    ax3.annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
        ,xycoords='axes fraction',fontsize='small')

    plt.gcf().text(0.75, 0.8, sim, fontsize=14, ha='center')
    plt.savefig(f'2diff_{xfield}_{yfield1}_{yfield2}_{sim}.pdf')
    print(np.min(diff1),np.min(diff2))
    
def fig_diff(selection_function=False):
    """make the diff figure for the paper"""
    delta_sfs=r"$\Delta \overline{SFS}$"
    delta_sfr=r"$\Delta \overline{SFR}(M_{H_2})$"
    delta_h2star=r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$"
    x1fields=['logMstar','logMstar','logMH2']
    y1fields=['logMH2','log_SFR','log_SFR']
    x2fields=['logMstar','logMH2','logMstar']
    y2fields=['log_SFR','log_SFR','logMH2']    
    xlabels=[delta_h2star,delta_sfs,delta_sfr]
    names=IQ.sim_names()
    names.insert(0,'xCOLDGASS')
    Nbins={'xCOLDGASS':10,'Eagle':10,'Mufasa':20,'TNG100':30,'Simba':30,'SC-SAM':40}
    f,axs=plt.subplots(nrows=3,ncols=6,sharex=True,sharey=True,figsize=(10,6))
    extra=''
    pad=[-10,-10,0]
    for j in range(3):
        title_axes=f.add_subplot(3,1,j+1,frameon=False)
        title_axes.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
        title_axes.set_xlabel(xlabels[j],labelpad=pad[j])
    for col,name in enumerate(names):
        data=IQ.get_data(name=name,mcut=9.0) #like xCOLDGASS
        #keep only detections of star forming and centrals
        if name=='xCOLDGASS':
            uplim=(data['uplim']==True).to_numpy()
            keep=np.invert(uplim)
        elif selection_function:
            extra='_sl'
            keep=np.logical_and(data['log_SFR'] > -2,
                                data['logMH2']-data['logMstar'] > -1.7) #log10(0.02)  
        else:
            keep=np.logical_and(data['log_SFR'] > -3,data['logMH2'] > 5)                                
        keep=np.logical_and(data['central']==True,keep)
        print("galaxies kept: {:.2f}".format(keep.sum()/len(data)))
        data=data[keep]

        for row in range(3):
            ax=axs[row][col]
            xvals1=data[x1fields[row]]
            xvals2=data[x2fields[row]]
            yvals1=data[y1fields[row]]
            yvals2=data[y2fields[row]]

            p1=fitmedian(xvals1,yvals1)
            diff1=perpdis(xvals1,yvals1,p1[0],p1[1])  #diff1=yvals1-(p1[0]*xvals1+p1[1])
            p2=fitmedian(xvals2,yvals2)
            diff2=perpdis(xvals2,yvals2,p2[0],p2[1])  #diff2=yvals2-(p2[0]*xvals2+p2[1])

            #SFS from paper1 fit using GMM
            if row==0:
                p2=IQ.SFS(ax,sim=name,params=True)
                diff2=perpdis(xvals2,yvals2,p2[0],p2[1])
            elif row==1:
                p1=IQ.SFS(ax,sim=name,params=True)
                diff1=perpdis(xvals1,yvals1,p1[0],p1[1])

            #plot the diffs
            plotdistribution(ax,diff1,diff2,bins=Nbins[name]) #diff1 on x-axis
            slope,intercept,r_val,p_val,std_err = stats.linregress(diff1,diff2)
            ax.annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
                ,xycoords='axes fraction',fontsize='xx-small')
            ax.annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
                ,xycoords='axes fraction',fontsize='xx-small')
            ax.annotate(name,xy=(0.05,0.9),xycoords='axes fraction',fontsize='small')
            ax.set_xlim([-1.4,1.4])
            ax.set_ylim([-1.4,1.4])

    axs[0][0].set_ylabel(delta_sfs)
    axs[1][0].set_ylabel(delta_sfr)
    axs[2][0].set_ylabel(delta_h2star)
    plt.subplots_adjust(wspace=0.0,hspace=0.25)    
    plt.savefig(f'diffs{extra}.pdf')    

def all_diff(xfield1,yfield1,xfield2,yfield2):
    names=IQ.sim_names()
    names.insert(0,'xCOLDGASS')
    if yfield2=='logMhalo':
        del names[0:2]
    xtit=diff_labels(xfield1,yfield1)
    ytit=diff_labels(xfield2,yfield2)
    f,axs=IQ.setup_multiplot(1,len(names),xtitle=xtit,ytitle=ytit,
        sharey=True,sharex=True,fs=(15,3.5))
    for name,ax in zip(names,f.axes):
        data=IQ.get_data(name=name)
        if name=='xCOLDGASS':
            uplim=(data['uplim']==True).to_numpy()
            keep=np.invert(uplim)
        else:
            keep=np.logical_and(data['log_SFR'] > -2,
                                data['logMH2']-data['logMstar'] > -1.7) #log10(0.02)
            keep=np.logical_and(data['logMHI']-data['logMstar'] > -1.7,keep)
        keep=np.logical_and(data['central']==True,keep)
        data=data[keep]
        xvals1=data[x1field1]
        xvals2=data[x2field2]
        yvals1=data[y1field1]
        yvals2=data[y2field2]
        if xfield1=='logMstar' and yfield1=='log_SFR':
            p1=IQ.SFS(ax,sim=name,params=True)
        else:
            p1=fitmedian(xvals1,yvals1)
        diff1=yvals1-(p1[0]*xvals1+p1[1])
        p2=fitmedian(xvals2,yvals2)
        diff2=yvals2-(p2[0]*xvals2+p2[1])
        
        #plot the diffs
        plotdistribution(ax,diff1,diff2)
        slope,intercept,r_val,p_val,std_err = stats.linregress(diff1,diff2)
        ax.annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
            ,xycoords='axes fraction',fontsize='small')
        ax.annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
            ,xycoords='axes fraction',fontsize='small')
        ax.annotate(name,xy=(0.05,0.92),xycoords='axes fraction')

    ax.set_xlim([-1.9,1.9])
    ax.set_ylim([-1.9,1.9])
    plt.tight_layout()
    plt.savefig(f'diff_{xfield1}_{yfield1}_{xfield2}_{yfield2}.pdf')


def triple_diff(logMgas,logSFR,logMstar,sim='',plot=False): 
    #logMgas, logSFR, logMstar, if sim not set assume 2nd not logSFR
    #plot shows the relations with color showing distance from median
    p=fitmedian(logMgas,logSFR)
    print(sim)
    print("The SFR-MH2 slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffSFR=perpdis(logMgas,logSFR,p[0],p[1])
    if sim:
        p=IQ.SFS(logMstar,sim=sim,Plot=False,params=True)
    else:
        p=fitmedian(logMstar,logSFR)
    print("The SFS slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffSFS=perpdis(logMstar,logSFR,p[0],p[1])
    p=fitmedian(logMstar,logMgas)
    print("The Mgas-Mstar slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffMgas=perpdis(logMstar,logMgas,p[0],p[1])
    if plot:
        f,axs=plt.subplots(nrows=1,ncols=3,figsize=(10,4))
        axs[0].scatter(logMgas,logSFR,marker=',',s=1,c=diffSFR)
        axs[1].scatter(logMstar,logSFR,marker=',',s=1,c=diffSFS)
        axs[2].scatter(logMstar,logMgas,marker=',',s=1,c=diffMgas)
        plt.savefig(sim+"3.pdf")
    return diffSFR,diffSFS,diffMgas

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Make a two diff plot")
    parser.add_argument("xfield1",action='store',
        help="The field used on the x-axis")
    parser.add_argument("yfield1",action='store',
        help="The first y-axis field")
    parser.add_argument("xfield2",action='store',
        help="The second x-axis field")        
    parser.add_argument("yfield2",action='store',
        help="The second y-axis field")       
    parser.add_argument("-c",action='store_true',
        help="Use only central galaxies")  
    parser.add_argument("-s",action='store_true',
        help="Use only satellite galaxies")                 
    args=parser.parse_args() 

    fig_diff(selection_function=False)
#    all_diff(args.xfield1,args.yfield1,args.xfield2,args.yfield2)