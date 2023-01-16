import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import gass
import read_sam as rs
import observations as obs

#colors for plots, 4 catagorical colors for models, 4 catagorical colors for observations
#3 sequential for 2d density per area
#Eagle, TNG100, Simba and SC-SAM
c_models=['tab:blue','tab:orange','tab:green','tab:red']
#Detected centrals, detected sats, uplim centrals, uplim sats
c_obs=['tab:purple','tab:cyan','tab:gray','tab:olive']
#c=['tab:brown','tab:pink'] left over

def quiescent():
    dname="/Users/ari/Dropbox/CCA Quenched Isolated Galaxies Workshop 2017-05-11/DATA/"
    dname=dname+'/quiescentSelections_ClairesPaper/'
    names=['EAGLE','MUFASA','SCSAM','SIMBA','TNG']
    # GroupID, SubID, Mass, Isolated&Quiescent, IsolatedInProjection&Quiescent, 
    #Central&Quiescent, Quiescent, Isolated, IsolatedInProjection, Central
    for name in names:
        fname=dname+f'quenchedselection_Claire_{name}.txt'
        gid,sid,mass,quiescent=np.loadtxt(fname,usecols=(0,1,2,4),unpack=True)
    
def sim_names():
    ''' Names of the simulations being analayzed, and a color for each name'''
    names=['Eagle','TNG100','Simba','SC-SAM']
    color={names[0]:c_models[0],names[1]:c_models[1],names[2]:c_models[2],names[3]:c_models[3]}
    return names,color

def log_with_inf(array):
    '''take the log10 of an array and make 0.0 go to np.inf with no warnings'''
    answer=-np.inf*np.ones(len(array))
    answer[array > 0.0]=np.log10(array[array > 0.0])
    return answer

def get_sim(sim=None,sample='all',mcut=8.0):
    '''routine to load simulation data into pandas dataframe'''
    dname="/Users/ari/Dropbox/CCA Quenched Isolated Galaxies Workshop 2017-05-11/DATA/"
    fname={'Illustris':'Illustris_with_hih2.dat',
#            'Eagle':'EAGLE_RefL0100HashPhotfix_MHIH2HIIRhalfmassSFRT1e4_allabove1.8e8Msun.txt',
            'Eagle':'EAGLE_RefL0100HashDave2020GK_MHIH2HIIRhalfmassSFRT1e4_allabove1.8e8Msun.txt',
            'Mufasa':'halos_m50n512_z0.0.dat',
            'TNG100':'TNG_with_hih2.dat',
            'Simba':'halos_m100n1024_z0.0.dat',
            'OLD-SAM':'SCSAMgalprop_updatedVersion.dat'}
    cols={'Illustris':[1,2,3,4,5], #cent, sfr, Mstar, MHI, MH2
            'Eagle':[2,10,15,3], #logMstar, logMHI, logMH2,rdisk - for 70kpc
            'Mufasa':[1,2,3,4,5,6,7],#cent, sfr, Mstar, MHI, MH2, Mvir
            'TNG100':[1,2,3,4,5,6,7], #cent, sfr, Mstar, MHI, MH2,rdisk, Mvir
            'Simba':[1,2,3,4,5,6,7], #cent, sfr, Mstar, MHI, MH2, rdisk, Mvir
            'OLD-SAM':[3,20,7,15,16]} #cent, Mstar, MHI, MH2, sfr
    
    if sim=='Eagle': #eagle has sfr and cent in a different file so need to add those
        #new Eagle file has different aperatures, all=[6,11],2R50 = [8,13],30=[9,14],70=[10,15]
        Mstar,mHI,mH2,rdisk=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        fname2='EAGLE_RefL0100_MstarMcoldgasSFR_allabove1.8e8Msun.txt'
        sfr,cent=np.loadtxt(dname+fname2,usecols=(4,5),unpack=True) #check that instant
        c=cent.astype(bool)
        mgas=log_with_inf(10**(mHI)+10**(mH2))
        data=pd.DataFrame({'central':c,'logMstar':Mstar,'log_SFR':log_with_inf(sfr)
        ,'logMHI':mHI,'logMH2':mH2,'logMgas':mgas,'r_disk':rdisk})   
    elif sim=='SC-SAM':
        data=rs.read_ilsam_galaxies('/Users/ari/Data/tng-sam/',snapshot=99) #99 is z=0
        data['central']=np.invert(data['sat_type'].astype(bool))
        data['logMstar']=log_with_inf(data['mstar'])
        data['log_SFR']=log_with_inf(data['sfr'])
        data['logMHI']=log_with_inf(data['mHI'])
        data['logMH2']=log_with_inf(data['mH2'])
        data['logMgas']=log_with_inf((data['mHI']+data['mH2']))
        data['logMhalo']=log_with_inf(data['mhalo'])  
    elif sim=='OLD-SAM':
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        bad = mHI < 0.0 #shouldn't be here
        mHI[bad]=0.0
        c=np.invert(cent.astype(bool))
        data=pd.DataFrame({'central':c,'logMstar':log_with_inf(Mstar*1.e9),'log_SFR':np.log10(sfr)
        ,'logMHI':log_with_inf(mHI*1.e9),'logMH2':log_with_inf(mH2*1.e9)
        ,'logMgas':np.log10((mHI+mH2)*1.e9)})
    else:
        cent,sfr,Mstar,mHI,mH2,rdisk,M200=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=cent.astype(bool)
        data=pd.DataFrame({'central':c,'logMstar':log_with_inf(Mstar),'log_SFR':log_with_inf(sfr),
        'logMHI':log_with_inf(mHI),'logMH2':log_with_inf(mH2),'logMgas':log_with_inf(mHI+mH2),
        'r_disk':rdisk,'logMhalo':np.log10(M200)})

    logRstar=log_with_inf(data['r_disk'])
    data['logRstar']=logRstar
    h2frac=(10**data['logMH2'])/(10**data['logMgas'])
    bad=np.isnan(h2frac)
    h2frac[bad]=0.0 # set 0/0 to be 0. 
    data.insert(5,'H2frac',h2frac)
    data['logctime']=data['logMH2']-data['log_SFR']+np.log10(1.333) #add He
    NaN=np.invert(np.isfinite(data['logctime']))
    data['logctime'].loc[NaN,:]=0.0
    data=data[data['logMstar'] > mcut]
    data.reset_index
    if sample=='cent':
        keep=(data['central']==True)
        print(f"{sim} central galaxies {keep.sum()}")
        data=data[keep]
        data.reset_index
    elif sample=='sat':
        keep=(data['central']==False)
        print(f"{sim} satellite galaxies {keep.sum()}") 
        data=data[keep]
        data.reset_index
    else:
        print(f"{sim} total galaxies {data.shape[0]}")
    return data

def get_data(name='xCO-GASS',sample='all',mcut=8.0):
    ''' get observational or simulation data'''
    if name=='xCOLDGASS' or name=='xGASS' or name=='xCO-GASS':
        data=gass.get_GASS(name=name,sample=sample)
    else:
        data=get_sim(sim=name,sample=sample,mcut=mcut)
    return data

def get_data_dict():
    '''creates a dictionary with all of the data for the paper, data_dict[name]
    is the data for the named simulation or observation'''
    names,_=sim_names()
    df_eagle=get_sim(sim=names[0])
    df_tng=get_sim(sim=names[1])
    df_simba=get_sim(sim=names[2])
    df_sam=get_sim(sim=names[3])
    df_xcold=get_data(name='xCOLDGASS')
    df_xgass=get_data(name='xGASS')
    data_dict={names[0]:df_eagle,names[1]:df_tng,names[2]:df_simba,names[3]:df_sam,
            'xGASS':df_xgass,'xCOLDGASS':df_xcold,'xC0-GASS':df_xcold}
    return data_dict

def pull_dataset(data_dict,name):
    '''function to either select dataframe from data_dict or load the frame from disk'''
    if data_dict:
        data=data_dict[name]
    else:
        names,_=sim_names()
        if name in names:
            data=get_sim(sim=name)
        else:
            data=gass.get_GASS(name=name)
    return data

#
def SFS(sim=None,axis=None,params=False,xaxis=None,**kwargs):
    ''' returns the paramerts or plots the line of the SFS from from Chang 2020 and J20'''
    p={'Illustris':[1.01,0.65], 'Eagle':[0.90,0.21],'Mufasa':[0.82,0.63],
        'TNG100':[0.90,0.52],'Simba':[0.84,0.49],'SC-SAM':[0.75,0.46],
        'xGASS':[0.656,0.162],'xCOLDGASS':[0.656,0.162]}
        #xGASS from J20 converted from sSFS to SFS by m'=1+m, b'=1.5m+10.5+b
    if axis:
        logM=np.linspace(8.0,11.5,num=25)
        SFS=p[sim][0]*(logM-10.5)+p[sim][1]
        axis.plot(logM,SFS,**kwargs)
    if params:
        return [p[sim][0],p[sim][1]-(10.5*p[sim][0])]
    else:
        SFS=p[sim][0]*(xaxis-10.5)+p[sim][1]
        return SFS

#functions used in figures 
def setup_multiplot(Nr,Nc,xtitle=None,ytitle=None,ytitle2=None,fs=(15,5),**kwargs):
    '''return the axes for a multiplot with x and y titles'''
    f,axs=plt.subplots(Nr,Nc,figsize=fs,**kwargs)
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    ax=f.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(xtitle,fontsize='x-large')
    ax.set_ylabel(ytitle,fontsize='x-large',labelpad=20)
    if ytitle2:
        ax2=ax.twinx()
        ax2.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False) 
        ax2.set_ylabel(ytitle2,fontsize='x-large',labelpad=20)   
    return f,axs

def binned_fractions(x,y,Nbins=10,xrange=None,sum=False,fraction=False):
    '''returns the median or sum of y in bins of x '''
    if xrange is None:
        xrange=[np.min(x),np.max(x)]
    xedges=np.linspace(xrange[0],xrange[1],Nbins)
    xmids=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0
    result=np.zeros(Nbins-1)
    for i in range(Nbins-1):
        bin=bin_number==i
        if fraction:
            result[i]=np.sum(y[bin])/len(y[bin])
        elif sum:
            result[i]=np.sum(y[bin])   
        else:
            result[i]=np.median(y[bin])
    return result,xmids

def get_rticks():
    rticks={'Eagle':False,'TNG100':False,'Simba':False,'SC-SAM':False}
    return rticks

def histogram(axs,values,**kwargs):
    '''plot a cumulative sum histogram'''
    h,bin_edges=np.histogram(values,density=True)
    bin_mids=0.5*(bin_edges[0:-1]+bin_edges[1:])
    axs.plot(bin_mids,np.cumsum(h),**kwargs)

def plot_with_uplim(axis,x,y,umark='v',uplim=None,sat=False,s=3):
    '''plot values and upper limits if given. Uplim is usually a boolean array
    with True for upper limits; however it is also possible to pass a int array 
    with 1 for upper limits, 2 for lower limtis and 3 for no limit. Setting sat to
    True changes the color scheme.'''
    col_detect=c_obs[0]
    col_uplim=c_obs[2]
    if sat:
        col_detect=c_obs[1]
        col_uplim=c_obs[3]

    if uplim.dtype=='bool':
        notup=np.invert(uplim)
        axis.scatter(x[notup],y[notup],color=col_detect,marker='o',s=s)
        axis.scatter(x[uplim],y[uplim],marker=umark,s=5,linewidths=0.5,
            facecolors='none',edgecolors=col_detect,alpha=0.5)
    else:
        notup=uplim==0
        axis.scatter(x[notup],y[notup],color=col_detect,marker='o',s=s)
        axis.scatter(x[uplim==1],y[uplim==1],marker='^',s=s,linewidths=0.5,
            facecolors='none',edgecolors=col_detect,alpha=0.5)
        axis.scatter(x[uplim==2],y[uplim==2],marker='v',s=s,linewidths=0.5,
            facecolors='none',edgecolors=col_detect,alpha=0.5)
        axis.scatter(x[uplim==3],y[uplim==3],marker='_',s=s,linewidths=0.5,
            color=col_detect,alpha=0.5)

def hist2dplot_with_limit(axis,x,y,uplim=None,fill=False,**kwargs):
    if uplim:
        upper_limits=y < uplim
        y[upper_limits]=uplim
    hist2dplot(axis,x,y,fill=fill,**kwargs)

def hist2dplot(axis,x,y,fill=True,**kwargs):
    h,xed,yed=np.histogram2d(x,y,**kwargs)
    h=np.transpose(h)
    total=h.sum()
    h=h/total
    hflat=np.sort(np.reshape(h,-1)) #makes 1D and sorted 
    csum=np.cumsum(hflat)
    values=1.0-np.array([0.9973,0.9545,0.6827,0.0])
    levels=[]
    for val in values:
        idx = (np.abs(csum - val)).argmin()
        levels.append(hflat[idx])

    if fill:
        colors=['#f7f7f7','#cccccc','#969696','#525252'] #gray
#        colors=['#ffffcc','#c2e699','#78c679','#238443'] #green 
        axis.contourf(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]])
    else:
        colors=['#fdcc8a','#fc8d59','#d7301f']
        axis.contour(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]]) 

def medianline(axis,x,y,xrange=None,N=10,uplim=None,invert=False,
        plot_upfrac=False,**kwargs):
    '''bins an array x and then calculates median of y in those bins
    also calculates fraction above some limit in those bins'''
    if np.any(np.isnan(y)):
        print(f"contains {(np.isnan(y)).sum()} NaN in y")
    if xrange==None:
        xrange=[np.min(x),np.max(x)]
    xedges=np.linspace(xrange[0],xrange[1],N)
    xmids=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0
    med=np.zeros(N-1)
    upfrac=np.zeros(N-1)
    if isinstance(uplim,float):
        upper_limits=(y < uplim).to_numpy()
    elif isinstance(uplim,np.ndarray):
        upper_limits=uplim
    elif uplim is None:
        upper_limits=None
    else:
        print(f"type not recognized: {type(uplim)}")

    for i in range(N-1):
        bin=bin_number==i
        med[i]=np.median(y[bin])
        if isinstance(upper_limits, np.ndarray):
            upfrac[i]=np.sum(upper_limits[bin] > 0)/np.sum(bin)
    
    good=(upfrac < 0.5)
    if type(uplim)==float:
        low=med < uplim
        med[low]= uplim
    if plot_upfrac:
        axis.plot(med,upfrac,**kwargs)
    else:
        if invert:
            axis.plot(med,xmids,**kwargs)   
        else:
            axis.plot(xmids,med,**kwargs)
    return xmids,med,upfrac

def myboxplot(xarray,ydistribution,axis,uplims=False):
    '''a version of a box plot which can show upper/lower limts'''
    #TODO: add uplims, probably need to pass uplim distribution
    N=len(xarray)
    y=np.zeros(N)
    percents=100*stats.norm().cdf((-2,-1,0,1,2)) #percents not 0-1
    one_sigma=np.zeros((2,N))
    two_sigma=np.zeros((2,N))   
    percentile=np.zeros((N,5))
    for i,x in enumerate(xarray):
        yvals=ydistribution[i] #y vals for one bin
        levels=np.percentile(yvals,percents) #interpolates value to percent
        two_sigma[0,i]=levels[2]-np.min(yvals) #levels[0]
        one_sigma[0,i]=levels[2]-levels[1]
        y[i]=levels[2]
        one_sigma[1,i]=levels[3] - levels[2] 
        two_sigma[1,i]=np.max(yvals) - levels[2] #levels[4] 
    
#    axis.violinplot(np.transpose(ydistribution),positions=xarray)
    axis.errorbar(xarray,y,yerr=two_sigma,fmt='none',ecolor='red',
        capsize=3.0,capthick=2,uplims=uplims)
#    axis.errorbar(xarray,y,yerr=one_sigma,fmt='none',ecolor='purple',capsize=3.0,capthick=2)
#    axis.plot(xarray,y,color='tab:red')
#    axis.plot(xarray,percentile[:,0],color='tab:red',linestyle='dotted')
#    axis.plot(xarray,percentile[:,4],color='tab:red',linestyle='dotted')

def gass_panel(data,xfield,yfield,axis,umark='v',xrange=[9,11.5],ncol=1,
        satcent=False,frac=False,name=None,loc='lower right',shade=False):
    '''make a plot using the GASS data on one axis in a multi axis plot'''
    uplim= data['uplim'].to_numpy() #boolean for only HI or H2, int for combined catalog

    #median lines
    if satcent:
        cent=data['central']==True
        sat=data['central']==False
        plot_with_uplim(axis,data[xfield][cent],data[yfield][cent],uplim=uplim[cent],umark=umark)
        plot_with_uplim(axis,data[xfield][sat],data[yfield][sat],uplim=uplim[sat],sat=True,umark=umark)
        uplim_legend(axis,fontsize='xx-small',ncol=ncol,umark=umark,loc=loc,
            frameon=False,markerscale=0.6,satcent=satcent)
        if umark=='<':
            ymids,meds,upfrac=medianline(axis,data[yfield][cent],data[xfield][cent],uplim=uplim[cent],
                xrange=[-1,1],linestyle='--',N=8,invert=True)
            ymids_sat,meds_sat,upfrac_sat=medianline(axis,data[yfield][sat],data[xfield][sat],
                xrange=[-1,0.75],uplim=uplim[sat],linestyle='--',N=6,invert=True)
        else:
            xmids,meds,upfrac=medianline(axis,data[xfield][cent],data[yfield][cent],uplim=uplim[cent],
                xrange=xrange,linestyle='',N=8)
            xmids_sat,meds_sat,upfrac_sat=medianline(axis,data[xfield][sat],data[yfield][sat],
                uplim=uplim[sat],xrange=xrange,linestyle='',N=6)
    elif shade:
        plot_with_uplim(axis,data[xfield],data[yfield],uplim=uplim,umark=umark)
        xmids,meds1,upfrac1=medianline(axis,data[xfield],data[yfield],N=7,
            xrange=[9.0,11.25],linestyle='dashed')
        lowlimit=data[yfield].to_numpy()
        lowlimit[uplim]=shade
        xmids,meds2,upfrac2=medianline(axis,data[xfield],lowlimit,N=7,
            xrange=[9.0,11.25],linestyle='dashed')
        axis.fill_between(xmids,meds2,meds1,color='gray',hatch='-',linestyle='dashed',alpha=0.25)     
    else:
        plot_with_uplim(axis,data[xfield],data[yfield],uplim=uplim,umark=umark)
        uplim_legend(axis,fontsize='xx-small',ncol=ncol,umark=umark,loc=loc,
            frameon=False,markerscale=0.6,satcent=satcent)
        xmids,meds,upfrac=medianline(axis,data[xfield],data[yfield],uplim=uplim,
            xrange=xrange,linestyle='dashed',N=8)   

    if frac:
        axis2=axis.twinx()
        axis2.set_ylim([0,1])  
#        axis2.set_yticks([])
        axis2.plot(xmids,upfrac,linestyle='dashed',color=c_obs[2])
        axis2.plot(xmids_sat,upfrac_sat,linestyle='dashed',color=c_obs[3])
       
    if name:  
        if loc=='upper left' or loc=='upper right':
            axis.annotate(name,xy=(0.05,0.8),xycoords='axes fraction')
        else:            
            axis.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')

def bootstrap_panel(data,xfield,yfield,axes,xrange=[9.0,11.5],satcent=True):
    '''function for bootstrap resampled values for observations'''
    c={'cent':'black','sat':'gray'}
    uplim=(data['uplim']==True).to_numpy()
    if satcent:
        cent=data['central']==True
        sat=data['central']==False
        res_cent,x_cent=obs.bootstrap_resample(data[xfield][cent],data[yfield][cent],
            upper_limits=uplim[cent],xrange=xrange)
        res_sat,x_sat=obs.bootstrap_resample(data[xfield][sat],data[yfield][sat],
            upper_limits=uplim[sat],xrange=xrange) 
    else:
        res_cent,x_cent=obs.bootstrap_resample(data[xfield],data[yfield],
            upper_limits=uplim,xrange=xrange)        
        
    for axis in axes: 
        axis.boxplot(res_cent,positions=x_cent,widths=0.06,whis=(5,95),showfliers=False,
            manage_ticks=False,patch_artist=False,boxprops=dict(color=c['cent']),
            whiskerprops=dict(color=c['cent']),capprops=dict(color=c['cent']))  
        myboxplot(x_cent,res_cent,axis)
        if satcent: 
            axis.boxplot(res_sat,positions=x_sat,widths=0.06,whis=(5,95),showfliers=False,
                manage_ticks=False,patch_artist=False,boxprops=dict(color=c['sat']),
                whiskerprops=dict(color=c['sat']),capprops=dict(color=c['sat'])) 

def sim_panel(data,xfield,yfield,axs,name=None,xrange=[8.5,11.45],yrange=[7.5,11.5],color=None,
                uplim=None,fbelow=False,right_ticks=True,line_legend=False,buffer=0,select=None):  
    if isinstance(select,np.ndarray):
        data_all=data.copy(deep=True)
        data=data[select]  
    cent=data['central']==True
    sat=data['central']==False 
    hist2dplot_with_limit(axs,data[xfield][cent],data[yfield][cent],fill=True,uplim=uplim,
        bins=30,range=[xrange,yrange])
    xrange_mids=[xrange[0]+buffer,xrange[1]-0.15] #need to shave a little off, few gals a high end       
    xmids_c,meds_c,upfrac_c=medianline(axs,data[xfield][cent],data[yfield][cent],uplim=uplim,
        xrange=xrange_mids,color=color,linestyle='solid',N=25) 
    xmids_s,meds_s,upfrac_s=medianline(axs,data[xfield][sat],data[yfield][sat],uplim=uplim,
        xrange=xrange_mids,color=color,linestyle='dotted',N=15)
    if fbelow: #problem with tick marks on far left axis
        if isinstance(select,np.ndarray):
            cent=data_all['central']==True
            sat=data_all['central']==False
            xmids_c,meds_c,upfrac_c=medianline(axs,data_all[xfield][cent],data_all[yfield][cent],
                uplim=uplim,xrange=xrange_mids,linestyle='None',N=25) 
            xmids_s,meds_s,upfrac_s=medianline(axs,data_all[xfield][sat],data_all[yfield][sat],
                uplim=uplim,xrange=xrange_mids,linestyle='None',N=15)  
  
        axs2=axs.twinx()
        axs2.set_ylim([0,0.98])
        axs2.plot(xmids_c,upfrac_c,linestyle='dashed',color=c_obs[2])  
        axs2.plot(xmids_s,upfrac_s,linestyle='dashed',color=c_obs[3]) 
        if not right_ticks:
            axs2.set_yticks([])
    if line_legend:
        add_line_legend(axs,ncol=2,fontsize='xx-small',frac=fbelow)
    axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction',color=color)
    return xmids_c,meds_c,xmids_s,meds_s

def add_line_legend(axis,ncol=1,loc='upper right',fontsize='xx-small',frac=False):
    legend_lines=[mpl.lines.Line2D([],[],color='black',linestyle='solid'),
        mpl.lines.Line2D([],[],color='black',linestyle='dotted')]
    legend_names=['median cent.','median sat.']
    if frac:
        substring='below'
        legend_lines.append(mpl.lines.Line2D([],[],color=c_obs[2],linestyle='dashed'))                
        legend_lines.append(mpl.lines.Line2D([],[],color=c_obs[3],linestyle='dashed'))
        legend_names.append('$f_{{{}}}$ cent.'.format(substring))
        legend_names.append('$f_{{{}}}$ sat.'.format(substring))
        
    axis.legend(legend_lines,legend_names,ncol=ncol,loc=loc,fontsize=fontsize,handlelength=2.5)

def uplim_legend(axis,umark='v',satcent=True,**kwargs):
    if satcent:
        legend_lines=[mpl.lines.Line2D([],[],color=c_obs[0],marker='o',linestyle=''),
                mpl.lines.Line2D([],[],color=c_obs[0],marker=umark,linestyle='',
                    markerfacecolor='none',linewidth=0.5,alpha=0.5),
                mpl.lines.Line2D([],[],color=c_obs[1],marker='o',linestyle=''),
                mpl.lines.Line2D([],[],color=c_obs[1],marker=umark,linestyle='',
                    markerfacecolor='none',linewidth=0.5,alpha=0.5)]
        legend_names=['cent. detect','cent. up. lim.','sat. detect','sat. up. lim.']
    else:
        legend_lines=[mpl.lines.Line2D([],[],color=c_obs[0],marker='o',linestyle=''),
                mpl.lines.Line2D([],[],color=c_obs[0],marker=umark,linestyle='',
                    markerfacecolor='none',linewidth=0.5,alpha=0.5)]
        legend_names=['detections','upper limits']
    axis.legend(legend_lines,legend_names,**kwargs)

#test figures
def just_gass():
    dataHI=get_data(name='xGASS')
    dataH2=get_data(name='xCOLDGASS')
    data=get_data()

    f,axes=plt.subplots(nrows=2,ncols=6,figsize=(18,6))
    axes[0][0].set_title('$M_* vrs. M_{HI}$')
    gass_panel(dataHI,'logMstar','logMHI',axes[0][0],satcent=False)
    bootstrap_panel(dataHI,'logMstar','logMHI',[axes[0][0]],xrange=[9.0,11.5],satcent=False)
    gass_panel(dataHI,'logMstar','logMHI',axes[1][0],satcent=True)
    bootstrap_panel(dataHI,'logMstar','logMHI',[axes[1][0]],xrange=[9.0,11.5],satcent=True)
    
    axes[0][1].set_title('$M_* vrs. M_{H_2}$')   
    gass_panel(dataH2,'logMstar','logMH2',axes[0][1],satcent=False)
    bootstrap_panel(dataH2,'logMstar','logMH2',[axes[0][1]],xrange=[9.0,11.5],satcent=False)
    gass_panel(dataH2,'logMstar','logMH2',axes[1][1],satcent=True)
    bootstrap_panel(dataH2,'logMstar','logMH2',[axes[1][1]],xrange=[9.0,11.5],satcent=True)

    axes[0][2].set_title('$M_* vrs. f_{H_2}$') 
    gass_panel(data,'logMstar','H2frac',axes[0][2],shade=0.1,xrange=[9.0,11.45],satcent=False)
    bootstrap_panel(data,'logMstar','H2frac',[axes[0][2]],xrange=[9.0,11.5],satcent=False)
    gass_panel(data,'logMstar','H2frac',axes[1][2],shade=0.1,xrange=[9.0,11.45],satcent=True)
    bootstrap_panel(data,'logMstar','H2frac',[axes[1][2]],xrange=[9.0,11.5],satcent=True)

    axes[0][3].set_title('$M_{HI} vrs. SFR$')
    gass_panel(dataHI,'logMHI','log_SFR',axes[0][3],umark='<',xrange=[7.8,10.],satcent=False)
    gass_panel(dataHI,'logMHI','log_SFR',axes[1][3],umark='<',xrange=[7.8,10.],satcent=True)

    axes[0][4].set_title('$M_{H_2} vrs. SFR$')
    gass_panel(dataH2,'logMH2','log_SFR',axes[0][4],umark='<',xrange=[7.8,10.],satcent=False)
    gass_panel(dataH2,'logMH2','log_SFR',axes[1][4],umark='<',xrange=[7.8,10.],satcent=True)

    axes[0][5].set_title(r'$M_{*} vrs. \tau_{H_2}$')
    gass_panel(dataH2,'logMstar','logctime',axes[0][5],umark='v',xrange=[9.0,11.45],satcent=False,
        shade=8.5,ncol=2) 
    gass_panel(dataH2,'logMstar','logctime',axes[1][5],umark='v',xrange=[9.0,11.45],satcent=True,
        shade=8.5,ncol=2)
#    axes[0].set_ylim([7.5,10.8])
#    axes[0].set_xlim([8.5,11.45])
    f.align_labels()
    plt.subplots_adjust(bottom=0.1,hspace=0.3,wspace=0.4)
    plt.savefig('xGASS.pdf')


#figures for the paper
def fig1(sample='cent'):
    '''Gas masses of quenched versus star forming galaxies'''
    names,_ = sim_names()
    f,axes=setup_multiplot(2,2,sharex=True,sharey=True, 
        xtitle='$M_{gas} / M_*$', ytitle='$\log_{10}$ pdf',fs=(6.5,5))
    for name,axs in zip(names,f.axes):
        data=get_sim(sim=name,sample=sample,mcut=9)
        sfs_sfr=SFS(xaxis=data['logMstar'],sim=name)
        diff=data['log_SFR'] - sfs_sfr
        star_forming= diff > -0.3
        quenched=diff < -1.0
        hifrac=10**(data['logMHI']-data['logMstar'])
        h2frac=10**(data['logMH2']-data['logMstar'])
        mx=0.75
        axs.hist(h2frac[star_forming],color='blue',bins=20,range=[0,mx],
            density=True, histtype='bar',alpha=0.6)
        axs.hist(hifrac[star_forming],color='blue',linestyle='dashed',bins=20,range=[0,mx],
            density=True, histtype='step')
        axs.hist(h2frac[quenched],color='red',bins=20,range=[0,mx-0.2],
            density=True, histtype='bar',alpha=0.5)
        axs.hist(hifrac[quenched],color='red',linestyle='dashed',bins=20,range=[0,mx-0.2],
            density=True, histtype='step')
        axs.annotate(name,xy=(0.7,0.9),xycoords='axes fraction')
    axs.set_xlim([0,mx])
    axs.set_yscale('log')
    axs.set_ylim([0.2,50])
    if sample=='cent':
        plt.savefig('Fig1.pdf')
    else:
        plt.savefig(f'Fig1_{sample}.pdf')

def fig2(sample='cent'):
    '''Plot of Quenched Fraction in terms of gas mass, compare to Fig 4. in Dickey 2021'''
    names,colors = sim_names()
    mtypes=['logMH2','logMHI']
    for name in names:
        xbins=np.arange(8,11,0.25)
        data=get_data(name=name,sample=sample,mcut=8.5)
        quenched_h2= data['logMH2'] < data['logMstar']-1.7
        y,xmids=binned_fractions(data['logMstar'],quenched_h2,fraction=True, xrange=[8.75,11.25])
        plt.plot(xmids,y,label=name,color=colors[name])
 #       quenched_hi= data['logMHI'] < data['logMstar']-1.7
 #       y,xmids=binned_fractions(data['logMstar'],quenched_hi,fraction=True, xrange=[8.75,11.25])
 #       plt.plot(xmids,y,linestyle='dashed',color=colors[name])       

    plt.legend()
    plt.xlabel('$\log M_* \,\,\, [M_{\odot}]$')
    plt.ylabel('Quiescent Fraction ')
    plt.xlim=[9,11]
    plt.ylim(bottom=0.0,top=1.0)
    if sample=='cent':
        plt.savefig('Fig2.pdf')
    else:
        plt.savefig(f'Fig2_{sample}.pdf')

def fig34(data_dict=None,mtype='logMH2',sample='all',mcut=8): #fig3 and fig4
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=fr"$log (M_{mname[mtype]})"+r"[M_{\odot}]$" 
    ytit2=r"$f_{below}$"
    f,axes=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,ytitle2=ytit2,sharex=True,sharey=True,fs=(9,6))
    names,colors=sim_names()
    names.insert(2,gname[mtype])
    nc={'logMH2':2,'logMHI':1}
    loc={'logMH2':'upper left','logMHI':'lower right'}
    rticks=get_rticks() #turns off right tick marks for all sims
    for name,axs in zip(names,f.axes):
        data=pull_dataset(data_dict,name)
        if name==names[2]:
            gass_panel(data,'logMstar',mtype,axs,xrange=[9.0,11.5],name=gname[mtype],ncol=nc[mtype],
                satcent=True,frac=True,loc=loc[mtype])
            bootstrap_panel(data,'logMstar',mtype,[axes[1][2],axes[0][2]],xrange=[9.0,11.5],satcent=True)
        else:
            line_legend=False
            if name=='Eagle':
                line_legend=True #add a line legend to this panel
            xc,yc,xs,ys=sim_panel(data,'logMstar',mtype,axs,xrange=[8.5,11.45],yrange=[7.5,10.8],
                color=colors[name],
                name=name,fbelow=True,uplim=7.5,line_legend=line_legend)
            axes[1][2].plot(xc,yc,color=colors[name],label=name)
    #        axes[1][2].plot(xs,ys,linestyle='dotted',color=colors[name])

    axes[1][2].legend(ncol=2,fontsize=8)
    axs.set_ylim([7.5,10.8])
    axs.set_xlim([8.5,11.45])
    f.align_labels()
#    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"{mtype}_Mstar.pdf") 

def fig5(data_dict=None):
    '''plot of H_2 fraction vrs stellar mass'''
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=r"$H_{2}$ fraction" 
    f,axes=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True,fs=(9,6))
    names,colors=sim_names()
    names.insert(2,'xC0-GASS')
    for name,axs in zip(names,f.axes):
        data=pull_dataset(data_dict,name)
        if name=='xC0-GASS':
            gass_panel(data,'logMstar','H2frac',axs,xrange=[9.0,11.5],name=name,ncol=2,
                loc='upper right',shade=0.01)
            bootstrap_panel(data,'logMstar','H2frac',[axes[1][2],axes[0][2]],
                xrange=[9.0,11.5],satcent=True)
        else:
            line_legend=False
            if name=='Simba':
                line_legend=True
            xc,yc,xs,ys=sim_panel(data,'logMstar','H2frac',axs,color=colors[name],name=name,
                xrange=[8.5,11.45],yrange=[0.0,1.0],line_legend=line_legend)
            axes[1][2].plot(xc,yc,color=colors[name],label=name)
    
    axs.set_xlim([8.5,11.45])
    axs.set_ylim([0,0.98])
    f.align_labels()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("H2frac_Mstar.pdf") 

def fig6(data_dict=None,mtype='logMH2'):
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit=r"$log \, M_{H_2} [M_{\odot}]$"
    nstr=mname[mtype]
    ytit=r"$log \, SFR \, \, [M_{\odot}/yr]$"
    f,axes=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True,fs=(9,6))
    names,colors=sim_names()
    names.insert(2,gname[mtype])
    for name,axs in zip(names,f.axes):
        data=pull_dataset(data_dict,name)
        if name==names[2]:
            gass_panel(data,mtype,'log_SFR',axs,umark='<',xrange=[7.8,10.],satcent=True,name=name)
        else:
            line_legend=False
            if name=='Eagle':
                line_legend=True
            xc,yc,xs,ys=sim_panel(data,mtype,'log_SFR',axs,color=colors[name],name=name,
                xrange=[7.5,10.5],yrange=[-2,1.5],line_legend=line_legend)
            axes[1][2].plot(xc,yc,color=colors[name],label=name)
    
    axs.set_xlim([7.5,10.45])
    axs.set_ylim([-2,1.45])
    plt.savefig(f"sfr_{mtype}.pdf")

def fig7(data_dict=None,mtype='logMH2'):
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit="$log \, M_{*} [M_{\odot}]$"
    nstr=mname[mtype]
    ytit=r"$log \, \tau^{"+nstr+"}_{depl}$  [yr]"
    f,axes=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,ytitle2=r'$f_{no\_sfr}$',
        sharex=True,sharey=True,fs=(9,6))
    names,colors=sim_names()
    names.insert(2,gname[mtype])
    rticks=get_rticks() #what does this do?
    for name,axs in zip(names,f.axes):
        data=pull_dataset(data_dict,name)
        if name==names[2]:
            gass_panel(data,'logMstar','logctime',axs,umark='v',xrange=[7.8,10.],
                satcent=False,name=name,shade=8.5,ncol=2)  
        else:
            #needs to have medianline excluding zeros and upfrac
            line_legend=False
            if name=='Simba':
                line_legend=True
            notdead = np.isfinite(data['logctime'])
            xc,yc,xs,ys=sim_panel(data,'logMstar','logctime',axs,uplim=7.0,xrange=[8.5,11.2],
                yrange=[7.5,11.0],buffer=0.3,name=name,select=notdead,color=colors[name],
                fbelow=True,right_ticks=rticks[name],line_legend=line_legend)
            axes[1][2].plot(xc,yc,color=colors[name])

    axs.set_xlim([8.5,11.45])
    axs.set_ylim([7.5,11.4])
    f.align_labels()
    plt.savefig(f"depltion_time_{mtype}.pdf")

if __name__=='__main__':
#    rng = np.random.default_rng(seed=42)
#    N=1000
#    xrand=rng.random(size=N)
#    yrand=xrand+rng.normal(loc=0.0,scale=0.2,size=N)
#    results,xarray=obs.bootstrap_resample(xrand,yrand,Nsamples=100,xrange=[0,1])
#    results=np.transpose(results)
#    f,axis=plt.subplots()
#    axis.scatter(xrand,yrand,marker=',',s=1,alpha=0.5)
#    myboxplot(xarray,results,axis)
#    plt.show()
    just_gass()
#    data_dict=get_data_dict()
#    fig34(mtype='logMHI',data_dict=data_dict)
#    fig34(mtype='logMH2',data_dict=data_dict)
#    fig5(data_dict=data_dict)
#    fig6(data_dict=data_dict)
#    fig6(data_dict=data_dict,mtype='logMHI')
#    fig7(data_dict=data_dict)