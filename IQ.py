import sys
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import pickle as pkl
import read_sam as rs
import observations as obs
from astropy.table import Table

#entropy = k_b*T*n^(-2./3.)
#fivethirtyeight color style
c=['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
name=['b','r','y','g','s','p'] 
color={name[0]:c[0],name[1]:c[1],name[2]:c[2],name[3]:c[3],name[4]:c[4],name[5]:c[5]}
#change plot style in here until I understand how to use style sheet

#functions for getting the data from the simulations 
#data and sims different since likely to loop over sims
#setup returns dictionaries for looping

def log10_with_inf(array):
    bad = array < 0.0
    if np.any(bad):
        print(array[bad])
    answer=np.zeros(array.shape)
    zeros = array==0.0
    not_zero=np.invert(zeros)
    answer[not_zero] = np.log10(array[not_zero])
    answer[zeros]= -np.inf
    print(type(answer))
    return answer

def test_sample():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df1 = dat.to_pandas()
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df2 = dat.to_pandas()
    df=pd.merge(df1,df2,left_on='GASS',right_on='ID')
    return df,df1,df2

def get_xGASS():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want=['GASS','SDSS','HI_FLAG','lgMstar','lgMHI','SFR_best','HIconf_flag', 'NYU_id', 'env_code_B'] #no HI error            
    discard=list(set(df.columns.values.tolist()).difference(want))  
    df=df.drop(columns=discard) 
    df.rename(columns={'lgMstar':'logMstar','lgMHI':'logMHI'},inplace=True)
    return df

def get_xCOLDGASS(): 
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want = ['ID','OBJID','FLAG_CO','LOGMH2','XCO_A17','LCO_COR','R50KPC',
            'LOGMSTAR','LOGMH2_ERR','LIM_LOGMH2','LOGSFR_BEST']
    discard=list(set(df.columns.values.tolist()).difference(want))
    df=df.drop(columns=discard)
    what=df['R50KPC'] < 0.0   #one radii is negative
    df=df[np.invert(what)]
    df.reindex
    df.rename(columns={'R50KPC':'r_disk'},inplace=True)
    df['logRstar']=np.log10(df['r_disk'])
    df['logSigma']=df['LOGMSTAR']-2.0*df['logRstar']
    logMH2=np.zeros(df.shape[0])
    yes_CO= df['FLAG_CO'] < 2
    no_CO = df['FLAG_CO']==2
    logMH2[yes_CO]=df['LOGMH2'][yes_CO]+np.log10(0.75) #remove He
    logMH2[no_CO]=df['LIM_LOGMH2'][no_CO]+np.log10(0.75) #remove He
    df.insert(3,'logMH2',logMH2)
    df=df.drop(columns=['LOGMH2','LIM_LOGMH2'])
    return df

def check_alpha():
    df=get_xCOLDGASS()
    f,axes=setup_multiplot(1,2,sharey=True)
    yes_CO= df['FLAG_CO'] < 2
    no_CO = df['FLAG_CO']==2   
    print(df['LCO_COR'][no_CO][0])
    axes[0].scatter(df['logMH2'][yes_CO],df['LOGSFR_BEST'][yes_CO],
        label='detections')
    axes[0].scatter(df['logMH2'][no_CO],df['LOGSFR_BEST'][no_CO], 
        label='non-dectections')   
    axes[1].scatter(np.log10(df['LCO_COR'][yes_CO]),df['LOGSFR_BEST'][yes_CO])
    axes[1].scatter(np.log10(df['LCO_COR'][no_CO]),df['LOGSFR_BEST'][no_CO])   
    axes[0].set_ylabel("log SFR")
    axes[0].set_xlabel(r"log $M_{H_2}$")
    axes[1].set_xlabel(r"log $CO_{cor}$")
    axes[0].legend()
    plt.show()

def get_GASS(name='xCOLDGASS',sample='all'):
    #returns the HI catalog, H2 catalog or the HI+H2 catalog,xGASS,xCOLDGASS,xCO
    HI=True
    H2=False
    if name=='xCOLDGASS':
        H2=True
    
    if H2: #trying to get coldgass only, but for now need HI for sat/cent
        dfH2=get_xCOLDGASS()
#        dfH2['log_SFR']=dfH2['LOGSFR_BEST']
        
    if HI:
        dfHI=get_xGASS()
        Nconf=(dfHI['HIconf_flag']==1).sum()
        #remove the galaxies with strong HI confusion, 108 in xGASS sample, still 20 slightly confused 0.1,0.2
        dfHI=dfHI[dfHI['HIconf_flag']!=1] #
        dfHI.reindex
        #remove SFR non-detections (8) and take log of SFR
        dfHI=dfHI[dfHI['SFR_best']!=-99]
        dfHI.reindex 
        dfHI['log_SFR']=np.log10(dfHI['SFR_best'])
        
    if name=='xCOLDGASS': #combined catalog
        df=pd.merge(dfHI,dfH2,left_on='GASS',right_on='ID')
        logMgas=np.log10(10**df['logMHI']+10**df['logMH2'])
        df.insert(15,'logMgas',logMgas)
        H2frac=10**(df['logMH2']-df['logMgas'])
        df.insert(16,'H2frac',H2frac)
        no_detect=df['FLAG_CO']==2
        df['logctime']= df['logMH2']-df['log_SFR']+np.log10(1.333) #add He
    else:
        no_detect=dfHI['HI_FLAG']==99
        df=dfHI
    #define upper limits based on HI or CO, Note for xCO not keeping HI upperlimits 
    uplim=np.zeros((df.shape[0]),dtype=bool)
    uplim[no_detect]=1
    df.insert(5,'uplim',uplim)
    df['log_sSFR']=df['log_SFR']-df['logMstar']
    #define central and remove 14/4 ungrouped galaxies (fix with Tinker?)
    #only works with HI, need to find values for gals not in xGASS 
    df=df[df['env_code_B'] > -1]
    df.reindex
    group=np.zeros(df.shape[0],dtype=np.bool)
    sat=df['env_code_B']==0
    cent=np.logical_or(df['env_code_B']==1,df['env_code_B']==2)
    group[sat]=False
    group[cent]=True
    df.insert(0,'central',group)
    if sample=='cent':
        df=df[df['central']==True]
    elif sample=='sat':
        df=df[df['central']==False]
    df.reindex
    df=df.drop(columns=['GASS','SDSS','SFR_best','HIconf_flag','env_code_B'])
    return df

def get_sim(sim=None,sample='all',mcut=8.0):
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
        mgas=np.log10(10**(mHI)+10**(mH2))
        data=pd.DataFrame({'central':c,'logMstar':Mstar,'log_SFR':np.log10(sfr)
        ,'logMHI':mHI,'logMH2':mH2,'logMgas':mgas,'r_disk':rdisk})   
    elif sim=='PAST-SAM':
            data=rs.read_samh5('tng_sam_galprop-0.h5')
            data['logMstar']=np.log10(data['mstar'])
            data['logMHI']=np.log10(data['mHI'])
            data['logMH2']=np.log10(data['mH2'])
            data['log_SFR']=np.log10(data['sfr'])
            data['logMgas']=np.log10(data['mHI']+data['mH2'])
            data['logMhalo']=np.log10(data['mhalo'])
            data=data.astype({'sat_type':bool})
            data['central']=np.invert(data['sat_type'])
    elif sim=='SC-SAM':
        data=rs.read_ilsam_galaxies('/Users/ari/Data/tng-sam/',snapshot=99) #99 is z=0
        data['central']=np.invert(data['sat_type'].astype(bool))
        data['logMstar']=np.log10(data['mstar'])
        data['log_SFR']=np.log10(data['sfr'])
        data['logMHI']=np.log10(data['mHI'])
        data['logMH2']=np.log10(data['mH2'])
        data['logMgas']=np.log10((data['mHI']+data['mH2']))
        data['logMhalo']=np.log10(data['mhalo'])  
    elif sim=='OLD-SAM':
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        bad = mHI < 0.0 #shouldn't be here
        mHI[bad]=0.0
        c=np.invert(cent.astype(bool))
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar*1.e9),'log_SFR':np.log10(sfr)
        ,'logMHI':np.log10(mHI*1.e9),'logMH2':np.log10(mH2*1.e9)
        ,'logMgas':np.log10((mHI+mH2)*1.e9)})
    else:
        cent,sfr,Mstar,mHI,mH2,rdisk,M200=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=cent.astype(bool)
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar),'log_SFR':np.log10(sfr),
        'logMHI':np.log10(mHI),'logMH2':np.log10(mH2),'logMgas':np.log10(mHI+mH2),
        'r_disk':rdisk,'logMhalo':np.log10(M200)})

    logRstar=np.log10(data['r_disk'])
    data['logRstar']=logRstar
    h2frac=(10**data['logMH2'])/(10**data['logMgas'])
    bad=np.isnan(h2frac)
    h2frac[bad]=0.0 # set 0/0 to be 0. 
    data.insert(5,'H2frac',h2frac)
    data['logctime']=data['logMH2']-data['log_SFR']+np.log10(1.333) #add He
    NaN=np.invert(np.isfinite(data['logctime']))
    data['logctime'].loc[NaN]=0.0
    data=data[data['logMstar'] > mcut]
    data.reset_index
    if os.path.isfile(sim+'_rank.pkl'):
        with open(sim+'_rank.pkl','rb') as rpf:
            rank=pkl.load(rpf)
        data.insert(5,'rank',rank)
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
    
def get_data(name='xCOLDGASS',sample='all',mcut=8.0):
    if name=='xCOLDGASS' or name=='xGASS':
        data=get_GASS(name=name,sample=sample)
    else:
        data=get_sim(sim=name,sample=sample,mcut=mcut)
    return data

def satfrac():
    data=get_xGASS(sample='all')
    Nbins=10
    xedges=np.linspace(8.5,11.5,Nbins+2)
    xmids=0.5*(xedges[0:-2]+xedges[1:-1])
    sfrac=np.zeros(Nbins)
    bin_number=np.digitize(data['logMstar'],xedges)-1 #start with 0
    for i in range(Nbins):
        bin= (i==bin_number)
        sfrac[i]=np.sum(data[bin]['env_code_B']==0)/np.sum(bin)
    plt.scatter(xmids,sfrac,label='SDSS')
    names=sim_names()
    Nbins=20
    xedges=np.linspace(8.5,11.5,Nbins+2)
    xmids=0.5*(xedges[0:-2]+xedges[1:-1])
    sfrac=np.zeros(Nbins)
    for name in names:
        data=get_sim(sim=name,sample='all')
        bin_number=np.digitize(data['logMstar'],xedges)-1 #start with 0
        for i in range(Nbins):
            bin= (i==bin_number)
            sfrac[i]=1.0 - np.sum(data['central'][bin])/np.sum(bin)
        plt.step(xmids,sfrac,label=name,where='mid')

    plt.legend()
    plt.xlabel(r"$log M_{stellar} [M_{\odot}]$")
    plt.ylabel("satellite fraction ")
    plt.savefig('satfrac.pdf')
        
def gas_fractions(sample='all'):
    names = sim_names()
    f,axs=plt.subplots(nrows=2,ncols=len(names),figsize=(12,3),sharey=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample=sample)
        H2frac=(10**data['logMH2'])/(10**data['logMgas'])
        axs[0][i].scatter(data['logMstar'],H2frac,s=1)
        axs[0][1].set_xlim([8.0,12.0])
        axs[1][i].scatter(data['logMgas'],H2frac,s=1)  
        axs[1][i].set_xlim([7.0,10.0])     

    plt.savefig('gfrac.pdf')

def get_MARVEL(sample='cent'):
    dname="/Users/ari/Dropbox/CCA Quenched Isolated Galaxies Workshop 2017-05-11/DATA/"
    fname=dname+'brooks_updated_catalog.dat'
    names=['logMstar','log_SFR_10Myr','log_SFR_1Gyr','logMHI','parent']
    #using SAM field instantanous for 10Myr SFR
    data=np.loadtxt(fname,usecols=(2,4,5,7,14),skiprows=1)
    tab=Table(data,names=names)
    tab['logMstar']=np.log10(tab['logMstar'])
    tab['logMHI']=np.log10(tab['logMHI'])
    tab['log_SFR_10Myr']=np.log10(tab['log_SFR_10Myr'])
    tab['log_SFR_1Gyr']=np.log10(tab['log_SFR_1Gyr'])
    if sample=='cent':
        keep=(tab['parent']==0)
        print(f"MARVEL central galaxies {keep.sum()}")
        tab=tab[keep]
    elif sample=='sat':
        keep=(tab['parent'] > 0)
        print(f"MARVEL satellite galaxies {keep.sum()}")
        tab=tab[keep]  
    else:
        print(f"MARVEL total galaxies {len(tab)}")
    
    return tab

#routines for making the plots 
def perpdis(x1,y1,m,b):
    #returns the perpinduclar distance between the points x1,y1 and the line given by y=mx+b
    dis=(-m*x1+y1-b)/np.sqrt(m*m+1)
    return dis

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
        colors=['#ffffcc','#c2e699','#78c679','#238443']
        axis.contourf(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]])
    else:
        colors=['#fdcc8a','#fc8d59','#d7301f']
        axis.contour(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]])        

def test():
    names=sim_names()
    f,axs=plt.subplots(nrows=2,ncols=len(names),figsize=(12,3),sharey=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample=all)
        subset=np.floor(len(data)*np.random.random(5000))
        axs[0][i].scatter(data['logMstar'][subset],data['logMH2'][subset],marker='.',s=1)
        axs[0][i].set_xlim([8.5,11.5])
        axs[0][i].set_ylim([8,10.5])
        hist2dplot(axs[1][i],data['logMstar'],data['logMH2'],range=[[8.5,11.5],[8,10.5]])

    plt.show()

def set_uplim(mtype):
    values={'logMH2':['FLAG_CO',2],
            'logMHI':['HI_FLAG',99]
            ,'logMgas':['uplim',1]}
    return values[mtype]

def set_values(name):
    values={'HIcent':['cent','logMHI','HI_FLAG',99],
            'HIsat':['sat','logMHI','HI_FLAG',99],
            'H2cent':['cent','logMH2','FLAG_CO',2],
            'H2sat':['sat','logMH2','FLAG_CO',2],
            'Totalcent':['cent','logMgas','FLAG_CO',2],
            'Totalsat':['sat','logMgas','FLAG_CO',2]}
    return values[name]

def log_with_lowerlimit(array,ll=8):
    #computes log_10 of array, but puts ll as lowest value, no -inf
    above = array > 10**ll
    below = array < 10**ll
    answer=np.zeros(len(array))
    answer[above]=np.log10(array[above])
    answer[below]=ll
    return answer

def subsample(xarray,Nbins=10,Nmax=500):
    N=len(xarray)
    ids=np.arange(0,N,1,dtype=int)
    if N > Nmax:
        xedges=np.linspace(np.min(xarray),np.max(xarray),Nbins+2)
        bin_number=np.digitize(xarray,xedges)-1 #start with 0
        keep=np.empty(0,int)
        Nkeep=Nmax//Nbins
        for i in range(Nbins):
            indices=ids[bin_number==i] #the indices of the data in this bin
            if len(indices) > Nkeep:
                keep=np.concatenate((keep,np.random.choice(indices,size=Nkeep,replace=False)))
            else:
                keep=np.concatenate((keep,indices))
    else:
        keep=ids
    return keep

def random_subsample(data,fields,Nmax=500):
    N=len(data)
    if N > Nmax:
        Nbins=10
        xedges=np.linspace(np.min(data['logMstar']),np.max(data['logMstar']),Nbins+2)
        bin_number=np.digitize(data['logMstar'],xedges)-1 #start with 0
        keep=np.empty(0,int)
        Nkeep=Nmax//Nbins
        for i in range(Nbins):
            indices=data[bin_number==i].index #the indices of the data in this bin
            if len(indices) > Nkeep:
                keep=np.concatenate((keep,np.random.choice(indices,size=Nkeep,replace=False)))
            else:
                keep=np.concatenate((keep,indices))        
    else:
        keep=np.arange(0,N,1,dtype=int) #all
    answer=[]
    for f in fields:
        answer.append(np.copy(data[f][keep].values))
    return answer

def binned_rank(xarray,yarray,xrange=None,Nbins=10):
    #returns the rank of yarray in bins of xarray
    if xrange is None:
        xrange=[np.min(xarray),np.max(xarray)]

    xedges=np.linspace(xrange[0],xrange[1],Nbins+2)
    indices=np.digitize(xarray,xedges)-1 #start with 0
    rank=np.full(len(yarray),0.5)
    y25=np.zeros(Nbins)
    y50=np.zeros(Nbins)
    y75=np.zeros(Nbins)
    for i in range(Nbins):
        bin=(indices==i)
        N=int(bin.sum())
        if N==0:
            print("N is zero:",i)
            break
        idx=np.arange(0,N,1)
        order=np.zeros(N)
        asort=np.argsort(yarray[bin])
        order[asort]=idx
        rank[bin]=order/N
        levs=(np.floor(np.array([0.25,0.5,0.75])*N)).astype(int)
        y25[i],y50[i],y75[i]=np.array(yarray[bin])[asort[levs]]
        rank[bin]=yarray[bin]-y50[i]

    return xedges,rank,y25,y50,y75

def plot_percent(axis,xarray,yarray,xrange=None,Nbins=10,name=None,fit=False):
    #plot the 25/50/75 value of yarray in bins of xarray
    xedges,rank,y25,y50,y75=binned_rank(xarray,yarray,xrange=xrange,Nbins=Nbins)
    xmid=0.5*(xedges[1:-1]+xedges[0:-2])
    if name:
        with open(name+'_rank.pkl','wb') as wfb:
            pkl.dump(rank,wfb)
   
    axis.plot(xmid,y25,color='m',linestyle='-.')
    axis.plot(xmid,y50,color='r')
    axis.plot(xmid,y75,color='m',linestyle='-.')
    if fit:
        plot_linfit(axis,xmid,y50,color='g')

def fitmedian(x,y,xrange=None,axis=None,**kwargs):
    if np.any(np.isnan(y)):
        print(f"contains {(np.isnan(y)).sum()} NaN in y")
    N=10
    if xrange==None:
        Num=x.size
        xsort=np.sort(x)
        xrange=[xsort[int(0.1*Num)],xsort[int(0.9*Num)]]
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
#        axis.scatter(xmids,med,marker='+')
    return p

def medianline(axis,x,y,xrange=None,N=10,uplim=None,invert=False,
        plot_upfrac=False,**kwargs):
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
            upfrac[i]=np.sum(upper_limits[bin])/np.sum(bin)
    
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

def bootstrap(xarray,yarray,xrange=None,Nbins=10):
    #bootstrap resample the median value of y in bins of x
    if xrange is None: #insure the bins don't change
        xrange=[np.min(xarray),np.max(xarray)]
    Nresample=100
    Npoints=len(xarray)
    medians=np.zeros((Nresample,Nbins))
    np.random.seed(seed=12345)
    for i in range(Nresample):
        sample=(np.random.uniform(0,Npoints,Npoints)).astype(int)
        x,rank,y25,y50,y75=binned_rank(xarray[sample],yarray[sample],xrange=xrange,Nbins=Nbins)
        medians[i]=y50

    return x,np.transpose(medians)

def bootstrap_medians(xarray,yarray,xrange=None,Nbins=10):
    xedges,medians=bootstrap(xarray,yarray,xrange=xrange,Nbins=Nbins)
    xmid=0.5*(xedges[1:-1]+xedges[0:-2])
    xerr=xmid[0]-xedges[0]
    ym,y,yp=np.percentile(medians,[5.0,50.0,95.0],axis=1)
    yerr=[y-ym,yp-y]
    return xmid,y,xerr,yerr
    
def plot_with_uplim(axis,logMstar,logMHI,umark='v',uplim=None,sat=False):
    #plot values and upper limits, if uplim given it is a boolean array
    #of which indices are upper limits. Only using on observational data.
    s=3
    fc=None
    cold=color['b']
    colu=color['y']
    if sat:
        fc=None
        cold='cyan'
        colu=color['s']

    notup=np.invert(uplim)
    axis.scatter(logMstar[notup],logMHI[notup],
        color=cold,marker='o',s=s,facecolors=fc)
    axis.scatter(logMstar[uplim],logMHI[uplim],
        color=colu,marker=umark,s=s,facecolors=fc)

def uplim_legend(axis,umark='v',**kwargs):
    coldc=color['b']
    coluc=color['y']
    colds='cyan'
    colus=color['s']
    legend_lines=[mpl.lines.Line2D([],[],color=coldc,marker='o',linestyle=''),
                mpl.lines.Line2D([],[],color=coluc,marker=umark,linestyle=''),
                mpl.lines.Line2D([],[],color=colds,marker='o',linestyle=''),
                mpl.lines.Line2D([],[],color=colus,marker=umark,linestyle='')]
    legend_names=['cent. detect','cent. up. lim.','sat. detect','sat. up. lim.']
    axis.legend(legend_lines,legend_names,**kwargs)

def upfrac_lines(axs,xmids_c,upfrac_c,xmids_s,upfrac_s):
    axs2=axs.twinx()
    axs2.set_ylim([0,1])  
    axs2.set_yticks([])
    axs2.plot(xmids_c,upfrac_c,linestyle=':',color=color['y'])
    axs2.plot(xmids_s,upfrac_s,linestyle=':',color=color['s']) 

def setup_multiplot(Nr,Nc,xtitle=None,ytitle=None,ytitle2=None,fs=(15,5),**kwargs):
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

def plot_selectfunc(axis,mtype='logMHI',frac=False):
    xselect=np.linspace(9.4,11.4,80)
    yselect=xselect+np.log10(0.02)
    yselect[yselect < 8] = 8.0
    if mtype=='logMH2':
#        yselect=0.5*xselect+3.5
        yselect=xselect+np.log10(0.015)
        yselect[xselect < 10]= xselect[xselect < 10]+np.log10(0.025)   
    if frac is True:
        gfrac=np.log10((10**yselect)/(10**xselect))
        axis.plot(xselect,gfrac,linestyle='--',color='orange')
    else:
        axis.plot(xselect,yselect,linestyle='--',color='orange')

def plot_linfit(axis,x,y,**kwargs):
    p=np.polyfit(x,y,1)
    std=np.sum((y-(p[1]+p[0]*x))**2)/len(x)
    print(std)
    x=np.linspace(np.min(x),np.max(x))
    axis.plot(x,p[1]+p[0]*x,linestyle='--',**kwargs) 
    axis.plot(x,p[1]+p[0]*x+std,linestyle=':',**kwargs) 
    axis.plot(x,p[1]+p[0]*x-std,linestyle=':',**kwargs) 
    return p

def set_lower_limits(array,limit):
    copy=np.copy(array)
    low=array < limit
    copy[low]=limit
    return copy

def xgass_boot_mids(sample='cent',mtype='logHI',Nbins=10):
    cold=False
    if mtype=='logMH2':
        cold=True
    data=get_xGASS(sample=sample,cold=cold)
    logMstar=np.copy(data['logMstar'].values)
    logMgas=np.copy(data[mtype].values)
    xmid,y,xerr,yerr=bootstrap_medians(logMstar,logMgas,Nbins=Nbins)
    return xmid,y,xerr,yerr

def sim_names():
#    names=['Illustris','Eagle','Mufasa','TNG','Simba','SC-SAM']
#    names=['Eagle','Mufasa','TNG100','Simba','SC-SAM']
    names=['Eagle','TNG100','Simba','SC-SAM']
    return names

def get_rticks():
    rticks={'Eagle':False,'Mufasa':True,'TNG100':False,'Simba':False,'SC-SAM':True}
    return rticks

def field_labels(field):
    #return a label for a given field
    labels={'logMstar':r"$\log (M_{*}) \, [M_{\odot}]$",
            'logMHI': r"$\log (M_{HI}) \, [M_{\odot}]$",
            'logMH2':r"$\log (M_{H_2}) \, [M_{\odot}]$",
            'logMgas':r"$\log (M_{gas}) \, [M_{\odot}]$",
            'logMhalo':r"$\log (M_{halo}) \, [M_{\odot}]$",
            'log_SFR':r"$\log (SFR)  \, \, [yr^{-1}]$",
            'r_disk': r"$ R_{50}$  [kpc]",
            'logRstar': r"$\log \, (R_{50}) \, \, [kpc]$"}
    try:
        ans=labels[field]
    except:
        print(f"field {field} not known")
    return ans

def SFS(axs,sim=None,Plot=False,params=False,**kwargs):
    #from Chang and J20
    p={'Illustris':[1.01,0.65], 'Eagle':[0.90,0.21],'Mufasa':[0.82,0.63],
        'TNG100':[0.90,0.52],'Simba':[0.84,0.49],'SC-SAM':[0.75,0.46],
        'xGASS':[0.656,0.162],'xCOLDGASS':[0.656,0.162]}
        #xGASS from J20 converted from sSFS to SFS by m'=1+m, b'=1.5m+10.5+b
    if Plot:
        logM=np.linspace(8.0,11.5,num=25)
        SFS=p[sim][0]*(logM-10.5)+p[sim][1]
        axs.plot(logM,SFS,**kwargs)
    if params:
        return [p[sim][0],p[sim][1]-(10.5*p[sim][0])]
    else:
        SFS=p[sim][0]*(axs-10.5)+p[sim][1]
        return SFS

def xyhistplot(array,axis,xrange=None,weight=1,log=True,Nbins=10,**kwargs):
    #plots histogram of array with x and y error bars
    if not xrange:
        xrange=[np.min(array),np.max(array)]
    xedges=np.linspace(xrange[0],xrange[1],num=Nbins+1,endpoint=True)
    indices=np.digitize(array,xedges)
    x=np.zeros(Nbins)
    y=np.zeros(Nbins)
    for i in range(Nbins):
        x[i]=np.mean(array[indices==i+1])
        y[i]=np.sum(indices==i+1)

    xerr_left=x-xedges[0:-1]
    xerr_right=xedges[1:]-x
    xerr=[xerr_left,xerr_right]
    Deltax=xedges[1:]-xedges[0:-1]
    yerr=np.sqrt(y)*weight/Deltax
    y=y*weight/Deltax
    if log==True: #take the log of y (pass true value of y)
        yerr_top=np.log10(y+yerr)-np.log10(y)
        yerr_bot=np.log10(y)-np.log10(y-yerr)
        y=np.log10(y)
        yerr=[yerr_top,yerr_bot]

    axis.plot(x,y,**kwargs)
#    axis.errorbar(x,y,xerr=xerr,yerr=yerr,linestyle='',label=label)

def check_finite(array):
    bad=(np.invert(np.isfinite(array))).sum()
    if bad > 0:
        print(f"This array has {bad} non finite values")
        return True
    else:
        return False

def quad_diff(logMgas,logSFR,logMstar,rdisk,sim='',plot=False):
    diffs=triple_diff(logMgas,logSFR,logMstar,sim=sim)
    p=fitmedian(logMstar,np.log10(rdisk))
    diff4=rdisk-10**(p[0]*logMstar+p[1])
    return diffs[0],diffs[1],diffs[2],diff4

def triple_diff(logMgas,logSFR,logMstar,sim='',plot=False): 
    #logMgas, logSFR, logMstar, if sim not set assume 2nd not logSFR
    #plot shows the relations with color showing distance from median
    p=fitmedian(logMgas,logSFR)
#    print(sim)
#    print("The SFR-MH2 slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffSFR=perpdis(logMgas,logSFR,p[0],p[1])
    if sim:
        p=SFS(logMstar,sim=sim,Plot=False,params=True)
    else:
        p=fitmedian(logMstar,logSFR)
#    print("The SFS slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffSFS=perpdis(logMstar,logSFR,p[0],p[1])
    p=fitmedian(logMstar,logMgas)
#    print("The Mgas-Mstar slope is {:3.2f} and intercept is {:3.2f}.".format(p[0],p[1]))
    diffMgas=perpdis(logMstar,logMgas,p[0],p[1])
    if plot:
        f,axs=plt.subplots(nrows=1,ncols=3,figsize=(10,4))
        axs[0].scatter(logMgas,logSFR,marker=',',s=1,c=diffSFR)
        axs[1].scatter(logMstar,logSFR,marker=',',s=1,c=diffSFS)
        axs[2].scatter(logMstar,logMgas,marker=',',s=1,c=diffMgas)
        plt.savefig(sim+"3.pdf")
    return diffSFR,diffSFS,diffMgas

def add_line_legend(axis,ncol=1,loc='upper right',fontsize='xx-small',frac=False):
    legend_lines=[mpl.lines.Line2D([],[],color=color['p'],linestyle='--'),
        mpl.lines.Line2D([],[],color='magenta',linestyle='--')]
    legend_names=['median cent.','median sat.']
    if frac:
        substring='no\_sfr'
        legend_lines.append(mpl.lines.Line2D([],[],color=color['y'],linestyle=':'))                
        legend_lines.append(mpl.lines.Line2D([],[],color=color['s'],linestyle=':'))
        legend_names.append('$f_{{{}}}$ cent.'.format(substring))
        legend_names.append('$f_{{{}}}$ sat.'.format(substring))
        
    axis.legend(legend_lines,legend_names,ncol=ncol,loc=loc,fontsize=fontsize,handlelength=2.5)

### code for generating the figures
def testsam():
    data1=get_sim(sim='NEW-SAM',sample='all',mcut=8.0)
    data2=get_sim(sim='OLD-SAM',sample='all',mcut=8.0)
    boxsize1=110.7
    boxsize2=147.5
    f,axis=setup_multiplot(1,3,sharey=True,ytitle='$\phi \,[Mpc^{-3}]$',fs=(12,4))
    mtypes=['logMstar','logMHI','logMH2']
    v1=(boxsize1)**3
    v2=(boxsize2)**3
    for i,mtype in enumerate(mtypes):
        keep=np.logical_and(data1['logMstar'] > 8.0,data1[mtype] > 7.0)
        mass=(data1[mtype][keep]).to_numpy()
        xyhistplot(mass,axis[i],weight=1./v1,Nbins=50,log=True,c=color['b'],label='SC-SAM')
        keep=np.logical_and(data1['central']==True,data1[mtype] > 7.0)
        mass=(data1[mtype][keep]).to_numpy()
        xyhistplot(mass,axis[i],weight=1./v1,Nbins=50,log=True,c=color['b'],label='SC-SAM cent.')       
        keep=np.logical_and(data2['logMstar'] > 8.0,data2[mtype] > 7.0)
        mass=(data2[mtype][keep]).to_numpy()           
        xyhistplot(mass,axis[i],weight=1./v2,Nbins=50,log=True,c=color['r'],label='OLD-SAM')
        keep=np.logical_and(data2['central']==True,data2[mtype] > 7.0)
        mass=(data2[mtype][keep]).to_numpy()           
        xyhistplot(mass,axis[i],weight=1./v2,Nbins=50,log=True,c=color['r'],label='OLD-SAM cent.')       
    axis[0].set_xlabel(r'$\log M_{*}$',fontsize='x-large')
    axis[1].set_xlabel(r'$\log M_{HI}$',fontsize='x-large')
    axis[2].set_xlabel(r'$\log M_{H_2}$',fontsize='x-large')
    axis[0].legend(fontsize='x-small',loc='lower left')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('testsam.pdf') 

def testsam2():
    names=['NEW-SAM','OLD-SAM']
    mtypes=['logMHI','logMH2','H2frac','logctime']
    yranges=[[7.5,10.8],[7.5,10.8],[0.0,0.95],[8.0,10.8]]
    fb=[True,True,False,False]
    uplims=[7.5,7.5,None,None]
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit_row=[r"$log (M_{HI})"+r"[M_{\odot}]$",r"$log (M_{H2})"+r"[M_{\odot}]$",
                r"$H_{2}$ fraction",r"$log \, \tau^{H_2}_{depl}$  [yr]"]
#    ytit2=r"$f_{below}$"
    f,axs=setup_multiplot(4,2,xtitle=xtit,sharex=True,sharey='row',fs=(6,9))
    right_ticks=[False,True]
    for i in range(2):
        data=get_sim(sim=names[i],sample='all')
        for j in range(4):
            tit=''
            if j==0:
                tit=names[i]
            ll=False
            if i==0 and j==3:
                ll=True
            sim_panel(data,'logMstar',mtypes[j],axs[j][i],xrange=[8.5,11.45],yrange=yranges[j],
                name=tit,fbelow=fb[j],right_ticks=right_ticks[i],line_legend=ll,uplim=uplims[j])
            axs[j][0].set_ylabel(ytit_row[j])
            axs[j][0].set_ylim(yranges[j])

    axs[0][0].set_xlim([8.5,11.45])
    f.align_labels()
#    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("samcomp.pdf")


def nfig1():
    color=plt.rcParams['axes.prop_cycle'].by_key()['color']
    color=['red','blue','green','purple','orange','magenta']
    boxsize={'Eagle':100,'Mufasa':73.5,'TNG100':110.7,'Simba':147,'SC-SAM':110.7}
    Nb={'Eagle':30,'Mufasa':20,'TNG100':30,'Simba':30,'SC-SAM':30}
    names=sim_names()
    f,axis=setup_multiplot(1,3,sharey=True,ytitle='$\phi \,[dex^{-1} Mpc^{-3}]$',fs=(12,4))
    
    mtypes=['logMstar','logMHI','logMH2']
    for j,name in enumerate(names):
        data=get_sim(sim=name,sample='all',mcut=8.0)
        mstar=(data['logMstar']).to_numpy()
        volume=(boxsize[name])**3
        for i,mtype in enumerate(mtypes):
            keep=np.logical_and(data['logMstar'] > 8.0,data[mtype] > 7.0)
            mass=(data[mtype][keep]).to_numpy()
            xyhistplot(mass,axis[i],weight=1./volume,Nbins=Nb[name],log=True,linestyle=':',c=color[j])
            keep=np.logical_and(data['logMstar'] > 9.0,data[mtype] > 7.0)
            mass=(data[mtype][keep]).to_numpy()           
            xyhistplot(mass,axis[i],weight=1./volume,Nbins=Nb[name],log=True,label=name,c=color[j])         

    logm,phi,dx,dy=obs.gmf_GAMA(Wright=True)
    axis[0].plot(logm,np.log10(phi),linestyle='--',c='black',label='Observed')
    logm,logphi=obs.HImf_ALFALFA()
    axis[1].plot(np.log10(logm),np.log10(logphi),linestyle='--',c='black')
#    logm,logphi=obs.H2mf(model='constant')
#    axis[2].plot(logm,logphi,linestyle='--',c='black')    
    logm,logphi=obs.H2mf(model='luminosity')
    axis[2].plot(logm,logphi,linestyle='dashdot',c='black')
    m,phi=obs.H2mf_OR(type='orig')
    axis[2].plot(np.log10(m),np.log10(phi),linestyle='dashdot',c='black')
    #shade between curves - stop at higher y value for two curves
    xend=m[np.log10(phi) > logphi[-1]][-1]
    x=np.append(logm,np.log10(xend))
    y1=np.append(logphi,logphi[-1])
    y2=np.interp(x,np.log10(m),np.log10(phi))
    axis[2].fill_between(x,y1,y2,color='gray',alpha=0.25)
#    m,phi=obs.H2mf_OR(type='ref')
#    axis[2].plot(np.log10(m),np.log10(phi),linestyle='--',c='black')
    m,phi=obs.H2mf_xCOLDGASS()
    axis[2].plot(np.log10(m),np.log10(phi),linestyle='--',c='black')   
#    axis[0].set_ylim([-4.0,-1.5])
    axis[0].set_ylim([-4.7,-1.5])
    axis[0].set_xlim([8.5,12.6])
    axis[1].set_xlim([8.5,11.4])
    axis[2].set_xlim([8.5,11.2])
    axis[0].set_xlabel(r'$\log M_{*}$',fontsize='x-large')
    axis[1].set_xlabel(r'$\log M_{HI}$',fontsize='x-large')
    axis[2].set_xlabel(r'$\log M_{H_2}$',fontsize='x-large')
    axis[0].legend(ncol=2,fontsize='small')
    legend_lines=[mpl.lines.Line2D([0],[0],color='gray',linestyle='-'),
                mpl.lines.Line2D([0],[0],color='gray',linestyle=':')]
    legend_names=[r'$M_* > 10^9 M_{\odot}$',r'$M_* > 10^8 M_{\odot}$']
    axis[1].legend(legend_lines,legend_names,loc='lower left')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('gasmf.pdf')

def nfig2():
    color=plt.rcParams['axes.prop_cycle'].by_key()['color']
    names=sim_names()
    f,axs=plt.subplots()
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='cent',mcut=8.0)
        data=data[data['logMH2'] > 7.5]
        medianline(axs,data['logMH2'],data['log_SFR'],N=20,label=name,c=color[i])
        data=data[data['logMstar'] > 9.0]
        medianline(axs,data['logMH2'],data['log_SFR'],N=20,linestyle='--',c=color[i])
    plt.legend(ncol=2)
    plt.savefig('nfig2.pdf')

def diff_SFS(mtype='logMH2',mcut=8):
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=fr"$log (M_{mname[mtype]})"+r"[M_{\odot}]$" 
    ytit2=r"$f_{below}$"
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,ytitle2=ytit2,sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,gname[mtype])  
    for name,axs in zip(names,f.axes):
        if name==names[0]:
            data=get_GASS(name=name,sample=sample)
            uplim=(data['uplim']==True).to_numpy()
            detect=np.invert(uplim)
            diffSFS=data['log_SFR']-SFS(data['logMstar'],sim='xGASS',Plot=False)
            p=fitmedian(data['log_SFR'][detect],data[mtype][detect],axis=axs) 
            diff_SFR=data[mtype]-(p[0]*data['log_SFR']+p[1]) 
            plot_with_uplim(axs,diffSFS[cent],diff_mtype[cent],uplim=uplim[cent])  
            plot_with_uplim(axs,diffSFS[sat],diff_mtype[sat],uplim=uplim[sat],sat=True)     
            xmids,meds,upfrac=medianline(axs,data['logMstar'][cent],data[mtype][cent],uplim=uplim[cent],
                xrange=[8.5,11.5],color=color['p'],linestyle='--',N=8)
    
def fig_xGASS(mcut=8):
    #makes the xCOLDGASS panel of the figures for all figures
    fsize=5
    mtypes=['logMHI','logMH2']
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'} 
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=r"$log (M_{gas}) \, [M_{\odot}]$"
    f,axis=plt.subplots(2,4,sharex='col',figsize=(12,6))          
    for i,mtype in enumerate(mtypes):
        data=get_GASS(name=gname[mtype],sample='all')
        uplim=(data['uplim']==True).to_numpy()
        detect=np.invert(uplim)
        cent=data['central']==True
        sat=data['central']==False
        #Mgas vs. Mstar plots  Fig 2 and 3
        axs=axis[i][0]
        gass_panel(data,'logMstar',mtype,axs,xrange=[9.0,11.5],name=gname[mtype],ncol=1,
            satcent=True,frac=True)
        pc=fitmedian(data['logMstar'][cent],data[mtype][cent],axis=axs) 
        ps=fitmedian(data['logMstar'][sat],data[mtype][sat],axis=axs)  
        print(f"slope for {mtype}-Mstar line is {pc[0]} for centrals and {ps[0]} for satellites") 

        #sfr vrs Mgas plot Fig 4
        axs=axis[i][1]
        gass_panel(data,mtype,'log_SFR',axs,umark='<',xrange=[7.8,10.],ncol=1,name=gname[mtype])
        p=fitmedian(data[mtype][cent],data['log_SFR'][cent],axis=axs)  
        print(f"slope for sfr-{mtype} line is {p[0]}")                  
  
        #depletion time plots 
        axs=axis[i][2]
        logctime = data[mtype]-data['log_SFR']+np.log10(1.33)
        logctime_lim=np.copy(logctime)
        logctime_lim[uplim]=8.0
        xmids,meds1,upfrac1=medianline(axs,data['logMstar'][cent],logctime[cent],N=8,
            xrange=[9.0,11.25],color=color['p'],linestyle=':')
        xmids,meds2,upfrac2=medianline(axs,data['logMstar'][cent],logctime_lim[cent],N=8,
            xrange=[9.0,11.25],color=color['p'],linestyle=':')
        xmids,meds1,upfrac1=medianline(axs,data['logMstar'],logctime,N=8,
            xrange=[9.0,11.25],color=color['p'],linestyle='--')   
        p=fitmedian(data['logMstar'],logctime,xrange=[9.0,11.25],axis=axs)
        print(f"slope for stellar mass - {mtype} depletion time is {p[0]}") 
        xmids2,meds,upfrac1=medianline(axs,data['logMstar'][sat],logctime[sat],N=7,
            xrange=[9.0,11.25],color='cyan',linestyle='--')
        xmids2,meds,upfrac2=medianline(axs,data['logMstar'][sat],logctime_lim[sat],N=6,
            xrange=[9.0,11.25],color='cyan',linestyle='--')
        axs.fill_between(xmids,meds2,meds1,color=color['p'],hatch='-',linestyle='--',alpha=0.25)
        plot_with_uplim(axs,data['logMstar'][cent],logctime[cent],uplim=uplim[cent])
        plot_with_uplim(axs,data['logMstar'][sat],logctime[sat],uplim=uplim[sat],sat=True)
        uplim_legend(axs,fontsize=fsize,ncol=2,loc='lower right')
        #H2 frac
        if i==1:
            axs=axis[0][3]
            plot_with_uplim(axs,data['logMstar'][cent],data['H2frac'][cent],uplim=uplim[cent])
            plot_with_uplim(axs,data['logMstar'][sat],data['H2frac'][sat],uplim=uplim[sat],sat=True)  
            xmids,meds,upfrac=medianline(axs,data['logMstar'][cent],data['H2frac'][cent],uplim=uplim[cent],
                xrange=[9.0,11.25],color=color['p'],linestyle='--',N=6)
            xmids,meds,upfrac=medianline(axs,data['logMstar'][sat],data['H2frac'][sat],uplim=uplim[sat],
                xrange=[9.0,11.25],color='cyan',linestyle='--',N=6)
            axs=axis[1][3]         
            plot_with_uplim(axs,data['logMH2'][cent],data['H2frac'][cent],uplim=uplim[cent])
            plot_with_uplim(axs,data['logMH2'][sat],data['H2frac'][sat],uplim=uplim[sat],sat=True) 

    axis[1][0].set_xlabel(r'log $M_{stellar} \, [M_{\odot}$]')           
    axis[1][1].set_xlabel(r'log ($M_{H_2})  [M_{\odot}$]')  
    axis[1][2].set_xlabel(r'log $M_{stellar} \, [M_{\odot}$]')
    f.align_labels()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("xGASS.pdf")

def nfig3(mtype='logMH2',sample='all',mcut=8): #fig3 and fig4
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=fr"$log (M_{mname[mtype]})"+r"[M_{\odot}]$" 
    ytit2=r"$f_{below}$"
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,ytitle2=ytit2,sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,gname[mtype])
    nc={'logMH2':2,'logMHI':1}
    loc={'logMH2':'upper left','logMHI':'lower right'}
    rticks=get_rticks()
    for name,axs in zip(names,f.axes):
        if name==names[0]:
            data=get_GASS(name=name,sample=sample)
            gass_panel(data,'logMstar',mtype,axs,xrange=[9.0,11.5],name=gname[mtype],ncol=nc[mtype],
                satcent=True,frac=True,loc=loc[mtype])
        else:
            data=get_sim(sim=name,sample=sample)
            line_legend=False
            if name=='Eagle':
                line_legend=True #add a line legend to this panel
            sim_panel(data,'logMstar',mtype,axs,xrange=[8.5,11.45],yrange=[7.5,10.8],
                name=name,fbelow=True,uplim=7.5,right_ticks=rticks[name],line_legend=line_legend)

    axs.set_ylim([7.5,10.8])
    axs.set_xlim([8.5,11.45])
    f.align_labels()
#    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"{mtype}_Mstar.pdf")   

def fig_h2frac(sample='all',mcut=8.0):
    xtit=r"$log (M_{*}) [M_{\odot}]$"
    ytit=r"$H_{2}$ fraction" 
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,'xCOLDGASS')
    for name,axs in zip(names,f.axes):
        if name=='xCOLDGASS':
            data=get_GASS(name='xCOLDGASS',sample=sample)
            gass_panel(data,'logMstar','H2frac',axs,xrange=[9.0,11.5],name='xCO-GASS',ncol=2,
                loc='upper right',shade=0.01)
        else:
            data=get_sim(sim=name,sample=sample,mcut=mcut)
            if name=='Eagle':
                add_line_legend(axs,ncol=1)
            sim_panel(data,'logMstar','H2frac',axs,xrange=[8.5,11.45],yrange=[0.0,1.0])

    axs.set_xlim([8.5,11.45])
    axs.set_ylim([0,0.98])
    f.align_labels()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("H2frac_Mstar.pdf")    

def fig_h2f_SFS(sample='all',mcut=8.0):
    xtit=r"$\Delta SFSlog$"
    ytit=r"$H_{2}$ fraction" 
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,'xCOLDGASS')
    for name,axs in zip(names,f.axes):
        if name=='xCOLDGASS':
            data=get_GASS(name='xCOLDGASS',sample=sample)
            uplim=data['FLAG_CO']==2
            cent=data['central']==True
            sat=data['central']==False
            diffSFS=data['log_SFR']-SFS(data['logMstar'],sim='xGASS',Plot=False)
            axs.scatter(diffSFS,data['H2frac'],marker='o',s=4)
        else:
            data=get_sim(sim=name,sample=sample,mcut=mcut)
            diffSFS=data['log_SFR']-SFS(data['logMstar'],sim=name,Plot=False)
            hist2dplot(axs,diffSFS,data['H2frac'],fill=True,
                bins=30,range=[[-2,1.5],[0.0,1.0]]) 
        axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
    axs.set_xlim([-2,1])
#    axs.set_ylim([0,0.98])
    f.align_labels()
#    plt.subplots_adjust(bottom=0.1)
    plt.savefig("H2frac_SFS.pdf")
    
def nfig5(sample='all',mcut=8):
    names=sim_names()
    names.insert(0,'xGASS')
    xtit=r'$\log M_{H_2} [M_{\odot}]$'
    ytit=r'$\log SFR [yr{-1}]$'
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True)
    yfit,xfit=obs.H2_SFR()
    for name,axs in zip(names,f.axes):
        if name=='xGASS':
            data=get_xCOLDGASS(sample=sample)
            uplim=data['FLAG_CO']==2
            detect=np.invert(uplim)
            axs.scatter(data['logMH2'][detect],data['log_SFR'][detect],s=4,marker='o',color='blue')
            axs.scatter(data['logMH2'][uplim],data['log_SFR'][uplim],s=4,marker='x',color='red')
        else:
            data=get_sim(sim=name,sample=sample,mcut=mcut)
            axs.scatter(data['logMH2'],data['log_SFR'],s=1,marker=',')
            medianline(axs,data['logMH2'],data['log_SFR'],xrange=[7.0,10.5],color='pink')
            
        x=np.linspace(7.1,10.6)
        axs.plot(x,x-9,linestyle=':',c='black')    
        axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
        axs.plot(xfit,yfit,linestyle='--',color='black')
    axs.set_xlim([7.1,10.6])
    axs.set_ylim([-2.1,1.6])
    plt.savefig('nfig5.pdf')

def gass_panel(data,xfield,yfield,axis,umark='v',xrange=[9,11.5],ncol=1,
        satcent=False,frac=False,name=None,loc='lower right',shade=False):
    uplim=(data['uplim']==True).to_numpy()
    cent=data['central']==True
    sat=data['central']==False
    plot_with_uplim(axis,data[xfield][cent],data[yfield][cent],uplim=uplim[cent],umark=umark)
    plot_with_uplim(axis,data[xfield][sat],data[yfield][sat],uplim=uplim[sat],sat=True,umark=umark)
    uplim_legend(axis,fontsize='xx-small',ncol=ncol,umark=umark,loc=loc,
        frameon=False,markerscale=0.6)
    #median lines
    if satcent:
        if umark=='<':
            ymids,meds,upfrac=medianline(axis,data[yfield][cent],data[xfield][cent],uplim=uplim[cent],
                xrange=[-1,1],color=color['p'],linestyle='--',N=8,invert=True)
#            axis.plot(meds[good],ymids[good],color=color['p'],linestyle='--')
            ymids_sat,meds_sat,upfrac_sat=medianline(axis,data[yfield][sat],data[xfield][sat],
                xrange=[-1,0.75],uplim=uplim[sat],color='magenta',linestyle='--',N=6,invert=True)
#            axis.plot(meds_sat[good],ymids_sat[good],color='magenta',linestyle='--')   
        else:
            xmids,meds,upfrac=medianline(axis,data[xfield][cent],data[yfield][cent],uplim=uplim[cent],
                xrange=xrange,color=color['p'],linestyle='--',N=8)
            xmids_sat,meds_sat,upfrac_sat=medianline(axis,data[xfield][sat],data[yfield][sat],
                uplim=uplim[sat],xrange=xrange,color='magenta',linestyle='--',N=6)
    elif shade:
        xmids,meds1,upfrac1=medianline(axis,data[xfield],data[yfield],N=7,
            xrange=[9.0,11.25],color=color['p'],linestyle='--')
        lowlimit=data[yfield]
        lowlimit[uplim]=shade
        xmids,meds2,upfrac2=medianline(axis,data[xfield],lowlimit,N=7,
            xrange=[9.0,11.25],color=color['p'],linestyle='--')
        axis.fill_between(xmids,meds2,meds1,color=color['p'],hatch='-',linestyle='--',alpha=0.25)     
    else:
        xmids,meds,upfrac=medianline(axis,data[xfield],data[yfield],uplim=uplim,
            xrange=xrange,color=color['p'],linestyle='--',N=8)              
    if frac:
        axis2=axis.twinx()
        axis2.set_ylim([0,1])  
        axis2.set_yticks([])
        axis2.plot(xmids,upfrac,linestyle=':',color=color['y'])
        axis2.plot(xmids_sat,upfrac_sat,linestyle=':',color=color['s'])
       
    if name:  
        if loc=='upper left' or loc=='upper right':
            axis.annotate(name,xy=(0.05,0.8),xycoords='axes fraction')
        else:            
            axis.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')

#make the panel for simulations in figs 2,3,4 and 5
def sim_panel(data,xfield,yfield,axs,name=None,xrange=[8.5,11.45],yrange=[7.5,11.5],
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
        xrange=xrange_mids,color=color['p'],linestyle='--',N=25) 
    xmids_s,meds_s,upfrac_s=medianline(axs,data[xfield][sat],data[yfield][sat],uplim=uplim,
        xrange=xrange_mids,color='magenta',linestyle='--',N=15)
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
        axs2.plot(xmids_c,upfrac_c,linestyle=':',color=color['y'])  
        axs2.plot(xmids_s,upfrac_s,linestyle=':',color=color['s']) 
        if not right_ticks:
            axs2.set_yticks([])
    if line_legend:
        add_line_legend(axs,ncol=2,fontsize='xx-small',frac=fbelow)
    axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')

def test_sim_panel():
    names=sim_names()
    data=get_sim(sim='Mufasa')
    f,axs=setup_multiplot(2,3,sharex=True,sharey=True,fs=(9,6))
    rt=[False,False,True,False,False,True]
    for i,axs in enumerate(f.axes):
        if i==6:
            break
        sim_panel(data,'logMstar','logMH2',axs,fbelow=True,right_ticks=rt[i])
    plt.savefig('testpanel.pdf')

def fig_sfr_mgas(mtype='logMH2',sample='all',mcut=8):
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit=r"$log \, M_{H_2} [M_{\odot}]$"
    nstr=mname[mtype]
    ytit=r"$log \, SFR \, \, [M_{\odot}/yr]$"
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,gname[mtype])
    for name,axs in zip(names,f.axes):
        if name==names[0]:
            data=get_GASS(name=name,sample=sample)
            gass_panel(data,mtype,'log_SFR',axs,umark='<',xrange=[7.8,10.],satcent=True,name=name)       
        else:
            data=get_sim(sim=name,sample=sample,mcut=mcut)
            sat=data['central']==False 
            hist2dplot_with_limit(axs,data[mtype],data['log_SFR'],
                bins=30,range=[[7.5,11.5],[-2,1.5]],fill=True)
            medianline(axs,data[mtype],data['log_SFR'],xrange=[7.5,10.],linestyle='--',
                color=color['p']) 
            medianline(axs,data[mtype][sat],data['log_SFR'][sat],xrange=[7.5,10.],linestyle='--',
                color='magenta') 
            if name=='Eagle':
                add_line_legend(axs,ncol=1,loc='lower right')
            axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
    axs.set_xlim([7.5,10.45])
    axs.set_ylim([-2,1.45])
    plt.savefig(f"sfr_{mtype}.pdf")

def sim_stats():
    names=sim_names()
    for name in names:
        data=get_sim(sim=name)
        noH2=np.invert(np.isfinite(data['logMH2']))
        nosfr=np.invert(np.isfinite(data['log_SFR']))
        both=np.logical_or(noH2,nosfr)
        nodtime=data['logctime']==0.0
        print(noH2.sum(),nosfr.sum(),both.sum(),nodtime.sum())

def fig_depl_time(mtype='logMH2',sample='all',mcut=8):
    mname={'logMH2':'{H_2}','logMHI':'{HI}','logMgas':'{H}'}
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    xtit="$log \, M_{*} [M_{\odot}]$"
    nstr=mname[mtype]
    ytit=r"$log \, \tau^{"+nstr+"}_{depl}$  [yr]"
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,ytitle2=r'$f_{no\_sfr}$',
        sharex=True,sharey=True,fs=(9,6))
    names=sim_names()
    names.insert(0,gname[mtype])
    rticks=get_rticks()
    for name,axs in zip(names,f.axes):
        if name==names[0]:
            data=get_GASS(name=name,sample=sample)
            gass_panel(data,'logMstar','logctime',axs,umark='v',xrange=[7.8,10.],
            satcent=False,name=name,shade=8.5,ncol=2)  
        else:
            #needs to have medianline excluding zeros and upfrac
            data=get_sim(sim=name,sample=sample,mcut=mcut)
            if name=='Eagle':
                add_line_legend(axs,ncol=1,frac=True)
            notdead = np.isfinite(data['logctime'])
            sim_panel(data,'logMstar','logctime',axs,uplim=7.0,xrange=[8.5,11.2],
                yrange=[7.5,11.0],buffer=0.3,name=name,select=notdead,
                fbelow=True,right_ticks=rticks[name])
            
    axs.set_xlim([8.5,11.45])
    axs.set_ylim([7.5,11.4])
    f.align_labels()
    plt.savefig(f"ctime_{mtype}.pdf")

def nfig6(mtype='logMH2'):
    f=plt.figure(constrained_layout=True,figsize=(6,6))
    spec=f.add_gridspec(ncols=3,nrows=4,width_ratios=[3,3,3],height_ratios=[3,3,3,0.3])
#    f,axs=plt.subplots(4,3,sharex='col',figsize=(6,6))
#    plt.subplots_adjust(hspace=0.0)
    tit_Mstar=r"$\log \, M_{stellar}$"
    tit_MH2=r"$\log \, M_{H_2}$"
    tit_sfr=r"$\log \, SFR$"
    #set up colormap
    col1=['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']
    col2=['#c51b7d','#e9a3c9','#fde0ef','#e6f5d0','#a1d76a','#4d9221']
    col3=['#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']
    bounds = [-0.6, -0.3, 0.00, 0.3, 0.6]
    cmap1 = mpl.colors.ListedColormap(col1[1:-1])
    cmap1.set_over(col1[-1])
    cmap1.set_under(col1[0])
    norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    cmap2 = mpl.colors.ListedColormap(col2[1:-1])
    cmap2.set_over(col2[-1])
    cmap2.set_under(col2[0])
    norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N)
    cmap3 = mpl.colors.ListedColormap(col3[1:-1])
    cmap3.set_over(col3[-1])
    cmap3.set_under(col3[0])
    norm3 = mpl.colors.BoundaryNorm(bounds, cmap3.N)

    data=get_GASS()
    uplim=(data['uplim']==True).to_numpy()
    detect=np.invert(uplim)
    cent=data['central']==True
    sat=data['central']==False

    ax = f.add_subplot(spec[0,0])
    p=fitmedian(data[mtype][detect],data['log_SFR'][detect],axis=ax)
    diffMgas=perpdis(data[mtype],data['log_SFR'],p[0],p[1])
    pcm1=ax.scatter(data[mtype][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffMgas[detect],cmap=cmap1,norm=norm1)

    ax = f.add_subplot(spec[0,1])
    SFS(ax,sim='xGASS',Plot=True)
    diffSFS=data['log_SFR']-SFS(data['logMstar'],sim='xGASS',Plot=False)
    ax.scatter(data['logMstar'][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffSFS[detect],cmap=cmap2,norm=norm2)

    ax = f.add_subplot(spec[0,2])
    p=fitmedian(data['logMstar'][detect],data[mtype][detect],axis=ax,linestyle=':') 
    diffMstar=perpdis(data['logMstar'],data[mtype],p[0],p[1])   
    ax.scatter(data['logMstar'][detect],data[mtype][detect],marker='o',s=3,
        c=diffMstar[detect],cmap=cmap3,norm=norm3)

    ax = f.add_subplot(spec[1,0])   
    ax.scatter(data[mtype][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffSFS[detect],cmap=cmap2,norm=norm2)
    ax = f.add_subplot(spec[1,1])    
    ax.scatter(data['logMstar'][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffMstar[detect],cmap=cmap3,norm=norm3)
    ax = f.add_subplot(spec[1,2])      
    ax.scatter(data['logMstar'][detect],data[mtype][detect],marker='o',s=3,
        c=diffMgas[detect],cmap=cmap1,norm=norm1)
    
    ax = f.add_subplot(spec[2,0])     
    ax.scatter(data[mtype][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffMstar[detect],cmap=cmap3,norm=norm3)
    ax.set_xlabel(tit_MH2)
            
    ax = f.add_subplot(spec[2,1]) 
    ax.scatter(data['logMstar'][detect],data['log_SFR'][detect],marker='o',s=3,
        c=diffMgas[detect],cmap=cmap1,norm=norm1)
    ax.set_xlabel(tit_Mstar)

    ax = f.add_subplot(spec[2,2]) 
    ax.scatter(data['logMstar'][detect],data[mtype][detect],marker='o',s=3,
        c=diffSFS[detect],cmap=cmap2,norm=norm2)
    ax.set_xlabel(tit_Mstar)

    ax = f.add_subplot(spec[3,0])     
    cb1=mpl.colorbar.ColorbarBase(ax, cmap=cmap1, norm=norm1,orientation='horizontal',extend='both')
    cb1.set_label(r"$\Delta \overline{SFR}(M_{H_2})$")
    ax = f.add_subplot(spec[3,1])     
    cb2=mpl.colorbar.ColorbarBase(ax, cmap=cmap2, norm=norm2,orientation='horizontal',extend='both')
    cb2.set_label(r"$\Delta \overline{SFS}$")
    ax = f.add_subplot(spec[3,2])     
    cb3=mpl.colorbar.ColorbarBase(ax, cmap=cmap3, norm=norm3,orientation='horizontal',extend='both')
    cb3.set_label(r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$")   
    plt.savefig('new.pdf')

    f,axes = setup_multiplot(2,2,sharex=True,sharey=True,fs=(4,4))
    slope, intercept, r_value, p_value, std_err = stats.linregress(diffSFS[detect],diffMstar[detect])
    axes[0][0].scatter(diffSFS[detect],diffMstar[detect],marker='.',s=3,c=diffMgas[detect])
    axes[0][0].annotate("R={:.2f}".format(r_value),xy=(0.05,0.9),xycoords='axes fraction')
    slope, intercept, r_value, p_value, std_err = stats.linregress(diffMgas[detect],diffMstar[detect])
    axes[0][1].scatter(diffMgas[detect],diffMstar[detect],marker='.',s=3,c=diffSFS[detect])
    axes[0][1].annotate("R={:.2f}".format(r_value),xy=(0.05,0.9),xycoords='axes fraction')   
    slope, intercept, r_value, p_value, std_err = stats.linregress(diffSFS[detect],diffMgas[detect])
    axes[1][0].scatter(diffSFS[detect],diffMgas[detect],marker='.',s=3,c=diffMstar[detect])
    axes[1][0].annotate("R={:.2f}".format(r_value),xy=(0.05,0.9),xycoords='axes fraction')
    x=np.linspace(-1.75,1.75)
    axes[1][0].plot(x,slope*x+intercept,color='red')
    p=fitmedian(data['logMstar'][detect],data['logctime'][detect])
    difftau=perpdis(data['logMstar'],data['logctime'],p[0],p[1])
    axes[1][1].scatter(diffMstar[detect],difftau[detect],marker='.',s=3,c=diffSFS[detect])
#    axes[1][1].axis('off')
    axes[1][0].set_xlim([-1.75,1.75])
    axes[1][0].set_ylim([-1.75,1.75])    
    axes[1][0].set_xlabel(r"$\Delta \overline{SFS}$")
    axes[1][0].set_ylabel(r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$")
    axes[0][0].set_ylabel(r"$\Delta \overline{SFR}(M_{H_2})$")
    axes[0][1].set_xlabel(r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$")
    plt.savefig('nfig6_diffs.pdf')

def fig_all_relations(mtype='logMH2'):
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    names=sim_names()
    names.insert(0,gname[mtype])
    for name in names:
        fig_relations(sim=name,mtype=mtype)

def fig_relations(sim='xGASS',mtype='logMH2',diffs=False):
    mtype2='logMHI'
    mstar=r"$\log (M_{stellar}) \, \, \,  [M_{\odot}]$"
    sfr =r"$\log (SFR) \, \, \, [yr^{-1}]$"
    mgas =r"$\log (M_{H_2}) \, \, \, [M_{\odot}]$"
    ctime=r"$\log (\tau^{c}_{H_2}) \, \, \, [yr^{-1}]$"
    f,axs=plt.subplots(nrows=3,ncols=2,sharex='col',sharey='row',figsize=(7,10))
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    xr_mstar=[8.9,11.7]
    xr_mgas=[7.55,10.2]
    yr_sfr=[-2.3,1.5]
    yr_mgas=[7.55,10.6]
    yr_ctime=[8.2,10.48]
    xfield=['logMstar',mtype]
    xfield=['logMhalo',mtype]
    yfield=['log_SFR',mtype2,'logctime']
    yfield=['log_SFR''logMstar','r_disk']
    xr=[xr_mstar,xr_mgas]
    yr=[yr_sfr,yr_mgas,yr_ctime]
    xrfit=[[9.0,10.5],[8.0,9.5]]
    um=['v','<']
    if sim=='xCOLDGASS':
        data=get_GASS(name='xCOLDGASS')
        uplim=data['uplim']==True
        detect=np.invert(uplim)
        for j in range(3):
            for i in range(2):
                plot_with_uplim(axs[j][i],data[xfield[i]],data[yfield[j]],uplim=uplim,umark=um[i])
                if i==0 and j==0:
                    SFS(axs[j][i],sim='xGASS',Plot=True,linestyle='--',c='red')
                    diff=SFS(data[xfield[i]],Plot=False)
                else:
                    p=fitmedian(data[xfield[i]][detect],data[yfield[j]][detect],xrange=xr[i])
                    x=np.linspace(xr[i][0],xr[i][1],num=10)
                    axs[j][i].plot(x,p[0]*x+p[1],linestyle='--',c='red') 
    else:
        data=get_sim(sim=sim)
        for j in range(3):
            for i in range(2):
                hist2dplot(axs[j][i],data[xfield[i]],data[yfield[j]],
                    fill=True,bins=20,range=[xr[i],yr[j]])
                if (j==0 and i==0):
                    SFS(axs[j][i],sim=sim,Plot=True,linestyle='--',c='red')
                else:
                    p=fitmedian(data[xfield[i]],data[yfield[j]],xrange=xrfit[i])
                    x=np.linspace(xrfit[i][0],xrfit[i][1],num=10)
                    axs[j][i].plot(x,p[0]*x+p[1],linestyle='--',c='red') 

    axs[0][0].set_xlim(xr_mstar)   
    axs[0][1].set_xlim(xr_mgas)  
    axs[0][0].set_ylim(yr_sfr)
    axs[1][0].set_ylim(yr_mgas)
    axs[2][0].set_ylim(yr_ctime)
    axs[0][0].set_ylabel(sfr)
    axs[2][1].set_xlabel(mgas)
    axs[1][0].set_ylabel(mgas)
    axs[2][0].set_xlabel(mstar)
    axs[2][0].set_ylabel(ctime)
    plt.savefig(sim+"_relations.pdf")

def xgass_diffs():
    dsfs=r"$\Delta \overline{SFS}$"
    dsfr=r"$\Delta \overline{SFR}(M_{H_2})$"
    dmg=r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$"
    dtau=r"$\Delta \overline{\tau}_{H_2}^d (M_{stellar})$"
    dfrc=r"$\Delta \overline{f}_{H_2} (M_{stellar})$"
    dr50=r"$\Delta \log \overline{R}_{50} (M_{stellar})$"
    xlabels=[dsfs,dsfr,dmg,dtau,dr50]
    xfields=['logMstar','logMH2','logMstar','logMstar','logMstar']
    yfields=['log_SFR','log_SFR','logMH2','logctime','logRstar']
   
    data=get_GASS(name='xCOLDGASS')
    uplim=data['uplim']==True
    detect=np.invert(uplim)
    diffs=[]
    for i in range(len(xfields)):
        if i==0:
            p=SFS(data['logMstar'][detect],sim='xGASS',Plot=False,params=True)
        else:
            p=fitmedian(data[xfields[i]][detect],data[yfields[i]][detect])
        diffs.append(perpdis(data[xfields[i]],data[yfields[i]],p[0],p[1]))
    
    f,axs=plt.subplots(nrows=4,ncols=4,sharey=True,sharex=True,figsize=(9,9)) 
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    for i in range(4):
        for j in range(4):
            if j>=i:
                print(i,j,j+1)
                axs[j][i].scatter(diffs[i][detect],diffs[j+1][detect],marker='.',s=3)
            else:
                axs[j][i].axis('off')
        axs[3][i].set_xlabel(xlabels[i])
        axs[i][0].set_ylabel(xlabels[i+1])

    plt.savefig("xgass_diffs.pdf")
    
def fig1diff(data,yfield1,yfield2,sim='xGASS',xfield='logMstar',axis=False):
    #asssume we are evaluating both against same xfield, default to Mstellar
    if yfield1=='log_SFR':
        p=SFS(data[xfield],sim=sim,Plot=False,params=True)
    else:
        p=fitmedian(data[xfield],data[yfield1])
    diff1=data[yfield1]-(p[0]*data[yfield1]+p[1])
    if yfield2=='log_SFR':
        p=SFS(data[xfield],sim=sim,Plot=False,params=True)
    else:
        p=fitmedian(data[xfield],data[yfield2])
    diff2=data[yfield2]-(p[0]*data[yfield1]+p[1])
    if plot:
        if len(diff1) < 2000:
            axis.scatter(diff1,diff2,marker='o',s=3)
        else:
            hist2dplot(axis,diff1,diff2)
    return diff1,diff2

def fig_diffs(mtype='logMH2'):
    delta_sfs=r"$\Delta \overline{SFS}$"
    delta_sfr=r"$\Delta \overline{SFR}(M_{H_2})$"
    delta_gas_mass=r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$"
    dtau=r"$\Delta \overline{\tau}_{H_2}^d (M_{stellar}$"
    xlabels=[delta_gas_mass,delta_sfr,delta_sfs]
    f,axs=plt.subplots(nrows=3,ncols=6,sharey=True,sharex=True,figsize=(9,6))
    pad=[-10,-10,0]
    for j in range(3):
        title_axes=f.add_subplot(3,1,j+1,frameon=False)
        title_axes.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
        title_axes.set_xlabel(xlabels[j],labelpad=pad[j])

    plt.subplots_adjust(wspace=0.0,hspace=0.25)
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    names=sim_names()
    names.insert(0,gname[mtype])
    idx=[2,1,1]
    idy=[0,0,2]
    for i,name in enumerate(names):
        if i==0:
            data=get_GASS(name=name)
            cent_detect=np.logical_and(data['uplim']==False,data['central']==True)
            diffs=triple_diff(data[mtype][cent_detect],data['log_SFR'][cent_detect]
                ,data['logMstar'][cent_detect],sim='xGASS')  #SFR,SFS,Mg 
            for j in range(3):
                axs[j][i].scatter(diffs[idx[j]],diffs[idy[j]],marker=',',s=1) 
                slope,intercept,r_val,p_val,std_err = stats.linregress(diffs[idx[j]],diffs[idy[j]])
                axs[j][i].annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
                axs[j][i].annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
        else:
            data=get_sim(sim=name)
            keep=np.logical_and(data['log_SFR'] > -2,data[mtype] > 7.5)
            data=data[keep]
            data.reset_index(inplace=True)
            data=data[data['central']==True]
            data.reset_index(inplace=True)            
            diffs=triple_diff(data[mtype],data['log_SFR'],data['logMstar'],sim=name) #SFR,SFS,Mg
            for j in range(3):
                hist2dplot(axs[j][i],diffs[idx[j]],diffs[idy[j]],bins=25)
                slope,intercept,r_val,p_val,std_err = stats.linregress(diffs[idx[j]],diffs[idy[j]])
                axs[j][i].annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
                axs[j][i].annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
            axs[0][i].annotate(name,xy=(0.05,0.9),xycoords='axes fraction',fontsize='small')
    
    axs[0][0].annotate('xCOLDGASS',xy=(0.05,0.9),xycoords='axes fraction',fontsize='small')
    axs[0][0].set_xlim([-1.3,1.3])
    axs[0][0].set_ylim([-1.3,1.3])
    axs[1][0].set_ylim([-1.3,1.3])
    axs[2][0].set_ylim([-1.3,1.3])
    axs[0][0].set_ylabel(delta_sfr)
    axs[1][0].set_ylabel(delta_sfs)
    axs[2][0].set_ylabel(delta_gas_mass)
    plt.savefig('diffs.pdf')        

def fig_newdiffs(mtype='logMH2'):
    dsfs=r"$\Delta \overline{SFS}$"
    dsfr=r"$\Delta \overline{SFR}(M_{H_2})$"
    dmg=r"$\Delta \overline{{M}}_{H_2} (M_{stellar})$"
    dtau=r"$\Delta \overline{\tau}_{H_2}^d (M_{stellar}$"
    xlabels=[dmg,dsfr,dsfs]
    f,axs=plt.subplots(nrows=3,ncols=6,sharey=True,sharex=True,figsize=(9,6))
    pad=[-10,-10,0]
    for j in range(3):
        title_axes=f.add_subplot(3,1,j+1,frameon=False)
        title_axes.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
        title_axes.set_xlabel(xlabels[j],labelpad=pad[j])

    plt.subplots_adjust(wspace=0.0,hspace=0.25)
    gname={'logMH2':'xCOLDGASS','logMHI':'xGASS','logMgas':'xCOLDGASS'}
    names=sim_names()
    names.insert(0,gname[mtype])
    idx=[2,1,0]
    idy=[3,3,3] #[0,0,2]
    for i,name in enumerate(names):
        if i==0:
            data=get_GASS(name=name)
            cent_detect=np.logical_and(data['uplim']==False,data['central']==True)
            diffs=quad_diff(data[mtype][cent_detect],data['log_SFR'][cent_detect]
                ,data['logMstar'][cent_detect],data['r_disk'][cent_detect],sim='xGASS')
            for j in range(3):
                axs[j][i].scatter(diffs[idx[j]],diffs[idy[j]],marker=',',s=1) 
                slope,intercept,r_val,p_val,std_err = stats.linregress(diffs[idx[j]],diffs[idy[j]])
                axs[j][i].annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
                axs[j][i].annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
        else:
            data=get_sim(sim=name)
            keep=np.logical_and(data['log_SFR'] > -2,data[mtype] > 7.5)
            data=data[keep]
            data.reset_index(inplace=True)
            data=data[data['central']==True] #should match xGASS detection limit
            data.reset_index(inplace=True)    
            diffs=quad_diff(data[mtype],data['log_SFR'],data['logMstar'],data['r_disk'],sim=name)         
            for j in range(3):
                slope,intercept,r_val,p_val,std_err = stats.linregress(diffs[idx[j]],diffs[idy[j]])
                axs[j][i].annotate("r={:.2f}".format(r_val),xy=(0.05,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
                axs[j][i].annotate("m={:.2f}".format(slope),xy=(0.65,0.05)
                    ,xycoords='axes fraction',fontsize='xx-small')
            axs[0][i].annotate(name,xy=(0.05,0.9),xycoords='axes fraction',fontsize='small')
    
    axs[0][0].annotate('xCOLDGASS',xy=(0.05,0.9),xycoords='axes fraction',fontsize='small')
    axs[0][0].set_xlim([-1.3,1.3])
    axs[0][0].set_ylim([-1.3,1.3])
    axs[1][0].set_ylim([-1.3,1.3])
    axs[2][0].set_ylim([-1.3,1.3])
    axs[0][0].set_ylabel(dsfr)
    axs[1][0].set_ylabel(dsfs)
    axs[2][0].set_ylabel(dmg)
    plt.savefig('new_diffs.pdf')        

def fignew():
    data=get_xGASS(sample='cent',cold=True)
    detect=data['FLAG_CO']<2
    uplim=data['FLAG_CO']==2
    f,axs=plt.subplots(2,2,sharex=True)
    axs[0][0].scatter(data['logMstar'][detect],data['log_SFR'][detect],s=4,marker='o')
    axs[0][0].scatter(data['logMstar'][uplim],data['log_SFR'][uplim],s=4,marker='2',c='gray')
    SFS(axs[0][0],sim='xGASS',Plot=True)
    diffSFS=data['log_SFR']-SFS(data['logMstar'],sim='xGASS',Plot=False)
    axs[0][1].scatter(data['logMstar'][detect],diffSFS[detect],s=4,marker='o')
    axs[0][1].scatter(data['logMstar'][uplim],diffSFS[uplim],s=4,marker='2',c='gray')
    axs[1][0].scatter(data['logMstar'][detect],data['logMH2'][detect],s=4,marker='o')
    axs[1][0].scatter(data['logMstar'][uplim],data['logMH2'][uplim],s=4,marker='2',c='gray')
    p=fitmedian(data['logMstar'],data['logMH2'],axis=axs[1][0])
    diffH2=data['logMH2']-(p[0]*data['logMstar']+p[1])
    axs[1][1].scatter(data['logMstar'][detect],diffH2[detect],s=4,marker='o')
    axs[1][1].scatter(data['logMstar'][uplim],diffH2[uplim],s=4,marker='2',c='gray')
    plt.savefig('xGASSfits.pdf')
    f,axs=plt.subplots()
    axs.scatter(diffH2[detect],diffSFS[detect],s=6,marker='o')
    axs.scatter(diffH2[uplim],diffSFS[uplim],s=4,marker='2',c='gray')
    p=np.polyfit(diffH2[detect],diffSFS[detect],1)
    x=np.linspace(-0.75,1.00,num=21)
    axs.plot(x,p[0]*x+p[1],c='red')
    plt.savefig('xGASSres.pdf')

def fig_mstar_rdisk(xfield,yfield,plotone=False):
    names=sim_names()
    names.insert(0,'xGASS') 
    xlabel=field_labels(xfield)
    ylabel=field_labels(yfield)
    f,axs=setup_multiplot(2,3,fs=(9,6),xtitle=xlabel,ytitle=ylabel,
        sharex=True,sharey=True)
    xrange=[8,11.9] #stellar mass range
    yrange=[0,15] #rdisk range
    xvals=np.linspace(8,11.5)
    for ax,name in zip(f.axes,names):
        if name=='xGASS':
            data=get_GASS() #centrals and sats the same
            p=fitmedian(data[xfield],np.log10(data[yfield]))
            diff=data[yfield]-10**(p[0]*data[xfield]+p[1])
            ax.scatter(data[xfield],data[yfield],marker='o',s=3)
    #        medianline(ax,(data['logMstar']).to_numpy(),(data['R50KPC']).to_numpy(),
    #            linestyle='--',label='median',c='brown')
            ax.plot(xvals,10**(p[0]*xvals+p[1]),color='red')
        else:
            data=get_sim(sim=name)
            p=fitmedian(data[xfield],np.log10(data[yfield]))
            diff=data[yfield]-10**(p[0]*data[xfield]+p[1])            
            hist2dplot(ax,data[xfield],data[yfield],fill=True,bins=30,
                range=[xrange,yrange])
    #        medianline(ax,(data['logMstar']).to_numpy(),(data['r_disk']).to_numpy(),
    #            linestyle='--',label='median',c='brown')
            ax.plot(xvals,10**(p[0]*xvals+p[1]),color='red')
        ax.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
        ax.set_ylim(yrange)
    plt.savefig('fig_mstar_rdisk.pdf')  


def fig_mstar_mhalo():
    names=['Mufasa','TNG100','Simba','SC-SAM']
    f,axs=setup_multiplot(2,2,fs=(6,6),xtitle=r'log $M_{halo}$',ytitle=r'log $M_{stellar}$',
        sharex=True,sharey=True)
    xrange=[10,13.9]
    yrange=[8,11.9]
    for ax,name in zip(f.axes,names):
        data=get_sim(sim=name)
        cent=data['central']==True
        sat=data['central']==False 
        hist2dplot(ax,data['logMhalo'][cent],data['logMstar'][cent],fill=True,bins=30,
            range=[xrange,yrange])
        ax.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
    plt.savefig('fig_mstar_mhalo.pdf')

def fig1():
    xtit=r'$\log_{10}(M_{stellar}) \, \, [M_{\odot}]$'
    f,axs=setup_multiplot(2,2,xtitle=xtit,fs=(6,6),sharex=True,sharey=True)
    names=['HIcent','HIsat','H2cent','H2sat']
    for ax,name in zip(f.axes,names):
        samp,mtype,flag,bad=set_values(name)
        cold=False
        if mtype=='logMH2':
            cold=True
        data=get_xGASS(sample=samp,cold=cold)
        logMstar=np.copy(data['logMstar'].values)
        logMgas =np.copy(data[mtype].values)
        frac=np.copy(10**data['logMH2']/10**data['logMgas'])
        print(np.min(logMstar),np.min(logMgas))
        uplim = data[flag]==bad
        notup = np.invert(uplim)
        ax.scatter(logMstar[uplim],logMgas[uplim],marker='v',color='0.75',s=10) 
        ax.scatter(logMstar[notup],logMgas[notup],c='lightblue',marker='.',s=10)
        xmid,y,xerr,yerr=bootstrap_medians(logMstar,logMgas,Nbins=7)
        ax.errorbar(xmid,y,xerr=xerr,yerr=yerr,color='g',fmt='o')
#        plot_linfit(ax,xmid,y,color='m')
        plot_selectfunc(ax,mtype=mtype)
       
    axs[0][0].text(10.25,7.75,"centrals only")
    axs[0][1].text(10.25,7.75,"satellites only")
    axs[1][0].text(10.25,7.75,"centrals only")
    axs[1][1].text(10.25,7.75,"satellites only")
    axs[0][0].set_ylabel(r'$\log_{10}(M_{HI}) \, \, [M_{\odot}]$')
    axs[1][0].set_ylabel(r'$\log_{10}(M_{H_2}) \, \, [M_{\odot}]$')
    f.align_labels()
    plt.savefig('fig1.pdf')

def fig2a(sample='cent'):
    data=get_xGASS(sample=sample)
    uplim=data['HI_FLAG']==99
    notup=np.invert(uplim)
    frac=(10**data['logMHI']/(10**data['logMstar']+10**data['logMHI']))
    data2=get_xGASS(sample=sample,cold=True)
    uplim2=data2['FLAG_CO']==2
    notup2=np.invert(uplim2)
    frac2=(10**data2['logMH2']/10**data2['logMgas'])
    print(np.max(frac),np.max(frac2),np.mean(frac),np.mean(frac2))
    ytit=r'$\log_{10}(SFR) \, \, [M_{\odot}/yr]$'
    f,axs=setup_multiplot(2,2,ytitle=ytit,fs=(6,6),sharey=True)
    axs[0][0].scatter(data['logMstar'][uplim],data['log_SFR'][uplim]
        ,marker='<',c='0.75',cmap='inferno',s=1)   
    axs[0][0].scatter(data['logMstar'][notup],data['log_SFR'][notup]
        ,marker='.',c=frac[notup],cmap='inferno',s=5)  
    axs[0][1].scatter(data['logMHI'][uplim],data['log_SFR'][uplim]
        ,marker='<',c='0.75',cmap='inferno',s=1)    
    axs[0][1].scatter(data['logMHI'][notup],data['log_SFR'][notup]
        ,marker='.',c=frac[notup],cmap='inferno',s=5)
    axs[1][0].scatter(data2['logMstar'][uplim2],data2['log_SFR'][uplim2]
        ,marker='<',c='0.75',cmap='inferno',s=1)       
    axs[1][0].scatter(data2['logMstar'][notup2],data2['log_SFR'][notup2]
        ,marker='.',c=frac2[notup2],cmap='inferno',s=5)
    axs[1][1].scatter(data2['logMH2'][uplim2],data2['log_SFR'][uplim2]
        ,marker='<',c='0.75',cmap='inferno',s=1)        
    axs[1][1].scatter(data2['logMH2'][notup2],data2['log_SFR'][notup2]
        ,marker='.',c=frac2[notup2],cmap='inferno',s=5)
    axs[0][0].set_ylim([-2.4,1.5])
    axs[0][0].set_xlim(8.8,11.5)
    axs[0][1].set_xlim(7.5,10.5)
    axs[1][0].set_xlim(8.8,11.5)
    axs[1][1].set_xlim(7.5,10.5)
    axs[1][0].set_xlabel(r'$\log_{10}({M_{stellar}})$')
    axs[1][1].set_xlabel(r'$\log_{10}(M_{gas})$')
    f.align_labels()
    plt.savefig('fig2a.pdf')

def fig2(sample='all'):
    data=get_xGASS(sample=sample)
    keep=np.invert(np.isnan(data['log_SFR']))
    data=data[keep]
    data2=get_xGASS(cold=True,sample=sample)
    keep=np.invert(np.isnan(data2['log_SFR']))
    data2=data2[keep]
    uplim=data['HI_FLAG']==99
    uplim2=data2['FLAG_CO']==2
    notup=np.invert(uplim)
    notup2=np.invert(uplim2)
    ytit=r'$\log_{10}(SFR_{inst}) \, \, [M_{\odot}/yr]$'
    f,axs=setup_multiplot(2,2,ytitle=ytit,fs=(6,6),sharey=True)
    diffHI=data['logMHI'][notup]-(0.09975847*data['logMstar'][notup]+8.31558395)
    diffH2=data2['logMH2'][notup2]-(0.35934293*data2['logMstar'][notup2]+5.07840006)
#    diffHI=data['logMstar'][notup]-(0.41193177*data['logMHI'][notup]+6.50892838)
#    diffH2=data2['logMstar'][notup2]-(0.76338016*data2['logMH2'][notup2]+3.50012149)   
    axs[0][0].scatter(data['logMstar'][notup],data['log_SFR'][notup]
        ,marker='.',c=diffHI, cmap='inferno')
    axs[0][0].scatter(data['logMstar'][uplim],data['log_SFR'][uplim]
        ,marker='v',color='0.75',s=10)
    axs[0][0].text(9.0,1.0,'xGASS')
    
    axs[0][1].scatter(data['logMHI'][notup],data['log_SFR'][notup]
        ,marker='.',c=diffHI,cmap='inferno')
    axs[0][1].scatter(data['logMHI'][uplim],data['log_SFR'][uplim]
        ,marker='<',color='0.75',s=10)
    bhi,mhi=plot_linfit(axs[0][1],data['logMHI'][notup],data['log_SFR'][notup],color='r')
    std=np.std(data['log_SFR'][notup]-(bhi+mhi*data['logMHI'][notup]))
    print(f"std H2 = {std}")
    axs[0][1].text(10.0,-2.0,r'$HI$')
    
    axs[1][0].scatter(data2['logMstar'][notup2],data2['log_SFR'][notup2]
        ,marker='.',c=diffH2,cmap='plasma')
    axs[1][0].scatter(data2['logMstar'][uplim2],data2['log_SFR'][uplim2]
        ,marker='v',color='0.75',s=10)
    axs[1][0].text(9.0,1.0,'xCOLDGASS')
    
    axs[1][1].scatter(data2['logMH2'][notup2],data2['log_SFR'][notup2]
        ,marker='.',c=diffH2,cmap='plasma')
    axs[1][1].scatter(data2['logMH2'][uplim2],data2['log_SFR'][uplim2]
        ,marker='<',color='0.75',s=10)
    bh2,mh2=plot_linfit(axs[1][1],data2['logMH2'][notup2],data2['log_SFR'][notup2],color='r')
    std=np.std(data2['log_SFR'][notup2]-(bh2+mh2*data2['logMH2'][notup2]))
    print(f"std H2 = {std}")
    axs[1][1].text(10.0,-2.0,r'$H_2$')
    axs[1][0].set_xlabel(r'$\log_{10}({M_{stellar}})$')
    axs[1][1].set_xlabel(r'$\log_{10}(M_{gas})$')
    axs[0][0].set_ylim([-2.4,1.5])
    axs[0][0].set_xlim(8.8,11.5)
    axs[0][1].set_xlim(7.5,10.5)
    axs[1][0].set_xlim(8.8,11.5)
    axs[1][1].set_xlim(7.5,10.5)
    f.align_labels()
    plt.savefig('fig2.pdf')
    return bhi,mhi,bh2,mh2

def fig3(H2=False):
    names=sim_names()
    xtit=r"$log_{10}(M_{stellar}) [M_{\odot}]$"
    ytit=r"$log_{10}(M_{HI}) [M_{\odot}]$" 
    figname="fig3.pdf"
    name='HIcent'
    Nc=len(names)
    ylim=[7.9,10.5]
    xlim=[8.5,11.5]
    xtxt_cent=8.6
    xtxt_sat=8.6
    ytxt_cent=8.2
    ytxt_sat=10.2
    if H2:
        ytit=r"$log_{10}(M_{H_2}) [M_{\odot}]$" 
        figname="fig4.pdf"
        name='H2cent'
        ylim=[7.9,10.5]
        xlim=[8.5,11.5]
        xtxt_cent=9.5
        ytxt_cent=8.0
        ytxt_sat=10.2

    samp,mtype,flag,bad=set_values(name)
    f,axs=setup_multiplot(2,Nc,xtitle=xtit,ytitle=ytit,sharex=True,sharey=True)
    Nb=7
    xmid1,y1,xerr1,yerr1=xgass_boot_mids(sample='cent',mtype=mtype,Nbins=Nb)
    xmid2,y2,xerr2,yerr2=xgass_boot_mids(sample='sat',mtype=mtype,Nbins=Nb)
    
    for i,name in enumerate(names):
        data1=get_sim(sim=name,sample='cent')
        data2=get_sim(sim=name,sample='sat')
        upper_limit1=None
        upper_limit2=None
        Nbins1=30
        Nbins2=15

        logMstar,logMgas=random_subsample(data1,['logMstar',mtype],Nmax=1000)
        plot_with_uplim(axs[0][i],logMstar,logMgas,uplim=upper_limit1)
        logMstar=np.copy(data1['logMstar'].values)
        logMgas=np.copy(data1[mtype].values)
        plot_percent(axs[0][i],logMstar,logMgas,xrange=[9.0,11.5]
                ,Nbins=Nbins1,name="cent_"+name)
        plot_selectfunc(axs[0][i],mtype=mtype)
        axs[0][i].errorbar(xmid1,y1,xerr=xerr1,yerr=yerr1,color='g',fmt='o')
        axs[0][i].set_xlim(xlim)
        axs[0][i].set_ylim(ylim)
        axs[0][i].text(xlim[0]+0.15,ylim[1]-0.25,name)
        axs[0][i].tick_params(axis='both', which='major', labelsize=14)

        logMstar,logMgas=random_subsample(data2,['logMstar',mtype],Nmax=1000)
        plot_with_uplim(axs[1][i],logMstar,logMgas,uplim=upper_limit2)
        logMstar=np.copy(data2['logMstar'].values)
        logMgas=np.copy(data2[mtype].values)
        plot_percent(axs[1][i],logMstar,logMgas,xrange=[9.0,11.5]
                ,Nbins=Nbins2,name="sat_"+name)
        plot_selectfunc(axs[1][i],mtype=mtype)
        axs[1][i].errorbar(xmid2,y2,xerr=xerr2,yerr=yerr2,color='g',fmt='o')
        axs[1][i].set_xlim(xlim)
        axs[1][i].set_ylim(ylim)
        axs[1][i].tick_params(axis='both', which='major', labelsize=14)

    axs[0][0].text(xtxt_cent,ytxt_cent,"Centrals")
    axs[1][0].text(xtxt_sat,ytxt_sat,"Satellites")
    f.align_labels()
    plt.savefig(figname)    

def fig5(H2=False, Total=False, line=None):
    names=sim_names()
    ytit=r"$log_{10}(SFR_{inst}) [M_{\odot}/yr]$"
    xtit=r"$log_{10}(M_{HI}) [M_{\odot}]$" 
    figname="fig5.pdf"
    name='HIcent'
    Nc=len(names)
    lrange=[8.5,10.5]
    if H2:
        xtit=r"$log_{10}(M_{H_2}) [M_{\odot}]$" 
        figname="fig6.pdf"
        name='H2cent'
        lrange=[8.0,10.0]
    if Total:
        xtit=r"$log_{10}(M_{H_2} + M_{HI}) [M_{\odot}]$" 
        figname="fig6a.pdf"
        name='Totalcent'
        lrange=[8.0,10.0]     
    samp,mtype,flag,bad=set_values(name)
    f,axs=setup_multiplot(1,Nc,xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    sfr='log_SFR'
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='all')
        logM,logSFR=random_subsample(data,[mtype,sfr],Nmax=1000)
        logSFR=set_lower_limits(logSFR,-2.0)
        logM=set_lower_limits(logM,8.0)
#        hist,xedg,yedg=np.histogram2d(logM,logSFR,bins=50)
#        print(np.min(hist),np.max(hist))
#        axs[i].hist2d(logM,logSFR,bins=50,cmap=plt.cm.jet
#            ,range=[[8.0, 10.5], [-2.25, 1.5]],density=True,cmin=0.3)
        axs[i].set_xlim(7.95,10.4)
        axs[i].scatter(logM,logSFR,marker='.',color='c')
        lowlim=np.logical_or(logSFR < -1.999,logM < 8.01)
        axs[i].scatter(logM[lowlim],logSFR[lowlim],marker='v',color='0.75')
        axs[i].text(8.25,1.1,name,color='k')
        if line:
            x=np.linspace(lrange[0],lrange[1])
            axs[i].plot(x,line[1]+line[0]*x,
                linestyle='--',color='r',linewidth=5)
            good=np.logical_and(data[mtype] > 8.0,data['log_SFR'] > -1)
            b,m=plot_linfit(axs[i],data[mtype][good],data['log_SFR'][good],color='orange')
            print("slope {:3.1f}, intercept {:3.1f}".format(m,b))
    
    axs[0].set_ylim(-1.6,1.3)
    f.align_labels()
    plt.savefig(figname)
    
def fighist():
    data=get_sim(sim='Simba')
    massbins=np.arange(8.5,12.0,step=0.5)
    Nrows=len(massbins)-1
    f,axs=setup_multiplot(Nrows,2,sharey=True,fs=(3,6))  
    h2_range=[7.0,11]
    sfr_range=[-2,1.5]
    for i in range(Nrows): 
        keep = np.logical_and(data['logMstar'] > massbins[i], data['logMstar'] < massbins[i+1])
        axs[i][0].hist(data['logMH2'][keep],bins=30,range=h2_range,density=True,histtype='step')  
        axs[i][1].hist(data['log_SFR'][keep],bins=30,range=sfr_range,density=True,histtype='step')
    
    axs[Nrows-1][1].set_xlabel(r'$\log (SFR_{inst})$')
    axs[Nrows-1][0].set_xlabel(r'$\log M_{H_2}$')
    plt.savefig('fighist.pdf')

def fig7():
    names=sim_names()
    Nc=len(names)
    xtit=r'$\log M_{H_2}$'
    ytit=r'$\log SFR$'
    f,axs=setup_multiplot(1,Nc,xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='cent')
        frac=10**data['logMH2']/(10**data['logMgas'])
#        frac=data['logMH2']
        im = axs[i].scatter(data['logMHI'],data['log_SFR'],marker=',',s=1,c=frac, cmap='inferno')
        axs[i].text(8.0,0.8,name)
    axs[0].set_xlim([8.0,10.])
    axs[0].set_ylim([-2,1])
    f.colorbar(im,ax=axs[Nc-1])
    plt.savefig('fig7.pdf')

def fig_meds():
    ss=['cent','cent','sat','sat']
    ms=['logMHI','logMH2','logMHI','logMH2']
    f,ax=plt.subplots()
    for s,m in zip(ss,ms):
        xmid,y,xerr,yerr=xgass_boot_mids(sample=s,mtype=m,Nbins=10)
        ax.errorbar(xmid,y,xerr=xerr,yerr=yerr,label=m+" "+s)
    ax.legend()
    plt.savefig("figmeds.pdf")

def xgass_limits():
    f,axs=plt.subplots(ncols=2,sharey=True)
    mtypes=['logMHI','logMH2']
    for i,mtype in enumerate(mtypes):
        if mtype=='logMHI':
            data=get_xGASS(sample='all')
            uplim=data['HI_FLAG']==99 
        if mtype=='logMH2':
            data=get_xGASS(sample='all',cold=True)
            uplim=data['FLAG_CO']==2

        gas_fraction=np.log10((10**data[mtype])/(10**data['logMstar']))
        axs[i].scatter(data['logMstar'],gas_fraction,marker='+',color='cornflowerblue')
        axs[i].scatter(data['logMstar'][uplim],gas_fraction[uplim],marker='v',color='cyan')
        plot_selectfunc(axs[i],mtype=mtype,frac=True)

    axs[0].set_xlabel(r'$\log M_{stellar}$')
    axs[1].set_xlabel(r'$\log M_{stellar}$')   
    axs[0].set_ylabel('Gas Fraction')
    plt.savefig('limits.pdf')

def fig_xgass_flags(sample='cent'):
    data=get_xGASS(sample=sample)
    c=['c','b','g','m','r','k']
    f,ax=plt.subplots()
    up_lim=data['HI_FLAG']==99
    ax.scatter(data['logMstar'][up_lim],data['logMHI'][up_lim]
        ,marker='.',s=5,facecolors='none', edgecolors='y')
    for i in range(6):
        keep=data['HI_FLAG']==i
        print(i,keep.sum())
        ax.scatter(data['logMstar'][keep],data['logMHI'][keep]
            ,color=c[i],marker='.',s=1,label=f"HI_FLAG = {i}")

    ax.legend()
    plot_percent(ax,data['logMstar'],data['logMHI'],xrange=[9,11.5],Nbins=10)
    ax.set_ylabel(r"log$_{10} ({M_{HI}})$")
    ax.set_xlabel(r"log$_{10} (M_{star})$")
    plt.savefig("xgass_flags.pdf")

def sfr_gass(s='all'):
    mtypes=['logMstar','logMgas','logMHI','logMH2']
    names=sim_names()
    data=get_xGASS(cold=True,sample=s)
    f,axs=setup_multiplot(len(names),len(mtypes),ytitle='log SFR',sharey=True,fs=(8,8))
    for i,mtype in enumerate(mtypes):
        plot_with_uplim(axs[0][i],data[mtype],data['log_SFR'],uplim=data['FLAG_CO']==2)

    for j,name in enumerate(names):
        data=get_sim(sim=name)
        logMstar,logMgas,logMHI,logMH2,logSFR=random_subsample(data,['logMstar','logMgas','logMHI','logMH2','log_SFR'],Nmax=1000)
        fracH2=10**logMH2/(10**logMgas)
        axs[j+1][0].scatter(logMstar,logSFR,s=1,c=fracH2)   
        axs[j+1][1].scatter(logMgas,logSFR,s=1,c=fracH2)
        axs[j+1][2].scatter(logMHI,logSFR,s=1,c=fracH2)
        axs[j+1][3].scatter(logMH2,logSFR,s=1,c=fracH2)
#       axs[[i].set_xlabel(mtype) 
    plt.show()

def fig_xcoldGASS():
    f,axs=plt.subplots(2,1,sharex=True)
    samp=['cent','sat']
    for i,s in enumerate(samp):
        data=get_xGASS(cold=True,sample=s)
        uplim=data['FLAG_CO']==2
        Nbins=8
        logMstar=(data['logMstar']).to_numpy()
        logMH2=(data['logMH2']).to_numpy()
        plot_with_uplim(axs[i],logMstar,logMH2,uplim=uplim)
        plot_percent(axs[i],logMstar,logMH2,xrange=[9.0,11.5],Nbins=Nbins)
        xmid,y,xerr,yerr=bootstrap_medians(logMstar,logMH2,xrange=[9.0,11.5],Nbins=8)
        axs[i].errorbar(xmid,y,xerr=xerr,yerr=yerr,color='g',fmt='o')
        axs[i].set_ylim([7.75,10.5])
        
    axs[0].set_xlabel(r"log$_{10} (M_{star})\,\, [M_{\odot}]$")    
    axs[0].set_ylabel(r"log$_{10} ({M_{H_2}})\,\, [M_{\odot}]$")
    plt.savefig('xcoldgass.pdf')

def fig_msfr(sample='cent'):
    names={'logMHI','logMH2'}
    get={'logMHI':get_xGASS,'logMH2':get_xCOLDGASS}
    xtit=r"log$_{10} (M_{star}) \, \, [M_{\odot}]$"
    ytit=r"log$_{10} (SFR) \, \, [M_{\odot}/yr]$"
    f,axs=setup_multiplot(1,2,xtitle=xtit,ytitle=ytit)
    for i,name in enumerate(names):
        data=get[name](sample=sample)
        uplim=np.isnan(data['log_SFR'])
        data['log_SFR'][uplim]= -2.8
        plot_with_uplim(axs[i],data['logMstar'],data['log_SFR'],uplim=uplim)
        sc=axs[i].scatter(data['logMstar'],data['log_SFR']
            ,marker='.',c=data['rank'],cmap='plasma')
        
    axs[0].text(9,1,"HI")
    axs[1].text(9,1,r"$H_2$")
    cbaxes=f.add_axes([0.9,0.11,0.02,0.77])
    plt.colorbar(sc,cax=cbaxes)
    f.align_labels()
    plt.savefig("fig_msfr.pdf")


def fig_satcent(H2=False):
    mtype='logMHI'
    names=['xGASS','Illustris','EAGLE','MUFASA','SC-SAM']
    get={'xGASS':get_xGASS,'Illustris':get_Illustris,'EAGLE':get_EAGLE
    ,'MUFASA':get_MUFASA,'SC-SAM':get_SCSAM}
    xtit=r"$log_{10}(M_{stellar}) [M_{\odot}]$"
    ytit=r"$log_{10}(M_{HI}) [M_{\odot}]$" 
    figname="satcent_HI.pdf"
    flag='HI_FLAG'
    bad=99
    if H2:
        names[0]='xCOLDGASS'
        get={'xCOLDGASS':get_xCOLDGASS,'Illustris':get_Illustris,'EAGLE':get_EAGLE
            ,'MUFASA':get_MUFASA,'SC-SAM':get_SCSAM}
        mtype='logMH2'
        ytit=r"$log_{10}(M_{H_2}) [M_{\odot}]$" 
        figname="satcent_H2.pdf"
        flag='FLAG_CO'
        bad=2
    f,axs=setup_multiplot(2,5,xtitle=xtit,ytitle=ytit)
    Nbins=12
    xmid1=np.zeros(Nbins);xmid2=np.zeros(Nbins);y1=np.zeros(Nbins);y2=np.zeros(Nbins)
    xerr1=0.0;xerr2=0.0;yerr1=[];yerr2=[]
    for i,name in enumerate(names):
        data1=get[name](sample='cent')
        data2=get[name](sample='sat')
        upper_limit1=None
        upper_limit2=None
        if (name is 'xGASS' or name is 'xCOLDGASS'):
            upper_limit1=data1[flag]==bad
            upper_limit2=data2[flag]==bad
            xmid1,y1,xerr1,yerr1=bootstrap_medians(data1['logMstar']
                ,data1[mtype],xrange=[9.0,11.5],Nbins=Nbins)
            xmid2,y2,xerr2,yerr2=bootstrap_medians(data2['logMstar']
                ,data2[mtype],xrange=[9.0,11.5],Nbins=Nbins)
        else:
            upper_limit1=None
            upper_limit2=None
            Nbins=30

        logMstar,logMHI=random_subsample(data1,['logMstar',mtype],Nmax=5000)
        plot_with_uplim(axs[0][i],logMstar,logMHI,uplim=upper_limit1)
        plot_percent(axs[0][i],data1['logMstar'],data1[mtype],xrange=[9.0,11.5]
                ,Nbins=Nbins,name="cent_"+name)
        plot_selectfunc(axs[0][i])
        axs[0][i].errorbar(xmid1,y1,xerr=xerr1,yerr=yerr1,color='0.75',fmt='x')
        axs[0][i].set_xlim([8.5,11.5])
        axs[0][i].set_ylim([7.5,10.5])
        axs[0][i].text(8.75,10.25,name)
        axs[0][i].tick_params(axis='both', which='major', labelsize=14)

        logMstar,logMHI=random_subsample(data2,['logMstar',mtype],Nmax=5000)
        plot_with_uplim(axs[1][i],logMstar,logMHI,uplim=upper_limit2)
        plot_percent(axs[1][i],data2['logMstar'],data2[mtype],xrange=[9.0,11.5]
                ,Nbins=Nbins,name="sat_"+name)
        plot_selectfunc(axs[1][i])
        axs[1][i].errorbar(xmid2,y2,xerr=xerr2,yerr=yerr2,color='0.75',fmt='x')
        axs[1][i].set_xlim([8.5,11.5])
        axs[1][i].set_ylim([7.75,10.5])
        axs[1][i].tick_params(axis='both', which='major', labelsize=14)

    axs[0][0].text(10.5,8.1,"Centrals")
    axs[1][0].text(10.5,8.1,"Satellites")
    f.align_labels()
    plt.savefig(figname)

def fig_allmsfr(sample='all',mtype='logMstar'):
    get,names=setup()
    if mtype=='logMHI':
        xtit=r"$log_{10}(M_{HI}) [M_{\odot}]$"
    else:
        xtit=r"$log_{10}(M_{stellar}) [M_{\odot}]$"
    ytit="$log_{10}(SFR_{inst}) [M_{\odot}/yr]$"
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit)
    for i,name in enumerate(names):
        data=get[name](sample=sample)
        logM,log_SFR,rank=random_subsample(data,[mtype,'log_SFR','rank'])
        sc=axs[i//3][i%3].scatter(logM,log_SFR,marker='.',c=rank,cmap='plasma')
        axs[i//3][i%3].text(9.0,1.0,name)
        axs[i//3][i%3].set_ylim([-2.5,1.5])  
        axs[i//3][i%3].set_xlim([8.6,11.4])
        
    axs[1][2].axis('off')
#    cbaxes=f.add_axes([0.7,0.11,0.02,0.38])
    cbaxes=f.add_axes([0.7,0.25,0.2,0.05])
    plt.colorbar(sc,cax=cbaxes, orientation='horizontal')
    f.align_labels()
    plt.savefig('msfr.pdf')

def plotmeans(axis,plot=True,type='H2SFR',**kwargs):
    p={'H2SFR':[0.81239644,-7.37196337], #H2 vrs SFR
        'HISFR':[ 0.62417987,-6.1490609 ], #HI vrs SFR
        'M*SFR':[0.656,-6.726], #from J20, M* vrs SFR
        'H2HI':[0.5983187, 3.26186368]} #H2 vrs HI
    if plot:
        x=np.linspace(8.0,10.0)
        axis.plot(x,p[type][0]*x+p[type][1],**kwargs)
    else:
        return p[type][0]*axis+p[type][1]

def figtalk(mtype='logMH2',conf=False,cfrac=False):
    if mtype=='logMHI':
        figname='HI_SFR.pdf'
        flag='HI_FLAG'
        bad=99
        xmax=10.5
        xlabel=r'log $M_{HI} \, [M_{\odot}]$'
    elif mtype=='logMgas':
        figname='gas_SFR.pdf'
        flag='good'
        bad=0
        xmax=10.5
        xlabel=r'log $M_{gas} \, [M_{\odot}]$'      
    else:
        figname='H2_SFR.pdf'
        flag='FLAG_CO'
        bad=2
        xmax=10
        xlabel=r'log $M_{H_2} \, [M_{\odot}]$'
    dfGASS=get_xGASS()
    if conf:
        dfGASS=dfGASS[dfGASS['HIconf_flag']==0]
        figname='nconf'+figname
    cent=dfGASS['central']==True
    sat=dfGASS['central']==False
    cd=np.logical_and(dfGASS[flag] <2,cent)
    cu=np.logical_and(dfGASS[flag]==bad,cent)
    sd=np.logical_and(dfGASS[flag] <2,sat)
    su=np.logical_and(dfGASS[flag]==bad,sat)
    uplim=(dfGASS[flag]==bad).to_numpy()
    detect=np.invert(uplim)
    
    f,axs=plt.subplots()
    if cfrac:
        figname='h2frac_'+figname
        good=np.logical_or(cd,sd)
        up=np.logical_or(cu,su)
        axs.scatter(dfGASS[mtype][good],dfGASS['log_SFR'][good],marker='o',s=6,
            c=dfGASS['H2frac'][good])
        axs.scatter(dfGASS[mtype][up],dfGASS['log_SFR'][up],marker='<',s=6,
            c=dfGASS['H2frac'][up])
    else:
        axs.scatter(dfGASS[mtype][cd],dfGASS['log_SFR'][cd],marker='o',s=6,c='blue')
        axs.scatter(dfGASS[mtype][cu],dfGASS['log_SFR'][cu],marker='<',s=6,c='gray')
        axs.scatter(dfGASS[mtype][sd],dfGASS['log_SFR'][sd],marker='o',s=6,c='skyblue')
        axs.scatter(dfGASS[mtype][su],dfGASS['log_SFR'][su],marker='<',s=6,c='silver')    
        medianline(axs,(dfGASS[mtype][cent]).to_numpy(),(dfGASS['log_SFR'][cent]).to_numpy()
            ,uplim=uplim[cent],linestyle='--',label='median centrals',c='brown')
        medianline(axs,(dfGASS[mtype][sat]).to_numpy(),(dfGASS['log_SFR'][sat]).to_numpy()
            ,uplim=uplim[sat],linestyle='-.',label='median satellites',c='orange')
        
    p=np.polyfit(dfGASS[mtype][detect],dfGASS['log_SFR'][detect],1)
    print(p)
    x=np.linspace(8.0,xmax,num=20)
    axs.plot(x,p[0]*x+p[1],linestyle='--',c='red',label='mean all')
    axs.set_ylabel(r'log SFR $\, [yr^{-1}$]',fontsize='xx-large')
    axs.set_xlabel(xlabel,fontsize='xx-large')
    plt.legend()
    plt.savefig(figname)

def figtalk3():
    dfGASS=get_xGASS()
    good=np.logical_and(dfGASS['FLAG_CO'] < 2,dfGASS['HI_FLAG']  < 3)
    good=np.logical_and(good,dfGASS['HIconf_flag']==0)
    bad=np.invert(good)
    cd=np.logical_and(good,dfGASS['central']==True)
    cu=np.logical_and(bad,dfGASS['central']==True)
    sd=np.logical_and(good,dfGASS['central']==False)
    su=np.logical_and(bad,dfGASS['central']==False)
    f,axs=plt.subplots()
    axs.scatter(dfGASS['logMstar'][cd],dfGASS['H2frac'][cd],marker='o',s=6,c='blue')
    axs.scatter(dfGASS['logMstar'][cu],dfGASS['H2frac'][cu],marker='x',s=6,c='gray')
    axs.scatter(dfGASS['logMstar'][sd],dfGASS['H2frac'][sd],marker='o',s=6,c='skyblue')
    axs.scatter(dfGASS['logMstar'][su],dfGASS['H2frac'][su],marker='x',s=6,c='silver')
    medianline(axs,dfGASS['logMstar'][cd],dfGASS['H2frac'][cd],linestyle='--',c='brown')
    medianline(axs,dfGASS['logMstar'][sd],dfGASS['H2frac'][sd],linestyle='-.',c='orange')
    axs.set_ylabel('molecular fraction',fontsize='xx-large')
    axs.set_xlabel('log $M_* [M_{\odot}]$',fontsize='xx-large')
    plt.savefig("gasfrac.pdf")
    #sims
    names=sim_names()
    xtit=r'log $M_{*} \, [M_{\odot}]$'
    ytit='molecular fraction'
    f,axs=setup_multiplot(1,len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True,fs=(12,4))
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='all')
        sub_idx=subsample((data['logMstar']).to_numpy())
        data=data.loc[sub_idx,:]
        H2frac=(10**data['logMH2']/10**data['logMgas'])
        H2frac[np.isnan(H2frac)]=0.0
        cent=(data['central']==True)
        sat=(data['central']==False)
        axs[i].scatter(data['logMstar'][cent],H2frac[cent],marker='o',s=6,c='blue')
        axs[i].scatter(data['logMstar'][sat],H2frac[sat],marker='o',s=6,c='skyblue')   
        medianline(axs[i],data['logMstar'].to_numpy(),H2frac.to_numpy(),
            xrange=[8,11.5],linestyle='--',c='brown')
        axs[i].text(8.1,0.95,name)
    plt.savefig('gfrac_all.pdf')     

def resplot(sample='cent'):
    names=sim_names()
    xtit=r'log $M_{H_2} \, [M_{\odot}]$'
    ytit=r'$\Delta SFS $'
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    norm=plt.Normalize(-1,1)
    for name,axs in zip(names,f.axes):
        data=get_sim(sim=name,sample=sample)
        diffSFS=data['log_SFR']-SFSsim(data['logMstar'],sim=name) 
        axs.scatter(data['logMH2'],diffSFS,marker=',',s=1)
        axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
    axs.set_xlim([8,10.5])
    axs.set_ylim([-2,2])
    plt.savefig('res.pdf')

def figtalk9(sample='cent'):
    sfr=False
    names=sim_names()
    xtit=r'log $M_{*} \, [M_{\odot}]$'
    ytit='log SFR $ \, [M_{\odot}/yr]$'
    f,axs=setup_multiplot(2,3,xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    norm=plt.Normalize(-1,1)
    for name,axs in zip(names,f.axes):
        data=get_sim(sim=name,sample=sample)
        if sfr:
            good=np.logical_and(data['logMH2'] > 8, data['log_SFR'] > - 1.5 )
            p=np.polyfit(data['logMH2'][good],data['log_SFR'][good],1)
            diff=data['log_SFR']- (p[0]*data['logMH2']+p[1])
        else:
            diff=data['log_SFR']-SFSsim(data['logMstar'],sim=name)           
        axs.scatter(data['logMstar'],data['log_SFR'],marker=',',s=1,c=diff,norm=norm)
        SFSsim(axs,sim=name,Plot=True,linestyle='--',color='gray')
        axs.annotate(name,xy=(0.05,0.9),xycoords='axes fraction')
    axs.set_ylim([-2,2])
    axs.set_xlim([8,11.8])
    plt.savefig('allsfs.pdf')

def figtalk4(mtype='logMH2'):
    if mtype=='logMHI':
        flag='HI_FLAG'
        bad=99
        ytit=r'$\log M_{HI} \, [M_{\odot}$]'
        fname='HI_M*.pdf'
    else:
        flag='FLAG_CO'
        bad=2
        ytit=r'$\log M_{H_2} \, [M_{\odot}$]'
        fname='H2_M*.pdf'        
    dfGASS=get_xGASS()
    cd=np.logical_and(dfGASS[flag] < 2,dfGASS['central']==True)
    sd=np.logical_and(dfGASS[flag] < 2,dfGASS['central']==False)  
    cu=np.logical_and(dfGASS[flag]==bad,dfGASS['central']==True)
    su=np.logical_and(dfGASS[flag]==bad,dfGASS['central']==False) 
    cent=dfGASS['central']==True
    sat=dfGASS['central']==False
    f,axs=plt.subplots()
    axs.scatter(dfGASS['logMstar'][cd],dfGASS[mtype][cd],marker='o',s=6,c='blue')
    axs.scatter(dfGASS['logMstar'][sd],dfGASS[mtype][sd],marker='o',s=6,c='skyblue')
    axs.scatter(dfGASS['logMstar'][cu],dfGASS[mtype][cu],marker='v',s=6,c='gray')
    axs.scatter(dfGASS['logMstar'][su],dfGASS[mtype][su],marker='v',s=6,c='silver')   
    medianline(axs,dfGASS['logMstar'][cent],dfGASS[mtype][cent],linestyle='--',c='brown')
    medianline(axs,dfGASS['logMstar'][sat],dfGASS[mtype][sat],linestyle=':',c='orange')
    axs.set_xlabel(r'$\log M_*  \, [M_{\odot}$]',fontsize='xx-large')
    axs.set_ylabel(ytit,fontsize='xx-large')
    plt.savefig(fname)

def figtalk5():
    dfGASS=get_xGASS()
    uplimHI=dfGASS['HI_FLAG']==99
    goodHI=np.logical_and(np.invert(uplimHI),dfGASS['HIconf_flag']==0)
    confHI=np.logical_and(np.invert(uplimHI),dfGASS['HIconf_flag']!=0)
    uplimH2=dfGASS['FLAG_CO']==2
    goodH2=np.invert(uplimH2)
    cent=dfGASS['central']==True
    sat=dfGASS['central']==False
    f,axs=setup_multiplot(1,2,ytitle='molecular fraction',sharey=True,fs=(8,4.5))
    diffH2=dfGASS['log_SFR']-plotmeans(dfGASS['logMH2'],plot=False,type='H2SFR')
    diffHI=dfGASS['log_SFR']-plotmeans(dfGASS['logMHI'],plot=False,type='HISFR')
    axs[0].scatter(diffH2[goodH2],dfGASS['H2frac'][goodH2],marker='o',c='blue')
    axs[0].scatter(diffH2[uplimH2],dfGASS['H2frac'][uplimH2],marker='2',c='gray')
    axs[1].scatter(diffHI[goodHI],dfGASS['H2frac'][goodHI],marker='o',c='blue')
    axs[1].scatter(diffHI[confHI],dfGASS['H2frac'][confHI],marker='o',c='green')   
    axs[1].scatter(diffHI[uplimHI],dfGASS['H2frac'][uplimHI],marker='2',c='gray')
    axs[0].set_xlabel(r'$\Delta SFR (M_{H_2})$',fontsize='x-large')
    axs[1].set_xlabel(r'$\Delta SFR (M_{HI})$',fontsize='x-large')
    plt.savefig('res1.pdf')
    #for sims
    masses=['logMH2','logMHI']
    names=sim_names()
    xtit=r'$\Delta SFR(M_{gas})$'
    ytit=r'$H_2$ gas fraction'
    f,axs=setup_multiplot(len(masses),len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='all')
        detect=np.logical_and(data[masses[0]] > 7.5,data[masses[1]] > 7.5)
        detect=np.logical_and(detect,data['log_SFR'] > -1.5)
        data=data[detect]
        data.reindex()
        for j,mtype in enumerate(masses):
            p=np.polyfit(data[mtype],data['log_SFR'],1)
            diff=data['log_SFR'] - (p[0]*data[mtype]+p[1])
            sub_idx=subsample((data['logMstar']).to_numpy())
            d2=data.loc[sub_idx,:]
            diff=diff[sub_idx]
            H2frac=(10**d2['logMH2']/10**d2['logMgas'])
            H2frac[np.isnan(H2frac)]=0.0
            axs[j][i].scatter(diff,H2frac,marker='o',s=4)
            axs[0][i].text(-1.4,0.8,name)
    axs[0][0].text(-1.4,0.0,'$H_2$')
    axs[1][0].text(-1.4,0.9,'$HI$')
    plt.savefig('allres1.pdf')
    
def figtalk8():
    mtype='logMH2'
    names=sim_names()
    xtit=r'$\log M_{*} \, [M_{\odot}]$'
    ytit=r'$log SFR [M_{\odot}/yr]$'
    f,axs=setup_multiplot(1,len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True,fs=(14,4))
    norm=plt.Normalize(-2.5,1.5)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='cent')
        detect=np.logical_and(data[mtype] > 7.5,data['log_SFR'] > -1.5)
        data=data[detect]
        data.reindex()
        logMstar,logSFR,logMH2=random_subsample(data,['logMstar','log_SFR','logMH2'])
        p=np.polyfit(logMH2,logSFR,1)
        diff=logSFR - (p[0]*logMH2+p[1])
        axs[i].scatter(logMstar,logSFR,marker='o',s=4,c=diff,norm=norm)
        axs[i].text(9.0,1.0,name)
    axs[0].set_ylim([-1.5,1.5])
    axs[0].set_xlim([8,11.5])
    plt.savefig('all_sfr.pdf')

def figtalk6():
    mtype='logMH2'
    names=sim_names()
    xtit=r'log $M_{star} \, [M_{\odot}]$'
    ytit=r'log $M_{H_2} \, [M_{\odot}]$'
    f,axs=setup_multiplot(1,len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='cent')
        axs[i].scatter(data['logMstar'],data['logMH2'],marker='.',s=1)
        uplim=data['logMH2'] < 7.0
        axs[i].scatter(data['logMstar'][uplim],7.05+0.0*data['logMstar'][uplim],marker=',',s=1,c='gold')
        axs[i].text(8,10.5,name)
    axs[0].set_ylim([7,11])
    plt.savefig("allH2vsM*.pdf")

def figtalk7():
    mtype='logMH2'
    names=sim_names()
    xtit=r'log $M_{star} \, [M_{\odot}]$'
    ytit=r'log $M_{H_2} \, [M_{\odot}]$'
    f,axs=setup_multiplot(1,len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='cent')
        p=np.polyfit(data['logMstar'],data['logMH2'])
        axs[i].scatter(data['logMstar'],data['logMH2'],marker='.',s=1)
        uplim=data['logMH2'] < 7.0
        axs[i].scatter(data['logMstar'][uplim],7.05+0.0*data['logMstar'][uplim],marker=',',s=1,c='gold')
        axs[i].text(8,10.5,name)
    axs[0].set_ylim([7,11])
    plt.savefig("allH2vsM*.pdf")

def figtalk2():
    masses=['logMH2','logMHI']
    names=sim_names()
    xtit=r'log $M_{gas} \, [M_{\odot}]$'
    ytit=r'log SFR $\, [yr^{-1}$]'
    f,axs=setup_multiplot(len(masses),len(names),xtitle=xtit,ytitle=ytit,sharey=True,sharex=True)
    type=['H2SFR','HISFR']
    for i,name in enumerate(names):
        data=get_sim(sim=name,sample='all')
        H2up=data['logMH2'] < 7.75
        sub_idx=subsample((data['logMstar']).to_numpy())
        data=data.loc[sub_idx,:]
        cent=(data['central']==True)
        sat=np.invert(cent)
        upsfr=data['log_SFR'] < -1.5
        upval=np.full(upsfr.sum(),-1.45)
        for j,mtype in enumerate(masses):
            up=data[mtype] < 7.5
            upvalm=np.full(up.sum(),7.55)
            axs[j][i].scatter(data[mtype][cent],data['log_SFR'][cent],marker='o',s=6,c='blue')
            axs[j][i].scatter(data[mtype][upsfr],upval,marker='v',s=6,c='gray')
            axs[j][i].scatter(upvalm,data['log_SFR'][up],marker='<',s=6,c='gray')           
            axs[j][i].scatter(data[mtype][sat],data['log_SFR'][sat],marker='o',s=6,c='lightblue')           
            plotmeans(axs[j][i],type=type[j],linestyle='--',c='red')
            medianline(axs[j][i],(data[mtype]).to_numpy(),(data['log_SFR']).to_numpy()
            ,xrange=[8.0,10.0],linestyle='--',c='brown')
            detect=np.logical_and(data[mtype] > 7.5,data['log_SFR'] > -1.5)
            p=np.polyfit(data[mtype][detect],data['log_SFR'][detect],1)
            x=np.linspace(8.0,10.0)
            axs[j][i].plot(x,p[0]*x+p[1],linestyle='-',c='gold')
        axs[0][i].text(7.75,1.0,name)
    axs[0][0].text(10,-1.3,r'$H_2$')
    axs[1][0].text(8,1.2,'HI')
    axs[0][0].set_ylim([-1.5,1.5])
    axs[0][0].set_xlim([7.5,10.5])
    plt.savefig("MvrsSFR.pdf")

def main(args):
    s='cent'
    m='logMstar'
    if args.sat:
        s='sat'
    if args.all:
        s='all'
    if args.HI:
        m='logMHI'
    if args.H2:
        m='logMH2'
    if args.test:
        test()
    if args.check:
        check_alpha()
    if args.sam:
        testsam()
        testsam2()
    if args.new:
        nfig1()
    if args.figs:
        nfig1()
        nfig3(mtype='logMHI')
        nfig3(mtype='logMH2')
        fig_h2frac()
        fig_sfr_mgas(mtype='logMH2')
        fig_depl_time(mtype='logMH2')
        fig_diffs(mtype='logMH2')
    if args.xg:
        if args.H2:
            fig_xcoldGASS()
        else:
            fig_xgass(sample=s)
    if args.msfr:
        fig_msfr(sample=s)
    if args.gs:
        fig_allgasstar(mtype=m,sample=s,use_rank=True)
    if args.sc:
        fig_satcent(H2=args.H2)
    if args.sfr:
        fig_allmsfr(sample=s,mtype=m)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Do Something.")
    parser.add_argument("--test",action='store_true'
        ,help="Test the loading functions")
    parser.add_argument("--check",action='store_true'
        ,help="Run the check routines that make plots checking things")
    parser.add_argument("--figs",action='store_true'
        ,help="Make the figures for the paper") 
    parser.add_argument("--xg",action='store_true'
        ,help="xGASS data plots from paper")
    parser.add_argument("--sat",action='store_true'
        ,help="Use only the satellite sample of galaxies")
    parser.add_argument("--all",action='store_true'
        ,help="Use all galaxies")   
    parser.add_argument("--msfr",action='store_true'
        ,help="Stellar Mass vrs Star Formation Rate")
    parser.add_argument("--gs",action='store_true'
        ,help="Make a gas mass vrs stellar mass plot, set HI or H2")
    parser.add_argument("--sc",action='store_true'
        ,help="Make a gas mass vrs stellar mass plot for centrals and satellites, use --h2 for H2")      
    parser.add_argument("--sfr",action='store_true'
        ,help="Make a sfr vrs stellar mass plot")              
    parser.add_argument("--HI",action='store_true'
        ,help="Use HI instead of stellar in plots, requires other arguments to be set")
    parser.add_argument("--H2",action='store_true'
        ,help="Use H2 instead of HI in plots, requires other arguments to be set")    
    parser.add_argument("--new",action='store_true'
        ,help="Run the newest plot I'm working on")
    parser.add_argument("--sam",action='store_true'
        ,help="Run the plots comparing new and old sam outputs")    
    args=parser.parse_args()
    main(args)
    
