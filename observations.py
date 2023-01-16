import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM,Planck13
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import scipy.stats as stats
# from hmf import cosmo,MassFunction

#Results from others for comparison to SAMs mostly observations but
#also from N-body etc.

def Delta_vir(z,Om0=0.30711,Ob0=0.048,h=0.7):
    #Om0=0.30711,Ob0=0.048,h=0.7 values for BolshoiP according to
    #https://www.cosmosim.org/cms/simulations/bolshoip/
    #Bryan & Norman 99
    cosmo = FlatLambdaCDM(H0=100*h, Om0=Om0, Ob0=Ob0)
    x=cosmo.Om(z)-1
    return 18*np.pi**2+82*x-39*x**2

#halo mass function
def halo_massfunc(z,mrange=[10.0,14],dlog10m=0.2,hmf_model='Tinker08',cum=False):
    h=Planck13.h
    hmf=MassFunction()
    hmf=MassFunction(Mmin=mrange[0],Mmax=mrange[1],dlog10m=dlog10m,hmf_model=hmf_model)
    hmf.update(z=z)
    #y values are h.dmdm,h.dmdlnm,h.dndlog10m
    #returns comoving and M/h, Mpc/h values so in physical
    x=hmf.m/h
    y=hmf.dndlog10m*h**3
    if (cum==True):
        y=hmf.ngtm*h**3
    return(np.log10(x),np.log10(y))

#galaxy stellar mass Function
def plot_mf_region(axis,gfm='GAMA',**kwargs):
    x,y,dx,dy=gmf_GAMA()
    ytop=np.log10(y+dy)
    bad = dy >= y
    dy[bad]=0.9*dy[bad]
    ybot=np.log10(y-dy)
    axis.fill_between(x,ytop,ybot,**kwargs)

def gmf_GAMA(Wright=True,Baldry=False): 
    #Wright et al 2017 https://academic.oup.com/mnras/article/470/1/283/3815542
    #Baldry et al 2012 https://academic.oup.com/view-large/17354749
    if Wright:
        logM,phi,dphi_minus,dphi_plus=np.loadtxt('GAMAII_BBD_GSMFs.csv',
            usecols=(0,1,2,3),unpack=True,delimiter=',')
        return logM,phi,dphi_minus,dphi_plus
    else:
        logM,dlogM,phi,phi_err=np.loadtxt('baldry.dat',usecols=(0,1,2,3),unpack=True)
        phi=phi*1.e-3 #given in 1.e-3 units dex^-1 Mpc^-3
        phi_err=phi_err*1.e-3
        return logM,phi,dlogM,phi_err

def galaxy_massfunc(z):
    zarray=np.array([0.2,0.5,0.8,1.1,1.5,2.0,2.5,3.0,3.5,4.5,5.5])
    idx = np.searchsorted(zarray, z, side="left")
    if idx == 0:
        print("Warning: galaxy mass function valid for z > 0.2")
        idx=1
    if idx >= 10:
        print("Warning: galaxy mass function valid for z < 5.5")
        idx=9
    params={0.2:[10.78,-1.38,1.187e-3,-0.43,1.92e-3],
    0.5:[10.77,-1.36,1.070e-3,0.03,1.68e-3],
    0.8:[10.56,-1.31,1.428e-3,0.51,2.19e-3],
    1.1:[10.62,-1.28,1.069e-3,0.29,1.21e-3],
    1.5:[10.51,-1.28,0.969e-3,0.82,0.064e-3],
    2.0:[10.60,-1.57,0.295e-3,0.07,0.45e-3],
    2.5:[10.59,-1.67,0.228e-3,-0.08,0.21e-3],
    3.0:[10.83,-1.76,0.090e-3,0.0,0.0],
    3.5:[11.10,-1.98,0.016e-3,0.0,0.0],
    4.5:[11.30,-2.11,0.003e-3,0.0,0.0]}
    p=params[zarray[idx]]
    logMstar=np.linspace(7,13,60)
    m=10**(logMstar-p[0])
    phi=(p[2]*m**p[1]+p[4]*m**p[3])*np.exp(-m)
    return logMstar,np.log10(phi)

#halo mass stellar mass relation
def hmsm_relation(z):  #Rodríguez-Puebla, et al 2017 https://doi.org/10.1093/mnras/stx1172
    range={0.1:[10.3,15.0],0.5:[10.8,14.3],1:[11.0,14.1],2:[11.5,13.7],
        3:[10.6,13.3],4:[10.2,12.3],5:[10.2,12.0],6:[10.2,11.7],7:[10.2,11.4]}
    logMstar=np.linspace(7,13,60)
    #params to hmsm fit
    p={0.1:[12.58,10.90,0.48,0.29,1.52],0.25:[12.61,10.93,0.48,0.27,1.46],
       0.5:[12.68,10.99,0.48,0.23,1.39],0.75:[12.77,11.08,0.50,0.18,1.33],
       1:[12.89,11.19,0.51,0.12,1.27],1.25:[13.01,11.31,0.53,0.03,1.22],
       1.5:[13.15,11.47,0.54,-0.10,1.17],1.75:[13.33,11.73,0.55,-0.34,1.16],
       2:[13.51,12.14,0.55,-0.44,0.92],3:[14.02,12.73,0.59,-0.44,0.92],
       4:[14.97,14.31,0.60,-0.44,0.92],5:[14.86,14.52,0.58,-0.44,0.92],
       6:[17.43,19.69,0.55,-0.44,0.92],7:[17.27,20.24,0.52,-0.44,0.92]}
    m=10**(logMstar-p[z][1])
    logMhalo=p[z][0]+p[z][2]*np.log10(m)+((m**p[z][3])/(1+m**(-1*p[z][4])))+0.5
    return logMhalo,logMstar

#mass size relation
def add_mass_size(axis,type='late'): #Lange 2015 https://academic.oup.com/mnras/article/447/3/2603/985449#equ2
    #Seric cut of 2.5 between types
    #late types a,siga,b,sigb
    params=[27.72e-3,3.93e-3,0.21,0.02]
    logMstar=np.linspace(8.5,10.5,20)
    if type is 'early':
        params=[8.37e-5,0.62e-5,0.44,0.02]
        logMstar=np.linspace(9.5,11.3,18)
    mstar=10**logMstar
    r_eff=params[0]*(mstar)**params[2]
    r_eff_top=(params[0]+params[1])*mstar**params[2]
    r_eff_bot=(params[0]-params[1])*mstar**params[2]
    axis.fill_between(logMstar,r_eff_top,r_eff_bot,facecolor='cyan',alpha=0.5)
    return logMstar,r_eff

def mass_size(): #Hashemizadeh 2022 z=0 GAMA survey
    '''linear fit to logmass-logsize relation for GAMA survey'''
    name = ['Total', 'Pure-Disk', 'Disk', 'pseudo-Bulge', 
            'classical-Bulge', 'Elliptical']
    c=['black','blue','cyan','green','orange','red']
    a = [0.264, 0.232, 0.272, 0.323, 0.506, 0.411]
    a_err= [0.032, 0.014, 0.015, 0.036, 0.054, 0.014]
    b = [2.095, 1.695, 2.105, 2.870, 4.971, 3.916]
    b_err = [0.313, 0.135, 0.149, 0.343, 0.533, 0.149]
    sigma= [0.365, 0.194, 0.168, 0.257, 0.294, 0.143]
    sigma_err= [0.004, 0.003, 0.004, 0.007, 0.008, 0.004]
    mrange=[[9,11.5],[9,10],[10,11.5],[9,10],[10,11.5],[9.5,11]]
    for i in range(1,6):
        x=np.linspace(mrange[i][0],mrange[i][1])
        axis.plot(x,a[i]*x-b[i],label=name[i],color=c[i])

    axis.legend()
    plt.show()

#HI mass function
def alfalfa_sample():
    df=pd.read_csv('a100.code12.table2.190808.txt')
    dfgood=df.loc[df['HIcode']==1] #only good data
    #vertix cut
    #spring sample
    df['RAdeg_HI']
    dec_range=[0,16,18,20,24,30,32,36]
    ra_range_left=[7.7,7.7,8.7,9.4,7.6,8.5,9.5]
    ra_range_right=[16.5,16.0,15.4,15.4,16.5,16.0,15.5]
    #fall_sample
    dec_range=[0,2,6,10,14,36]
    ra_right_range=[22,22.5,22,22,22]*15.0 #hours to degrees
    ra_left_edge=[3,3,3,2.5,3]*15.0
    dfvol=dfgood.loc[df['Vhelio'] < 15000]  #distance cut   0–16 	7.7-16.5 
    return dfvol

def HImf_ALFALFA(): #Jones et al 2018, MNRAS 477,2 https://academic.oup.com/mnras/article/477/1/2/4911535
    phi_star= 0.0045 #+/- 0.0002
    mstar = 10**9.94 #+/-0.01
    alpha = -1.25  #+/- 0.02
    x=10**np.arange(6.5,11.25,0.25)
    phi=np.log(10)*phi_star*(x/mstar)**(alpha+1)*np.exp(-x/mstar)
    return x,phi

#stellar mass HI mass relation
def HImstar_xGASS():  #http://xgass.icrar.org/data.html
    fname="xGASS_representative_sample.fits"
    hdulist=fits.open(fname)
    data=hdulist[1].data
    keep=data['HI_FLAG'] < 3
    data=data[keep]
    return data['lgMstar'],data['lgMHI']

#H2 mass function
def H2mf_xCOLDGASS(): #Fletcher 2021 https://academic.oup.com/mnras/article-abstract/501/1/411/5918004
    mstar = 10**9.59 #+0.11−0.10 
    alpha = -1.18 #+0.11−0.11
    phi_star = 2.34e-3 #+0.72-0.61
    x=10**np.arange(7.5,10.5,0.25)
    phi=np.log(10)*phi_star*(x/mstar)**(alpha+1)*np.exp(-x/mstar)
    return x,phi

def H2mf(model='luminosity'): #https://www.aanda.org/articles/aa/full_html/2020/11/aa38675-20/aa38675-20.html
    #log M then (# gal/Mpc−3/dex) model 1 XCO luminosity dependent, model 2 constant
    logM=[6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75,10.0]
    logphi1=[-2.19,-2.14,-2.08,-2.01,-1.92,-1.85,-1.85,-1.96,-2.17,-2.42,-2.68,-2.94,-3.20,-3.46,-3.62]
    logphi2=[-2.36,-2.14,-2.02,-1.96,-1.93,-1.94,-1.98,-2.04,-2.12,-2.24,-2.40,-2.64,-3.78,-5.2,-6.00]
    if model=='constant':
        return logM,logphi2
    else:
        return logM,logphi1

def H2mf_OR(type='orig'): #https://academic.oup.com/mnras/article/394/4/1857/1200684
    #params = [phi_star,mstar,alpha] 2 fits, one using a correction to Xfactor by stellar mass
    h=0.7
    param={'ref':[0.0243*h**3,7.5e8/h**2,-1.07], 'orig':[0.0089*h**3, 2.81e9/h**2,-1.18]}
    x=10**np.arange(6.5,11.25,0.25)
    phi=np.log(10)*param[type][0]*(x/param[type][1])**(param[type][2]+1)*np.exp(-x/param[type][1])
    return x,phi  

#H2-SFR relation
def H2_SFR(): #from Fletcher 2020 http://arxiv.org/abs/2002.04959
    slope=0.85 #+/-0.03
    intercept=8.92 #+/-0.02
    logSFR=np.linspace(-1.5,1.5,num=30)
    logMH2=slope*logSFR+intercept
    return logSFR,logMH2

def line_fit_with_errors(axis,x,m,b,merr,berr):
    y=m*x+b
    axis.plot(x,y)
    axis.plot(x,(m+merr)*x+b+berr,label="++")
    axis.plot(x,(m-merr)*x+b-berr,label="--")
    axis.plot(x,(m+merr)*x+b-berr,label="+-")
    axis.plot(x,(m-merr)*x+b+berr,label="-+")   
    plt.legend()
    plt.show()

#bootstrap resample with errors
def binned_statistic(x,y,Nbins=10,xrange=None,sum=False,fraction=False,upper_limits=False):
    '''returns the median or sum of y in bins of x if upper_limits is passed determines 
    if more than half the values in a bin or upper_limits and if so returns min(y[bin])'''
    #TODO allow the statistic to be passed
    if xrange is None:
        xrange=[np.min(x),np.max(x)]
    xedges=np.linspace(xrange[0],xrange[1],Nbins+1)
    xmids=0.5*(xedges[0:-1]+xedges[1:])
    bin_number=np.digitize(x,xedges)-1 #start with 0
    result=[]
    for i in range(Nbins):
        bin=bin_number==i
        if fraction:
            result.append(np.sum(y[bin])/len(y[bin]))
        elif sum:
            result.append(np.sum(y[bin]))
        else:
            result.append(np.median(y[bin]))

    return result,xmids

def bootstrap_resample(x,y,dx=0.0,dy=0.0,Nsamples=500,upper_limits=False,
        Nbins=10,xrange=None,sum=False,fraction=False,**kwargs):
    '''bootstrap resample a 2d data set including errors and returns Nsamples of values
    if upper limits is set and 50% of the values are upper limits the median is set to  '''
    rng = np.random.default_rng(seed=42)
    N=x.size
    if not xrange:
        print("you must set xrange, so that the bins are the same for all resamplings")
    res,xmids=binned_statistic(x,y,Nbins=Nbins,xrange=xrange)
    results=np.zeros((Nsamples,Nbins))
    for i in range(Nsamples):
        Nrand=rng.integers(low=0,high=N,size=N)
#        xboot=x[Nrand]+dx[Nrand]*Nnormalx
#        yboot=y[Nrand]+dy[Nrand]*Nnormaly
        res,_=binned_statistic(x[Nrand],y[Nrand],Nbins=Nbins)
        results[i]=res

    if isinstance(upper_limits,np.ndarray):
        ones=np.ones(len(upper_limits))
        frac_uplim,_=binned_statistic(x,upper_limits,Nbins=Nbins,xrange=xrange,fraction=True)   
        total,_= binned_statistic(x,ones,Nbins=Nbins,xrange=xrange,sum=True)  
        total=np.array(total)
        frac_uplim=np.array(frac_uplim)
        prob_uplim=stats.binom.cdf(0.5*total,total,frac_uplim)
        print(frac_uplim)
        return results,xmids,prob_uplim
    else:
        return results,xmids

def resample_plot(x,y,upper_limits=False,xrange=[9.,11.3]):
    results,xvals=bootstrap_resample(x,y,upper_limits=upper_limits,
        Nbins=10,xrange=xrange)
    f,ax=plt.subplots(nrows=1,ncols=1)
    ax.scatter(x,y,marker='.')
    c='red'
    bplot=ax.boxplot(results,widths=0.06,whis=(5,95),showfliers=False,positions=xvals,
        manage_ticks=False,patch_artist=False,boxprops=dict(color=c),
        whiskerprops=dict(color=c),capprops=dict(color=c))
    plt.show()

    
if __name__=='__main__':
    f,axis=plt.subplots(figsize=(8,4))
    mass_size()
    print('making all plots of observational data...')