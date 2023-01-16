import numpy as np
import matplotlib.pyplot as plt
import IQ
import observations as obs

boxsize={'Illustris':106.5,'Eagle':100,'Mufasa':73.5,
    'TNG100':110.7,'Simba':147,'SC-SAM':110.7}
N={'Eagle':30,'Mufasa':25,'TNG100':30,'Simba':30,'SC-SAM':30}
def test_hmf():
    names=['Mufasa','TNG100','Simba','SC-SAM']
    f,ax=plt.subplots(1,1)
    for name in names:
        volume=boxsize[name]**3
        data=IQ.get_sim(sim=name)
        cent=data['central']==True
        IQ.xyhistplot(data['logMhalo'][cent]+offset[name],ax,weight=1/volume,
            label=name,Nbins=N[name],xrange=[11,14])
    
    ax.legend()
    plt.savefig('hmf.pdf')

#make fig2 in Dave 2020
def fig234_dave():
    names=['Eagle','TNG100','Simba']
    f,ax=plt.subplots(3,1,figsize=(5,7))
    plt.subplots_adjust(hspace=0.6)
    c={'Eagle':'green','TNG100':'red','Simba':'mediumslateblue'} 
    for name in names:
        volume=boxsize[name]**3
        data=IQ.get_sim(sim=name)
        IQ.xyhistplot(data['logMstar'],ax[0],weight=1/volume,
            label=name,Nbins=20,color=c[name])
        keep=data['logMHI'] > 7.5
        IQ.xyhistplot(data['logMHI'][keep],ax[1],weight=1/volume,Nbins=10,
            xrange=[9,11],color=c[name])
        keep=data['logMH2'] > 7.5
        IQ.xyhistplot(data['logMH2'][keep],ax[2],weight=1/volume,Nbins=10,
            xrange=[8.5,11],color=c[name])        
    
    x,y,ym,yp=obs.gmf_GAMA()
    ax[0].scatter(x,np.log10(y),marker='x',color='black')
    x,y=obs.HImf_ALFALFA()
    ax[1].scatter(np.log10(x),np.log10(y),marker='o',color='black')
    x,y=obs.H2mf_xCOLDGASS()
    ax[2].scatter(np.log10(x),np.log10(y),marker='o',color='black')
    ax[0].set_xlim([9.1,12.4])
    ax[1].set_xlim([9.,11.5])
    ax[2].set_xlim([8.5,11.1])
    ax[0].set_ylim([-5.6,-1.5])
    ax[1].set_ylim([-4.2,-1.5])
    ax[2].set_ylim([-4.8,-1.5])
    ax[0].set_xlabel(r'log M$_* [M_{\odot}]$')
    ax[1].set_xlabel(r'log M$_{HI} [M_{\odot}]$')
    ax[2].set_xlabel(r'log M$_{H_2} [M_{\odot}]$')
    ax[0].set_ylabel(r'$\phi_* [Mpc^{-3}]$')
    ax[1].set_ylabel(r'$\phi_{HI} [Mpc^{-3}]$')
    ax[2].set_ylabel(r'$\phi_{H_2} [Mpc^{-3}]$')
    ax[0].legend(loc='lower left')
    plt.savefig('Dave2020_Fig234.pdf')

fig234_dave()