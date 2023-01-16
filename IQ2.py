import IQ as iq
import numpy as np
import matplotlib.pyplot as plt 

def newfig():
    plt.set_cmap('Dark2')
    df=iq.get_sim(sim='SC-SAM',sample='cent',mcut=9.0)
    logMhalo=np.log10(df['mhalo'])
    diffgashalo,diffstarhalo,diffMgas=iq.triple_diff(df['logMH2'],logMhalo,df['logMstar'])
    diffgasdisk,diffstardisk,diffMgas=iq.triple_diff(df['logMH2'],df['r_disk'],df['logMstar'])
    f,axis=plt.subplots(nrows=3,ncols=1,figsize=(5,8))
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    axis[0].scatter(df['logMstar'],df['logMH2'],s=1,marker=',',c=diffMgas)
    axis[0].set_ylim([7.55,10.5])
    axis[0].set_ylabel(r'$\log M_{H_2}$')
    axis[1].scatter(df['logMstar'],logMhalo,s=1,marker=',',c=diffstarhalo)
    axis[1].set_ylim([10.55,14])
    axis[1].set_ylabel(r'$\log M_{halo}$')
    axis[2].scatter(df['logMstar'],df['r_disk'],s=1,marker=',',c=diffstardisk)
    axis[2].set_ylim([0,15])
    axis[2].set_ylabel(r'$r_{disk} [kpc]$')
    axis[2].set_xlabel(r'$\log M_{stellar}$')    
    plt.savefig('new.pdf')

newfig()