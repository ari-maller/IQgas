import numpy as np 
import pandas as pd 

def get_sim(sim=None,sample='cent'):
    dname="/Users/ari/Dropbox/CCA Quenched Isolated Galaxies Workshop 2017-05-11/DATA/"
    fname={'Illustris':'Illustris_with_hih2.dat',
            'Eagle':'EAGLE_RefL0100HashPhotfix_MHIH2HIIRhalfmassSFRT1e4_allabove1.8e8Msun.txt',
            'Mufasa':'halos_m50n512_z0.0.dat',
            'TNG':'TNG_with_hih2.dat',
            'Simba':'halos_m100n1024_z0.0.dat',
            'SC-SAM':'SCSAMgalprop_updatedVersion.dat'}
    cols={'Illustris':[1,2,3,4,5], #cent, sfr, Mstar, MHI, MH2
            'Eagle':[2,6,10], #logMstar, logMHI, logMH2
            'Mufasa':[1,2,3,4,5],#cent, sfr, Mstar, MHI, MH2, #old Mstar, MHI, MH2, sfr, cent
            'TNG':[1,2,3,4,5], #cent, sfr, Mstar, MHI, MH2
            'Simba':[1,2,3,4,5], #Mstar, MHI, MH2, sfr, cent
            'SC-SAM':[3,20,7,15,16]} #cent, Mstar, MHI, MH2, sfr
    
    if sim=='Eagle': #eagle has sfr and cent in a different file so need to add those
        Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        fname2='EAGLE_RefL0100_MstarMcoldgasSFR_allabove1.8e8Msun.txt'
        sfr,cent=np.loadtxt(dname+fname2,usecols=(4,5),unpack=True) #check that instant
        c=cent.astype(bool)
        mgas=np.log10(10**(mHI)+10**(mH2))
        data=pd.DataFrame({'central':c,'logMstar':Mstar,'log_SFR':np.log10(sfr)
        ,'logMHI':mHI,'logMH2':mH2,'logMgas':mgas})   
    elif sim=='SC-SAM':
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=np.invert(cent.astype(bool))
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar*1.e9),'log_SFR':np.log10(sfr)
        ,'logMHI':np.log10(mHI*1.e9),'logMH2':np.log10(mH2*1.e9)
        ,'logMgas':np.log10((mHI+mH2)*1.e9)})
    else:
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=cent.astype(bool)
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar),'log_SFR':np.log10(sfr)
        ,'logMHI':np.log10(mHI),'logMH2':np.log10(mH2),'logMgas':np.log10(mHI+mH2)})

    data=data[data['logMstar'] > 8.0]
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