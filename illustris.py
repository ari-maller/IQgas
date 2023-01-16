import os
import sys
import requests
import h5py
import numpy as np 
import pandas as pd
import groupcat
import matplotlib.pyplot as plt

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"128c8b59f9c4eaa4e1841e31ca2bde32"}

basePath1='/Users/ari/Data/illustris/Illustris-1'
basePath2='/Users/ari/Data/illustris/TNG100-1'

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r
'''
#can be used to build the url if not sure what it is
r = get(baseUrl)
names = [sim['name'] for sim in r['simulations']] #array of simulation names
i = names.index('TNG100-1') #get the index for name 
#i = names.index('Illustris-1')
sim = get( r['simulations'][i]['url'] )
snaps = get( sim['snapshots'] ) 
snap = get( snaps[-1]['url'] ) #get z=0 snapshot
subs = get( snap['subhalos'] )#subhalos for snapshot
'''
def get_filenames(sim='TNG',boxsize=100):
    '''get the filenames and paths for stellar and gas data
    returns path, snapshot number and full path to gas file
    TODO: add z not 0 as an option'''
    if not (sim=='Illustris' or sim=='TNG'):
        print('Only takes Illustris or TNG as sim')
    basePath1='/Users/ari/Data/illustris/Illustris-1'
    basePath2='/Users/ari/Data/illustris/TNG100-1'
    basePath3='/Users/ari/Data/illustris/TNG50-1'
    gasfile={'Illustris':'hih2_galaxy_135.hdf5','TNG':'hih2_galaxy_099.hdf5'}
    path={'Illustris':basePath1,'TNG':basePath2}
    snap={'Illustris':135,'TNG':99}
    dname={'Illustris':'/groups_135/','TNG':'/groups_099/'}
    gaspath=path[sim]+dname[sim]+gasfile[sim]
    return path[sim],snap[sim],gaspath

def parseTNGgas(sim='Illustris'): #this combines the TNG hi/h2 files and subhalos
    path,snap,gaspath=get_filenames(sim=sim)

    #read subhalo catalog SubhaloSFRinRad SubhaloMassInRadType better according to Dave 2020
    fields=['SubhaloSFRinRad','SubhaloMassInRadType','SubhaloHalfmassRadType','SubhaloMass','SubhaloGrNr']
    subs=groupcat.loadSubhalos(path[sim], snap[sim], fields=fields)
    halos=groupcat.loadHalos(path[sim],snap[sim],fields=['Group_M_Crit200']) #returns 1D narray
    masses=np.transpose(subs['SubhaloMassInRadType'])
    Mgas=masses[0]  #0 is gas in gadget
    Mstar=masses[4] #4 is stars in gadget
    N=len(Mstar)
    M200=halos[subs['SubhaloGrNr']]
    radii=np.transpose(subs['SubhaloHalfmassRadType'])
    Rgas=radii[0]
    Rstar=radii[4]
    #add hi_h2 file
    hf = h5py.File(gaspath,'r')
    idx = np.array(hf.get('id_subhalo'),dtype=np.int)
    cent = np.array(hf.get('is_primary'),dtype=np.int)
    mH = np.array(hf.get('m_neutral_H'))
    mH2 = np.array(hf.get('m_h2_GD14_vol'))
    mHI = np.array(hf.get('m_hi_GD14_vol'))
    hf.close()
    h=0.704
    sfr=subs['SubhaloSFRinRad'][idx]
    Mgas=Mgas[idx]*1.e10/h #removing h for m,masses
    Mstar=Mstar[idx]*1.e10/h
    M200=M200[idx]*1.e10/h
    Rgas=Rgas[idx]/h #also comoving so *(1+z) if z not 0
    Rstar=Rstar[idx]/h #also comoving so *(1+z) if z not 0 
    df=pd.DataFrame({'GID':idx,'central':cent,'Mstar':Mstar,'sfr':sfr,'mHI':mHI,'mH2':mH2,
        'r_disk':Rstar,'M200':M200})
    return df

def gas_sizes(sim='TNG',plotone=False):
    '''get gas sizes for TNG or illustris'''
    #some 10k gals seem to have 0 for gas profile but still a total gas mass
    path,snap,gaspath=get_filenames(sim=sim)
    model='GD14'
    proj='3d' # or '2d' needs to match model type of vol or map
    hf = h5py.File(gaspath,'r')
    mH2 = np.array(hf.get(f'm_h2_{model}_vol'))
    mHI = np.array(hf.get(f'm_hi_{model}_vol'))
    profile_bins=np.array(hf.get('profile_bins')) 
    profile_gas_rho=np.array(hf.get(f'profile_gas_rho_{proj}'))
    profile_f_neutral=np.array(hf.get(f'profile_f_neutral_H_{proj}'))
    profile_f_mol=np.array(hf.get(f'profile_f_mol_{model}_{proj}'))
    Ngal,Nbin=profile_bins.shape
    hir50=np.zeros(Ngal)
    h2r50=np.zeros(Ngal)
    count=0
    for i in range(Ngal):
        if profile_bins[i][-1] > 30.0 and mHI[i] > 0.0:
            profile_HI=profile_gas_rho[i]*profile_f_neutral[i]*(1.-profile_f_mol[i])
            mass_HI_inR = 2.0*np.pi*np.cumsum(profile_HI)*profile_bins[i]
            idx=np.where((mass_HI_inR/mass_HI_inR[-1]) >= 0.9)
            hir50[i]=profile_bins[i][idx[0][0]]
        else:
            count=count+1
            '''
        if mH2[i] > 0.0 and i!=4220:
            profile_H2=profile_gas_rho[i]*profile_f_neutral[i]*profile_f_mol[i]
            mass_H2_inR = 2.0*np.pi*np.cumsum(profile_H2)*profile_bins[i]
            idx=np.where((mass_H2_inR/mass_H2_inR[-1]) >= 0.5)
            if idx[0].size==0:
                print(i,mH2[i],profile_f_neutral[i],profile_f_mol[i])
            h2r50[i]=profile_bins[i][idx[0][0]]
        if mHI[i] > 0.0 and i!=4220:
            profile_HI=profile_gas_rho[i]*profile_f_neutral[i]*(1.-profile_f_mol[i])
            mass_HI_inR = 2.0*np.pi*np.cumsum(profile_HI)*profile_bins[i]
            idx=np.where((mass_HI_inR/mass_HI_inR[-1]) >= 0.5)
            if idx[0].size==0:
                print(idx,mass_HI_inRR)
            hir50[i]=profile_bins[i][idx[0][0]]
            '''
    if plotone:
        i=433
        profile_H2=profile_gas_rho[i]*profile_f_neutral[i]*profile_f_mol[i]
        profile_HI=profile_gas_rho[i]*profile_f_neutral[i]*(1.-profile_f_mol[i])
        plt.plot(profile_bins[i],profile_gas_rho[i]*profile_f_neutral[i],label='neutral')
        plt.plot(profile_bins[i],profile_HI,label='atomic')
        plt.plot(profile_bins[i],profile_H2,label='molecular')
        plt.legend()
        plt.show()

    print(count,Ngal)
    return hir50,h2r50

def test(data):
    f,axs=plt.subplots(nrows=3,ncols=1,figsize=(6,8),sharex=True)
    axs[0].scatter(np.log10(data['Mstar']),np.log10(data['sfr']),s=1,marker='.')
    axs[0].set_xlim([9,12.0])
    axs[0].set_ylim([-3,1.5])
    axs[1].scatter(np.log10(data['Mstar']),np.log10(data['mHI']),s=1,marker='.')
    axs[1].set_ylim([8,11.5])
    axs[2].scatter(np.log10(data['Mstar']),np.log10(data['mH2']),s=1,marker='.')
    axs[2].set_ylim([8,11.5])
    plt.show()

def write_sim(data,name='tmp.dat'):
    f=open(name,'w')
    f.write("# Group cent.  sfr_inst   Mstar      MHI       MH2     r_disk    M200\n")
    for i in range(data.shape[0]):
        f.write("    {}    {}   {:8.4f}  {:6.4e}  {:6.4e}  {:6.4e}  {:6.3f} {:6.4e} \n".format(
            data['GID'][i],data['central'][i],data['sfr'][i],data['Mstar'][i],data['mHI'][i],
            data['mH2'][i],data['r_disk'][i],data['M200'][i]))

    f.close()

if __name__=='__main__':
    name=input("Generate a new ascii catalog for Illustris or TNG? ")
    if name=='size':
        hir50,h2r50=gas_sizes()
        print((hir50 > 70.0).sum())

    elif name!='Illustris' and name!='TNG':
        print("Must choose Illustris or TNG")
    else:
        data=parseTNGgas(sim=name)
#        mcut = data['Mstar'] > 1.e8
        write_sim(data,name=name+'_with_hih2.dat')
