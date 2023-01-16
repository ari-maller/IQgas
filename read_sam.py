#adding support for new TNG-SAM file format by austen
import site
import sys
import h5py
import numpy as np
import pandas as pd
import astropy.units as u
site.addsitedir('/Users/ari/Code/') #directory where illustris_sam is located
import illustris_sam as ilsam #imports directory files can be called with ilsam.file

def read_sam(fname):
    #try to parse how to read based on the file name
    if fname=='/Users/ari/Data/tng-sam/':#the fname is a basePath for tng-sam
        galprop,haloprop=read_ilsam(fname)

    return galprop,haloprop

def read_ilsam_galaxies(basePath,snapshot=99,print_fields=False):
    #https://drive.google.com/file/d/1cQGgqp6F-Y9RyYe6Z0Mn1E52np0UHGFk/view
    n=5
    subvolume_list = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                subvolume_list.append([i, j, k])
    fields=['GalpropMBH', 'GalpropMH2', 'GalpropMHI', 'GalpropMHII', 'GalpropMbulge', 'GalpropMcold', 
        'GalpropMstar', 'GalpropMvir', 'GalpropRbulge', 'GalpropRdisk', 'GalpropSatType', 'GalpropSfr', 
        'GalpropZcold', 'GalpropZstar','GalpropHaloIndex_Snapshot']
    sam_gals = ilsam.groupcat.load_snapshot_subhalos(basePath, snapshot, subvolume_list, fields=fields)
    #this returns ['GalpropSubfindIndex_DM', 'GalpropSubfindIndex_FP']
    galprop_matches=ilsam.groupcat.load_matches(basePath, subvolume_list, 'Galprop')
    #this returns ['HalopropFoFIndex_DM', 'HalopropFoFIndex_FP']
    #halo_matches=ilsam.groupcat.load_matches(basePath, subvolume_list, 'Haloprop')
    if print_fields:
        print(list(sam_gals))
    halo_fields=['HalopropC_nfw','HalopropMhot','HalopropMvir','HalopropSpin','HalopropMaccdot_pristine']
    sam_halos = ilsam.groupcat.load_snapshot_halos(basePath,snapshot,subvolume_list,fields=halo_fields)
    galprop=pd.DataFrame()
    galprop['mhalo']=(sam_halos['HalopropMvir'][sam_gals['GalpropHaloIndex_Snapshot']])*1.e9
    galprop['spin'] = sam_halos['HalopropSpin'][sam_gals['GalpropHaloIndex_Snapshot']]
    galprop['Cnfw'] = sam_halos['HalopropC_nfw'][sam_gals['GalpropHaloIndex_Snapshot']]
    galprop['Mhot'] =(sam_halos['HalopropMhot'][sam_gals['GalpropHaloIndex_Snapshot']])*1.e9
    galprop['Macc'] =(sam_halos['HalopropMaccdot_pristine'][sam_gals['GalpropHaloIndex_Snapshot']])
    galprop['mstar']=sam_gals['GalpropMstar']*1.e9
    galprop['mHI']=sam_gals['GalpropMHI']*1.e9
    galprop['mH2']=sam_gals['GalpropMH2']*1.e9
    galprop['mbulge']=sam_gals['GalpropMbulge']*1.e9
    galprop['r_bulge']=sam_gals['GalpropRbulge']
    galprop['r_disk']=sam_gals['GalpropRdisk']
    galprop['sfr']=sam_gals['GalpropSfr']
    galprop['Zcold']=sam_gals['GalpropZcold']
    galprop['Zstar']=sam_gals['GalpropZstar']   
    galprop['sat_type']=sam_gals['GalpropSatType']
    galprop['subfind_id']=sam_gals['GalpropSubfindIndex_FP']
    return galprop

def info(fname):
    #returns info about the SAM hdf5 File
    hf=h5py.File(fname,'r')
    print(f"Simulation: {hf.attrs['simulation']}")
    print(f"Arbor: {hf.attrs['arbor']}")
    print(f"Version: {hf.attrs['version']}")
    print(f"Boxsize: {hf.attrs['boxsize']}")

def fields(fname):
    hf=h5py.File(fname,'r')
    f=[]
    for key in hf.keys():
        f.append(key)
    return f

def read(fname,fields=[],units=False):
    assign_unit={'':1, 'Msun':u.Msun, 'kpc':u.kpc, 'Mpc':u.Mpc, 'Gyr':u.Gyr,
            'km/s':u.km/u.s,'Msun/yr':u.Msun/u.yr, 'Zsun*Msun':u.Msun,
            'Msun/Zsun/yr':u.Msun/u.yr, 'cMpc':u.Mpc}
    hf=h5py.File(fname,'r')
    results=[]
    if (fields==[]):#if no fields specified load all fields
        for key in hf.keys():
            fields.append(key)
    for name in fields:
        dset=hf[name]
        unit=dset.attrs['units']
        data=dset[:]
        if units is True:
            data=data*assign_unit[unit]
        results.append(data)

    if len(results)==1:
        results=results[0]
    return results

def logsum(array):
    array=np.array(array)
    return np.sum(10**array)

def gal2halo(fname,field='mcold'):
    gdata=read_all(fname)
    hids,hids_indices=np.unique(gdata['halo_id'],return_index=True)
    Nhalos=len(hids)
    halo_sum=np.zeros(Nhalos)
    for i in range(Nhalos-1):
        halo_sum[i]=logsum(gdata[field][hids_indices[i]:hids_indices[i+1]])

    return halo_sum

if __name__ == '__main__':
    base_path='/Users/ari/Data/tng-sam'
    df=read_ilsam_galaxies(base_path,print_fields=True)
    df.to_hdf('tng-sam.h5',key='s',mode='w')
