''' Create new xGASS and xCOLDGASS files adding information from NYU-VAGC'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.table import Table
import astropy.units as u
import astropy.constants as cons
import astropy.coordinates as coord
from astropy.cosmology import Planck15 as cosmo

import read_catalog as rc

xGASS_fname="xGASS_representative_sample.fits"
xCOLD_GASS_fname="xCOLDGASS_PubCat.fits"
vagc_dname='/Users/ari/Data/vagc-dr7/'
sdss_fname=vagc_dname+'object_catalog.fits'

def fits2pandas(fname):
    '''read a fits table and return a pandas dataframe'''
    table = Table.read(fname, format='fits')
    for name in table.colnames:
        if len(table[name].shape) > 1: #fits tables can have arrays under one column name
            table.remove_column(name)  
    df = table.to_pandas()
    return df

def create_xGASS_catalog(info = False):
    '''create the xGASS and xCOLDGASS merged catalog '''
    H0=70
    df_xGASS=fits2pandas(xGASS_fname)
    df_xCOLD=fits2pandas(xCOLD_GASS_fname)
    dirname = '/Users/ari/Data/'
    df_tinker = rc.tinker_catalog_pandas(dirname+'vagc-dr7/')
    if info:
        print(df_xGASS.columns.values)
        print(df_xCOLD.columns.values)
        print(df_tinker.columns.values)
        return

    cat_xGASS = coord.SkyCoord(ra = df_xGASS['RA'].to_numpy()*u.degree, 
        dec = df_xGASS['DEC'].to_numpy()*u.degree, 
        distance = cosmo.comoving_distance(df_xGASS['zSDSS']))
    cat_xCOLD = coord.SkyCoord(ra = df_xCOLD['RA'].to_numpy()*u.degree, 
        dec = df_xCOLD['DEC'].to_numpy()*u.degree,
        distance = cosmo.comoving_distance(df_xCOLD['Z_SDSS']))
    
#    df_simard=rc.simard_catalog_pandas(dirname+'SDSS/', spectral=True)
    cat_tinker = coord.SkyCoord(ra = df_tinker['RA'].to_numpy() * u.degree, 
        dec = df_tinker['Dec'].to_numpy() * u.degree, 
        distance = cosmo.comoving_distance(df_tinker['z']))

    idx_xGASS, d2d_xGASS, d3d_xGASS = cat_xGASS.match_to_catalog_3d(cat_tinker)
    idx_xCOLD, d2d_xCOLD, d3d_xCOLD = cat_xCOLD.match_to_catalog_3d(cat_tinker)
    print(f'worst match to xGASS is {np.max(d3d_xGASS)}')
    print(f"worst match to xCOLD is {np.max(d3d_xCOLD)}")    

    #new columns to add
    df_xGASS['d3d'] = d3d_xGASS
    df_xGASS['P_sat'] = df_tinker['P_sat'].to_numpy()[idx_xGASS]
    df_xGASS['M_halo'] = df_tinker['M_halo'].to_numpy()[idx_xGASS]
    df_xGASS['c90_c50'] = df_tinker['c90_c50'].to_numpy()[idx_xGASS]
    df_xGASS['logMstar'] = df_tinker['logMstar'].to_numpy()[idx_xGASS]

    df_xCOLD['d3d'] = d3d_xCOLD
    df_xCOLD['P_sat'] = df_tinker['P_sat'].to_numpy()[idx_xCOLD]
    df_xCOLD['M_halo'] = df_tinker['M_halo'].to_numpy()[idx_xCOLD]
    df_xCOLD['c90_c50'] = df_tinker['c90_c50'].to_numpy()[idx_xCOLD]
    df_xCOLD['logMstar'] = df_tinker['logMstar'].to_numpy()[idx_xCOLD]

    #write hdf5 files 
    df_xGASS.to_hdf('xGASS.hdf',key='df',mode='w')
    df_xCOLD.to_hdf('xCOLD.hdf',key='df',mode='w')

def match_tinker_simard(info=False):
    H0 = 70
    dirname = '/Users/ari/Data/'
    df_tinker = rc.tinker_catalog_pandas(dirname+'vagc-dr7/')
    df_simard=rc.simard_catalog_pandas(dirname+'SDSS/', spectral=True)
    if info:
        print(df_tinker.columns.values)
        print(df_simard.columns.values)
    neg = df_simard['z'] < 0
    df_simard['z'][neg] = 0.0
#    bad = np.isnan(df_simard['z'])
    bad = np.arange(0,len(df_simard))[np.isnan(df_simard['z'])]
    print(len(bad))
    df_simard.drop(labels=bad,axis=0,inplace=True)
    print((np.isnan(df_simard['z'])).sum())
    print((np.isnan(df_tinker['z'])).sum())    
    cat_tinker = coord.SkyCoord(ra = df_tinker['RA'].to_numpy() * u.degree, 
        dec = df_tinker['Dec'].to_numpy() * u.degree, 
        distance = cosmo.comoving_distance(df_tinker['z']))
    cat_simard = coord.SkyCoord(ra = df_simard['_RA'].to_numpy()*u.degree,
        dec = df_simard['_DE'].to_numpy() * u.degree, 
        distance = cosmo.comoving_distance(df_simard['z']))
#    idx, d2d, d3d = cat_simard.match_to_catalog_3d(cat_tinker)
    print(np.max(d3d))

def distance2ways():
    H0 = 70.0 #Mpc/km/s
    dirname = '/Users/ari/Data/'
    df_simard=rc.simard_catalog_pandas(dirname+'SDSS/', spectral=True)
    d1=cosmo.angular_diameter_distance(df_simard['z'])
    d2=df_simard['Scale']*360.*60.*60./(2*np.pi)/1.e3
    plt.scatter(d1,d2,marker='.', s=1)
    plt.axis('equal')
    plt.show()

if __name__=='__main__':
#    df=tinker_catalog_pandas(vagc_dname)
#    print(df.columns.values)
    create_xGASS_catalog(info = True) 
#    match_tinker_simard(info = True)