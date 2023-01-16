import numpy as np
import pandas as pd

from astropy.table import Table

def fits2pandas(fname):
    '''read a fits table and return a pandas dataframe'''
    table = Table.read(fname, format='fits')
    for name in table.colnames:
        if len(table[name].shape) > 1: #fits tables can have arrays under one column name
            table.remove_column(name)  
    df = table.to_pandas()
    return df

def simard_catalog_pandas(dname,spectral=True):
    '''read in catalogs from Simard 2011 and remove
    some problemtic rows'''
    if dname==None:
        dname='/Users/ari/Data/SDSS/'
    df_simard=fits2pandas(dname+'Simard_Table1.fit')
    if spectral:
        df_simard=df_simard[df_simard['Sp'] > 0]
    else:
        df_simard=df_simard[df_simard['Sp'] > -2]
    return df_simard

def mass_catalog_pandas(dname):
    '''read in Mendel catalog and combine with Simard'''
    if dname==None:
        dname='/Users/ari/Data/SDSS/'
    df_s = simard_catalog_pandas(dname)
    del df_s['Sp']  #they are all spectra
    df_m = fits2pandas(dname+'Mendel_dusty.fit')
    del df_m['z']   #same as in Simard
    del df_m['PpS'] #same as in Simard
    df=df_m.set_index('objID').join(df_s.set_index('objID'),
        how='inner',rsuffix='_s')
    return df

def tinker_catalog_pandas(dname):
    '''read in catalogs from Tinker 2021 found here
    https://www.galaxygroupfinder.net/catalogs
    and return as a pandas data frame'''
    if dname==None:
        dname='/Users/ari/Data/vagc-dr7/'
    fname_groups=dname+'sdss_kdgroups_v1.0.dat'
    fname_galprops=dname+'sdss_galprops_v1.0.dat'
    colnames=['id','RA','Dec','z','L_gal','V_max','P_sat','M_halo','N_sat','L_tot','igrp','???']
    galprop_names=['Mag_g','Mag_r','sigma_v','DN4000','c90_c50','logMstar']
    data=np.loadtxt(fname_groups)
    df=pd.DataFrame(data, columns=colnames)
    data2=np.transpose(np.loadtxt(fname_galprops))
    for i,name in enumerate(galprop_names):
        df[name]=data2[i]
    df = df.astype({'id': int, 'N_sat': int, 'igrp': int})  
    return df
