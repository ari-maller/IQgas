import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#catalogs from https://gax.sjtu.edu.cn/data/Group.html


def get_fnames(survey,L=True,plus=False,dname='/Users/ari/Data/Yang_Groups/lim17_catalogs/'):
    keep=[]
    letter='M'
    if L:
        letter='L'  
    if survey=='dr4' or survey=='dr7':
        if survey=='dr4': #using b for now
            galfile='/Users/ari/Data/Yang_Groups/Groups_DR4/isdss4b_1'
            grpfile='/Users/ari/Data/Yang_Groups/Groups_DR4/sdss4b_group'
        else:
            galfile='/Users/ari/Data/Yang_Groups/group_DR7/imodelB_1'
            grpfile='/Users/ari/Data/Yang_Groups/group_DR7/modelB_group'         
    else:
        if survey=='sdss':
            survey='SDSS'
        if survey=='2mrs':
            survey='2MRS'
        files=os.listdir(dname)
        for file in files:
            if survey in file:
                if letter in file:
                    if '+' in file:
                        if plus:
                            keep.append(file)
                    else:
                        if not plus:
                            keep.append(file)
        galfile=dname+keep[0]
        grpfile=dname+keep[1]
    return galfile,grpfile

#read galaxy files
def read_lim17_galaxy(fname): #plus=True didn't work
    names=['gal_id','survey_id','group_id','ra','dec','l','b',
        'z_cmb','z_edd','z_comp','z_src','dist_NN','logL','logMstar','color']
    df=pd.read_csv(fname,comment='#',names=names,delim_whitespace=True)
    return df

def read_yang07_galaxy(fname):
    names=['gal_id','survey_id','group_id','central'] #only take central by mass
    cols=[0,1,2,4] #both dr4 and dr7
    df=pd.read_csv(fname,names=names,usecols=cols,delim_whitespace=True,engine='python')
    return df

#read group files
def read_lim17_group(fname):
    names=['group_id','cent_id','ra','dec','z','Mass','Nmem','fedg','i-o']
    cols=[0,1,2,3,4,5,6,7,8]
    df=pd.read_csv(fname, comment='#', skiprows=12, names=names, 
        delim_whitespace=True, usecols=cols, engine='python')
    return df

def read_yang07_group(fname,sample='dr7'):
    names=['group_id','ra','dec','z','g_stellar_mass','halo_mass','f_edg']
    if sample=='dr4':
        cols=[0,1,2,3,5,7,8] #dr4
    else:
        cols=[0,1,2,3,5,7,10] #dr7
    df=pd.read_csv(fname,names=names,usecols=cols,engine='python',delim_whitespace=True)
    return df

def match_yang2lim():
    ynames=get_fnames('dr7')
    lnames=get_fnames('sdss')
    dfgal_yang=read_yang07_galaxy(ynames[0])
    dfgal_lim =read_lim17_galaxy(lnames[0])
    print(dfgal_yang['gal_id'][0],dfgal_yang['survey_id'][0],dfgal_yang['group_id'][0])
    print(dfgal_lim['gal_id'][0],dfgal_lim['survey_id'][0],dfgal_lim['group_id'][0])

if __name__=='__main__':
#    fnames=get_fnames('sdss')
#    dfgal=read_lim17_galaxy(fnames[0])
#    dfgroup=read_lim17_group(fnames[1])
    match_yang2lim()
