import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from astropy.table import Table

import read_lim17_cat as rlim


def read_xGASS():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    df.rename(columns={'GASS':'ID'},inplace=True)
    return df

def read_xCOLDGASS():
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    df['SDSS']=df['SDSS'].str.decode('UTF-8')
    df.rename(columns={'Z_SDSS':'zSDSS'},inplace=True)
    return df

def gass_for_skysurver():
    df=read_xGASS()
    f=open('gass.dat','w')
    f.write("name       ra       dec \n")
    for i,row in df.iterrows():
        f.write("GASS{}  {}  {}\n".format(row['ID'],row['RA'],row['DEC']))
    f.close()
    df=read_xCOLDGASS()
    f=open('coldgass.dat','w')
    f.write("name        ra       dec \n")
    for i,row in df.iterrows():
        f.write("CGASS{}  {}  {}\n".format(row['ID'],row['RA'],row['DEC']))
    f.close()

def match_by_objid(sample='xGASS'):
    fname='SDSS_GASS.csv'
    names=['name','objid','ra','dec','specObjid','z']
    df=pd.read_csv(fname,names=names,skiprows=2)
    fnames=rlim.get_fnames('sdss')
    dfgal=rlim.read_lim17_galaxy(fnames[0])
    print(dfgal['survey_id'].dtype)
    dfgroup=rlim.read_lim17_group(fnames[1])
    f=open('gass_match.dat','w')
    f.write('#GASS_ID   SDSS_OBJID   RA   DEC   z   SURVEY_ID    RA   DEC   z   GROUP_ID\n')
    for i,row in df.iterrows():
        match=(np.where(dfgal['survey_id']==row['objid']))[0]
        if match.size > 0:
            gal=dfgal.iloc[match[0]]
            print(gal['survey_id'].dtype)
            f.write("{}  {:d} {:5.2f} {:5.2f} {:5.3f} {:d} {:5.2f} {:5.2f} {:5.3f} {:d}\n".format(
                row['name'],row['objid'],row['ra'],row['dec'],row['z'],
                gal['survey_id'],gal['ra'],gal['dec'],gal['z_cmb'],gal['group_id']))
        else:
            pass
            f.write("{} {:d} {:5.2f} {:5.2f} {:5.3f} {} {} {} {} {}\n".format(
                row['name'],row['objid'],row['ra'],row['dec'],row['z'],0,0.0,0.0,0.0,0))           
    f.close()

def close_in_slice(vx,vy,vz,x,y,z,dr=0.5,dz=0.01):
    #finds closest in x and y after being sliced in z, returns index
    indices=np.arange(0,len(vz))
    cut=np.abs(vz-z) < dz
    dis=np.abs(vx[cut]-x)+np.abs(vy[cut]-y)
    a=dis.idxmin()
    return a

def match2halos_yang(df,survey='dr7'):
    fnames=rlim.get_fnames(survey)
    dfgal=rlim.read_yang07_galaxy(fnames[0])
    dfgroup=rlim.read_yang07_group(fnames[1])
    f=open("match.dat",mode='w')
    count=0
    for i,row in df.iterrows():
        a=close_in_slice(dfgroup['ra'],dfgroup['dec'],dfgroup['z'],
            row['RA'],row['DEC'],row['zSDSS'])
        if row['group_id_B']!=dfgroup['group_id'][a]:
            count=count+1
            f.write("{}  {:6.3f}  {:6.3f}  {:5.3f}  {:5.3f}  {:6.4f}  {:6.4f}  {}  {}\n".format(
                row['NYU_id'],row['RA'],dfgroup['ra'][a],row['DEC'],dfgroup['dec'][a],row['zSDSS'],
                dfgroup['z'][a],row['group_id_B'],dfgroup['group_id'][a]))
    f.close()
    print(f"Bad matches = {count}")

def match2halos_lim(df,survey='sdss',L=True,plus=False):
    N=df.shape[0]
    sid=np.zeros(N,dtype=np.str)
    gid=np.zeros(N,dtype=np.int)
    cent=np.zeros(N,dtype=np.bool)
    fnames=rlim.get_fnames(survey,L=L,plus=plus)
    dfgal=rlim.read_lim17_galaxy(fnames[0])
    dfgroup=rlim.read_lim17_group(fnames[1])
    dfgal=dfgal[dfgal['z_cmb'] < 0.07] #only keep gals in xGASS redshift range
    ralims=[60,300]
    print(dfgal.keys())
    for i,row in df.iterrows():
        if (row['RA'] > ralims[0] and row['RA'] < ralims[1]):
            dis=np.abs(row['RA']-dfgal['ra'])+np.abs(row['DEC']-dfgal['dec'])
            try:
                a=dis.idxmin()
            except ValueError:
                print(f"ValueError: for row {i} \n {row}")
            if ((row['zSDSS']-dfgal['z_cmb'][a] > 0.02) or (dis[a] > 5.e-3)):
                print("Failed to match: dz={:3f}, da={:3f}".format(row['zSDSS']-dfgal['z_cmb'][a],dis[a]))
            else:
                pass
            print("{}  {:6.3f}  {:6.3f}  {:5.3f}  {:5.3f}  {:6.4f}  {:6.4f}  {}  {}  {}".format(row['ID'],row['RA'],dfgal['ra'][a],row['DEC'],dfgal['dec'][a],row['zSDSS'],
                dfgal['z_cmb'][a],row['SDSS'],dfgal['survey_id'][a],dfgal['group_id'][a]))
#                sid[i]=dfgal['survey_id'][a]  #these are strings for 2mrs objID for sdss
#                gid[i]=int(dfgal['group_id'][a])
#                group=(dfgroup.loc[dfgroup['group_id']==gid[i]]) #this is a row
#                if (group['cent_id']).values==(dfgal['gal_id'][a]):
#                    cent[i]=True                

#    print(f"Matched {(sid!='0').sum()}, missed {(sid=='0').sum()}")
#    print(f"central galaxies {cent.sum()}")
#    df.insert(0,'survey_id',sid) #non-matches have id 0
#    df.insert(1,'group_id',gid)  #non-matches have id 0
#    df.insert(2,'central',cent)
    return df

def matchxGASSsamples(survey='2mrs'):
    df=read_xGASS()
    print(f"In strip {(np.logical_or(df['RA'] < 50,df['RA'] > 300)).sum()}")
    dfmatch=match2halos(df,survey=survey)
    dfmatch.to_hdf("xGASS.h5",key='df',mode='w')
    dfcold=read_xCOLDGASS()   
    print(f"In strip {(np.logical_or(dfcold['RA'] < 50,dfcold['RA'] > 300)).sum()}") 
    dfmatch2=match2halos(dfcold,survey=survey)  
    dfmatch2.to_hdf("xGASSCOLD.h5",key='df',mode='w')
    dfCO=pd.merge(dfmatch,dfmatch2,left_on='GASS',right_on='ID')
    dfCO.to_hdf("xGASS-CO.h5",key='df',mode='w')

def zdistribution():
    df=read_xGASS()
    plt.hist(df['zSDSS'],label='xGASS',histtype='step',density=True)
    df=read_xCOLDGASS()
    plt.hist(df['zSDSS'],label='xCOLDGASS',histtype='step',density=True)
    fnames=rlim.get_fnames('2mrs')
    dfgal=rlim.read_lim17_galaxy(fnames[0])
    plt.hist(dfgal['z_cmb'],label='2MRS',histtype='step',density=True,range=[0,0.2])
    fnames=rlim.get_fnames('sdss')
    dfgal=rlim.read_lim17_galaxy(fnames[0])
    plt.hist(dfgal['z_cmb'],label='SDSS',histtype='step',density=True,range=[0,0.2])   
    plt.xlim([0,0.2])
    plt.legend()
    plt.show()

def skydistribution():
#    plt.figure()
#    plt.subplot(111,projection='hammer')
#    plt.grid(True)
    df=read_xGASS()
    plt.scatter(df['RA'],df['DEC'],marker=',',s=1,label='xGASS')
    df=read_xCOLDGASS()
    plt.scatter(df['RA'],df['DEC'],marker=',',s=1,label='xCOLDGASS')
    fnames=rlim.get_fnames('2mrs')
    dfgal=rlim.read_lim17_galaxy(fnames[0])  
    dfgal=dfgal[dfgal['z_cmb'] < 0.06]
    plt.scatter(dfgal['ra'],dfgal['dec'],marker=',',s=1,label='SDSS')  
    plt.legend()
    plt.show()

def check_keys(survey):
    fnames=rlim.get_fnames(survey)
    dfgal=rlim.read_lim17_galaxy(fnames[0])
    dfgroup=rlim.read_lim17_group(fnames[1])
    print(dfgal.keys())
    print(dfgroup.keys())

if __name__=='__main__':
#    gass_for_skysurver()
#    df=read_xGASS()
    match_by_objid(sample='xGASS')
#    match2halos_yang(df,survey='dr7')
    #zdistribution()
    #skydistribution()
