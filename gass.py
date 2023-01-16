import numpy as np
import pandas as pd
from astropy.table import Table
'''Notes on finding xCOLDGASS galaxy not in xGASS
example 1, line 7 in xCOLDGASS, ID=3318, RA=15.6595,DEC=15.1685
'''
def get_xGASS():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want=['GASS','SDSS','HI_FLAG','lgMstar','lgMHI','SFR_best','SFRerr_best',
        'HIconf_flag', 'NYU_id', 'env_code_B'] #no HI error            
    discard=list(set(df.columns.values.tolist()).difference(want))  
    df=df.drop(columns=discard) 
    df.rename(columns={'lgMstar':'logMstar','lgMHI':'logMHI'},inplace=True)
    uplimHI=np.zeros((df.shape[0]),dtype=bool)
    uplimHI[df['HI_FLAG']==99]=1 #set true if HI upper limit  
    df.insert(6,'uplimHI',uplimHI) #375 upper limits
    return df

def get_xCOLDGASS(): 
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want = ['ID','OBJID','FLAG_CO','LOGMH2','XCO_A17','LCO_COR','R50KPC',
            'LOGMSTAR','LOGMH2_ERR','LIM_LOGMH2','LOGSFR_BEST','LOGSFR_ERR']
    discard=list(set(df.columns.values.tolist()).difference(want))
    df=df.drop(columns=discard)
    good=df['R50KPC'] > 0.0   # 64 is NaN and 415 is -3622.7014
    df=df[good]
    df.reindex
    df.rename(columns={'R50KPC':'r_disk'},inplace=True)
    df['logRstar']=np.log10(df['r_disk'])
    df['logSigma']=df['LOGMSTAR']-2.0*df['logRstar']
    yes_CO= df['FLAG_CO'] < 2
    no_CO = df['FLAG_CO']==2
    logMH2=np.zeros(df.shape[0])
    logMH2[yes_CO]=df['LOGMH2'][yes_CO]+np.log10(0.75) #remove He
    logMH2[no_CO]=df['LIM_LOGMH2'][no_CO]+np.log10(0.75) #remove He upper limits
    df.insert(3,'logMH2',logMH2)
    uplimH2=np.zeros((df.shape[0]),dtype=bool)
    uplimH2[no_CO]=1 #set true if H2 is upper limit  
    df.insert(8,'uplimH2',uplimH2) # 199 upperlimits
    df=df.drop(columns=['LOGMH2','LIM_LOGMH2'])
    return df

def get_GASS(name='xCO-GASS',sample='all',shape=False):
    '''returns the HI catalog, H2 catalog or the HI+H2 catalog,xGASS,xCOLDGASS,xCO-GASS'''
    if name=='xCOLDGASS' or name=='xCO-GASS':
        dfH2=get_xCOLDGASS()
      
    if name=='xGASS' or name=='xCO-GASS' or name=='xCOLDGASS':
        dfHI=get_xGASS()
        Nconf=(dfHI['HIconf_flag']==1).sum()
        #remove the galaxies with strong HI confusion, 108 in xGASS sample, still 20 slightly confused 0.1,0.2
#        dfHI=dfHI[dfHI['HIconf_flag']!=1] #
#        dfHI.reindex
        #SFR non-detections (8) have been set to -99
        dfHI['log_SFR']=-np.inf #first set all to nondetection     
        detections = dfHI['SFR_best'] > -99
        dfHI.loc[detections,'log_SFR']=np.log10(dfHI['SFR_best'][detections])

    if name=='xCOLDGASS' or name=='xCO-GASS': #combined catalog TODO only xCO in future 
        df=pd.merge(dfHI,dfH2,left_on='GASS',right_on='ID')
        logMgas=np.log10(10**df['logMHI']+10**df['logMH2'])
        df.insert(15,'logMgas',logMgas)
        H2frac=10**(df['logMH2']-df['logMgas'])
        df.insert(16,'H2frac',H2frac)

        df['logctime']= df['logMH2']-df['log_SFR']+np.log10(1.333) #add He
        if name=='xCO-GASS':
            df['uplim']=1*df['uplimHI']+2*df['uplimH2']
        else:
            df['uplim']=df['uplimH2']
    else:
        dfHI['uplim']=dfHI['uplimHI']
        df=dfHI

    df['log_sSFR']=df['log_SFR']-df['logMstar']
    #define central and remove 14/4 ungrouped galaxies (fix with Tinker?)
    #only works with HI, need to find values for gals not in xGASS 
    df=df[df['env_code_B'] > -1]
    df.reindex
    group=np.zeros(df.shape[0],dtype=np.bool)
    sat=df['env_code_B']==0
    cent=np.logical_or(df['env_code_B']==1,df['env_code_B']==2)
    group[sat]=False
    group[cent]=True
    df.insert(0,'central',group)
    if sample=='cent':
        df=df[df['central']==True]
    elif sample=='sat':
        df=df[df['central']==False]
    df.reindex
    df=df.drop(columns=['GASS','SDSS','SFR_best','HIconf_flag','env_code_B'])
    if shape:
        print(dfHI.shape,dfH2.shape,df.shape)
    return df