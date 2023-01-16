import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.table import Table
import pandas as pd
from sklearn import decomposition
import IQ
import observations as obs
import myplots as my

def read_xGASS(info=False):
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want=['GASS','SDSS','HI_FLAG','lgMstar','lgMHI','HIconf_flag','NYU_id','env_code_B','SFR_best']
    discard=list(set(df.columns.values.tolist()).difference(want))
    df=df.drop(columns=discard)
    df.rename(columns={'lgMstar':'logMstar','lgMHI':'logMHI'},inplace=True)
    logSFR=np.zeros(df.shape[0])
    sfr_ok= df['SFR_best']!=-99 # -99 used for nondetections
    logSFR[sfr_ok]=np.log10(df['SFR_best'][sfr_ok])
    df.insert(6,'log_SFR',logSFR)
    df=df[sfr_ok]
    df.reindex
    logsSFR=df['log_SFR']-df['logMstar']
    df.insert(7,'log_sSFR',logsSFR)
    if info:
        print(f"HI_FLAGs are {np.unique(df['HI_FLAG'])}")
        print(f"Galaxies with good HI detections {(df['HI_FLAG'] < 2).sum()}")
        print(f"Galaxies with good/ok HI detections {(df['HI_FLAG'] < 3).sum()}")
        print(f"HI not confused galaxies {(df['HIconf_flag']==0).sum()}")
        goodHI=np.logical_and(df['HI_FLAG'] < 2,df['HIconf_flag']==0)
        print(f"good not confused {goodHI.sum()}")
        print(f"Galaxies with no HI detection {(df['HI_FLAG']==99).sum()}")
        print(f"Isolated central galaxies {(df['env_code_B']==1).sum()}")
        print(f"Group central galaxies {(df['env_code_B']==2).sum()}")
        print(f"Galaxies with no group identification {(df['env_code_B']==-1).sum()}")
        print(f"Satallite galaxies {(df['env_code_B']==0).sum()}")
    return df

def read_xCOLDGASS(info=False):
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    want = ['ID','SDSS','OBJID','LOGMSTAR','LOGSFR_BEST','FLAG_CO','LOGMH2',
        'LOGMH2_ERR','LIM_LOGMH2','LOGSFR_ERR']
    discard=list(set(df.columns.values.tolist()).difference(want))
    df=df.drop(columns=discard)
    df.rename(columns={'LOGMSTAR':'logMstar','LOGSFR_BEST':'log_SFR'},inplace=True)
    df=df[np.isfinite(df['log_SFR'])] #drop two -inf SFRs
    df.reindex
    logMH2=np.zeros(df.shape[0])
    yes_CO= df['FLAG_CO'] < 2
    no_CO = df['FLAG_CO']==2
    logMH2[yes_CO]=df['LOGMH2'][yes_CO]
    logMH2[no_CO]=df['LIM_LOGMH2'][no_CO]
    df.insert(7,'logMH2',logMH2)
    logsSFR=df['log_SFR']-df['logMstar']
    df.insert(8,'log_sSFR',logsSFR)
    return df

def get_xGASS():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df1 = dat.to_pandas()
    want=['GASS','SDSS','HI_FLAG','lgMHI','HIconf_flag', 'NYU_id', 'env_code_B']
    discard=list(set(df1.columns.values.tolist()).difference(want))  
    df1=df1.drop(columns=discard)    
    fname="xCOLDGASS_PubCat.fits"
    dat = Table.read(fname, format='fits')
    df2 = dat.to_pandas()
    want = ['ID','OBJID','LOGMSTAR','LOGSFR_BEST','FLAG_CO','LOGMH2','LIM_LOGMH2']
    discard=list(set(df2.columns.values.tolist()).difference(want))
    df2=df2.drop(columns=discard)
    df=pd.merge(df1,df2,left_on='GASS',right_on='ID')
    df.rename(columns={'LOGMSTAR':'logMstar','lgMHI':'logMHI','LOGSFR_BEST':'log_SFR'},inplace=True)
    logMH2=np.zeros(df.shape[0])
    yes_CO= df['FLAG_CO'] < 2
    no_CO = df['FLAG_CO']==2
    logMH2[yes_CO]=df['LOGMH2'][yes_CO]
    logMH2[no_CO]=df['LIM_LOGMH2'][no_CO]
    df.insert(14,'logMH2',logMH2)
    group=np.empty(df.shape[0],dtype=np.bool)
    sat=df['env_code_B']==0
    cent=np.logical_or(df['env_code_B']==1,df['env_code_B']==1)
    group[sat]=False
    group[cent]=True
    df.insert(0,'central',group)
    df=df[np.isfinite(df['log_SFR'])]
    df.reindex
    logsSFR=df['log_SFR']-df['logMstar']
    df.insert(15,'log_sSFR',logsSFR)
    return df

#sanity checks to xGASS papers
#Janowiecki 2020
def J20SFS(SFR=False):
    logMstar=np.linspace(9,11.5,num=50)
    fit_sfms=[-0.344,-9.822]
    fit_sfs=[0.088,0.188]
    logsSFR=fit_sfms[0]*(logMstar-9.0)+fit_sfms[1]
    if SFR:
        logsSFR=logsSFR+logMstar
    return logMstar,logsSFR

def plot_J20SFS(axis,plot_sigma=False,SFR=False):
    if SFR:
        logMstar,logsSFR=J20SFS(SFR=True)
    else:
        logMstar,logsSFR=J20SFS()
    axis.plot(logMstar,logsSFR,linestyle='--',color='red')
    if plot_sigma:
        sigma=fit_sfs[0]*(logMstar-9.0)+fit_sfs[1]
        axis.plot(logMstar,logsSFR+sigma,linestyle=':',color='purple')
        axis.plot(logMstar,logsSFR-sigma,linestyle=':',color='purple')

def J20_fig1(sSFR=True):
    data=read_xGASS()
    f,ax=plt.subplots()
    good=data['HIconf_flag']==0
    bad=data['HIconf_flag'] > 0
    if sSFR:
        ax.scatter(data['logMstar'][good],data['log_sSFR'][good],color='grey',marker='o',facecolor='none')
        ax.scatter(data['logMstar'][bad],data['log_sSFR'][bad],color='grey',marker='x')
        plot_J20SFS(ax,plot_sigma=True)       
        ax.set_ylabel('log (sSFR)')
        ax.set_ylim([-13.5,-8.5])
    else:
        ax.scatter(data['logMstar'],data['log_SFR'])
        plot_J20SFS(ax,plot_sigma=True,SFR=True)
        ax.set_ylabel('log (SFR)')

    ax.set_xlabel(r'$\log M_{stellar}$')
    plt.show()

def J20_fig2(SFR=False):
    data1=read_xGASS()
    data2=read_xCOLDGASS()
    f,axs=plt.subplots(2,2,figsize=(8,8),sharex=True,sharey='row')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    HI=data1['HI_FLAG'] < 3
    noHI=data1['HI_FLAG']==99
    CO=data2['FLAG_CO'] < 2
    noCO=data2['FLAG_CO']==2
    #caluculate SFS offset
    logMstar_SFS,logSFR_SFS=J20SFS(SFR=True)
    diff1=data1['log_SFR']-(np.interp(data1['logMstar'],logMstar_SFS,logSFR_SFS))
    diff2=data2['log_SFR']-(np.interp(data2['logMstar'],logMstar_SFS,logSFR_SFS))
    norm=plt.Normalize(-2.5,1.5)
    if SFR:
        logSFR1=data1['log_SFR']
        logSFR2=data2['log_SFR']
    else:
        logSFR1=data1['log_sSFR']
        logSFR2=data2['log_sSFR']
    plt.set_cmap(plt.cm.get_cmap('cool_r',4))
    #subplot 1
    cb=axs[0][0].scatter(data1['logMstar'][HI],logSFR1[HI],
        marker='o',s=6,c=diff1[HI],label='HI detections',norm=norm)
    axs[0][0].scatter(data1['logMstar'][noHI],logSFR1[noHI],norm=norm,
        marker='x',s=10,c=diff1[noHI],linewidth=1,label='HI nondetections')
    plot_J20SFS(axs[0][0],SFR=SFR)
    axs[0][0].legend(fontsize='small',frameon=False)
    #subplot 2
    axs[0][1].scatter(data2['logMstar'][CO],logSFR2[CO],norm=norm,
        marker='s',s=6,c=diff2[CO],label=r'$H_2$(CO) detections')
    axs[0][1].scatter(data2['logMstar'][noCO],logSFR2[noCO],norm=norm,
        marker='x',s=10,c=diff2[noCO],linewidth=1,label=r'$H_2$(CO) nondetections')
    plot_J20SFS(axs[0][1],SFR=SFR)
    axs[0][1].legend(fontsize='small',frameon=False)
    #subplot 3
    axs[1][0].scatter(data1['logMstar'][HI],data1['logMHI'][HI],norm=norm,
        marker='o',s=6,c=diff1[HI])
    axs[1][0].scatter(data1['logMstar'][noHI],data1['logMHI'][noHI],norm=norm,
        marker='x',s=10,c=diff1[noHI],linewidth=1)
    #subplot 4
    axs[1][1].scatter(data2['logMstar'][CO],data2['logMH2'][CO],norm=norm,
        marker='s',s=6,c=diff2[CO])
    axs[1][1].scatter(data2['logMstar'][noCO],data2['logMH2'][noCO],norm=norm,
        marker='x',s=10,c=diff2[noCO],linewidth=1)

    axs[0][0].set_ylim([-3.25,1.5])
    axs[0][1].set_ylim([-3.25,1.5])
    axs[1][0].set_ylim([7.5,10.75])
    axs[1][1].set_ylim([7.5,10.75])
    axs[1][1].set_xlim([8.8,11.45])
    axs[0][0].set_ylabel('log (SFR)',fontsize='xx-large')
    axs[1][0].set_ylabel(r'$\log (M_{gas})$',fontsize='xx-large')
    axs[1][0].set_xlabel(r'$\log M_*$',fontsize='xx-large')
    axs[1][1].set_xlabel(r'$\log M_*$',fontsize='xx-large')
    #colorbar
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.83, 0.25, 0.03, 0.5]) #left,bottom,width,height
    f.colorbar(cb, cax=cbar_ax)
    plt.savefig('J20fig2.pdf')

#sanity checks from Fletcher 2020 paper
def F20_fig1():
    mpl.rc('lines', linewidth=1)
    data=read_xCOLDGASS()
    uplim=data['FLAG_CO']==2
    detect=np.invert(uplim)
    print(f"uplim: {uplim.sum()}, detections: {detect.sum()}")
#    plt.scatter(data['log_SFR'][uplim],data['logMH2'][uplim],s=2,marker='o',c='red')
#    plt.scatter(data['log_SFR'][detect],data['logMH2'][detect],s=2,marker='o',c='blue')
    plt.errorbar(data['log_SFR'][uplim],data['logMH2'][uplim],yerr=data['LOGMH2_ERR'][uplim],
        xerr=data['LOGSFR_ERR'][uplim],c='red',linestyle='',label='non-detections')
    plt.errorbar(data['log_SFR'][detect],data['logMH2'][detect],yerr=data['LOGMH2_ERR'][detect],
        xerr=data['LOGSFR_ERR'][detect],c='blue',linestyle='',label='detections')  
    x,y=obs.H2_SFR()
    plt.plot(x,y,c='brown')
    plt.legend()  
    plt.xlim([-3,2])
    plt.ylim([7,11])
    plt.xlabel(r"$\log SFR [yr^{-1}]$")
    plt.ylabel(r"$\log M_{H_2} [M_{\odot}]$")
    plt.savefig('f20_fig1.pdf')

def fig_deplete():
    data1=read_xGASS()
    data2=read_xCOLDGASS()
    f,axs=plt.subplots(1,2,figsize=(8,4))
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    HI=data1['HI_FLAG'] < 3
    noHI=data1['HI_FLAG']==99
    CO=data2['FLAG_CO'] < 2
    noCO=data2['FLAG_CO']==2
    #caluculate SFS offset
    logMstar_SFS,logSFR_SFS=J20SFS(SFR=True)
    diff1=data1['log_SFR']-(np.interp(data1['logMstar'],logMstar_SFS,logSFR_SFS))
    diff2=data2['log_SFR']-(np.interp(data2['logMstar'],logMstar_SFS,logSFR_SFS))
    dtime1=np.log10(10**data1['logMHI']/10**data1['log_SFR'])
    dtime2=np.log10(10**data2['logMH2']/10**data2['log_SFR'])
    axs[0].scatter(data1['logMstar'],dtime1)
    axs[1].scatter(data2['logMstar'],dtime2)
    plt.show()

def diff(data,sample='xGASS'):
    pH2=[0.81239644,-7.37196337] #H2 vrs SFR
    pHI=[0.57942498,-5.59053751] #HI vrs SFR
    pSFS=[0.656,-6.726] #from J20, M* vrs SFR
    pHI2=[0.5983187, 3.26186368] #H2 vrs HI
#    pHIM=
#    pH2M=
    diffH2=data['log_SFR']-(pH2[0]*data['logMH2']+pH2[1])
    diffHI=data['log_SFR']-(pHI[0]*data['logMHI']+pHI[1])
    diffSFS=data['log_SFR']-(pSFS[0]*data['logMstar']+pSFS[1])
    diffH2_HI=data['logMH2']-(pHI2[0]*data['logMHI']+pHI2[1])
    return diffH2,diffHI,diffSFS,diffH2_HI    

def fig_sfr():
    data=get_xGASS()
    f,axs=plt.subplots(1,3,figsize=(10,4),sharey=True)
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    HI=data['HI_FLAG'] < 3
    noHI=data['HI_FLAG']==99
    CO=data['FLAG_CO'] < 2
    noCO=data['FLAG_CO']==2
    fields=['logMH2','logMHI','logMstar']
    name=[r'$\log M_{H_2}$',r'$\log M_{HI}$',r'$\log M_*$']
    name2=[r'$\Delta H_2$',r'$\Delta HI$',r'$\Delta SFS$']
    logM=np.linspace(8,10.5,num=25)
    diffs=diff(data)
    for i,field in enumerate(fields):
        axs[i].scatter(data[field][CO],data['log_SFR'][CO],marker='o',s=6,c=diffs[i][CO],cmap='inferno')
        axs[i].scatter(data[field][noCO],data['log_SFR'][noCO],marker='x',s=6,c=diffs[i][noCO],linewidth=1) 
        p=np.polyfit(data[field][CO],data['log_SFR'][CO],1) 
        if i < 2:
            axs[i].plot(logM,p[0]*logM+p[1],linestyle='--',color='green')
        else:
            plot_J20SFS(axs[2],SFR=True)
        axs[i].set_xlabel(name[i])

    axs[0].set_ylabel('log SFR')
    plt.savefig('figSFR.pdf')

def fig_all():
    data=get_xGASS()
    f,axs=plt.subplots(3,3,figsize=(7,7),sharey='row',sharex='col')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    detect=np.logical_and(data['FLAG_CO'] < 2,data['HI_FLAG'] < 3)
    uplim=np.invert(detect)
    xfield=['logMstar','logMHI','logMH2']
    yfield=['logMHI','logMH2','log_SFR']
    xnames=[r'$\log \, M_*$',r'$\log \, M_{HI}$',r'$\log \, M_{H_2}$']
    ynames=[r'$\log \, M_{HI}$',r'$\log \, M_{H_2}$',r'$\log \, SFR$']
    logM=np.linspace(8,10.5,num=25)
    logM2=np.linspace(9,11.5,num=25)
    diffs=diff(data)
    for i in range(0,3):
        for j in range(0,3):
            if j>i:
                axs[i][j].axis('off')
            else:
                axs[i][j].scatter(data[xfield[j]][detect],data[yfield[i]][detect],marker='o',s=6)
                axs[i][j].scatter(data[xfield[j]][uplim],data[yfield[i]][uplim],marker='v',s=6,linewidth=1)           
                p=np.polyfit(data[xfield[j]][detect],data[yfield[i]][detect],1) 
                print(i,j,xfield[j],yfield[i],p)
                if j==0:
                    axs[i][j].plot(logM2,p[0]*logM2+p[1],linestyle='--',color='green')
                    axs[i][j].set_ylabel(ynames[i])
                else:
                    axs[i][j].plot(logM,p[0]*logM+p[1],linestyle='--',color='green')
                if i==2:
                    axs[i][j].set_xlabel(xnames[j])
    plot_J20SFS(axs[2][0],SFR=True)
    #correlation matrices - drop everything but 4 fields for correlation function
    data.drop(['GASS','SDSS','HI_FLAG','HIconf_flag','NYU_id','env_code_B'],axis=1,inplace=True)
    data.drop(['ID','OBJID','FLAG_CO','LOGMH2','LIM_LOGMH2','log_sSFR'],axis=1,inplace=True)
    corr = data[detect].corr()
    xvals=axs[0][1].get_xlim()
    yvals=axs[0][1].get_ylim()
    axs[0][1].text(xvals[0]+0.5,yvals[1]-1.0,corr.to_string(),color='blue')
    corr = data.corr()  
    xvals=axs[0][1].get_xlim()
    yvals=axs[0][1].get_ylim()
    axs[0][1].text(xvals[0]+0.5,yvals[1]-2.5,corr.to_string(),color='orange')   
    plt.savefig('figall.pdf')
    Ncomponents=3
    pca = decomposition.PCA(n_components=Ncomponents)
    pca.fit(data)
    print('PCA analysis')
    for i in range(Ncomponents):
        print(pca.explained_variance_[i],pca.components_[i])

def figdiffs():
    data=get_xGASS()
    f,axs=plt.subplots(3,3,figsize=(7,7),sharey='row',sharex='col')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    detect=np.logical_and(data['FLAG_CO'] < 2,data['HI_FLAG'] < 3)
    uplim=np.invert(detect)
    diffs=diff(data)
    ylabels=[r'\Delta HI(M_*)',r'$\Delta H_2 (M_*)',r'$\Delta SFR (M_*)']
    xlabels=[r'$\Delta SFR (M_*)',r'$\Delta SFR (M_{HI})',r'$\Delta SFR (M_{H_2})']
    for i in range(3):
        for j in range(3):
            if j>i:
                axis[i][j].axis('off')
            else:
                axs[i][j].scatter(diffHI[detect],diffH2[detect],marker='o',s=6)
                axs[i][j].scatter(diffHI[uplim],diffH2[uplim],marker='x',s=6,linewidth=1)

    plt.show()

def figcorr():
    data=get_xGASS()
    f,axs=plt.subplots(1,2)
#    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    HI=data['HI_FLAG'] < 3
    noHI=data['HI_FLAG']==99
    CO=data['FLAG_CO'] < 2
    noCO=data['FLAG_CO']==2
    fields=['logMH2','logMHI','logMstar']
    name=[r'$\log M_{H_2}$',r'$\log M_{HI}$',r'$\log M_*$']
    logM=np.linspace(8,10.5,num=25)
    pH2=[0.81239644,-7.37196337]
    diffH2=data['log_SFR']-(pH2[0]*data['logMH2']+pH2[1])
    pHI=[0.57942498,-5.59053751]
    diffHI=data['log_SFR']-(pHI[0]*data['logMH2']+pHI[1])
    logMstar_SFS,logSFR_SFS=J20SFS(SFR=True)
    diffsfs=data['log_SFR']-(np.interp(data['logMstar'],logMstar_SFS,logSFR_SFS))
    axs[0].scatter(diffHI[CO],diffH2[CO],marker='o')
    axs[0].scatter(diffHI[noCO],diffH2[noCO],marker='x',linewidth=1)
    axs[0].plot([-2.0,1.0],[-2.0,1.0])
    axs[0].axis('equal')
    axs[0].set_xlabel('$\Delta HI$')
    axs[0].set_ylabel('$\Delta H_2$')
    axs[1].scatter(diffsfs[CO],diffH2[CO],marker='o')
    axs[1].scatter(diffsfs[noCO],diffH2[noCO],marker='x',linewidth=1)
    axs[1].plot([-2.0,1.0],[-2.0,1.0])
    axs[1].axis('equal')
    axs[1].set_xlabel('$\Delta SFS$')
    plt.show()

F20_fig1()
