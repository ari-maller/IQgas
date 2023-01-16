''' An test of using plotly for plots '''
import numpy as np 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import IQ

#first compare EAGLE apatures
def EAGLE_compare():
    dname="/Users/ari/Dropbox/CCA Quenched Isolated Galaxies Workshop 2017-05-11/DATA/"
    fname='EAGLE_RefL0100HashDave2020GK_MHIH2HIIRhalfmassSFRT1e4_allabove1.8e8Msun.txt' #new file
    mstar,rdisk,mHI,mHI_2r50,mHI_30,mHI_70,mH2,mH2_2r50,mH2_30,mH2_70=np.loadtxt(dname+fname,usecols=[2,3,6,8,9,10,11,13,14,15],unpack=True) #Mstar,R,all,2R50,30,70
    frac=10**(mHI_2r50-mHI_70)
    print(np.isinf(mHI_30).sum())
    print(f"less that 0.95 {(frac < 0.95).sum()/len(frac)}")
    print(f"less that 0.90 {(frac < 0.90).sum()/len(frac)}")
    plt.hist(frac,range=[0.0,1.0],bins=100)
    plt.show()


#fig 1

def fig1():
    color=['red','blue','green','purple','orange','magenta']
    boxsize={'Eagle':100,'Mufasa':73.5,'TNG100':110.7,'Simba':147,'SC-SAM':110.7}
    Nb={'Eagle':30,'Mufasa':20,'TNG100':30,'Simba':30,'SC-SAM':30}
    names=IQ.sim_names()
    mtypes=['logMstar','logMHI','logMH2']
    for j,name in enumerate(names):
        data=get_sim(sim=name,sample='all',mcut=8.0)
        mstar=(data['logMstar']).to_numpy()
        volume=(boxsize[name])**3
        for i,mtype in enumerate(mtypes):
            keep=np.logical_and(data['logMstar'] > 8.0,data[mtype] > 7.0)
            mass=(data[mtype][keep]).to_numpy()
            IQ.xyhistplot(mass,axis[i],weight=1./volume,Nbins=Nb[name],log=True,linestyle=':',c=color[j])
            keep=np.logical_and(data['logMstar'] > 9.0,data[mtype] > 7.0)
            mass=(data[mtype][keep]).to_numpy()           
            IQ.xyhistplot(mass,axis[i],weight=1./volume,Nbins=Nb[name],log=True,label=name,c=color[j])         

    logm,phi,dx,dy=obs.gmf_GAMA(Wright=True)
    axis[0].plot(logm,np.log10(phi),linestyle='--',c='black',label='Observed')
    logm,logphi=obs.HImf_ALFALFA()
    axis[1].plot(np.log10(logm),np.log10(logphi),linestyle='--',c='black')
#    logm,logphi=obs.H2mf(model='constant')
#    axis[2].plot(logm,logphi,linestyle='--',c='black')    
    logm,logphi=obs.H2mf(model='luminosity')
    axis[2].plot(logm,logphi,linestyle='dashdot',c='black')
    m,phi=obs.H2mf_OR(type='orig')
    axis[2].plot(np.log10(m),np.log10(phi),linestyle='dashdot',c='black')
    x=np.log10(m[np.log10(m) <= logm[-1]])
    logphi2=np.interp(x,logm,logphi)
    x=np.append(x,np.log10(m[-1]))
    logphi2=np.append(logphi2,np.log10(phi[-1]))
    N=len(x)
    axis[2].fill_between(x,np.log10(phi[:N]),logphi2,color='gray',alpha=0.25)
#    m,phi=obs.H2mf_OR(type='ref')
#    axis[2].plot(np.log10(m),np.log10(phi),linestyle='--',c='black')
    m,phi=obs.H2mf_xCOLDGASS()
    axis[2].plot(np.log10(m),np.log10(phi),linestyle='--',c='black')   
#    axis[0].set_ylim([-4.0,-1.5])
    axis[0].set_ylim([-4.7,-1.5])
    axis[0].set_xlim([8.5,12.6])
    axis[1].set_xlim([8.5,11.4])
    axis[2].set_xlim([8.5,11.2])
    axis[0].set_xlabel(r'$\log M_{*}$',fontsize='x-large')
    axis[1].set_xlabel(r'$\log M_{HI}$',fontsize='x-large')
    axis[2].set_xlabel(r'$\log M_{H_2}$',fontsize='x-large')
    axis[0].legend(ncol=2,fontsize='small')
    legend_lines=[mpl.lines.Line2D([0],[0],color='gray',linestyle='-'),
                mpl.lines.Line2D([0],[0],color='gray',linestyle=':')]
    legend_names=[r'$M_* > 10^9 M_{\odot}$',r'$M_* > 10^8 M_{\odot}$']
    axis[1].legend(legend_lines,legend_names,loc='lower left')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('gasmf.pdf')
'''
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
'''

'''
df = IQ.get_GASS()
fig=px.scatter(df,x='logMstar',y='logMH2',color='central',
    labels={'logMstar':r'$\log M_{*}$','logMH2':r'$\log M_{H_2}$',
    'central':'Legend','True':'Central','False':'Satellite'})
fig.update_xaxes(title_font_family="Arial",title_font_size=18,tickfont_size=4)
fig.update_yaxes(title_font_family="Arial",title_font_size=18,tickfont_size=4)
fig.show()
'''
EAGLE_compare()