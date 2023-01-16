import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
import IQ

def setup_multiplot(Nr,Nc,xtitle=None,ytitle=None,xtitle2=None,ytitle2=None,**kwargs):
    ratio=Nr/Nc
    fs=(np.round(Nr/Nc*8),np.round(Nc/Nr*8))
    print(fs)
    f,axs=plt.subplots(Nr,Nc,figsize=fs,**kwargs)
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    ax=f.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(xtitle,fontsize='x-large')
    ax.set_ylabel(ytitle,fontsize='x-large',labelpad=20)
    if xtitle2:
        ax2=ax.twiny()
        ax2.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False) 
        ax2.set_xlabel(xtitle2,fontsize='x-large',labelpad=20)               
    if ytitle2:
        ax2=ax.twinx()
        ax2.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False) 
        ax2.set_ylabel(ytitle2,fontsize='x-large',labelpad=20)   
    return f,axs

dfgass=IQ.get_GASS()

#countour plot
a=np.arange(1,11)
c=np.outer(a,a)
fig = go.Figure(data = go.Contour( z=c,colorscale='Hot',contours=dict(start=1,end=100,size=10,),))
fig.show()

#scatter plot
fig = px.scatter(dfgass,x='logMstar',y='logctime',color='central',
    hover_data=['log_SFR']) #color can be int or floats
fig.show()

#tenary plot
fig = px.scatter_ternary(dfgass, a="logMstar", b="logMHI", c="logMH2", hover_name="log_SFR",
    color="central", size="H2frac", size_max=15,
    color_discrete_map = {True: "blue", False: "green"} )
fig.show()

