import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn as skl 
plt.style.use('fivethirtyeight')
c=list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
name=['b','r','y','g','s','p']
color={name[0]:c[0],name[1]:c[1],name[2]:c[2], name[3]:c[3],name[4]:c[4],name[5]:c[5]}

def test_colors():
    c=list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    print(c)
    x=np.linspace(0,1)
    for i in range(len(c)+1):
        plt.plot(x,i*x)

    plt.savefig('test.pdf')

def hist2dplot(axis,x,y,contour=True,image=True,**kwargs):
    colors=['#ffffcc','#c2e699','#78c679','#238443']
    h,xed,yed=np.histogram2d(x,y,**kwargs)
    total=h.sum()
    h=h/total
    hflat=np.sort(np.reshape(h,-1)) #makes 1D and sorted 
    csum=np.cumsum(hflat)
    values=1.0-np.array([0.9973,0.9545,0.6827,0.0])
    levels=[]
    for val in values:
        idx = (np.abs(csum - val)).argmin()
        levels.append(hflat[idx])

    if contour:
        axis.contourf(h,levels,colors=colors,extent=[xed[0],xed[-1],yed[0],yed[-1]])
    
def xnormed_hist2d(ax,x,y,Nbins=25,range=None,contour=True,image=True):
    Ndata=len(x)
    Ncut=Ndata/(10*Nbins**2)
    h,xed,yed=np.histogram2d(x,y,bins=Nbins,range=range)
    Nx=(h.shape)[0]
    idx=np.arange(0,Nx)
    for i in idx: #equal x weighing
        total=np.sum(h[i])
        h[i]= h[i]/total

    h=np.transpose(h)
    if image:
        hmask=np.ma.masked_where( h < Ncut, h)
        ax.imshow(hmask,origin='lower',extent=[xed[0],xed[-1],yed[0],yed[-1]])
    if contour:
        levs=[0.001,0.01,0.1]
        ax.contour(h,levs,origin='lower',extent=[xed[0],xed[-1],yed[0],yed[-1]])


def fig_residual(data,figname='',labels=None,residual=True):
    plt.set_cmap('jet')
    N=data.shape[1]-1
    Nres=np.int((N+1)*N/2)
    res_array=np.zeros((Nres,data.shape[0]))
    resarray=[]
    namearray=[]
    if labels:
        if len(labels)!=N+1:
            print("must pass same number of labels as vectors")
            print(len(labels),N,data.index)
            exit(1)
    else:
        labels=["vector"+str(i) for i in range(N+1)]
    f,axs=plt.subplots(N,N,figsize=(7,7),sharey='row',sharex='col')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    col=list(data)
    for i in range(0,N):
        for j in range(0,N):
            if j>i:
                axs[i][j].axis('off')
            else:
                p = np.polyfit(data[col[j]],data[col[i+1]],1)  
                axs[i][j].annotate("m = {:.2f}".format(p[0]),xy=(0.05,0.9),xycoords='axes fraction')
                if type(data[col[0]])!=np.ndarray: 
                    res=(data[col[i+1]]-(p[0]*data[col[j]]+p[1])).to_numpy() #residual from linear fit
                else:
                    res=data[col[i+1]]-(p[0]*data[col[j]]+p[1])
                resarray.append(res) 
                ytmp=(labels[i+1]).replace('\log','')
                xtmp=(labels[j]).replace('\log','')
                namearray.append(f"$\Delta$ {ytmp}({xtmp})")         
                axs[i][j].scatter(data[col[j]],data[col[i+1]],s=5,c=res)
                xlim=axs[i][j].get_xlim()
                xpts=xlim[0]+(xlim[1]-xlim[0])*np.linspace(0,1.0)
                axs[i][j].plot(xpts,p[0]*xpts+p[1],linestyle='--',color='orange')
                if j==0:
                    axs[i][j].set_ylabel(labels[i+1])
                if i==(N-1):
                    axs[i][j].set_xlabel(labels[j])
    
    corr=(data.corr()).values
    cstr=[]
    for i in range(corr.shape[0]):
        rowstr=[]        
        for  j in range(corr.shape[1]):
            tmp='{:.2f}'.format(corr[i][j])
            rowstr.append(tmp)
        cstr.append(rowstr)
    axs[0][N-1].table(cellText=cstr,rowLabels=labels,colLabels=labels,loc=9,fontsize='xx-large')
    if residual:
        plt.savefig(figname+'1.pdf')
        df=pd.DataFrame(resarray)
        df=df.transpose()
        fig_residual(df,figname=figname,labels=namearray,residual=False)
    else:
        plt.savefig(figname+'2.pdf')

def test_hist():
    x=np.random.normal(size=1000000)
    y=np.random.normal(size=1000000)
    f,axis=plt.subplots()
    hist2dplot(axis,x,y,bins=30,range=[[-3.5,3.5],[-3.5,3.5]])
    plt.savefig('test_contour.pdf')

if __name__=='__main__':
    test_hist()