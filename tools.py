import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import IQ 

data=IQ.get_sim(sim='TNG')
data.drop(['central','logMgas'],axis=1,inplace=True)
keep=np.logical_and(data['logMHI'] > 6.0,data['logMH2'] > 6.0)
corr = data[keep].corr()
print(corr)
f,ax=plt.subplots()
map = sns.heatmap(corr,vmin=-1,vmax=1,center=0,
    cmap=sns.diverging_palette(20,220,n=200),square=True,ax=ax)
print(type(ax))
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,
    horizontalalignment='right')
plt.savefig('corr.pdf')
