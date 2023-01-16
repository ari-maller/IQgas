import caesar
import numpy as np
import matplotlib.pyplot as plt
import IQ
import observations as obs

groups=caesar.load('/Users/ari/Data/simba/m100n1024_151.hdf5')
galaxies=groups.galaxies
stellar_masses=[g.masses['stellar_30kpc'] for g in galaxies ]
f,ax=plt.subplots(1,1)
#ax.hist(np.log10(stellar_masses),bins=50,range=[8.5,12],density=True)
vol=147**3
IQ.xyhistplot(np.log10(stellar_masses),ax,xrange=[8.5,12],log=True,
    weight=1/vol,Nbins=50,label='Simba')
logm,phi,dphi_m,dphi_p=obs.gmf_GAMA(Wright=True)
ax.plot(logm,np.log10(phi),linestyle='--',c='black',label='Observed')
ax.plot(logm,np.log10(phi-dphi_m),linestyle=':',c='black')
ax.plot(logm,np.log10(phi+dphi_p),linestyle=':',c='black')
ax.set_xlim([8,11.5])
ax.set_ylim([-4,-1.5])
plt.savefig('mass_hist.pdf')