import matplotlib
import pylab as plt
import numpy as np
from astropy.table import Table
import read_lim17_cat as rl
#must activate with conda activate skymap
import skymap


def get_xGASS():
    fname="xGASS_representative_sample.fits"
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()
    return df

smap = skymap.Skymap(projection='hammer',lon_0=180, lat_0=0)
df_xGASS=get_xGASS()
zmax=np.max(df_xGASS['zSDSS'])
print("xGASS maximum z is {:4.3f}".format(zmax))
ra=df_xGASS['RA'].to_numpy()
dec=df_xGASS['DEC'].to_numpy()
smap.scatter(ra,dec,latlon=True,label='xGASS',s=3)

survey=['2MRS']
fname=rl.get_fnames(survey[0],plus=False)
df_gal = rl.read_lim17_galaxy(fname[0])
N=len(df_gal)
z=df_gal['z_cmb']
print(np.max(z))
#print(f"{survey[0]}, max z = {np.max(df_gal['z_cmb'].to_numpy())}, N gals = {N}")
#ra=df_group['ra'].to_numpy()
#dec=df_group['dec'].to_numpy()


plt.savefig('skymap.pdf')