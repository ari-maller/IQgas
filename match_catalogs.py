import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord

xGASS_fname="xGASS_representative_sample.fits"
xCOLD_GASS_fname="xCOLDGASS_PubCat.fits"
vagc_dname='/Users/ari/Data/vagc-dr7/'
sdss_fname=vagc_dname+'object_catalog.fits'


df_xGASS=fits2pandas(xGASS_fname)
print(df_xGASS.columns.values)
df_xCOLD=fits2pandas(xCOLD_GASS_fname)
df_sdss=fits2pandas(sdss_fname)
df_group=tinker_catalog_pandas(group_fname)


c = coord.SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, distance=distance1*u.kpc)
catalog = coord.SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, distance=distance2*u.kpc)
idx, d2d, d3d = c.match_to_catalog_sky(catalog)