# >>> execfile('simba_IQ_forAri.py')

## python simba_IQ_forAri.py  >  log/IQ.log 2>&1 &

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as os
import sys as sys
import numpy as np
#import fsps
from scipy.optimize import curve_fit
from scipy import stats
import h5py as h5py
#import gadget as g
#import utilities as util
#import daa_lib as daa
#from daa_constants import *
import IQ_lib as iq

import imp
imp.reload(iq)
#imp.reload(daa)
#imp.reload(util)



simdir = '/mnt/ceph/users/dangles/simba/m100n1024/s50j7k/'

outdir = simdir + 'IQ'
if not os.path.exists(outdir):
  try:
    os.mkdir(outdir)
  except OSError:
    try:
      os.makedirs(outdir)
    except OSError:
      print('...could not create output directory: ' + outdir)



#snaplist = [ 151, 114, 98, 87, 78, 71, 65 ]
snaplist = [ 151 ]


skipreading = 1


#--- constants
MSUN = 1.989e33                    # in g
UnitMass_in_g            = 1.989e43        # 1.e10/h solar masses
UnitMass_in_Msun         = UnitMass_in_g / MSUN
SolarAbundance_Simba = 0.0134
#-------


# --- loop over snapshot list
for snapnum in snaplist:

   if skipreading == 0:
    # --- read snapshot - STARS
    P = iq.readsnap( simdir, snapnum, 4, cosmological=1, snapshot_name='snap_m100n1024' )          
    m_star = P['m'][:] * UnitMass_in_Msun
    # ---read snapshot - GAS
    P = iq.readsnap( simdir, snapnum, 0, cosmological=1, snapshot_name='snap_m100n1024' )         
    m_gas = P['m'][:] * UnitMass_in_Msun
    sfr = P['sfr'][:]
    # --- read caesar file
    gr = iq.load_dict_from_hdf5( simdir + 'Groups/m100n1024_' + iq.snap_ext(snapnum) + '.hdf5' )   

   # galaxy properties from caesar file
   gr_ID = gr['galaxy_data']['GroupID']
   gr_nstar = gr['galaxy_data']['nstar']
   gr_ngas = gr['galaxy_data']['ngas']
   gr_StarMass = gr['galaxy_data']['dicts']['masses.stellar']
   gr_GasMass = gr['galaxy_data']['dicts']['masses.gas']
   gr_sfr = gr['galaxy_data']['sfr']
   gr_central = gr['galaxy_data']['central']
   gr_pos = gr['galaxy_data']['pos']
   gr_vel = gr['galaxy_data']['vel']
   gr_gasmetal = gr['galaxy_data']['dicts']['metallicities.sfr_weighted']
   ngal = gr_ID.size

   # define output dict
   data = { 'redshift':P['redshift'], 
            'ID':gr_ID, 'Central':gr_central, 'SFR':gr_sfr, 'pos':gr_pos, 'vel':gr_vel, 'gasmetal':gr_gasmetal,
            'Nstar':np.zeros(ngal,dtype=np.int32), 'StarMass':np.zeros(ngal),
            'Ngas':np.zeros(ngal,dtype=np.int32), 'GasMass':np.zeros(ngal)
          }

   # --- loop over all galaxies
   for n in range(ngal):

      # --- get star particle list in group
      slist_start = gr['galaxy_data']['slist_start'][n]
      slist_end = gr['galaxy_data']['slist_end'][n]

      m_star_ingal = np.array( [ m_star[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )
      Nstar = m_star_ingal.size
      if Nstar > 0:
        StarMass = np.sum(m_star_ingal)
      else:
        StarMass = 0.

      # --- get gas particle list in group
      glist_start = gr['galaxy_data']['glist_start'][n]
      glist_end = gr['galaxy_data']['glist_end'][n]

      m_gas_ingal = np.array( [ m_gas[k] for k in gr['galaxy_data']['lists']['glist'][glist_start:glist_end] ] )
      Ngas = m_gas_ingal.size
      if Ngas > 0:
        GasMass = np.sum(m_gas_ingal)
      else:
        GasMass = 0.

      data['Nstar'][n] = Nstar
      data['StarMass'][n] = StarMass
      data['Ngas'][n] = Ngas
      data['GasMass'][n] = GasMass

   # --- check a few numbers
   print('\nsnapnum =', snapnum, '  z =', np.round(P['redshift'],decimals=2))
   print('nstar =', np.min(gr_nstar.astype(float)/data['Nstar']), np.max(gr_nstar.astype(float)/data['Nstar']), '  StarMass =', np.min(gr_StarMass/data['StarMass']), np.max(gr_StarMass/data['StarMass']))
   ind_gas = data['Ngas'] > 0
   print('gas =', np.min(gr_ngas[ind_gas].astype(float)/data['Ngas'][ind_gas]), np.max(gr_ngas[ind_gas].astype(float)/data['Ngas'][ind_gas]), '  GasMass =', np.min(gr_GasMass[ind_gas]/data['GasMass'][ind_gas]), np.max(gr_GasMass[ind_gas]/data['GasMass'][ind_gas]))

   # --- write data to ouput file
   outfile = outdir + '/IQtest.hdf5'  
   iq.write_hdf5_from_dict( data, outfile )











print('\nDone!!\n')






   
