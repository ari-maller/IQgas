# >>> execfile('simba_IQ.py')

## python simba_IQ.py  >  log/IQ.log 2>&1 &

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as os
import sys as sys
import numpy as np
import fsps
from scipy.optimize import curve_fit
from scipy import stats
import h5py as h5py
import gadget as g
import utilities as util
import daa_lib as daa
from daa_constants import *

import imp
imp.reload(daa)
imp.reload(util)



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

dt_sfr = 100. # Myr

logmetal_bins = np.array( [ -2.2, -1.9, -1.6, -1.3, -1, -0.7, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5 ] )
Nbin_metal = logmetal_bins.size + 1
mid_logmetal = np.zeros(Nbin_metal)
mid_logmetal[1:-1] = (logmetal_bins[:-1] + logmetal_bins[1:] ) / 2.
mid_logmetal[0] = logmetal_bins[0];  mid_logmetal[-1] = logmetal_bins[-1]


ageGyr_bins = np.array( [ 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.55, 0.65, 0.75, 0.85, 0.95, 1.125, 1.375, 1.625, 1.875, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75 ] )
Nbin_age = ageGyr_bins.size + 1

skipreading = 0


# --- initialize FSPS
print('initializing FSPS...')
fsps_ssp = fsps.StellarPopulation( sfh=0,           # single stellar population
                                   zcontinuous=1,   # the SSPs is interpolated to the value of logzsol
                                   logzsol=0.0,     # metallicity in units of log(Z/Z_solar)
                                   imf_type=1 )       # Chabrier IMF

mass_remaining = np.zeros( [ Nbin_metal, fsps_ssp.stellar_mass.size ] )
for i in range(Nbin_metal):
   print('logzsol =', mid_logmetal[i])
   fsps_ssp.params["logzsol"] = mid_logmetal[i] 
   mass_remaining[i,:] = fsps_ssp.stellar_mass
   # --- not allowing the mass remaining to decrease...
   mbin = mass_remaining.shape[1]
   for j in range(mbin-1):
      mass_change = mass_remaining[i,mbin-1-j] - mass_remaining[i,mbin-2-j]
      if mass_change > 0:
        mass_remaining[i,mbin-2-j] += mass_change
print('done with FSPS')
SolarAbundance_Simba = 0.0134
SolarAbundance_FSPS = 0.0190


# --- loop over snapshot list
for snapnum in snaplist:

   if skipreading == 0:
    P = g.readsnap( simdir, snapnum, 4, cosmological=1, snapshot_name='snap_m100n1024' )            # snapshot
    gr = daa.load_dict_from_hdf5( simdir + 'Groups/m100n1024_' + g.snap_ext(snapnum) + '.hdf5' )    # caesar file

   m_star_now = P['m'][:] * UnitMass_in_Msun
   met_star = P['z'][:,0]
   logmet_star = np.log10( met_star / SolarAbundance_Simba )
   aform = P['age'][:]
   zform = 1./aform - 1.
   tform = 1e3 * util.age_of_universe( zform, h=P['hubble'], Omega_M=P['omega_matter'] )  # in Myr
   tnow = 1e3 * util.age_of_universe( P['redshift'], h=P['hubble'], Omega_M=P['omega_matter'] )  # in Myr
   age_star = (tnow-tform) / 1e3  # in Gyr

   metbin_star = np.digitize( logmet_star, logmetal_bins, right=False)
   agebin_star = np.digitize( age_star, ageGyr_bins, right=False)
   

   # --- get original stellar mass at time of formation ---
   m_star = m_star_now.copy()
   for i in range(m_star.size): 
      if i % 100000 == 0:
        print(i)
        sys.stdout.flush()
      this_mass_remaining = mass_remaining[metbin_star[i],:]
      m_star[i] = m_star[i] / np.interp( np.log10( (tnow-tform[i])*1e6 ), fsps_ssp.ssp_ages, this_mass_remaining )


   gr_ID = gr['galaxy_data']['GroupID']
   gr_nstar = gr['galaxy_data']['nstar']
   gr_StarMass = gr['galaxy_data']['dicts']['masses.stellar']
   gr_sfr = gr['galaxy_data']['sfr']
   gr_central = gr['galaxy_data']['central']
   gr_pos = gr['galaxy_data']['pos']
   gr_vel = gr['galaxy_data']['vel']
   gr_gasmetal = gr['galaxy_data']['dicts']['metallicities.sfr_weighted']
   ngal = gr_ID.size

   data = { 'redshift':P['redshift'], 'logmetal_bins':logmetal_bins, 'ageGyr_bins':ageGyr_bins, 'ssp_ages':fsps_ssp.ssp_ages, 'mass_remaining':mass_remaining,
            'ID':gr_ID, 'Central':gr_central, 'SFR':gr_sfr, 'pos':gr_pos, 'vel':gr_vel, 'gasmetal':gr_gasmetal,
            'Nstar':np.zeros(ngal,dtype=np.int32), 'StarMass':np.zeros(ngal), 'SFR_100Myr':np.zeros(ngal), 'NstarNew_100Myr':np.zeros(ngal,dtype=np.int32),
            'StarMassFormed_in_metal_age_Bin':np.zeros([ngal,Nbin_metal,Nbin_age]), 'StarMassNow_in_metal_age_Bin':np.zeros([ngal,Nbin_metal,Nbin_age]) }

   # --- loop over all galaxies
   for n in range(ngal):
   #for i in range(3):

      slist_start = gr['galaxy_data']['slist_start'][n]
      slist_end = gr['galaxy_data']['slist_end'][n]

      m_star_now_ingal = np.array( [ m_star_now[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )
      Nstar = m_star_now_ingal.size
      if Nstar > 0:
        StarMass = np.sum(m_star_now_ingal)
      else:
        StarMass = 0.

      tform_ingal = np.array( [ tform[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )
      m_star_ingal = np.array( [ m_star[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )
      dt = tnow - tform_ingal
      indt = np.where( dt < dt_sfr )[0]
      if indt.size > 0:
        SFR = np.sum( m_star_ingal[indt] ) / (dt_sfr*1e6)
      else:
        SFR = 0.

      StarMassFormed_in_metal_age_Bin = np.zeros([Nbin_metal,Nbin_age])
      StarMassNow_in_metal_age_Bin = np.zeros([Nbin_metal,Nbin_age])
      metbin_star_ingal = np.array( [ metbin_star[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )
      agebin_star_ingal = np.array( [ agebin_star[k] for k in gr['galaxy_data']['lists']['slist'][slist_start:slist_end] ] )

      for i in range(Nstar):
         StarMassFormed_in_metal_age_Bin[metbin_star_ingal[i],agebin_star_ingal[i]] += m_star_ingal[i]
         StarMassNow_in_metal_age_Bin[metbin_star_ingal[i],agebin_star_ingal[i]] += m_star_now_ingal[i]
      
      data['Nstar'][n] = Nstar
      data['StarMass'][n] = StarMass
      data['SFR_100Myr'][n] = SFR 
      data['NstarNew_100Myr'][n] = indt.size
      data['StarMassFormed_in_metal_age_Bin'][n,:,:] = StarMassFormed_in_metal_age_Bin[:,:]
      data['StarMassNow_in_metal_age_Bin'][n,:,:] = StarMassNow_in_metal_age_Bin[:,:]

   print('\nsnapnum =', snapnum, '  z =', np.round(P['redshift'],decimals=2), '  nstar =', np.min(gr_nstar.astype(float)/data['Nstar']), np.max(gr_nstar.astype(float)/data['Nstar']), '  StarMass =', np.min(gr_StarMass/data['StarMass']), np.max(gr_StarMass/data['StarMass']), np.min(np.sum(data['StarMassNow_in_metal_age_Bin'],axis=(1,2))/data['StarMass']), np.max(np.sum(data['StarMassNow_in_metal_age_Bin'],axis=(1,2))/data['StarMass']))

   # --- write data to ouput file
   outfile = outdir + '/SimbaIQ_m100n1024_s' + g.snap_ext(snapnum) + '_z%.2f' % P['redshift'] + '.hdf5'  
   daa.write_hdf5_from_dict( data, outfile )











print('\nDone!!\n') 






   
