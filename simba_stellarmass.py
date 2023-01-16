import numpy as np
import pickle 
import fsps

logmetal_bins = np.array([-2.2,-1.9,-1.6,-1.3,-1,-0.7,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.5])
Nbin_metal = logmetal_bins.size + 1
mid_logmetal = np.zeros(Nbin_metal)
mid_logmetal[1:-1] = 0.5*(logmetal_bins[:-1] + logmetal_bins[1:])
mid_logmetal[0] = logmetal_bins[0]  
mid_logmetal[-1] = logmetal_bins[-1]

# --- initialize FSPS
print('initializing FSPS...')
fsps_ssp = fsps.StellarPopulation( sfh=0,           # single stellar population
                                   zcontinuous=1,   # the SSPs is interpolated to the value of logzsol
                                   logzsol=0.0,     # metallicity in units of log(Z/Z_solar)
                                   imf_type=1 )       # Chabrier IMF

mass_remaining = np.zeros([ Nbin_metal,fsps_ssp.stellar_mass.size])
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

print((fsps_ssp.ssp_ages).shape,mass_remaining.shape)
pickle.dump([fsps_ssp.ssp_ages,mass_remaining], open("fsps.pkl",'wb'))
print('done with FSPS')
