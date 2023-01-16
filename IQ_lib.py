#!/usr/bin/env python

import numpy as np
import h5py as h5py
import pandas as pd
from glob import glob
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math


# ------------------------------ CONSTANTS
CM_PER_KM = 1e5
CM_PER_MPC = 3.085678e24
CM_PER_KPC = CM_PER_MPC / 1000.
HUBBLE = 3.2407789e-18             # in h/sec
PROTONMASS = 1.6726e-24            # in g
MSUN = 1.989e33                    # in g
SEC_PER_YEAR =  3.15569e7
G_UNIV = 6.672e-8                       # [ cm^3 g^-1 s^-2 ]
K_BOLTZMANN = 1.38065812e-16       # cgs
SPEED_OF_LIGHT = 2.99792458e10        # speed of light (cm s^-1)
SIGMA_T = 6.6524e-25                   # cm^(-2) Thomson cross-section
GAMMA = 5.0/3.0
GAMMA_MINUS1 = GAMMA - 1.

# ------------------------------ UNITS
UnitMass_in_g            = 1.989e43        # 1.e10/h solar masses
UnitMass_in_Msun         = UnitMass_in_g / MSUN
UnitVelocity_in_cm_per_s = 1e5             # 1 km/s
UnitLength_in_cm         = 3.085678e21     # 1 kpc/h
UnitLength_in_Mpc        = 0.001
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s          # seconds / h
UnitTime_in_Gyr          = UnitTime_in_s / ( SEC_PER_YEAR * 1e9 )
UnitTime_in_Myr          = UnitTime_in_s / ( SEC_PER_YEAR * 1e6 )
UnitDensity_in_cgs       = UnitMass_in_g / UnitLength_in_cm**3.
UnitEnergy_in_cgs        = UnitMass_in_g * UnitLength_in_cm**2. / UnitTime_in_s**2.
UnitMdot_in_Msun_per_yr = UnitMass_in_Msun / (UnitTime_in_s / SEC_PER_YEAR)
G_UNIV_CodeUnits = G_UNIV / UnitLength_in_cm**3 * UnitMass_in_g * UnitTime_in_s**2
SolarAbundance           = 0.02

# ------------------------------ COSMOLOGY
h = 0.68
hinv = 1. / h
rho_crit = 3 * (HUBBLE*h)**2 / ( 8 * np.pi * G_UNIV )       # in cgs
Omega0 = 0.3
OmegaLambda = 0.7
OmegaBaryon = 0.048



def read_AHF_particles(file):
   f = open(file, "r")
   Nhalos = int( f.readline() )
   f.close()
   df = pd.read_csv(file, delim_whitespace=True, names=['id','Ptype'], header=0)
   Nlines = df['id'].size
   Ndata = Nlines - Nhalos
   data = { 'id':np.zeros(Ndata,dtype='int64'), 'Ptype':np.zeros(Ndata,dtype='int8'), 'HaloID':np.zeros(Ndata,dtype='int32') }
   jbeg = 0
   for i in xrange(0, Nhalos):
      jend = jbeg + df['id'][ jbeg+i ]
      data['id'][ jbeg : jend ] = df['id'][ jbeg+i+1 : jend+i+1 ].values
      data['Ptype'][ jbeg : jend ] = df['Ptype'][ jbeg+i+1 : jend+i+1 ].values
      data['HaloID'][ jbeg : jend ] = df['Ptype'][ jbeg+i ]
      jbeg = jend
   return data


def read_AHF_halos( simdir, snapnum, AHFdir='AHF', readheader=True, h=0.7 ):

   if len( glob( simdir + '/' + AHFdir + '/snap' + snap_ext(snapnum) + '*.AHF_halos') ) == 0:
     AHFdir = 'AHF/%05d/' % snapnum
     AHFile = glob( simdir + '/' + AHFdir + '*.AHF_halos')[0]
   else:
     AHFile = glob( simdir + '/' + AHFdir + '/snap' + snap_ext(snapnum) + '*.AHF_halos')[0]

   print('\nReading ' + AHFile)
   hfile = open( AHFile )
   htxt = np.loadtxt(hfile)
   hfile.close()

   if readheader:
     header = readsnap(simdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1)
     if header['k'] == -1:
        snapname = glob( simdir + '/snap*' + snap_ext(snapnum) + '*.hdf5')[0]
        snapname = snapname[ len(simdir)+1 : -9 ]
        header = readsnap(simdir,snapnum,0,cosmological=1,skip_bh=1,header_only=1,snapshot_name=snapname)

     redshift = header['redshift']
     h = header['hubble']
     print(snapnum, '  z = ', AHFile[-16:-10].split('z')[-1], header['redshift'], '  h =', h)
   else:
     redshift = float( AHFile[-16:-10].split('z')[-1] )
     print(snapnum, 'guessing  z = ', redshift, '  h =', h)

   ascale = 1. / ( 1. + redshift )
   hinv = 1./h

   halos = { 'z': redshift,
             'id': htxt[:,0].astype('int32'),
             'HostID': htxt[:,1].astype('int32'),
             'Mvir': htxt[:,3] * hinv,     # in Msun
             'Mgas': htxt[:,53] * hinv,
             'Mstar': htxt[:,73] * hinv,
             'nstar': htxt[:,72],
             'Rvir': htxt[:,11] * hinv * ascale,   # physical kpc
             'cNFW': htxt[:,42],
             'pos': np.array( [ htxt[:,5], htxt[:,6], htxt[:,7] ] ).T * hinv * ascale,     # physical kpc
             'vel': np.array( [ htxt[:,8], htxt[:,9], htxt[:,10] ] ).T,                     # peculiar velocity is already in km/s
             'p_mbp': np.array( [ htxt[:,46], htxt[:,47], htxt[:,48] ] ).T * hinv * ascale * 1e3,       # now in physical kpc
             'v_mbp': np.array( [ htxt[:,43], htxt[:,44], htxt[:,45] ] ).T
           }

   print('Done.\n')
   return halos



def medianbin( x, y, bins, value=[25,50,75] ):
   if checklen(bins) == 1:
     Nbins = bins
     Dbin = ( np.max(x) - np.min(x) ) / float(Nbins)
     XbinS = np.min(x) + Dbin * np.arange(Nbins+1)
     xbin = XbinS[0:Nbins] + 0.5 * Dbin
   else:
     Nbins = bins.size - 1
     XbinS = bins
     xbin = ( XbinS[:-1] + XbinS[1:] ) / 2.
   nval = len(value)
   ymed = np.zeros([nval,Nbins])
   NinBin = np.zeros(Nbins, dtype='int64')
   for n in range(Nbins):
      ind = np.where( ( x > XbinS[n] ) & ( x <= XbinS[n+1] ) )[0]
      NinBin[n] = ind.size
      if NinBin[n] > 0:
        ymed[:,n] = np.percentile( y[ ind ], value )
   return xbin, ymed, NinBin



# from http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def write_hdf5_from_dict( var, outfile ):
   if os.path.isfile(outfile):
     os.remove(outfile)
   file = h5py.File(outfile, 'w')
   for grname in var.keys():
      if type(var[grname]) is dict:
        group = file.create_group( grname )
        for keyname in var[grname].keys():
           group.create_dataset(keyname, data=var[grname][keyname])
      else:
        file.create_dataset(grname, data=var[grname])
   file.close()
   return




'''below some Routines for reading in GADGET/GIZMO snapshots.
@author: Phil Hopkins, Claude-Andre Faucher-Giguere, Zach Hafen, and maybe others (this file's been passed around a lot).
'''

def snap_ext(snum,four_char=0):
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        return ext;

## more robust length checker (will catch scalars)
def checklen(x):
    return len(np.array(x,ndmin=1));


def readsnap(sdir,snum,ptype, load_additional_ids=0, snapshot_name='snapshot', extension='.hdf5', h0=0,cosmological=0, skip_bh=0, four_char=0, header_only=0, loud=0):
    '''Reads in a particle snapshot.

    Args:
      sdir (str): Simulation directory to load from.
      snum (int): Snapshot to load.
      ptype (int): Particle type to load. Typically, 0=gas, 1=hi-res dark matter, 2=low-res dark matter, 4=stars.
      load_additional_ids (bool): Whether or not to also load the fields 'ParticleChildIDsNumber' and 'ParticleIDGenerationNumber'.
      snapshot_name (str): The base name for the snapshot.
      extension (str): The extension for the snapshot.
      h0 (bool): If True, use present day values.
      cosmological (bool): If True, convert from comoving to physical units, and return in physical units, with factors of h accounted for.
      skip_bh (bool): If True, don't read BH-specific fields for particles of type 5.
      four_char (bool): If True, then snapshot files have 4 digits, instead of three (e.g. 0020 instead of 020)
      header_only (bool): If True, return only the header, in stead of the full data.
      loud (bool): If True, print additional information.

    Returns:
      P (dict): A dictionary of numpy arrays containing the particle data.
    '''
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};

    #print "just past ptype check"

    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): 
      raise Exception( "Couldn't read snapshot {} in {}.\nCheck if this is the right location.".format( snum, sdir ) )
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at 
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)

    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    omega_matter = header_toparse["Omega0"]
    omega_lambda = header_toparse["OmegaLambda"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]

    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time
        hinv=1./hubble
    if (cosmological==0): 
        time*=hinv

    # CAFG: changed order of the following two lines, so that header
    # info is returned even if no particles.
    #if (header_only==1): file.close(); return {'k':0,'time':time,'hubble':hubble,'redshift':redshift};
    # DAA: return additional header info
    if (header_only==1): file.close(); return {'k':0,'time':time,'hubble':hubble,'Omega0':omega_matter,'OmegaLambda':omega_lambda,'redshift':redshift,
                                               'BoxSize':boxsize,'npartTotal':npartTotal,'massarr':massarr};
    if (header_only==2): file.close(); return header_toparse;

    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};


    # initialize variables to be read
    pos=np.zeros([npartTotal[ptype],3],dtype=float)
    vel=np.copy(pos)
    ids=np.zeros([npartTotal[ptype]],dtype=int)
    if load_additional_ids:
      child_ids=np.zeros([npartTotal[ptype]],dtype=int)
      id_gens=np.zeros([npartTotal[ptype]],dtype=int)
    mass=np.zeros([npartTotal[ptype]],dtype=float)
    if (ptype==0):
        ugas=np.copy(mass)
        rho=np.copy(mass)
        hsml=np.copy(mass) 
        if (flag_cooling>0): 
            nume=np.copy(mass)
            numh=np.copy(mass)
        if (flag_sfr>0): 
            sfr=np.copy(mass)
    if (ptype==0 or ptype==4) and (flag_metals > 0):
        metal=np.zeros([npartTotal[ptype],flag_metals],dtype=float)
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
        stellage=np.copy(mass)
    if (ptype==5) and (skip_bh==0):
        bhmass=np.copy(mass)
        bhmdot=np.copy(mass)

    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
                
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            bname = "PartType"+str(ptype)+"/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
            
        
        # now do the actual reading
        
        # CAFG: there can be an error here when particles of a certain
        # type (e.g., stars) are only in a subset of all files that
        # constitute the snapshot. Added 'if' on npart[ptype]>0.

        if npart[ptype]>0:
          nR=nL + npart[ptype]
          pos[nL:nR,:]=input_struct[bname+"Coordinates"]
          vel[nL:nR,:]=input_struct[bname+"Velocities"]
          ids[nL:nR]=input_struct[bname+"ParticleIDs"]
          if load_additional_ids:
            child_ids[nL:nR]=input_struct[bname+"ParticleChildIDsNumber"]
            id_gens[nL:nR]=input_struct[bname+"ParticleIDGenerationNumber"]
          mass[nL:nR]=massarr[ptype]
          if (massarr[ptype] <= 0.):
              mass[nL:nR]=input_struct[bname+"Masses"]
          if (ptype==0):
              ugas[nL:nR]=input_struct[bname+"InternalEnergy"]
              rho[nL:nR]=input_struct[bname+"Density"]
              hsml[nL:nR]=input_struct[bname+"SmoothingLength"]
              if (flag_cooling > 0): 
                  nume[nL:nR]=input_struct[bname+"ElectronAbundance"]
                  numh[nL:nR]=input_struct[bname+"NeutralHydrogenAbundance"]
              if (flag_sfr > 0):
                  sfr[nL:nR]=input_struct[bname+"StarFormationRate"]
          if (ptype==0 or ptype==4) and (flag_metals > 0):
              metal_t=input_struct[bname+"Metallicity"]
              if (flag_metals > 1):
                  if (metal_t.shape[0] != npart[ptype]): 
                      metal_t=np.transpose(metal_t)
              else:
                  metal_t=np.reshape(np.array(metal_t),(np.array(metal_t).size,1))
              metal[nL:nR,:]=metal_t
          if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
              stellage[nL:nR]=input_struct[bname+"StellarFormationTime"]
          if (ptype==5) and (skip_bh==0):
              bhmass[nL:nR]=input_struct[bname+"BH_Mass"]
              bhmdot[nL:nR]=input_struct[bname+"BH_Mdot"]
          nL = nR # sets it for the next iteration	

        ## correct to same ID as original gas particle for new stars, if bit-flip applied
    if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (1 << 31)

    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    boxsize *= hinv*ascale
    mass *= hinv
    vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
    if (ptype==0):
        rho *= (hinv/((ascale*hinv)**3))
        hsml *= hinv*ascale
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0) and (cosmological==0):
        stellage *= hinv
    if (ptype==5) and (skip_bh==0):
        bhmass *= hinv

    file.close();
    if (ptype==0):
       ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'u':ugas,'rho':rho,'h':hsml,'ne':nume,'nh':numh,'sfr':sfr}
       if flag_metals>0:
         ret_dict['z'] = metal
    elif (ptype==4):
        ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'age':stellage}
        if flag_metals>0:
          ret_dict['z'] = metal
    elif (ptype==5) and (skip_bh==0):
        ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'mbh':bhmass,'mdot':bhmdot}

    else:
      ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,}

    # Include optional ids
    if load_additional_ids:
      ret_dict['child_id'] = child_ids
      ret_dict['id_gen'] = id_gens

    return ret_dict

def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    '''Check if a file name exists, and if so return informatoun about it.

    Args:
      sdir (str): Simulation directory to load from.
      snum (int): Snapshot to load.
      snapshot_name (str): The base name for the snapshot.
      extension (str): The extension for the snapshot.
      four_char (bool): If True, then snapshot files have 4 digits, instead of three (e.g. 0020 instead of 020)

    Returns:
      fname_found (bool): If true, the filename was found.
      fname_base_found (str): The base for the found filename.
      fname_ext (str): The extension for the found filename.
    '''
    for extension_touse in [extension,'.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname

        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];

        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/snapdir_'+ext+'/'+snapshot_name+'_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snapdir_'+ext+'/'+'snap_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist! 
    return fname_found, fname_base_found, fname_ext;



def load_gadget_binary_header(f):
    '''Get the header information for a binary snapshot file.
  
    Args:
      f (binary file): The file to read the information from.

    Returns:
      header_info (dict): A dictionary of info contained in the header.
    '''
    ### Read header.
    import array
    # Skip 4-byte integer at beginning of header block.
    f.read(4)
    # Number of particles of each type. 6*unsigned integer.
    Npart = array.array('I')
    Npart.fromfile(f, 6)
    # Mass of each particle type. If set to 0 for a type which is present, 
    # individual particle masses from the 'mass' block are used instead.
    # 6*double.
    Massarr = array.array('d')
    Massarr.fromfile(f, 6)
    # Expansion factor (or time, if non-cosmological sims) of output. 1*double. 
    a = array.array('d')
    a.fromfile(f, 1)
    a = a[0]
    # Redshift of output. Should satisfy z=1/a-1. 1*double.
    z = array.array('d')
    z.fromfile(f, 1)
    z = float(z[0])
    # Flag for star formation. 1*int.
    FlagSfr = array.array('i')
    FlagSfr.fromfile(f, 1)
    # Flag for feedback. 1*int.
    FlagFeedback = array.array('i')
    FlagFeedback.fromfile(f, 1)
    # Total number of particles of each type in the simulation. 6*int.
    Nall = array.array('i')
    Nall.fromfile(f, 6)
    # Flag for cooling. 1*int.
    FlagCooling = array.array('i')
    FlagCooling.fromfile(f, 1)
    # Number of files in each snapshot. 1*int.
    NumFiles = array.array('i')
    NumFiles.fromfile(f, 1)
    # Box size (comoving kpc/h). 1*double.
    BoxSize = array.array('d')
    BoxSize.fromfile(f, 1)
    # Matter density at z=0 in units of the critical density. 1*double.
    Omega0 = array.array('d')
    Omega0.fromfile(f, 1)
    # Vacuum energy density at z=0 in units of the critical density. 1*double.
    OmegaLambda = array.array('d')
    OmegaLambda.fromfile(f, 1)
    # Hubble parameter h in units of 100 km s^-1 Mpc^-1. 1*double.
    h = array.array('d')
    h.fromfile(f, 1)
    h = float(h[0])
    # Creation times of stars. 1*int.
    FlagAge = array.array('i')
    FlagAge.fromfile(f, 1)
    # Flag for metallicity values. 1*int.
    FlagMetals = array.array('i')
    FlagMetals.fromfile(f, 1)

    # For simulations that use more than 2^32 particles, most significant word 
    # of 64-bit total particle numbers. Otherwise 0. 6*int.
    NallHW = array.array('i')
    NallHW.fromfile(f, 6)

    # Flag that initial conditions contain entropy instead of thermal energy
    # in the u block. 1*int.
    flag_entr_ics = array.array('i')
    flag_entr_ics.fromfile(f, 1)

    # Unused header space. Skip to particle positions.
    f.seek(4+256+4+4)

    return {'NumPart_ThisFile':Npart, 'MassTable':Massarr, 'Time':a, 'Redshift':z, \
    'Flag_Sfr':FlagSfr[0], 'Flag_Feedback':FlagFeedback[0], 'NumPart_Total':Nall, \
    'Flag_Cooling':FlagCooling[0], 'NumFilesPerSnapshot':NumFiles[0], 'BoxSize':BoxSize[0], \
    'Omega0':Omega0[0], 'OmegaLambda':OmegaLambda[0], 'HubbleParam':h, \
    'Flag_StellarAge':FlagAge[0], 'Flag_Metals':FlagMetals[0], 'Nall_HW':NallHW, \
    'Flag_EntrICs':flag_entr_ics[0]}

########################################################################

def gas_mu(num_e):
    '''Calculates the mean weight from gadget.'''

    XH=0.76; # we track this with metal species now, could do better...
    yhelium=(1.-XH)/(4.*XH); 
    return (1.+4.*yhelium)/(1.+yhelium+num_e);

def gas_temperature(u, num_e, keV=0):
    '''Returns gas particles temperature in Kelvin. Originally from PFH's gadget lib (maybe somewhere else before?).'''

    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    UnitMass_in_g= 1.989e43 # 1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53
    # note gadget units of energy/mass = 1e10 ergs/g,
    # this comes into the formula below

    mu = gas_mu(num_e);
    MeanWeight= mu*PROTONMASS
    Temp= MeanWeight/BoltzMann_ergs * g_minus_1 * u * 1.e10

    # do we want units of keV?  (0.001 factor converts from eV to keV)
    if (keV==1):
        BoltzMann_keV = 8.617e-8;
        Temp *= BoltzMann_keV;

    return Temp

## return age of universe (for a flat universe) to a given redshift
def age_of_universe(z,h=0.7,Omega_M=0.3):
    ## exact solution for a flat universe
    a=1./(1.+z); x=Omega_M/(1.-Omega_M) / (a*a*a);
    t=(2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) );
    t *= 13.777 * (0.71/h); ## in Gyr
    return t;

