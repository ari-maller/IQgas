#This script reads a caesar galaxy/halo file and outputs the data to an ascii file
#updated caesar output can be found in Data/simba
 
import numpy as np 
import h5py 
import pandas as pd
import argparse
import caesar
try:
    import caesar
except:
    print('Could not load ceaser need to use hdf5 file hack')

#ceaserfilename="m50n512_135.hdf5"
#first if ceaser works 
def read_caesar(caesarfilename):
    groups=caesar.load(caesarfilename)
    Ngal=len(groups.galaxies)
    idx=np.zeros(Ngal);cent=np.zeros(Ngal);sfr=np.zeros(Ngal);rdisk=np.zeros(Ngal)
    mHI=np.zeros(Ngal);mH2=np.zeros(Ngal);Mstar=np.zeros(Ngal);Mhalo=np.zeros(Ngal)
    for i,galaxy in enumerate(groups.galaxies):
        idx[i] = galaxy.GroupID
        cent[i] = (galaxy.central).astype(int)
        sfr[i] = galaxy.sfr
        rdisk[i]=galaxy.radii['stellar_half_mass']
        Mstar[i] = galaxy.masses['stellar_30kpc']
        mHI[i] = galaxy.masses['HI_30kpc']
        mH2[i] = galaxy.masses['H2_30kpc'] 
        Mhalo[i]=galaxy.halo.masses['total']
    
    df=pd.DataFrame({'GID':idx,'central':cent,'Mstar':Mstar,'sfr':sfr,'mHI':mHI,'mH2':mH2,
        'r_disk':rdisk,'Mhalo':Mhalo})
    return df

#if caesar doesn't work
# the next two routines are taken from 
# http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
# then I got it from Daniel

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

    
def read_nocaesar(caesarfilename):
    groups=load_dict_from_hdf5(caesarfilename)
    idx = groups['galaxy_data']['GroupID']
    cent = (groups['galaxy_data']['central']).astype(int)
    sfr = groups['galaxy_data']['sfr']
    Mstar = groups['galaxy_data']['dicts']['masses.stellar']
    mHI = groups['galaxy_data']['dicts']['masses.HI']
    mH2 = groups['galaxy_data']['dicts']['masses.H2']
    rdisk=groups['galaxy_data']['dicts']['radii.stellar_half_mass']
    parent_id=groups['galaxy_data']['parent_halo_index']
    Mhalo=groups['halo_data']['dicts']['masses.total'][parent_id]
    # sfr 0.0 was mysteriously set to 1.0 in mufasa
    epsilon=1.e-6
    zero=np.logical_and(sfr > 1.0-epsilon,sfr < 1.0+epsilon)
    sfr[zero]=0.0
    print(f"changed {zero.sum()} sfr 1.0s to 0.0")
    df=pd.DataFrame({'GID':idx,'central':cent,'Mstar':Mstar,'sfr':sfr,'mHI':mHI,'mH2':mH2,
        'r_disk':rdisk,'Mhalo':Mhalo})
    return df

#writing and parsing
def write_sim(data,name='tmp.dat'):
    f=open(name,'w')
    f.write("# Group central  sfr_inst     Mstar        MHI         MH2      R_disk   M_halo\n")
    for i in range(data.shape[0]):
        f.write("    {}       {}   {:8.4f}    {:6.4e}   {:6.4e}   {:6.4e}   {:6.4f}  {:6.4e}\n".format(
            data['GID'][i],data['central'][i],data['sfr'][i],data['Mstar'][i],data['mHI'][i],
            data['mH2'][i],data['r_disk'][i],data['Mhalo'][i]))

    f.close()

def parse(name):
    #mufasa names use box length, cube root of particle number and snap_num 
    mloc=name.find('m')
    nloc=name.find('n')
    sloc=name.find('_')
    boxsize=float(name[mloc+1:nloc])
    Nparticles=int(name[nloc+1:sloc])
    snap_num=int(name[sloc+1:sloc+4])
    return boxsize,Nparticles,snap_num

def main(args):
    caesarfilename=args.caesar
    start=caesarfilename.rfind('/')
    if start==-1:
        filename=caesarfilename
    else:
        filename=caesarfilename[start+1:-1]
    L,N,snap_num=parse(filename)
    if args.out is None:
        outname='tmp.dat'
    else:
        outname=args.out
    if L==100: #simba 
        df=read_caesar(caesarfilename)
    else:     #mufasa 
        df=read_nocaesar(caesarfilename)

    write_sim(df,name=outname)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Do Something.")
    parser.add_argument("--caesar",action='store'
        ,help="The caesar file to be converted to ascii")
    parser.add_argument("--out",action='store',default=None
        ,help="The file name to output to in ascii")   
    args=parser.parse_args()
    main(args)
    