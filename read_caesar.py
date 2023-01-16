import numpy as np 
import h5py

#from daniel 
def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()] #item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

if __name__=='__main__':
    boxsize=50
    Np=512
    z=0.0
    snapnum=135
    outfilename='halos_m{}n{}_z{}.txt'.format(boxsize,Np,z)
    caesarfile='m{}n{}_{}.hdf5'.format(boxsize,Np,snapnum)
    f=open(outfilename,mode='w')
    f.write("#GalID M_star  M_HI   M_H2 sfr central\n")
    gr = load_dict_from_hdf5(caesarfile) #this contains all the caesar info

    for i in range(gr['galaxy_data']['GroupID'].size):
        gid=gr['galaxy_data']['GroupID'][i]
        mstar=gr['galaxy_data']['dicts']['masses.stellar'][i]
        mHI=gr['galaxy_data']['dicts']['masses.HI'][i] 
        mH2=gr['galaxy_data']['dicts']['masses.H2'][i] 
        sfr=gr['galaxy_data']['sfr'][i]
        cent=gr['galaxy_data']['central'][i]
        f.write("{:5d} {:4.2e} {:4.2e} {:4.2e} {:4.2e} {:1d}\n"
        .format(gid,mstar,mHI,mH2,sfr,cent))

    f.close()

