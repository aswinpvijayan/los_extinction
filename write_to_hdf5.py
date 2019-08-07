import numpy as np
import h5py 

class HDF5_write(object):
    """
    Simple class to append value to a hdf5 file 
    
    Params:
        datapath: filepath of h5 file
        group: group name within the file
        dataset: dataset name within the file or group
        value: dataset value, used to create and append to the dataset
        dtype: numpy dtype
    
    Usage:
        shape = (20,3,)
        hdf5_store = HDF5Store('hdf5_store.hdf5')
        hdf5_store.create_grp('Stars')
        hdf5_store.create_dset(np.random.random(shape), 'X', 'Stars')
        for _ in range(10):
            hdf5_store.append(np.random.random(shape), 'X', 'Stars')
        
    Modified from https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, compression="gzip"):
        self.datapath = datapath
        self.compression = compression
        
    def create_grp(self, group):
        with h5py.File(self.datapath, mode='a') as h5f:
            if group not in list(h5f.keys()): 
                h5f.create_group(group)    
            else: 
                print("`{}` group already created".format(group))
            
    
    def create_dset(self, values, dataset, group = 'None', dtype=np.float32):
        with h5py.File(self.datapath, mode='a') as h5f:
            shape = np.shape(values)
            if group == 'None':
                dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape[1:],
                maxshape=(None, ) + shape[1:],
                dtype=dtype,
                compression=self.compression)
            else:
                try:
                    dset = h5f[group].create_dataset(
                    dataset,
                    shape=(0,) + shape[1:],
                    maxshape=(None,) + shape[1:],
                    dtype=dtype,
                    compression=self.compression)
                except:
                    print("Oh! Oh! something went wrong while creating {}/{} or it already exists.\nNo value was written into the dataset.".format(group, dataset))
                    
        self.append(values, dataset, group)
    
    def append(self, values, dataset, group = 'None'):
        with h5py.File(self.datapath, mode='a') as h5f:
            if group == 'None':
                dset = h5f[dataset]
            else:
                dset = h5f["{}/{}".format(group, dataset)]
            ini = len(dset)
            add = len(values)
            dset.resize(ini+add, axis = 0)
            dset[ini:] = values
            h5f.flush()


if __name__== "__main__": 

    shape = (20,3,)
    hdf5_store = HDF5Store('hdf5_store.hdf5')
    hdf5_store.create_grp('Stars')
    hdf5_store.create_dset(np.random.random(shape), 'X', 'Stars')
    for _ in range(10):
        hdf5_store.append(np.random.random(shape), 'X', 'Stars')

