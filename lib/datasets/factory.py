from datasets.vg_hdf5 import vg_hdf5
from datasets.vg_hdf5_multi import vg_hdf5_multi

def get_imdb(roidb_name, imdb_name, rpndb_name, data_dir, split=-1, num_im=-1):
    return vg_hdf5('%s.h5'%roidb_name, '%s-dicts.json'%roidb_name, imdb_name, rpndb_name, data_dir, split=split, num_im=num_im)

def get_imdb_multi(roidb_name, imdb_name, rpndb_name, data_dir, split=-1, num_im=-1):
    return vg_hdf5_multi('%s.h5'%roidb_name, '%s-dicts.json'%roidb_name, imdb_name, rpndb_name, data_dir, split=split, num_im=num_im)
