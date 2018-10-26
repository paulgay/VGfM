import sys
sys.path.append('../')
import utils_prep

db_dir = '/ssd_disk/gay/scenegraph/pers_quadrics/' # The output directory where the files will be written
im_size = 1296 # largest image size
imdir = '/ssd_disk/datasets/scannet/data/' # Scannet dataset directory

imdb_file = db_dir + 'imdb_' + str(im_size) + '.h5'
prop_file = db_dir + 'proposals.h5'
sg_file = db_dir + 'scannet-SGG.h5'
sg_dict_file = db_dir + 'scannet-SGG-dicts.json'
sg_dict_template = '/ssd_disk/gay/scenegraph/scannet-SGG-dicts.json'
#quadric_file = '/ssd_disk/gay/scenegraph/sparse_seqs/gt_bbx.txt'
quadric_file = '/ssd_disk/datasets/scannet/material_for_scene_graph/data/lfdc_bbx.txt' # contains the csv files with the quadric 3D bbox
seqdir = '/ssd_disk/datasets/scannet/material_for_scene_graph/data/seqdir/' # contains the detection for each sequence
relations_file = '/ssd_disk/datasets/scannet/material_for_scene_graph/data/spatial_sem_relations.txt' # contains the relations between the objects, this one contains spatial and semantic
pr = utils_prep.Prepare_seq_scannet(imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir, seqdir, relations_file, quadric_file) #, build_imdb=True) 
# building the imdb file
pr.imdb_from_seq()
# building the proposal.h5
pr.makeRpn()
# building the scannet-SGG.h5 and the scannet-SGG-dicts.json
pr.make_sg()
