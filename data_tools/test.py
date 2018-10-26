import h5py, json, numpy as np
from prettytable import PrettyTable
#datadir = '/ssd_disk/gay/scenegraph/synth_fus/'
#im_size = '875'

datadir = '//gay/scenegraph/gt_quadrics_final/'
im_size = '1296'
#dic = json.load(open(datadir + '/synth-SGG-dicts.json'))
#sgg = h5py.File(datadir + "/synth-SGG.h5","r")
dic = json.load(open(datadir + '/scannet-SGG-dicts.json'))
sgg = h5py.File(datadir + "/scannet-SGG.h5","r")
prop = h5py.File(datadir + '/proposals.h5')
imdb = h5py.File(datadir + '/imdb_'+im_size+'.h5')

num_im = len(sgg['img_to_first_box'])
i = 10
seq_name = prop['seq_names'][i]
im_idx = prop['seq_to_im_idx'][i]

print '----------------------displaying the quadric-----------------------------------------------'
imdb_idx = prop['im_to_imdb_idx'][im_idx]
im_path = imdb['im_paths'][imdb_idx]
q_idx = prop['im_to_roi_idx'][im_idx]
quadric = prop['quadric_rois'][q_idx]
print 'sequence', seq_name,  im_path, 'oid', sgg['roi_idx_to_scannet_oid'][q_idx]
print quadric 

fst_roi = sgg['img_to_first_box'][im_idx]
print '-----------------correspondance between the proposals and the gt -----------------------'
print 'gt, first and last roi:', fst_roi, sgg['img_to_last_box'][im_idx]
fst_roi_prop = prop['im_to_roi_idx'][im_idx]
print 'proposal, first and num of roi:', prop['im_to_roi_idx'][im_idx], prop['num_rois'][im_idx]
(cx, cy, w, h) = sgg['boxes_'+im_size][fst_roi]
print 'gt first roi:', cx-w/2, cy-h/2, cx+w/2, cy+h/2
print 'proposal, first roi:', prop['rpn_rois'][fst_roi_prop]
imdb_idx =  prop['im_to_imdb_idx'][im_idx]
print seq_name, imdb['im_paths'][imdb_idx]

print '---------------------checking the gt relations are correctly read --------------------------------'
for r in range(fst_roi,sgg['img_to_last_box'][im_idx]+1):
  print 'labels of the object :',r , dic['idx_to_label'][str(sgg['labels'][r][0])]

print('multi label array')
for r in range(sgg['img_to_first_rel'][im_idx],sgg['img_to_last_rel'][im_idx]+1):
  idx_ps = np.where(sgg['predicates_all'][r])[0]
  print r, sgg['roi_idx_to_scannet_oid'][sgg['relationships'][r][0]], 'is ', sgg['relationships'][r], [ dic['idx_to_predicate'][str(i)] for i in idx_ps ], 'of', sgg['roi_idx_to_scannet_oid'][sgg['relationships'][r][1]] 
print('mono label array')
for r in range(sgg['img_to_first_rel'][im_idx],sgg['img_to_last_rel'][im_idx]+1):
  idx_p = sgg['predicates'][r,0]
  print r, sgg['roi_idx_to_scannet_oid'][sgg['relationships'][r][0]], 'is ', sgg['relationships'][r], dic['idx_to_predicate'][str(idx_p)], 'of', sgg['roi_idx_to_scannet_oid'][sgg['relationships'][r][1]] 



print '---------------------getting some statistics on the dataset---------------------------'



pred_c_0 = {}
pred_c_1 = {}
pred_c_2 = {}
split_r = np.zeros((len(sgg['predicates'])))
for i in range(len(sgg['img_to_first_rel'])):
    split_r[sgg['img_to_first_rel'][i]:sgg['img_to_last_rel'][i]+1] = sgg['split'][i]
for (p,i) in dic['predicate_to_idx'].items():
    idx_p = np.where(sgg['predicates'][()]==i)[0]
    pred_c_0[p] = sum(np.logical_and(sgg['predicates'][()]==i,split_r==0))
    pred_c_1[p] = sum(np.logical_and(sgg['predicates'][()]==i,split_r==1))
    pred_c_2[p] = sum(np.logical_and(sgg['predicates'][()]==i,split_r==2))


kk = dic['predicate_to_idx'].keys()
t = PrettyTable( ['split'] + kk )
t.add_row( ['0'] + [ pred_c_0[p]  for p in kk] )
t.add_row( ['1'] + [ pred_c_1[p]  for p in kk] )
t.add_row( ['2'] + [ pred_c_2[p]  for p in kk] )
print t

pred_count = {}
for i in range(len(sgg['predicates'])):
  p = sgg['predicates'][i][0]
  pred =  dic['idx_to_predicate'][str(p)]
  if pred not in pred_count:
    pred_count[pred] = 0
  pred_count[pred] += 1
print 'for the predicates'
t = PrettyTable( kk + ['all'])
t.add_row( [ pred_count[p]  for p in kk] + [len(sgg['predicates'])]) 
print t
print pred_count

num_roi = len(sgg['roi_idx_to_scannet_oid'])
num_rel = len(sgg['predicates_all'])
pred_count_all = {}
for i in range(len(sgg['predicates_all'])):
  idx_ps = np.where(sgg['predicates_all'][i])[0]
  for p in idx_ps:
    pred =  dic['idx_to_predicate'][str(p)]
    if pred not in pred_count_all:
      pred_count_all[pred] = 0
    pred_count_all[pred] += 1
    if split
print 'for the predicates_all',num_roi,num_rel, pred_count_all

obj_count = {}
for i in range(len(sgg['labels'])):
  olab = dic['idx_to_label'][str(sgg['labels'][i][0])]
  if olab not in obj_count:
    obj_count[olab] = 0
  obj_count[olab] += 1
print obj_count
