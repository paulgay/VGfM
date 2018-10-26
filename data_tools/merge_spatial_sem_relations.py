import sys, numpy as np, pickle, re
import utils_prep

annot='/ssd_disk/gay/scenegraph/gt_quadrics/semantic_relations_17_01.pc'
seq_dir = '/ssd_disk/gay/scenegraph/sparse_seqs/'
datadir = '/ssd_disk/datasets/scannet/data/'
spatial_rel = '/ssd_disk/gay/scenegraph/sparse_seqs/spatial_relations.txt'

fout = open('/ssd_disk/gay/scenegraph/sparse_seqs/spatial_sem_relations.txt','w')

# reading the filesi and making dictionnaries
s_rels = utils_prep.read_rel(spatial_rel)
o2rels = pickle.load(open(annot,'rb'))

allps = set()
o2rels_new = {}
for ((sc,o1,o2),ps) in o2rels.items():
  pss = set()
  if (sc,o2,o1) not in o2rels_new:
    o2rels_new[(sc,o2,o1)] = set()
  ipss = o2rels_new[(sc,o2,o1)]
  if sc == 'fake_scene' or sc == 'bad_object':
    continue
  if 'attached' in ps or 'part_of' in ps or 'inside' in ps:
    pss.add('part_of')
    pss.add('same_set')
  if 'support_behind' in ps  or 'support_below' in ps or 'support_hidden' in ps:
    pss.add('support')
    pss.add('same_set')
  if 'same_plan' in ps:
    pss.add('same_plan')
    ipss.add('same_plan') 
  if 'same_set' in ps:
    pss.add('same_set')
    ipss.add('same_set')
  allps.update(pss)
  o2rels_new[(sc,o1,o2)] = pss
  o2rels_new[(sc,o2,o1)] = ipss

o2rels = {}
for  ((sc,o1,o2),ps) in o2rels_new.items():
  if sc == 'fake_scene' or sc == 'bad_object':
    continue
  for p in allps:
    if p not in ps:
      ps.add('not_'+p)
  o2rels[(sc,o1,o2)] = ps
#import pdb;pdb.set_trace() 'scene0180_00_7', 'frame-000346')
(seq2im, im2roi)  = utils_prep.read_seq(seq_dir, datadir)
sc2seq = {} # o2rels is indexed with scene, and the file contains relations by sequence, so building a map from scene to seq
for seq_name in seq2im.keys():
  scene = '_'.join(seq_name.split('_')[0:-1])
  if scene not in sc2seq:
    sc2seq[scene]=[]
  sc2seq[scene].append(seq_name)

seq2oid = {}
for (k,dets) in im2roi.items():
  seq2oid[k[0]] =  [ l[5] for l in dets]

for ((sc,o1,o2),ps) in o2rels.items():
  if sc == 'fake_scene' or sc == 'bad_object' or sc not in sc2seq:
    continue
  for seq_name in sc2seq[sc]:# for all sequences built on this room
    if o1 not in seq2oid[seq_name] or o2 not in seq2oid[seq_name]: # check that both objects are present
      continue
    for impath in seq2im[seq_name]:
      fname = impath.split('/')[-1].split('.')[0] 
      for predicate in ps:
        if predicate == 'None' or predicate == 'not known':
          continue
        fout.write(' '.join((seq_name,fname,o1,o2,predicate))+'\n')
# appending the spatial relations
for ((seq_name,frame), rels ) in s_rels.items(): # (o1,o2,predicate)
  assert(seq_name in seq2im)
  for (o1,o2,p) in rels: 
      fout.write(' '.join((seq_name,frame,o1,o2,p))+'\n')
fout.close()
