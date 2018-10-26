import scipy.io, glob, numpy as np, csv, json, h5py, os
from scipy.misc import imread, imresize 
from Queue import Queue
from threading import Thread, Lock

labels =['part_of', 'same_set', 'support', 'in_front_of', 'below', 'same_plan' , 'left']

def read_span(span_file):
    return dict([(seq.replace('.seq',''),float(span)) for (seq,span) in list(csv.reader(open(span_file), delimiter= ' '))  ])

def op_l(l):
    if 'not_' in l:
        l = l.replace('not_','')
    else:
        l = 'not_' + l
    return l


def cwh_to_xy(b):
    """
    from (c1,c2,width, height) to (x1,y1,x2,y2)
    """
    (cx, cy, w, h) = b
    xy = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
    return xy

def read_rel(relations_f):
  relations={}
  for (scene, frame, o1, o2, predicate) in list(csv.reader(open(relations_f), delimiter= ' ')):
  #for fields in list(csv.reader(open(relations_f), delimiter= ' ')):
    if (scene,frame) not in relations:
      relations[(scene,frame)] = []
    relations[(scene,frame)].append((o1,o2,predicate))
  return relations

def get_pose(fn):
  cam2world = np.loadtxt(open(fn, "rb"), delimiter=" ")
  if np.logical_not(np.isfinite(cam2world)).any() or  np.isnan(cam2world).any() or cam2world.shape[0]==0:
      print 'erroneous camera value, skipping', cam2world, cam_f
      return None
  world2cam = np.linalg.inv(cam2world)
  return world2cam

def rotate_points(pts,cam):
  if len(pts.shape)==1:
    if pts.shape[0]%3!=0:
      print('the shape is not for 3d points cloud')
      return None
    npts = pts.shape[0]/3
    pts = np.transpose(np.reshape(pts, (npts,3)))
  pts = np.dot(cam,np.vstack((pts,np.ones((1,pts.shape[1])))))
  return pts[0:3,:]

def read_seq(seq_dir, imdir):
    seq2im={}
    im2roi={}
    for seq_file in glob.glob(seq_dir+'/*.seq'):
      seq_name=seq_file.split('/')[-1].split('.')[0]
      seq2im[seq_name] = set() 
      for (fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)  in list(csv.reader(open(seq_file), delimiter= ' ')):
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        scene = '_'.join(seq_name.split('_')[0:-1])
        im_path = imdir+'/'+scene + '/' + fname + '.color.jpg'
        seq2im[seq_name].add(im_path)
        if (seq_name,im_path) not in im2roi:
          im2roi[(seq_name,im_path)]=[]
        im2roi[(seq_name,im_path)].append((fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2))
    return (seq2im, im2roi)

def read_obj(obj_file):
  objs={}
  for fields in list(csv.reader(open(obj_file), delimiter= ' ')):
    scene=fields[0]
    oid = fields[1]
    objs[(scene,oid)] = np.array(fields[2:]).astype('float')
  return objs

def path2fname(p):
    return p.split('/')[-1].split('.')[0]

def seq2scene(seq_name):
    return '_'.join(seq_name.split('_')[0:-1])

def read_orientated_obj(obj_file):
  objs={}
  for fields in list(csv.reader(open(obj_file), delimiter= ' ')):
    seq_name = fields[0]
    fname = fields[1]
    oid = fields[2]
    objs[(seq_name,fname,oid)] = np.array(fields[3:]).astype('float')# * 25 + 3
  return objs

def load_intrinsics(intr_file):
    ps = {}
    for fs in  list(csv.reader(open(intr_file), delimiter= ' ')):
        ps[fs[0]] = fs[2]
    K = np.array((( float(ps['fx_color']), 0, float(ps['colorWidth'])/2 ),(0, float(ps['fy_color']), float(ps['colorHeight'])/2)))
    return K



def read_seq(seq_dir, imdir):
    seq2im={}
    im2roi={}
    for seq_file in glob.glob(seq_dir+'/*.seq'):
      seq_name=seq_file.split('/')[-1].split('.')[0]
      seq2im[seq_name] = set()
      for (fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)  in list(csv.reader(open(seq_file), delimiter= ' ')):
        scene = '_'.join(seq_name.split('_')[0:-1])
        im_path = imdir+'/'+ scene + '/' + fname + '.color.jpg'
        seq2im[seq_name].add(im_path)
        if (seq_name,im_path) not in im2roi:
          im2roi[(seq_name,im_path)]=[]
        im2roi[(seq_name,im_path)].append((fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2))
        #im2roi[(seq_name,im_path)].append((fnum,fname,'label','label','label',oid,oc,x1,y1,x2,y2))
    return (seq2im, im2roi)

def get_im2seq(seq2im,num_im,seq_names):
  im2seq = {}
  for s in range(len(seq_names)):
    seq_name = seq_names[s]
    first_im = seq2im[s]
    for i in range(num_im[s]):
      im2seq[first_im + i ] = seq_name
  return im2seq

def get_opposite_labels(label_dict):
    opposite_labels = {}
    for i in label_dict.keys():
        for j in label_dict.keys():
            if j == 'not_' + i:
                opposite_labels[i] = j
                opposite_labels[j] = i
    return opposite_labels

class Set_splitter(object):
    """
    sub-optimal division of the data into split train, val and test. I try to respect class balance in each set 
    """
    def __init__(self,prop, img_to_last_rel, img_to_first_rel, predicates, sgg_dict, counts=None):
        if counts is None:
            self.counts = self.make_counts(sgg_dict, predicates)
        else:
            self.counts = counts
        self.prop = prop
        self.img_to_last_rel = img_to_last_rel
        self.img_to_first_rel = img_to_first_rel
        self.predicates = predicates
        self.sgg_dict = sgg_dict
        self.set2idx = {"train":0, "val":1, "test":2}

    def make_counts(self, dic, predicates):
        pred_count = {}
	for i in range(len(predicates)):
	    p = predicates[i][0]
	    pred =  dic['idx_to_predicate'][str(p)]
	    if pred not in pred_count:
	        pred_count[pred] = 0
	    pred_count[pred] += 1
        return pred_count
 
    def get_sc_split(self):
        self.sc2split = {}
        self.sc2rels = self.get_sc2rels()
        scene_done = set()
        (self.target_pred_set, self.hist_pred_set) = self.init_target_hist()
        for p in sorted(self.counts, key=self.counts.__getitem__):
            subset_sc = self.get_subset(self.sc2rels,p).difference(scene_done)
            for sc in subset_sc:
                if not self.add_('train',p,sc):
                    if not self.add_('val',p,sc):
                        self.add_('test',p,sc,force=True)
                scene_done.add(sc) 
        return self.sc2split

    def add_(self,dset,p,sc, force=False):
        if self.hist_pred_set[p][dset] < self.target_pred_set[p][dset] or force:
	    self.sc2split[sc] = self.set2idx[dset] 
	    for (r,c) in self.sc2rels[sc].items():
                self.hist_pred_set[r][dset] += c
            return True
        return False
   

    def get_sc2rels(self):
        sc2rels = {}
        for i in range(len(self.prop['seq_names'])):
            if i%1000 == 0:
                print 'buiding sc2rels',i,len(self.prop['seq_names'])
            seq_name = self.prop['seq_names'][i]
            first_im_idx = self.prop['seq_to_im_idx'][i]
            num_im = self.prop['num_ims'][i]
            sc = seq2scene(seq_name)
            if sc not in sc2rels:
                sc2rels[sc]={}
            for f in range(first_im_idx,first_im_idx+num_im):
                f_r = self.img_to_first_rel[f]
                l_r = self.img_to_last_rel[f]
                for r in range(f_r,l_r+1):
                    p=self.predicates[r]
                    pred = self.sgg_dict['idx_to_predicate'][str(p[0])]
                    if pred not in sc2rels[sc]:
                        sc2rels[sc][pred] = 0
                    sc2rels[sc][pred] += 1
        for (p,i) in self.counts.items():
            #if i != sum([d[p] for (sc,d) in sc2rels.items() if p in d ]):
            #    import pdb; pdb.set_trace()
            assert(i == sum([d[p] for (sc,d) in sc2rels.items() if p in d ] ) )
        return sc2rels

    def  get_subset(self,sc2rels,p):
        return set([sc for (sc,d) in sc2rels.items() if p in d])

    def init_target_hist(self):
        ntrain =  0.7
        nval =  0.12
        ntest =  0.18
        target_pred_set, hist_pred_set = {}, {}
        for (p,i) in self.counts.items():
            if p not in target_pred_set:
                target_pred_set[p] = {}
            target_pred_set[p]['train'] = self.counts[p]*ntrain
            target_pred_set[p]['val'] = self.counts[p]*nval
            target_pred_set[p]['test'] = self.counts[p]*ntest
            if p not in hist_pred_set:
                hist_pred_set[p]={}
            hist_pred_set[p]['train'] = 0
            hist_pred_set[p]['val'] = 0
            hist_pred_set[p]['test'] = 0
        return (target_pred_set, hist_pred_set)


class Prepare_sg_seq(object):
    def __init__(self, gt_file, dict_file_template, sgg_dict_file, imdb, prop, relations, seq2im, im2roi):
        self.sgg_dict_file = sgg_dict_file
	self.sgg_dict= json.load(open(dict_file_template,"r"))
        self.imdb = imdb
        self.fi = h5py.File(gt_file, "w")
        self.im_scale = max(self.imdb['image_widths'][0],self.imdb['image_heights'][0])
        self.prop = prop
        self.relations = relations 
        (self.seq2im, self.im2roi)  = (seq2im, im2roi)
        predicates = set([ pred for ((sc,frame),preds) in self.relations.items() for (o1,o2,pred) in preds ]) # a bit ugly 
        labels = set([ vglabel for (impath, rois) in im2roi.items()  for (fnum,imname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2) in rois ])
        self.sgg_dict['idx_to_predicate'] = dict( (str(i+1),p) for (i,p) in enumerate(predicates) )
        self.sgg_dict['predicate_to_idx'] = dict( (p, int(i)) for (i,p) in self.sgg_dict['idx_to_predicate'].items()  )
        self.sgg_dict['idx_to_label'] = dict( (str(i+1),p) for (i,p) in enumerate(labels) )
        self.sgg_dict['label_to_idx'] = dict( (p, int(i)) for (i,p) in self.sgg_dict['idx_to_label'].items()  )
        for i in range(len(self.sgg_dict['idx_to_label'])+1,151): # padding label dictionnary to fit with some precomputed tensors used later by the framework
            self.sgg_dict['idx_to_label'][str(i)] = str(i)
            self.sgg_dict['label_to_idx'][str(i)] = i
        self.counts = dict( (p, 0) for (i,p) in self.sgg_dict['idx_to_predicate'].items() )
        self.opposite_p = get_opposite_labels(self.sgg_dict['predicate_to_idx'])
        self.ndet = sum( [ len(bbxs) for (key,bbxs) in self.im2roi.items()] ) 
        self.numi = len(self.prop['im_to_imdb_idx'])
        assert(len([ (scene,im_path) for ((scene,im_path),x) in self.relations.items() ]) <= self.numi ) # maybe some images do not contains relations
        self.npred=sum( [ len(set([ (scene, im_path, o1, o2) for (o1,o2,p) in x])) for ((scene,im_path),x) in self.relations.items() ] )

    def run(self):
        self.init_arrays()
        self.fill_arrays()
        print('LABEL GENERATION filling arrays done')
        print self.counts
        set_s = Set_splitter(self.prop, self.img_to_last_rel, self.img_to_first_rel, self.predicates, self.sgg_dict)
        sc2split = set_s.get_sc_split()
        print('data split, train', len([ s  for (s,d)  in sc2split.items() if d==0]), 'val', len([ s  for (s,d)  in sc2split.items() if d==1]), 'test', len([ s  for (s,d)  in sc2split.items() if d==2])  )
        self.update_split(sc2split)
        self.write_arrays()

    def update_split(self, sc2split):
        for i in range(len(self.prop['seq_names'])):
            seq_name = self.prop['seq_names'][i]
            first_im_idx = self.prop['seq_to_im_idx'][i]
            num_im = self.prop['num_ims'][i]
            sc = seq2scene(seq_name)
            for f in range(first_im_idx,first_im_idx+num_im):
                self.split[f] = sc2split[sc]

    def init_arrays(self):
        self.aom = self.fi.create_dataset(u"active_object_mask", (self.ndet,1), dtype='b1')
        self.boxes = self.fi.create_dataset(u"boxes_"+str(self.im_scale), (self.ndet,4), dtype='i4')
        self.img_to_first_box = self.fi.create_dataset(u"img_to_first_box", (self.numi,), dtype='i4')
        self.img_to_first_rel = self.fi.create_dataset(u"img_to_first_rel", (self.numi,), dtype='i4')
        self.img_to_last_box = self.fi.create_dataset(u"img_to_last_box", (self.numi,), dtype='i4')
        self.img_to_last_rel = self.fi.create_dataset(u"img_to_last_rel", (self.numi,), dtype='i4')
        self.roi_idx_to_scannet_oid = self.fi.create_dataset(u"roi_idx_to_scannet_oid", (self.ndet,1), dtype='i4')
        self.roi_idx_to_scannet_oid[:]=-1
        self.labels = self.fi.create_dataset(u"labels", (self.ndet,1), dtype='i8')
        self.predicates = self.fi.create_dataset(u"predicates", (self.npred,1), dtype='i8')
        self.relationships = self.fi.create_dataset(u"relationships", (self.npred,2), dtype='i8')
        self.predicates_all = self.fi.create_dataset(u"predicates_all", (self.npred,len(self.sgg_dict['idx_to_predicate'])+1), dtype='i8') # +1 because, making place for the background class
        self.split = self.fi.create_dataset(u"split", (self.numi,), dtype='f')
        

    def get_split0_idx(self):
        """
        by default, the first 70% of the sequences are training set
        """
        return range(int(len(self.seqs)*0.7))

    def fill_arrays_no_split(self):
        self.seqs =  self.prop['seq_names']
        self.img_to_last_box[-1]=-1 # because later code assumes img_to_first_box[i] = img_to_last_box[i-1] + 1 and I want img_to_first_box[0] = 0
        self.img_to_last_rel[-1]=-1
        for s in range(len(self.seqs)):
            i1_idx = self.prop['seq_to_im_idx'][s] # using the idx computed for the proposal, this way, I am sure they will be the same
            i_num = self.prop['num_ims'][s]
            seq_name = self.prop['seq_names'][s]
            if s%1000 == 0:
               print 'processing', s, 'sequences over', len(self.seqs)
            for i_idx in range(i1_idx,i1_idx+i_num): # for all images in this sequence
		i_imdb = self.prop['im_to_imdb_idx'][i_idx]
		im_path = self.imdb['im_paths'][i_imdb]
		scene_oid2dataset_idx = self.add_rois(im_path, s, i_idx)
		self.add_rels(im_path, seq_name, scene_oid2dataset_idx, i_idx)
        self.make_splits()
        assert(self.img_to_last_rel[-1] == self.npred - 1)
        assert(self.img_to_last_box[-1] == self.ndet - 1)

    def fill_arrays(self):
        self.seqs =  self.prop['seq_names']
        split_0 = self.seqs[self.get_split0_idx()]
        self.img_to_last_box[-1]=-1 # because later code assumes img_to_first_box[i] = img_to_last_box[i-1] + 1 and I want img_to_first_box[0] = 0
        self.img_to_last_rel[-1]=-1
        for s in range(len(self.seqs)):
            i1_idx = self.prop['seq_to_im_idx'][s] # using the idx computed for the proposal, this way, I am sure they will be the same
            i_num = self.prop['num_ims'][s]
            seq_name = self.prop['seq_names'][s]
            if s%1000 == 0:
               print 'processing', s, 'sequences over', len(self.seqs)
            for i_idx in range(i1_idx,i1_idx+i_num): # for all images in this sequence
		if seq_name in split_0:
		    self.split[i_idx] = 0
		else:
		    self.split[i_idx] = 1
		i_imdb = self.prop['im_to_imdb_idx'][i_idx]
		im_path = self.imdb['im_paths'][i_imdb]
		scene_oid2dataset_idx = self.add_rois(im_path, s, i_idx)
		self.add_rels(im_path, seq_name, scene_oid2dataset_idx, i_idx)
        assert(self.img_to_last_rel[-1] == self.npred - 1)
        assert(self.img_to_last_box[-1] == self.ndet - 1)
            
    def add_rels(self,im_path, seq_name, scene_oid2dataset_idx, i_idx):
	im_name = im_path.split('/')[-1].split('.')[0]
	rels = {}
        #if (seq_name,im_name) not in self.relations:
        #    print (seq_name,im_name)
        #    return
        for (o1,o2, p) in self.relations[(seq_name,im_name)]:
            if (o1,o2) not in rels:
                rels[(o1,o2)] = []
            rels[(o1,o2)].append(p)
	num_rel = len( rels )
	self.img_to_first_rel[i_idx] = self.img_to_last_rel[i_idx-1] + 1
	self.img_to_last_rel[i_idx] = self.img_to_first_rel[i_idx] + num_rel - 1
	for (r,((o1, o2), predicates)) in enumerate(rels.items()):
            idx_r = self.img_to_first_rel[i_idx] + r
            for p in predicates:
                #if 'not_' in p:
                #    continue # don't need those for the multi label setting
    	        self.predicates_all[idx_r,self.sgg_dict['predicate_to_idx'][p]] = 1
            self.add_less_common(predicates,idx_r)
	    if o1 not in scene_oid2dataset_idx or o2 not in scene_oid2dataset_idx:
                print 'WARNING'
	    self.relationships[idx_r]  = (scene_oid2dataset_idx[o1],scene_oid2dataset_idx[o2]) 

    def add_less_common(self, predicates, idx_r):
        """
        select one label for the relation idx_r
        This chosen label is the less common so far in self.sgg_dict
        This rule is chosen to minimise the class imbalance among labels.
        very weird coding but no time 
        """
        ok_label = dict([(p,i) for (p,i) in self.counts.items() if self.counts[p] < 5 + self.counts[self.opposite_p[p]] and p in predicates  ])
        p = self.get_less_common(ok_label)
        if "" == p:
            counts = dict([(p,i) for (p,i) in self.counts.items() if p in predicates  ])
            p = self.get_less_common(counts)
        self.predicates[idx_r] = self.sgg_dict['predicate_to_idx'][p]
        self.counts[p] += 1

    def get_less_common(self, counts):
        for p in sorted(counts, key=counts.__getitem__):
	    return p
        return ""

    def add_rois(self, im_path, s, i_idx):
        rois = self.im2roi[(self.seqs[s], im_path)]
        num_roi = len(rois)
        self.img_to_first_box[i_idx] = self.img_to_last_box[i_idx-1] + 1
        self.img_to_last_box[i_idx] = self.img_to_first_box[i_idx] + num_roi - 1
        scene_oid2dataset_idx={}
	for (k,(fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)) in enumerate(rois): # will fix the label and bbx fields for each object in this image
	    (x1,y1,x2,y2) = (float(x1),float(y1),float(x2),float(y2))
	    xc=(x1+x2)/2
	    yc=(y1+y2)/2
	    w=x2-x1
	    h=y2-y1
	    assert(w>0 and h>0)
	    idx_k = self.img_to_first_box[i_idx] + k
            assert(oid not in scene_oid2dataset_idx)
	    scene_oid2dataset_idx[oid]=idx_k
	    self.roi_idx_to_scannet_oid[idx_k] = int(oid)
	    self.boxes[idx_k,:] = (xc,yc,w,h)
	    self.labels[idx_k] = self.sgg_dict['label_to_idx'][vglabel]
        return scene_oid2dataset_idx

    def write_arrays(self):
        self.fi.flush()
        self.fi.close()
        json.dump(self.sgg_dict,open(self.sgg_dict_file,"w"))


class Prepare_seq(object):
    def __init__(self, imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir):
        self.imdb_file = imdb_file
        self.prop_file = prop_file
        self.sg_file = sg_file
        self.sg_dict_file = sg_dict_file
        self.sg_dict_template = sg_dict_template
        self.image_size = im_size
        self.num_workers = 20
        self.imdir = imdir

    def make_sg(self):
        if not hasattr(self,'prop'):
            assert(os.path.isfile(self.prop_file))
            self.prop = h5py.File(self.prop_file)
        if not hasattr(self, 'imdb'):
            assert(os.path.isfile(self.imdb_file))
            self.imdb = h5py.File(self.imdb_file)
        self.pr = Prepare_sg_seq(self.sg_file, self.sg_dict_template, self.sg_dict_file, self.imdb, self.prop, self.relations, self.seq2im, self.im2roi)
        self.pr.run()

    def makeRpn(self, overWrite=False):
        if os.path.isfile(self.prop_file) and not overWrite:
            pass
        if not hasattr(self, 'imdb'):
            assert(os.path.isfile(self.imdb_file))
            self.imdb = h5py.File(self.imdb_file)
        self.make_rpn(self.seq2im, self.im2roi, self.quadrics)
        self.write_rpn()

    def write_rpn(self):
        """
        probably not the proper way to do it
        """
        self.prop.flush() # writing 
        self.prop.close() # probably, this should not bw prop in the first place
        self.prop = h5py.File(self.prop_file) # reloading 
        print 'Wrote RPN proposals to {}'.format(self.prop_file)       

    def make_rpn(self, seq2im, im2roi, quadrics):
        seqs = seq2im.keys()
        if not hasattr(self, 'imdb'):
          self.imdb = h5py.File(self.imdb_file)
        self.prop = h5py.File(self.prop_file, "w")
        num_seq = len(seqs)
        im_paths = self.imdb['im_paths'][()]
        numi=len(im_paths)
        self.numi_dupli = len(im2roi.keys())
        num_rois_tot=sum( [ len(bbxs) for (key,bbxs) in im2roi.items()] )
        self.prop.create_dataset('seq_names', data=[ a.encode('utf8') for a in seqs])
        rpn_rois = self.prop.create_dataset(u"rpn_rois", (num_rois_tot,4), dtype='f4')
        rpn_3dbox = self.prop.create_dataset(u"quadric_rois", (num_rois_tot,27), dtype='f4')
        rpn_scores = self.prop.create_dataset(u"rpn_scores",(num_rois_tot,1), dtype='f4')
        im_to_roi_idx = self.prop.create_dataset(u"im_to_roi_idx",(self.numi_dupli,), dtype='i8')
        num_rois = self.prop.create_dataset(u"num_rois",(self.numi_dupli,), dtype='i8')
        im_scales = self.prop.create_dataset(u"im_scales",(self.numi_dupli,), dtype='f8')
        im_to_imdb_idx = self.prop.create_dataset(u"im_to_imdb_idx",(self.numi_dupli,), dtype='i8')
        seq_to_im_idx = self.prop.create_dataset(u"seq_to_im_idx",(num_seq,), dtype='i8')
        num_ims = self.prop.create_dataset(u"num_ims",(num_seq,), dtype='i8')
        rpn_scores[:]=1
        im_scales[:]=1
        for i in range(num_seq):
          if i%1000 == 0:
              print 'processing', i, 'sequences over', num_seq
          seq_to_im_idx[i] = seq_to_im_idx[i-1] + num_ims[i-1]
          seq_name = seqs[i]
          scene = '_'.join(seq_name.split('_')[0:-1])
          ims = seq2im[seq_name]
          num_ims[i] = len(ims)
          for (j,path) in enumerate(ims):
            idx_j = seq_to_im_idx[i]+j
            imdb_idx = np.where(path == im_paths)[0][0]
            im_to_imdb_idx[idx_j] = imdb_idx
            rois = im2roi[(seq_name, path)]
            num_roi = len(rois)
            im_to_roi_idx[idx_j] = im_to_roi_idx[idx_j-1] + num_rois[idx_j-1]
            num_rois[idx_j] = num_roi
            for (k,(fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)) in enumerate(rois):
              #assert(np.imag(x1) == 0)
              #assert(not isinstance(y1, complex))
              #assert(not isinstance(x2, complex))
              #assert(not isinstance(y2, complex))
              (x1,y1,x2,y2) = (float(x1),float(y1),float(x2),float(y2))
              idx_k = im_to_roi_idx[idx_j] + k
              rpn_rois[idx_k,:] = (x1,y1,x2,y2)
              rpn_3dbox[idx_k] = quadrics[(seq_name, fname, oid)]

    def imdb_from_dir(self, ext):
        self.ext = ext
        im_data = self.im_data_dir(ext)
        self.add_images(im_data)

    def im_data_dir(self, ext):
        ims=[]
        for im_path in glob.glob(self.imdir+'/*.'+ext):
            imdict={}
            imdict['image_path'] = im_path
            ims.append(imdict)
        return ims       

    def im_data_seq(self, seq_dir):
        im_data=[]
        i=0
        seen=set()
        for seq_file in glob.glob(seq_dir+'/*.seq'):
          seq_name=seq_file.split('/')[-1].split('.')[0]
          for (fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)  in list(csv.reader(open(seq_file), delimiter= ' ')):
            im={}
            im['scene'] = seq2scene(seq_name)
            path = self.imdir+'/'+im['scene'] + '/' + fname + '.color.jpg'
            if path in seen:
              continue
            seen.add(path)
            im['image_path']=path
            im_data.append(im)
            i += i
        return im_data

    def add_images(self, im_data):
        self.imdb = h5py.File(self.imdb_file, 'w')
        fns = []; ids = []; idx = []
        for i, img in enumerate(im_data):
            filename = img['image_path'] 
            if os.path.exists(filename):
                fns.append(filename)
                #ids.append(img['image_id'])
                idx.append(i)

        ids = np.array(ids, dtype=np.int32)
        idx = np.array(idx, dtype=np.int32)
        #self.imdb.create_dataset('image_ids', data=ids)
        self.imdb.create_dataset('valid_idx', data=idx)
        self.imdb.create_dataset('im_paths', data=fns)
        num_images = len(fns)
        shape = (num_images, 3, self.image_size, self.image_size)
        image_dset = self.imdb.create_dataset('images', shape, dtype=np.uint8)
        original_heights = np.zeros(num_images, dtype=np.int32)
        original_widths = np.zeros(num_images, dtype=np.int32)
        image_heights = np.zeros(num_images, dtype=np.int32)
        image_widths = np.zeros(num_images, dtype=np.int32)
        
        print "starting to process ",len(fns),"images"
        lock = Lock()
        q = Queue()
        for i, fn in enumerate(fns):
            q.put((i, fn))

        def worker():
            while True:
                i, filename = q.get()
                if i % 1000 == 0:
                    print('processing %i images...' % i)
                img = imread(filename)
                # handle grayscale
                if img.ndim == 2:
                    img = img[:, :, None][:, :, [0, 0, 0]]
                H0, W0 = img.shape[0], img.shape[1]
                img = imresize(img, float(self.image_size) / max(H0, W0))
                H, W = img.shape[0], img.shape[1]
                # swap rgb to bgr. This can't be the best way right? #fail
                r = img[:,:,0].copy()
                img[:,:,0] = img[:,:,2]
                img[:,:,2] = r

                lock.acquire()
                original_heights[i] = H0
                original_widths[i] = W0
                image_heights[i] = H
                image_widths[i] = W
                image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
                lock.release()
                q.task_done()
        for i in range(self.num_workers):
            t = Thread(target=worker)
            t.daemon = True
            t.start()

        q.join()
        self.imdb.create_dataset('image_heights', data=image_heights)
        self.imdb.create_dataset('image_widths', data=image_widths)
        self.imdb.create_dataset('original_heights', data=original_heights)
        self.imdb.create_dataset('original_widths', data=original_widths)
        self.imdb.flush()
        self.imdb.close()
        self.imdb = h5py.File(self.imdb_file)
        return fns

    def add_centres(self):
        for ((scene, fname, oid), q) in self.quadrics.items():
            centre = np.mean(np.reshape(q,(8,3)),axis=0)
            quadrics[(scene, imname, oid)] =  np.hstack( (q, centre) )

class Prepare_seq_scannet(Prepare_seq):
    def __init__(self, imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir, seqdir, relations_file, quadric_file):
        super(Prepare_seq_scannet, self).__init__(imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir)
        self.relations_file = relations_file
        self.seq_dir = seqdir
        self.quadric_file = quadric_file
        self.load_data()

    def load_data(self):
        (self.seq2im, self.im2roi)  = read_seq(self.seq_dir, self.imdir)
        self.quadrics = read_orientated_obj(self.quadric_file)
        self.relations = read_rel(self.relations_file)
 
    def imdb_from_seq(self):
        im_data = self.im_data_seq(self.seq_dir)
        self.add_images(im_data)

class Prepare_seq_synth(Prepare_seq):
    def __init__(self, imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir, matfile, build_imdb=False):
        super(Prepare_seq_synth, self).__init__(imdb_file, prop_file, sg_file, sg_dict_file, sg_dict_template, im_size, imdir)
        self.matfile = matfile
        self.ext = 'png'
        if build_imdb:
            self.imdb_from_dir(self.ext)
        self.load_data()

    def load_data(self):
        if not hasattr(self,'imdb'):
            print('loading the imdb from: '+ self.imdb_file)
            assert(os.path.isfile(self.imdb_file))
            self.imdb = h5py.File(self.imdb_file) 
        (self.seq2im, self.im2roi, self.quadrics, self.relations, self.im2case) = self.read_mat_file(self.matfile)

    def makeRpn(self, overWrite=False):
        if os.path.isfile(self.prop_file) and not overWrite:
            pass
        if not hasattr(self, 'imdb'):
            assert(os.path.isfile(self.imdb_file))
            self.imdb = h5py.File(self.imdb_file)
        self.make_rpn(self.seq2im, self.im2roi, self.quadrics)
        self.add_cases()
        self.write_rpn()

    def add_cases(self):
        im_to_case = self.prop.create_dataset(u"im_to_case",(self.numi_dupli,), dtype='i8')
        im_to_outlier = self.prop.create_dataset(u"im_to_outlier",(self.numi_dupli,), dtype='i8')
        for i in range(self.numi_dupli):
            imdb_idx = self.prop['im_to_imdb_idx'][i]
            im_path = self.imdb['im_paths'][imdb_idx]
            im_name = im_path.split('/')[-1].split('.')[0]
            case, outlier = self.im2case[im_name]
            im_to_case[i] = case
            im_to_outlier[i] = outlier

    def read_mat_file(self, matfile):
        mat = scipy.io.loadmat(matfile)
        seq2im = {}
        im2roi = {}
        relations = {}
        quadrics = {}
        im2case = {}
        for i in range(len(mat['simuls'][0])):
            field_names=mat['simuls'][0][i][0][0].dtype.names
            obj_labels = mat['simuls'][0][i][0][0][field_names.index('obj_label')]
            conics=mat['simuls'][0][i][0][0][field_names.index('Cgt')]
            n_f=conics.shape[0]/3
            bbx3d=mat['simuls'][0][i][0][0][field_names.index('bbx3d_or')]
            extrinsic=mat['simuls'][0][i][0][0][field_names.index('M')]
            bbx2d=mat['simuls'][0][i][0][0][field_names.index('bbx2d')]
            names = mat['simuls'][0][i][0][0][field_names.index('imnames')]
            case = mat['simuls'][0][i][0][0][field_names.index('case')][0][0]
            if 'outlier' not in field_names:
                outlier = -1
            else:
                outlier = mat['simuls'][0][i][0][0][field_names.index('outlier')][0][0]
            seq_name = '_'.join(names[0,0][0][0][0][0].split('_')[0:2]).encode('utf8')
            scene = seq_name
            seq_name = seq_name + '_0'
            seq2im[seq_name] = []
            for im in range(names.shape[1]):
                imname = names[0,im][0][0][0][0].encode('utf8')
                imnum = int(imname.split('_')[-1])
                if imnum == outlier:
                    im2case[imname] = (case,outlier)
                else:
                    im2case[imname] = (case,-1)
                fnum = int(imname.split('_')[-1])               
                impath = os.path.join(self.imdir, imname) + '.' + self.ext
                seq2im[seq_name].append(impath.encode('utf8'))
                box2d = bbx2d[im,:,:]
                rels = dict([(n,mat['simuls'][0][i][0][0][field_names.index(n)][:,:,im]) for n in field_names if 'rels_' in n])
                colors = mat['simuls'][0][i][0][0][field_names.index('colors')][im,:]
                no = colors.shape[0]
                im2roi[(seq_name, impath)] = []
                #adding object
                oidx2oid = {}
                for o in range(no):
                    if obj_labels[0][o] == 1:
                        olabel = 'left'
                    else:
                        olabel = 'right'
                    oid = o
                    oidx2oid[o]=oid
                    shapelabel = olabel
                    vglabel = olabel
                    oc = 0
                    (x1, y1, x2, y2) = self.get_bbx(box2d[o])
                    im2roi[(seq_name, impath)].append((fnum,imname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2))
                    centre = np.mean(np.reshape(bbx3d[im,o,:],(8,3)),axis=0)
                    quadrics[(seq_name, imname, oid)] =  np.hstack( (bbx3d[im,o,:], centre) )
                #adding relations
                for o1 in range(no):
                    for o2 in range(no):
                        if o1 ==  o2:
                            continue
                        for (pred, values) in rels.items():
                            if (seq_name, imname) not in relations:
                                relations[(seq_name, imname)] = []
                            if values[o1,o2] == 1:
                                relations[(seq_name, imname)].append((oidx2oid[o1],oidx2oid[o2],pred))
                            else:
                                if values[o1,o2] == -1:
                                    relations[(seq_name, imname)].append((oidx2oid[o1],oidx2oid[o2],'not_'+pred))
        return (seq2im, im2roi, quadrics, relations, im2case)

    def get_data_from_mat(matfile):
        mat = scipy.io.loadmat(matfile)
        simuls = []
        for i in range(len(mat['simuls'][0])):
            simul = {}
            field_names=mat['simuls'][0][i][0][0].dtype.names
            for field in field_names:
                simul[field] = mat['simuls'][0][i][0][0][field_names.index(field)]
            seq_name = '_'.join(names[0,0][0][0][0][0].split('_')[0:2]).encode('utf8')
            simul['seq_name'] = seq_name
            n_f=conics.shape[0]/3
        return simuls

    def get_bbx(self, bbx2d):
        x1=max(min(bbx2d[0:bbx2d.shape[0]:2]),1)
        y1=max(min(bbx2d[1:bbx2d.shape[0]:2]),1)
        x2=min(max(bbx2d[0:bbx2d.shape[0]:2]),self.imdb['image_widths'][0])
        y2=min(max(bbx2d[1:bbx2d.shape[0]:2]),self.imdb['image_heights'][0])        
        return (x1, y1, x2, y2)
"""
import h5py, json
sgg = h5py.File("/ssd_disk/gay/scenegraph/gt_quadrics/scannet-SGG.h5","r")
prop = h5py.File('/ssd_disk/gay/scenegraph/gt_quadrics/proposals.h5')
dic = json.load(open('/ssd_disk/gay/scenegraph/gt_quadrics/scannet-SGG-dicts.json'))
print 'starting'
set_s = Set_splitter(prop, sgg['img_to_last_rel'], sgg['img_to_first_rel'], sgg['predicates'], dic)
print 'running'
sc2split = set_s.get_sc_split()
"""

