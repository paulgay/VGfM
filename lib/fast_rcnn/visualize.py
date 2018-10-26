# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""
Visualize a generated scene graph
"""

from fast_rcnn.config import cfg
from roi_data_layer.roidb import prepare_roidb
from roi_data_layer.layer_vid import RoIDataLayerVid
from fast_rcnn.test import im_detect, gt_rois, non_gt_rois, im_detect_vid
from datasets.viz import viz_scene_graph, draw_scene_graph
from datasets.eval_utils import ground_predictions
from networks.factory import get_network
import numpy as np
import tensorflow as tf
from utils.cpu_nms import cpu_nms
import matplotlib.pyplot as plt

import json, pickle
global idx
idx=0

def my_draw_graph_pred(im, boxes, cls_score, rel_score, gt_to_pred, roidb, im_num):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: predicted boxes
        cls_score: object classification scores
        rel_score: relation classification scores
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        roidb: roidb
    """
    gt_relations = roidb['gt_relations']
    im = im[:, :, (2, 1, 0)].copy()
    cls_pred = np.argmax(cls_score, 1)
    rel_pred_mat = np.argmax(rel_score, 2)
    rel_pred_scores = np.max(rel_score,2)
    sorteds=np.argsort(-rel_pred_scores,axis=None)
    indexes=np.unravel_index(np.argsort(-rel_pred_scores,axis=None),rel_pred_scores.shape)
    rel_pred = []
    all_rels = []
    import cv2
    good_det=np.argsort(-np.max(cls_score,1),axis=None)
    im2= im #.copy()[:, :, (2, 1, 0)].astype(np.uint8) #im #cv2.imread('/home/gay/ext_lib/scene-graph-TF-release/data/scannet/images/1.jpg')w
    for n in range(min(len(cls_pred),10)):
      lidx=cls_pred[good_det[n]]
      b=boxes[good_det[n],:]
      cv2.rectangle(im2,(int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (255, 0, 0), 5)
      cv2.rectangle(im2,(int(b[4]),int(b[5])), (int(b[6]),int(b[7])), (255, 0, 0), 5)
      cv2.rectangle(im2,(int(b[8]),int(b[9])), (int(b[10]),int(b[11])), (255, 0, 0), 5)
      cv2.rectangle(im2,(int(b[12]),int(b[13])), (int(b[14]),int(b[15])), (255, 0, 0), 5)
      cv2.rectangle(im2,(int(b[16]),int(b[17])), (int(b[18]),int(b[19])), (255, 0, 0), 5)
      l=cls_pred[n]
      cv2.putText(im2, cfg.ind_to_class[lidx],(int(b[0])+40,int(b[1])+40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2 ) 
    #cv2.imshow('inage',im)
    cv2.imwrite('visual'+str(im_num)+'.png',im2)
    #cv2.waitKey()
    for n in range(min(len(indexes[0]),7)):
      i=indexes[0][n]
      j=indexes[1][n]
      #if [i,j] in all_rels:
      #  continue
      rel_pred.append([i, j, rel_pred_mat[i,j], 1])
      all_rels.append([i, j])
    rel_pred = np.array(rel_pred)
    if rel_pred.size == 0:
        return
    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()
    # draw graph predictions
    #graph_dict = draw_scene_graph(cls_pred, pred_inds, rel_pred)
    viz_scene_graph(im, boxes, cls_pred, pred_inds, rel_pred, preprocess=False, save_f='graph'+str(im_num)+'.png')

def draw_graph_pred(im, boxes, cls_score, rel_score, gt_to_pred, roidb,idx):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: prediceted boxes
        cls_score: object classification scores
        rel_score: relation classification scores
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        roidb: roidb
    """
    gt_relations = roidb['gt_relations']
    im = im[:, :, (2, 1, 0)].copy()
    cls_pred = np.argmax(cls_score, 1)
    rel_pred_mat = np.argmax(rel_score, 2)
    rel_pred = []
    all_rels = []
    for i in xrange(rel_pred_mat.shape[0]):
        for j in xrange(rel_pred_mat.shape[1]):
            # find graph predictions (nodes and edges) that have
            # correspounding ground truth annotations
            # ignore nodes that have no edge connections
            for rel in gt_relations:
                if rel[0] not in gt_to_pred or rel[1] not in gt_to_pred:
                    continue
                # discard duplicate grounding
                if [i, j] in all_rels:
                    continue
                if i == gt_to_pred[rel[0]] and j == gt_to_pred[rel[1]]:
                    rel_pred.append([i, j, rel_pred_mat[i,j], 1])
                    all_rels.append([i, j])

    rel_pred = np.array(rel_pred)
    if rel_pred.size == 0:
        return

    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()

    # draw graph predictions
    graph_dict = draw_scene_graph(cls_pred, pred_inds, rel_pred)
    viz_scene_graph(im, boxes, cls_pred, pred_inds, rel_pred, preprocess=False)
    
    out_boxes = []
    for box, cls in zip(boxes[pred_inds], cls_pred[pred_inds]):
        out_boxes.append(box[cls*4:(cls+1)*4].tolist())

    graph_dict['boxes'] = out_boxes
    do_save='y'
    if do_save == 'y':
        scipy.misc.imsave('cherry/im_%i.png' % idx, im)
        fn = open('cherry/graph_%i.json' % idx, 'w+')
        json.dump(graph_dict, fn)
    print(idx)
    
def split_entry_by_img(out_dict, blob_seq,sg_entries):
  num_img = blob_seq['db_idx'].shape[0]
  sg_entries = []
  for i in range(num_img):
    obj_this_im = np.where(blob_seq['rois'][:,0]==i)[0]
    sg_entry = {}
    sg_entry['scores'] = out_dict['scores'][obj_this_im]
    sg_entry['relations'] = out_dict['relations'][obj_this_im][:,obj_this_im]
    sg_entry['boxes'] = out_dict['boxes'][obj_this_im]
    sg_entries.append(sg_entry)
  return sg_entries

def viz_net_vid(net_name, weight_name, imdb, dump_file, viz_mode='viz_cls'):
    sess = tf.Session()
    # set up testing mode
    rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    rel_rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    ims = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='ims')
    relations = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='relations')

    inputs = {'rois': rois,
              'rel_rois': rel_rois,
              'ims': ims,
              'relations': relations,
              'num_roi': tf.placeholder(name='num_roi', dtype=tf.int32, shape=[]),
              'num_rel': tf.placeholder(name='num_rel', dtype=tf.int32, shape=[]),
              'num_classes': imdb.num_classes,
              'num_predicates': imdb.num_predicates,
              'rel_mask_inds': tf.placeholder(name='rel_mask_inds', dtype=tf.int32, shape=[None]),
              'rel_segment_inds': tf.placeholder(name='rel_segment_inds', dtype=tf.int32, shape=[None]),
              'rel_pair_mask_inds': tf.placeholder(name='rel_pair_mask_inds', dtype=tf.int32, shape=[None, 2]),
              'rel_pair_segment_inds': tf.placeholder(name='rel_pair_segment_inds', dtype=tf.int32, shape=[None]),
              'quadric_rois':  tf.placeholder(name='quadric_rois',dtype=tf.float32, shape=[None, 28]),
              'n_iter': cfg.TEST.INFERENCE_ITER,
              'obj_fus_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'obj_fus_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_fus_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_fus_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              }
    bbox_dist = np.load(cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE).item()
    bbox_means = bbox_dist['means']
    bbox_stds = bbox_dist['stds']
    data_layer = RoIDataLayerVid(imdb, bbox_means, bbox_stds)
    test_seqs = imdb.seq2im_mask.keys()
    net = get_network(net_name)(inputs)
    net.setup()
    print ('Loading model weights from {:s}').format(weight_name)
    saver = tf.train.Saver()
    saver.restore(sess, weight_name)
    roidb = imdb.roidb
    prepare_roidb(roidb)
    num_images = len(imdb.image_index)
    if net.iterable:
        inference_iter = net.n_iter - 1
    else:
        inference_iter = 0
    print('=======================VIZ INFERENCE Iteration = '),
    print(inference_iter)
    print('=======================VIZ MODES = '),
    print(viz_mode)
    sg_entries=[]
    for s_i in range(min(300,len(test_seqs))): #,10):
        print('processing sequence ' +test_seqs[s_i]+' '+str(s_i)+'/'+str(len(test_seqs)))
        blob_seq = data_layer._get_next_minibatch_test(s_i)
        out_dict = im_detect_vid(sess, net, inputs, blob_seq, [inference_iter]) #, box_proposals, bbox_reg, [inference_iter], quadric_rois )
        entry_by_img = split_entry_by_img(out_dict[inference_iter], blob_seq,sg_entries)
        for (i, sg_entry) in enumerate(entry_by_img):
            roidb_idx = blob_seq['db_idx'][i] 
            gt_to_pred = ground_predictions(sg_entry, roidb[roidb_idx], 0.5)
            roidb2 = dict(roidb[roidb_idx])
            roidb2['gt_to_pred_object'] = gt_to_pred
            del roidb2['image']
            sg_entries.append((sg_entry, roidb2))
    print 'dump_file',dump_file, len(sg_entries)
    dump_f = open(dump_file,'wb')
    pickle.dump(sg_entries,dump_f)
    dump_f.close()

def viz_net2(net_name, weight_name, imdb, dump_file, viz_mode='viz_cls'):
    sess = tf.Session()
    # set up testing mode
    rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    rel_rois = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='rois')
    ims = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='ims')
    relations = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='relations')
    inputs = {'rois': rois,
              'rel_rois': rel_rois,
              'ims': ims,
              'relations': relations,
              'num_roi': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_rel': tf.placeholder(dtype=tf.int32, shape=[]),
              'num_classes': imdb.num_classes,
              'num_predicates': imdb.num_predicates,
              'rel_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'rel_pair_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None, 2]),
              'rel_pair_segment_inds': tf.placeholder(dtype=tf.int32, shape=[None]),
              'quadric_rois':  tf.placeholder(dtype=tf.float32, shape=[None, 28]),
              'rels_feat2d': tf.placeholder(dtype=tf.float32, shape=[None, 400]),
              'rels_feat3d': tf.placeholder(dtype=tf.float32, shape=[None, 400]),
              'n_iter': cfg.TEST.INFERENCE_ITER,
              'labels': tf.placeholder(dtype=tf.int32, shape=[None])
              }

    net = get_network(net_name)(inputs)
    net.setup()
    print ('Loading model weights from {:s}').format(weight_name)
    saver = tf.train.Saver()
    saver.restore(sess, weight_name)
    roidb = imdb.roidb
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    num_images = len(imdb.image_index)

    if net.iterable:
        inference_iter = net.n_iter - 1
    else:
        inference_iter = 0
    print('=======================VIZ INFERENCE Iteration = '),
    print(inference_iter)
    print('=======================VIZ MODES = '),
    print(viz_mode)
    #gts = np.zeros((0,5))
    sg_entries=[]
    for im_i in range(0,min(3000,num_images)): #num_images):
        print('processing image '+str(im_i)+'/'+str(num_images))
        im = imdb.im_getter(im_i)
        #from scipy.misc import imread, imsave
        first_box_idx = imdb.im_to_last_box[im_i]
        bbox_reg = True
        if viz_mode == 'viz_cls':
            # use ground truth bounding boxes
            bbox_reg = False
            box_proposals = gt_rois(roidb[im_i])
        elif viz_mode == 'viz_det':
            # use RPN-proposed object locations
            box_proposals, roi_scores = non_gt_rois(roidb[im_i])
            roi_scores = np.expand_dims(roi_scores, axis=1)
            nms_keep = cpu_nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                        cfg.TEST.PROPOSAL_NMS)
            nms_keep = np.array(nms_keep)
            num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
            keep = nms_keep[:num_proposal]
            box_proposals = box_proposals[keep, :]
        else:
            raise NotImplementedError('Incorrect visualization mode. Choose between [cls] and [det]')
        if box_proposals.size == 0 or box_proposals.shape[0] < 2:
            print 'skipping image',im_i
            continue
        quadric_rois = np.hstack([np.zeros((box_proposals.shape[0],1)),roidb[im_i]['quadric_rois']]) # this is because in the training phase, the image number is pre-pended, in the test case one image so index 0
        #rels_feat2d = roidb[im_i]['rel_geo_2d']
        #rels_feat3d = roidb[im_i]['rel_geo_3d']
        relations = roidb[im_i]['gt_relations'][:,:2]
        out_dict = im_detect(sess, net, inputs, im, box_proposals, bbox_reg, [inference_iter], quadric_rois, relations, roidb[im_i]['gt_classes'][:quadric_rois.shape[0]])
        sg_entry = out_dict[inference_iter]
        # ground predicted graphs to ground truth annotations
        gt_to_pred = ground_predictions(sg_entry, roidb[im_i], 0.5)
        roidb2 = dict(roidb[im_i])
        roidb2['gt_to_pred_object'] = gt_to_pred
        del roidb2['image']
        #import pdb; pdb.set_trace()
        sg_entries.append((sg_entry,roidb2))
    print 'saving results in file ',dump_file, 'number of entries: ',len(sg_entries)
    dump_f = open(dump_file,'wb')
    pickle.dump(sg_entries,dump_f)
    dump_f.close()
