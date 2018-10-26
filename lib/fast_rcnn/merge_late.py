# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from roi_data_layer.roidb import prepare_roidb
import roi_data_layer.data_utils as data_utils
from datasets.evaluator import SceneGraphEvaluator
from networks.factory import get_network
from utils.timer import Timer
from utils.cpu_nms import cpu_nms
import numpy as np
import scipy.ndimage
import tensorflow as tf
import os, pickle
from utils.blob import im_list_to_blob

"""
Test a scene graph generation network
"""

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        im = scipy.ndimage.interpolation.zoom(im_orig, (im_scale, im_scale, 1.0), order=1)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(sess, net, inputs, im, boxes, bbox_reg, multi_iter, quadric_rois):
    blobs, im_scales = _get_blobs(im, boxes)

    relations = []
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            if i != j:
                relations.append([i, j])
    relations = np.array(relations, dtype=np.int32) # all possible combinations
    num_roi = blobs['rois'].shape[0]
    num_rel = relations.shape[0]

    inputs_feed = data_utils.create_graph_data(num_roi, num_rel, relations)

    feed_dict = {inputs['ims']: blobs['data'],
                 inputs['rois']: blobs['rois'],
                 inputs['relations']: relations,
                 inputs['quadric_rois']: quadric_rois,
                 net.keep_prob: 1}

    for k in inputs_feed:
        feed_dict[inputs[k]] = inputs_feed[k]

    # compute relation rois
    feed_dict[inputs['rel_rois']] = \
        data_utils.compute_rel_rois(num_rel, blobs['rois'], relations)

    ops = {}

    ops['bbox_deltas'] = net.bbox_pred_output(multi_iter)
    ops['rel_probs'] = net.rel_pred_output(multi_iter)
    ops['cls_probs'] = net.cls_pred_output(multi_iter)
    ops_value = sess.run(ops, feed_dict=feed_dict)

    out_dict = {}
    for mi in multi_iter:
        rel_probs = None
        rel_probs_flat = ops_value['rel_probs'][mi]
        rel_probs = np.zeros([num_roi, num_roi, rel_probs_flat.shape[1]])
        for i, rel in enumerate(relations):
            rel_probs[rel[0], rel[1], :] = rel_probs_flat[i, :]

        cls_probs = ops_value['cls_probs'][mi]

        if bbox_reg:
            # Apply bounding-box regression deltas
            pred_boxes = bbox_transform_inv(boxes, ops_value['bbox_deltas'][mi])
            pred_boxes = clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, cls_probs.shape[1]))

        out_dict[mi] = {'scores': cls_probs.copy(),
                        'boxes': pred_boxes.copy(),
                        'relations': rel_probs.copy()}
    return out_dict

def non_gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    non_gt_inds = np.setdiff1d(np.arange(overlaps.shape[0]), gt_inds)
    rois = roidb['boxes'][non_gt_inds]
    scores = roidb['roi_scores'][non_gt_inds]
    return rois, scores

def gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    rois = roidb['boxes'][gt_inds]
    return rois

def get_seq2im(i2s):
  s2i = {}
  for (im,seq) in i2s.items():
    if seq not in s2i:
      s2i[seq] = []
    s2i[seq].append(im)
  return s2i

def merge(sg_entries, roidb, im2seq):
  s2i = get_seq2im(im2seq)
  sg_post = []
  for i in range(len(im2seq)):
    sg_post.append({})
  for (seq_name, im_idx) in s2i.items():
    no = sg_entries[im_idx[0]]['scores'].shape[0]
    nc = sg_entries[im_idx[0]]['scores'].shape[1]
    agg_scores = np.zeros((no,nc))
    for o in range(no):
      agg_scores_o = np.zeros((len(im_idx),nc))
      ii=0
      for i in im_idx:
        agg_scores_o[ii,:] = sg_entries[i]['scores'][o,:]
        ii+=1
      #agg_scores[o,:] = np.sum(agg_scores_o, 0)/len(im_idx)
      agg_scores[o,:] = np.max(agg_scores_o, 0)
    for i in im_idx:
      sg_post[i]['boxes'] = sg_entries[i]['boxes']
      sg_post[i]['relations'] = sg_entries[i]['relations']
      sg_post[i]['scores'] = agg_scores
  return sg_post 

def merge_late(score_file, imdb, mode, write_rel_f, max_per_image=100):
    sg_entries = pickle.load(open(score_file))
    # set up testing mode
    roidb = imdb.roidb
    sg_entries = merge(sg_entries, roidb, imdb.im2seq_sp)
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    num_images = len(imdb.image_index)

    # timers
    _t = {'im_detect' : Timer(), 'evaluate' : Timer()}

    if mode == 'all':
        eval_modes = ['pred_cls', 'sg_cls', 'sg_det']
    else:
        eval_modes = [mode]
    multi_iter = [1]  #FIXME,. maybe try 2
    print('Graph Inference Iteration ='),
    print(multi_iter)
    print('EVAL MODES ='),
    print(eval_modes)

    # initialize evaluator for each task
    evaluators = {}
    for m in eval_modes:
        evaluators[m] = {}
        for it in multi_iter:
            evaluators[m][it] = SceneGraphEvaluator(imdb, mode=m)
    sg_i=-1
    preds = np.zeros((0,1))
    f = open(write_rel_f,'w')
    """
    #pickle.dump(roidb,open('gt.pc','wb'))
    my_roidb = []
    for i in range(0,num_images):
      gt_en = {}
      gt_en['gt_classes'] = roidb[i]['gt_classes']
      gt_en['boxes'] = roidb[i]['boxes']
      gt_en['gt_relations'] = roidb[i]['gt_relations']
      my_roidb.append(gt_en)
    pickle.dump(my_roidb,open('gt.pc','wb'))
    import sys; sys.exit()
    """
    for im_i in xrange(0,num_images,1): #xrange(num_images):
        im_i_roidb_full = imdb._image_index[im_i]
        seq_name = imdb.im2seq[im_i_roidb_full]
        im_path = imdb.impaths[imdb.roidb_idx_to_imdbidx[im_i_roidb_full]]
        fname = im_path.split('/')[-1].replace('.color.jpg','')
        first_box_idx = imdb.im_to_first_box[im_i]
        im = imdb.im_getter(im_i)
        sg_i+=1
        sg_entry=sg_entries[sg_i]
        scores_rel = sg_entry['relations']
        gt_rel=roidb[im_i]['gt_relations']
        gt_rela = np.hstack([np.ones((gt_rel.shape[0],1))*im_i, gt_rel])
        pred_rel_vec = np.zeros((gt_rel.shape[0],1));
        for i in range(gt_rel.shape[0]):
          o1 = gt_rela[i,1]
          o2 = gt_rela[i,2]
          o1_oid = imdb.roidb_to_scannet_oid[first_box_idx + o1][0] # to check later
          o2_oid = imdb.roidb_to_scannet_oid[first_box_idx + o2][0]
          predicate = gt_rela[i,3]
          if scores_rel[int(o1),int(o2),int(predicate)] > scores_rel[int(o2),int(o1),int(predicate)]:
            pred_rel_vec[i] = 1
            f.write(' '.join((seq_name,fname,str(o1_oid),str(o2_oid),imdb.info['idx_to_predicate'][str(int(predicate))],'1\n')))
          else:
            pred_rel_vec[i] = 0
            f.write(' '.join((seq_name,fname,str(o1_oid),str(o2_oid),imdb.info['idx_to_predicate'][str(int(predicate))],'0\n')))
        preds = np.vstack([preds,pred_rel_vec])

        for mode in eval_modes:
            bbox_reg = True
            if mode == 'pred_cls' or mode == 'sg_cls':
                # use ground truth object locations
                bbox_reg = False
                box_proposals = gt_rois(roidb[im_i])
            else:
                # use RPN-proposed object locations
                box_proposals, roi_scores = non_gt_rois(roidb[im_i])
                roi_scores = np.expand_dims(roi_scores, axis=1)
                nms_keep = cpu_nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                            cfg.TEST.PROPOSAL_NMS)
                nms_keep = np.array(nms_keep)
                num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
                keep = nms_keep[:num_proposal]
                box_proposals = box_proposals[keep, :]


            if box_proposals.size == 0 or box_proposals.shape[0] < 2:
                # continue if no graph
                continue

            _t['im_detect'].tic()
            quadric_rois=np.vstack([roidb[im_i]['quadric_rois'],roidb[im_i]['quadric_rois']]) # this duplication, I don't know why, but the boxes are duplicated, so I did the same, maybe it is to evaluate different aspect
            quadric_rois = np.hstack([np.zeros((quadric_rois.shape[0],1)),quadric_rois]) #this is because in the training phase, the image number is appended
            #out_dict = im_detect(sess, net, inputs, im, box_proposals, bbox_reg, multi_iter, quadric_rois)
                
            _t['im_detect'].toc()
            _t['evaluate'].tic()
            for iter_n in multi_iter:
                evaluators[mode][iter_n].evaluate_scene_graph_entry(sg_entry, im_i, iou_thresh=0.5)
            _t['evaluate'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(im_i + 1, num_images, _t['im_detect'].average_time,
                      _t['evaluate'].average_time)

    # print out evaluation results
    for mode in eval_modes:
        for iter_n in multi_iter:
            evaluators[mode][iter_n].print_stats()
    print 'direct accuracy of the predicates',np.mean(preds)
    f.close()