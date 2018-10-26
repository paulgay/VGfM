from fast_rcnn.config import cfg
from roi_data_layer.minibatch_vid import get_minibatch
from roi_data_layer.minibatch_vid import get_minibatch_test
from roi_data_layer.roidb import prepare_roidb, add_bbox_regression_targets
import numpy as np


class RoIDataLayerVid:
    def __init__(self, imdb, bbox_means, bbox_stds):
        self.imdb = imdb
        self._roidb = imdb.roidb
        self._num_classes = imdb.num_classes
        self.seqs = imdb.seqs_mask
        self.seq2im = imdb.seq2im_mask
        self._shuffle_roidb_inds()
        self.bbox_means = bbox_means
        self.bbox_stds = bbox_stds

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self.seqs)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        if self._cur + 1 >= len(self.seqs):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur]
        self._cur += 1
        return db_inds

    def _get_next_minibatch(self, db_inds):
        next_seq = self.seqs[db_inds] # get next sequence indices
        minibatch_db = [self._roidb[i] for i in self.seq2im[next_seq]] # get the images for these sequences
        if cfg.TRAIN.USE_RPN_DB:
            minibatch_db = self.imdb.add_rpn_rois(minibatch_db)
        prepare_roidb(minibatch_db)
        add_bbox_regression_targets(minibatch_db, self.bbox_means,
                                    self.bbox_stds)
        blobs = get_minibatch(minibatch_db, self._num_classes)
        if blobs is not None:
            blobs['db_inds'] = self.seq2im[next_seq]
            blobs['seq_name'] = next_seq
        return blobs


    def _get_next_minibatch_test(self, db_inds):
        """
        same but without the bounding boxes regression
        """
        next_seq = self.seqs[db_inds]
        minibatch_db = [self._roidb[i] for i in self.seq2im[next_seq]]

        #if cfg.TRAIN.USE_RPN_DB:
        #    minibatch_db = self.imdb.add_rpn_rois(minibatch_db)
        #prepare_roidb(minibatch_db)
        #add_bbox_regression_targets(minibatch_db, self.bbox_means, self.bbox_stds)
        blobs = get_minibatch_test(minibatch_db, self._num_classes)
        if blobs is not None:
            blobs['db_inds'] = self.seq2im[next_seq]
            blobs['seq_name'] = next_seq
        return blobs

    def next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""
        batch = None
        while batch is None:
            db_inds = self._get_next_minibatch_inds()
            batch = self._get_next_minibatch(db_inds)
        return batch
