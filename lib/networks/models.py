# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------


from tensorflow.python.ops import nn_ops
import tensorflow as tf
from networks.network import Network
import losses
from fast_rcnn.config import cfg
import net_utils as utils
import numpy as np
"""
A TensorFlow implementation of the scene graph generation models introduced in
"Scene Graph Generation by Iterative Message Passing" by Xu et al.
"""

class basenet(Network):
    def __init__(self, data):
        self.inputs = []
        self.data = data
        self.ims = data['ims']
        self.rois = data['rois']
        self.iterable = False
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {}

    def _vgg16(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois})
        self._vgg_conv()
        self._vgg_fc()

    def _vgg_conv(self):
        (self.feed('ims')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .stop_gradient(name='conv_out'))

    def _vgg_fc(self):
        (self.feed('conv_out', 'rois')
             .roi_pool(7, 7, 1.0/16, name='pool5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='vgg_out'))

    def _union_rel_vgg_fc(self):
        (self.feed('conv_out', 'rel_rois')
             .roi_pool(7, 7, 1.0/16, name='rel_pool5')
             .fc(4096, name='rel_fc6')
             .dropout(self.keep_prob, name='rel_drop6')
             .fc(4096, name='rel_fc7')
             .dropout(self.keep_prob, name='rel_vgg_out'))

    # predictions
    def _cls_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'cls_score'+layer_suffix if new_var else 'cls_score'
        print(layer_name)
        (self.feed(input_layer)
             .fc(self.data['num_classes'], relu=False, name=layer_name,
                 reuse=reuse)
             .softmax(name='cls_prob'+layer_suffix)
             .argmax(1, name='cls_pred'+layer_suffix))

    def _bbox_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'bbox_pred'+layer_suffix if new_var else 'bbox_pred'
        (self.feed(input_layer)
             .fc(self.data['num_classes']*4, relu=False, name=layer_name,
                 reuse=reuse))

    def _rel_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'rel_score'+layer_suffix if new_var else 'rel_score'
        (self.feed(input_layer)
             #.fc(64, relu=True, name='just_before',reuse=reuse)
             .fc(self.data['num_predicates'], relu=False, name=layer_name,
                 reuse=reuse)
             .softmax(name='rel_prob'+layer_suffix)
             .argmax(1, name='rel_pred'+layer_suffix))

    # Losses
    def _sg_losses(self, ops={}, suffix=''):
        #ops = self._frc_losses(ops, suffix)
        ops = {}
        rel_score = self.get_output('rel_score'+suffix)
        ops['loss_rel'+suffix] = losses.sparse_softmax(rel_score, self.data['predicates'],
                                                name='rel_loss'+suffix, ignore_bg=True)
        return ops

    def _frc_losses(self, ops={}, suffix=''):
        # classification loss
        cls_score = self.get_output('cls_score'+suffix)
        ops['loss_cls'+suffix] = losses.sparse_softmax(cls_score, self.data['labels'], name='cls_loss'+suffix)

        # bounding box regression L1 loss
        if cfg.TRAIN.BBOX_REG:
            bbox_pred = self.get_output('bbox_pred'+suffix)
            ops['loss_box'+suffix]  = losses.l1_loss(bbox_pred, self.data['bbox_targets'], 'reg_loss'+suffix,
                                              self.data['bbox_inside_weights'])
        else:
            print('NO BBOX REGRESSION!!!!!')
        return ops

    def cls_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('cls_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('cls_prob')

        else:
            op = self.get_output('cls_prob')
        return op

    def bbox_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                op[i] = self.get_output('bbox_pred')

        else:
            op = self.get_output('bbox_pred')
        return op

    def rel_pred_output(self, iters=None):
        if iters is not None:
            op = {}
            for i in iters:
                if self.iterable and i != self.n_iter - 1:
                    op[i] = self.get_output('rel_prob_iter%i' % i)
                else:
                    op[i] = self.get_output('rel_prob')

        else:
            op = self.get_output('rel_prob')
        return op


class dual_graph_vrd(basenet):
    def __init__(self, data):
        basenet.__init__(self, data)
        self.num_roi = data['num_roi']
        self.num_rel = data['num_rel']
        self.rel_rois = data['rel_rois']
        self.iterable = True

        self.edge_mask_inds = data['rel_mask_inds']
        self.edge_segment_inds = data['rel_segment_inds']

        self.edge_pair_mask_inds = data['rel_pair_mask_inds']
        self.edge_pair_segment_inds = data['rel_pair_segment_inds']

        # number of refine iterations
        self.n_iter = data['n_iter']
        self.relations = data['relations']
        self.vert_state_dim = 512
        self.edge_state_dim = 512

    def setup(self):
        # extraction of feature
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cells()
        # feature refinement with message passing
        self._iterate()

    def _cells(self):
        """
        construct RNN cells and states
        vert are the object, and edges are the relations
        """
        # intiialize lstms
        self.vert_rnn = tf.nn.rnn_cell.GRUCell(self.vert_state_dim, activation=tf.tanh)
        self.edge_rnn = tf.nn.rnn_cell.GRUCell(self.edge_state_dim, activation=tf.tanh)

        # lstm states
        self.vert_state = self.vert_rnn.zero_state(self.num_roi, tf.float32)
        self.edge_state = self.edge_rnn.zero_state(self.num_rel, tf.float32)

    def _iterate(self):
        (self.feed('vgg_out')
             .fc(self.vert_state_dim, relu=False, name='vert_unary'))
        (self.feed('rel_vgg_out')
             .fc(self.edge_state_dim, relu=True, name='edge_unary'))
        vert_unary = self.get_output('vert_unary') # tensor matrix which is num_roi x 512
        edge_unary = self.get_output('edge_unary')
        vert_factor = self._vert_rnn_forward(vert_unary, reuse=False) # num_roi x 512 forward also gives the output of the cell, the state is still updated, it just not return by the function
        edge_factor = self._edge_rnn_forward(edge_unary, reuse=False) # num_rel x 512
        for i in xrange(self.n_iter):
            reuse = i > 0
            # compute edge states
            edge_ctx = self._compute_edge_context(vert_factor, edge_factor, reuse=reuse) # shape is num_rel x 512
            edge_factor = self._edge_rnn_forward(edge_ctx, reuse=True)
            # compute vert states
            vert_ctx = self._compute_vert_context(edge_factor, vert_factor, reuse=reuse)
            vert_factor = self._vert_rnn_forward(vert_ctx, reuse=True)
            vert_in = vert_factor
            edge_in = edge_factor
            self._update_inference(vert_in, edge_in, i)

    def _compute_edge_context_hard(self, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        if reduction_mode=='max':
            return tf.reduce_max(tf.gather(vert_factor, self.relations), [1])
        elif reduction_mode=='mean':
            return tf.reduce_mean(tf.gather(vert_factor, self.relations), [1])

    def _compute_vert_context_hard(self, edge_factor, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        edge_factor_gathered = utils.pad_and_gather(edge_factor, self.edge_mask_inds, None)

        vert_ctx = utils.padded_segment_reduce(edge_factor_gathered, self.edge_segment_inds,
                                               vert_factor.get_shape()[0], reduction_mode)

        return vert_ctx

    def _compute_edge_context_soft(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2 512] so I guess the corresponding object feature for each relation

        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs) # I guess putting the in (subject) on one part and the out (object) on the other part, because subject verb object
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor]) # sub_vert with the state of the predicate cell at tm1
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor]) # 24 x 1024

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score')) # passing the input messages from the relation to a fc layer
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score')) # passing the previous state of the object node to a fc layer

        sub_vert_w = self.get_output('sub_vert_score')
        obj_vert_w = self.get_output('obj_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        return weighted_sub + weighted_obj


    def _compute_edge_context_soft(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2 512] so I guess the corresponding object feature for each relation

        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs) # I guess putting the in (subject) on one part and the out (object) on the other part, because subject verb object
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor]) # sub_vert with the state of the predicate cell at tm1
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor]) # 24 x 1024

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score')) # passing the input messages from the relation to a fc layer
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score')) # passing the previous state of the object node to a fc layer

        sub_vert_w = self.get_output('sub_vert_score')
        obj_vert_w = self.get_output('obj_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        return weighted_sub + weighted_obj

    def _compute_vert_context_soft(self, edge_factor, vert_factor, reuse=False):
        """
        attention-based vertex(node) message pooling
        """
        out_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,0]) # from rel_pair_segment_inds and contains the indices of the relations which are going out
        in_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,1]) # 100 x 512 i.e. edge_factor[self.edge_pair_mask_inds[:,1],:] self.edge_pair_mask_inds[:,1] is 100,1 
        # gather correspounding vert factors
        vert_factor_gathered = tf.gather(vert_factor, self.edge_pair_segment_inds) #  will tell you which vert_factor correspond to which segment id maybe?

        # concat outgoing edges and ingoing edges with gathered vert_factors
        out_edge_w_input = tf.concat(concat_dim=1, values=[out_edge, vert_factor_gathered]) # 100 x 1024
        in_edge_w_input = tf.concat(concat_dim=1, values=[in_edge, vert_factor_gathered])

        # compute compatibility scores
        (self.feed(out_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='out_edge_w_fc')
             .sigmoid(name='out_edge_score'))
        (self.feed(in_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='in_edge_w_fc')
             .sigmoid(name='in_edge_score'))

        out_edge_w = self.get_output('out_edge_score')
        in_edge_w = self.get_output('in_edge_score')

        # weight the edge factors with computed weigths
        out_edge_weighted = tf.mul(out_edge, out_edge_w)
        in_edge_weighted = tf.mul(in_edge, in_edge_w)

        edge_sum = out_edge_weighted + in_edge_weighted
        vert_ctx = tf.segment_sum(edge_sum, self.edge_pair_segment_inds)
        return vert_ctx

    def _vert_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
        return vert_out

    def _edge_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
        return edge_out

    def _update_inference(self, vert_factor, edge_factor, iter_i):
        # make predictions
        reuse = iter_i > 0  # reuse variables

        iter_suffix = '_iter%i' % iter_i if iter_i < self.n_iter - 1 else ''
        self._cls_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._bbox_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._rel_pred(edge_factor, layer_suffix=iter_suffix, reuse=reuse)

    def losses(self):
        return self._sg_losses()


class vrd(basenet):
    """
    Baseline: the visual relation detection module proposed by
    Lu et al.
    """

    def __init__(self, data):
        basenet.__init__(self, data)
        self.rel_rois = data['rel_rois']

    def setup(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cls_pred('vgg_out')
        self._bbox_pred('vgg_out')
        self._rel_pred('rel_vgg_out')

    def losses(self):
        return self._sg_losses()


class dual_graph_vrd_maxpool(dual_graph_vrd):
    """
    Baseline: context-pooling by max pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='max')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='max')


class dual_graph_vrd_avgpool(dual_graph_vrd):
    """
    Baseline: context-pooling by avg. pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='mean')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='mean')


class dual_graph_vrd_final(dual_graph_vrd):
    """
    Our final model: context-pooling by attention
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_soft(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)


class dual_graph_vrd_2dbw(dual_graph_vrd):
    """
    2D baseline with a bag of word encoding as a 2D geometric representation
    """
    def __init__(self, data):
        dual_graph_vrd.__init__(self,data)
        self.geo_state =  tf.pack(data['rels_feat2d']) #tf.pack(data['quadric_rois'][:,1:])


    def _compute_edge_context_2d(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2*512] so I guess the corresponding object feature for each relation
        vis_geo = tf.concat(concat_dim=1, values=[vert_pairs, self.geo_state])
        # encoding the geomtery of the relation
        (self.feed(vis_geo)
             .fc(512, relu=False, reuse=reuse, name='geo_fc2_obj')
             .sigmoid(name='geo_encoded')) # passing the previous state of the object node to a fc layer
        geo_encoded = self.get_output('geo_encoded')
        return geo_encoded 

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_2d(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)

class dual_graph_vrd_3d(dual_graph_vrd):
    """
    Mono frame + ellipsoids
    """
    def __init__(self, data):
        dual_graph_vrd.__init__(self,data)
        self.geo_state =  tf.pack(data['quadric_rois'][:,1:]) 
        self.simple_state = tf.map_fn(self.get_simple_q,self.geo_state)

    def get_simple_q(self,q):
        w = tf.global_norm([q[:3] - q[3:6]])
        h = tf.global_norm([q[3:6] - q[6:9]])
        z = tf.global_norm([q[:3] - q[12:15]])
        return tf.concat(0, [q[24:27],tf.pack((w, h, z))])

    def _compute_edge_context_geo_only(self, vert_factor, edge_factor, reuse=False):
        """
        The message will only contains the geometric information
        """
        geo_pairs = utils.gather_vec_pairs(self.geo_state, self.relations)   #utils.gather_vec_pairs(self.geo_state, self.relations)
        self.geo_pairs = geo_pairs
        # compute compatibility scores
        (self.feed(geo_pairs)
             .fc(100, relu=False, reuse=reuse, name='geo_fc1')
             .sigmoid(name='geo_score_pre')
             .fc(512, relu=False, reuse=reuse, name='geo_fc2')
             .sigmoid(name='geo_score')) # passing the input messages from the relation to a fc layer

        geo_state_w = self.get_output('geo_score') 
        return geo_state_w

    def norm_geo_pairs(self, geo_pairs, sc=100):
        """
        normalise the pair of bounding boxes such that their mean is 0. i.e. we remove the location information
        """
        sh = tf.shape(geo_pairs)
        resh = tf.reshape(geo_pairs,tf.pack([sh[0], sh[1]/3, 3]))
        m = tf.reduce_mean(resh, axis=1)
        mr = tf.reshape(m, tf.pack([sh[0],1,3]))
        red = resh - mr
        return tf.reshape(red,sh)*sc


    def _compute_edge_context_geo_only_stand(self, vert_factor, edge_factor, reuse=False):
        """
        same as _compute_edge_context_geo_only, but the normalisation is added.
        """
        geo_pairs = utils.gather_vec_pairs(self.geo_state, self.relations)   #utils.gather_vec_pairs(self.geo_state, self.relations)
        geo_pairs = self.norm_geo_pairs(geo_pairs,sc=500)
        self.geo_pairs = geo_pairs
        # compute compatibility scores
        (self.feed(geo_pairs)
             .fc(100, relu=False, reuse=reuse, name='geo_fc1')
             .sigmoid(name='geo_score_pre')
             .fc(512, relu=False, reuse=reuse, name='geo_fc2')
             .sigmoid(name='geo_score')) # passing the input messages from the relation to a fc layer
        geo_state_w = self.get_output('geo_score') 
        return geo_state_w

    def _compute_edge_context_3d2(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        geo_pairs = utils.gather_vec_pairs(self.geo_state,self.relations)   #utils.gather_vec_pairs(self.geo_state, self.relations)
        self.geo_pairs = geo_pairs
        sub_geo, obj_geo = tf.split(split_dim=1, num_split=2, value=geo_pairs)

        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2*512] so I guess the corresponding object feature for each relation
        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs) # I guess putting the in (subject) on one part and the out (object) on the other part, because subject verb object, so each is 24 z 512
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor]) 
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor])

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score')) # passing the input messages from the relation to a fc layer
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score')) # passing the previous state of the object node to a fc layer

        # encoding the geomtery of the relation
        geo_pairs = self.norm_geo_pairs(geo_pairs,sc=1000)
        (self.feed(geo_pairs)
             .fc(100, relu=False, reuse=reuse, name='geo_fc1_obj')
             .sigmoid(name='geo_encoded_pre')
             .fc(512, relu=False, reuse=reuse, name='geo_fc2_obj')
             .sigmoid(name='geo_encoded')) # passing the previous state of the object node to a fc layer
        geo_encoded = self.get_output('geo_encoded')
        
        # then concatenating wiht the relation feature (edge_factor) and getting a weight out of that
        geo_encoded_w_input = tf.concat(concat_dim=1, values=[geo_encoded, edge_factor])
        (self.feed(geo_encoded_w_input)
             .fc(1, relu=False, reuse=reuse, name='geo_w_fc')
             .sigmoid(name='geo_vert_score')) # passing the previous state of the object node to a fc layer       
        sub_vert_w = self.get_output('sub_vert_score') #this guy is 24 x 1, and the relations vector is 24,which count for the 4 objects present in the 2 images.
        obj_vert_w = self.get_output('obj_vert_score')
        geo_vert_w = self.get_output('geo_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        weighted_geo = tf.mul(geo_encoded, geo_vert_w)
        return weighted_sub + weighted_obj + weighted_geo

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_3d2(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)


class dual_graph_simple(dual_graph_vrd_3d):
    """
    same as dual_graph_vrd_3d but use the get_simple_pairs function to gget the 3D geometric features
    """
    def get_simple_pairs(self, geo_pairs, sc=100):
        """
        extract the centers, the distances between the centers, and the axis length of the bounding boxes
        """
        m = (geo_pairs[:,:3] + geo_pairs[:,6:9])/2
        dist = tf.map_fn(nn_ops.l2_loss,geo_pairs[:,:3] - geo_pairs[:,6:9])
        t1 = (geo_pairs[:,:3] - m)
        t2 = (geo_pairs[:,6:9] - m)
        geo_pairs = tf.concat(1,[t1, t2, geo_pairs[:,3:6], geo_pairs[:,9:12], tf.reshape(dist,(tf.shape(dist)[0],1))])
        return geo_pairs*sc

    def _compute_edge_context_3d2(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        simple_pairs = utils.gather_vec_pairs(self.simple_state,self.relations)   #utils.gather_vec_pairs(self.geo_state, self.relations)
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2*512] so I guess the corresponding object feature for each relation
        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs) # I guess putting the in (subject) on one part and the out (object) on the other part, because subject verb object, so each is 24 z 512
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor]) 
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor])

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score')) # passing the input messages from the relation to a fc layer
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score')) # passing the previous state of the object node to a fc layer

        # encoding the geomtery of the relation
        geo_pairs = self.get_simple_pairs(simple_pairs,sc=500)
        #sub_geo, obj_geo = tf.split(split_dim=1, num_split=2, value=geo_pairs)
        #geo_conc = tf.concat(concat_dim=1, values=[sub_geo, obj_geo])
        (self.feed(geo_pairs)
             .fc(100, reuse=reuse, name='geo_fc1_obj')
             .dropout(0.5, name='drop_geo1')
             .sigmoid(name='geo_encoded_pre')
             .fc(512, relu=False, reuse=reuse, name='geo_fc2_obj')
             .dropout(0.5, name='drop_geo2')
             .sigmoid(name='geo_encoded')) # passing the previous state of the object node to a fc layer
        geo_encoded = self.get_output('geo_encoded')
        
        # then concatenating wiht the relation feature (edge_factor) and getting a weight out of that
        geo_encoded_w_input = tf.concat(concat_dim=1, values=[geo_encoded, edge_factor])
        (self.feed(geo_encoded_w_input)
             .fc(1, relu=False, reuse=reuse, name='geo_w_fc')
             .sigmoid(name='geo_vert_score')) # passing the previous state of the object node to a fc layer       
        sub_vert_w = self.get_output('sub_vert_score') #this guy is 24 x 1, and the relations vector is 24,which count for the 4 objects present in the 2 images.
        obj_vert_w = self.get_output('obj_vert_score')
        geo_vert_w = self.get_output('geo_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        weighted_geo = tf.mul(geo_encoded, geo_vert_w)
        return weighted_sub + weighted_obj + weighted_geo

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_3d2(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)


class dual_graph_vrd_2d(dual_graph_vrd_3d):
    """
    baseline with 2D bounding boxes 
    """
    def __init__(self, data):
        dual_graph_vrd.__init__(self,data)
        self.geo_state =  tf.pack(data['rois']) 
        self.geo_state = tf.pack(values=[self.geo_state[:,1], self.geo_state[:,2], self.geo_state[:,3], self.geo_state[:,4], (self.geo_state[:,1]+self.geo_state[:,3])/2, (self.geo_state[:,2]+self.geo_state[:,4])/2 ], axis=1)

    def norm_geo_pairs(self, geo_pairs, sc=100):
        return geo_pairs

class mono_multi(dual_graph_vrd_3d):
    """
    Extension with multi-label classification
    """
    def _rel_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'rel_score'+layer_suffix if new_var else 'rel_score'
        (self.feed(input_layer)
             .fc(self.data['num_predicates'], relu=False, name=layer_name,
                 reuse=reuse)
             .sigmoid(name='rel_prob'+layer_suffix)
             .argmax(1, name='rel_pred'+layer_suffix))

    def _sg_losses(self, ops={}, suffix=''):
        ops = self._frc_losses(ops, suffix)
        rel_score = self.get_output('rel_score'+suffix)
        ops['loss_rel'+suffix] = losses.multi_label_rel(rel_score, self.data['predicates'], name='rel_loss'+suffix, ignore_bg=True)
        return ops


class dual_graph_vrd_geo_only(dual_graph_vrd_3d):
    """
    using only geometric information
    """
    def __init__(self, data):
        dual_graph_vrd_3d.__init__(self,data)

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_geo_only(vert_factor, edge_factor, reuse)    

class dual_graph_vrd_geo_stand(dual_graph_vrd_3d):
    """
    using only normalised geometric information
    """
    def __init__(self, data):
        dual_graph_vrd_3d.__init__(self,data)

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_geo_only_stand(vert_factor, edge_factor, reuse)    


class dual_graph_vrd_fus(dual_graph_vrd_3d):
    """
    The early fusion
    """
    def __init__(self, data):
        dual_graph_vrd_3d.__init__(self,data)
        self.obj_fus_mask_inds = data['obj_fus_segment_inds']
        self.obj_fus_segment_inds = data['obj_fus_mask_inds']
        self.rel_fus_mask_inds = data['rel_fus_segment_inds']
        self.rel_fus_segment_inds = data['rel_fus_mask_inds']

    def _compute_vert_context_fus(self, edge_factor, vert_factor, reuse=False):
        """
        attention-based vertex(node) message pooling
        also collecting messages from the detections of this object in other frames
        """
        out_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,0]) # from rel_pair_segment_inds and contains the indices of the relations which are going out
        in_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,1]) # if I have 5 objects, each of them is connected to 4 objects, so 5 times 4, edge_pair_mask_inds has size 20, but there is 25, and this 20 coming every time. dans data_utils, there is this pad with pad with dummy edge ind 
        # gather correspounding vert factors
        vert_factor_gathered = tf.gather(vert_factor, self.edge_pair_segment_inds) # in Eq.3 you compute sum_j v_1*[h_i,hi->j]hi->j. So you want to concatenate each h_i with hi->j. Here, you are building the matrix vert_factor_gathered, contains the h_i part. Later you jus have to concatenate this matrix with out_edge which contains the hi->j. vert_factor_gathered will contains all the hi. vert_factor_gathered is also used to concatenate with in_edge, because self.edge_pair_segment_inds[k] is build to be the segment id corresponding to self.edge_pair_mask_inds[k,:]

        vert_factor_matched = utils.pad_and_gather(vert_factor, self.obj_fus_mask_inds[:]) # This tells where is this object matched in the other frames, and select the corresponding features
        vert_factor_gathered_vid = tf.gather(vert_factor, self.obj_fus_segment_inds)  # instead of having number of output +1 I need to have the number of frames
    
        # concat outgoing edges and ingoing edges with gathered vert_factors
        out_edge_w_input = tf.concat(concat_dim=1, values=[out_edge, vert_factor_gathered])
        in_edge_w_input = tf.concat(concat_dim=1, values=[in_edge, vert_factor_gathered])

        vid_edge_w_input =  tf.concat(concat_dim=1, values=[vert_factor_matched, vert_factor_gathered_vid]) 
        # compute compatibility scores
        (self.feed(out_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='out_edge_w_fc')
             .sigmoid(name='out_edge_score'))
        (self.feed(in_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='in_edge_w_fc')
             .sigmoid(name='in_edge_score'))

        (self.feed(vid_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='vid_vert_w_fc')
             .sigmoid(name='vid_vert_score'))

        out_edge_w = self.get_output('out_edge_score')
        in_edge_w = self.get_output('in_edge_score')
        vid_vert_w = self.get_output('vid_vert_score')
 
        # weight the edge factors with computed weigths
        out_edge_weighted = tf.mul(out_edge, out_edge_w)
        in_edge_weighted = tf.mul(in_edge, in_edge_w)
        vid_vert_weighted = tf.mul(vert_factor_matched, vid_vert_w)

        edge_sum = out_edge_weighted + in_edge_weighted 
        
        vert_ctx = tf.segment_sum(edge_sum, self.edge_pair_segment_inds) + tf.segment_sum(vid_vert_weighted, self.obj_fus_segment_inds)
        return vert_ctx

    def _compute_edge_context_3d_fus(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        this output will be the input edge_ctx of _edge_rnn_forward(edge_ctx, ...
        edge_factor and vert_factor are gru_cell
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations) #Tensor("Reshape:0", shape=(?, 1024), dtype=float32) if relation is [24 2], vert_pairs is [24 2*512] so I guess the corresponding object feature for each relation
        geo_pairs = utils.gather_vec_pairs(self.geo_state,self.relations)   #utils.gather_vec_pairs(self.geo_state, self.relations)
        self.geo_pairs = geo_pairs

        edge_factor_matched = utils.pad_and_gather(edge_factor, self.rel_fus_mask_inds[:]) # getting the output for this relation in every frame
        edge_factor_gathered_vid = tf.gather(edge_factor, self.rel_fus_segment_inds) # getting the same but in other order. This way it is just a matter of concatenate and sum
        vid_edge_w_input =  tf.concat(concat_dim=1, values=[edge_factor_matched, edge_factor_gathered_vid])

        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs) # I guess putting the in (subject) on one part and the out (object) on the other part, because subject verb object, so each is 24 z 512
        sub_geo, obj_geo = tf.split(split_dim=1, num_split=2, value=geo_pairs)
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor ]) 
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor ])

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score')) # passing the input messages from the relation to a fc layer
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score')) # passing the previous state of the object node to a fc layer

        # encoding the geomtery of the relation
        geo_pairs = self.norm_geo_pairs(geo_pairs,sc=500)
        #sub_geo, obj_geo = tf.split(split_dim=1, num_split=2, value=geo_pairs)
        #geo_conc = tf.concat(concat_dim=1, values=[sub_geo, obj_geo])
        (self.feed(geo_pairs)
             .fc(100, relu=False, reuse=reuse, name='geo_fc1_obj')
             .sigmoid(name='geo_encoded_pre')
             .fc(512, relu=False, reuse=reuse, name='geo_fc2_obj')
             .sigmoid(name='geo_encoded')) # passing the previous state of the object node to a fc layer
        geo_encoded = self.get_output('geo_encoded')

        # then concatenating wiht the relation feature (edge_factor) and getting a weight out of that
        geo_encoded_w_input = tf.concat(concat_dim=1, values=[geo_encoded, edge_factor])
        (self.feed(geo_encoded_w_input)
             .fc(1, relu=False, reuse=reuse, name='geo_w_fc')
             .sigmoid(name='geo_vert_score')) # passing the previous state of the object node to a fc layer       

        # will compute weigts from the messages in each relations
        (self.feed(vid_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='vid_edge_w_fc')
             .sigmoid(name='vid_edge_score'))

        sub_vert_w = self.get_output('sub_vert_score') #this guy is 24 x 1, and the relations vector is 24,which count for the 4 objects present in the 2 images.
        obj_vert_w = self.get_output('obj_vert_score')
        geo_vert_w = self.get_output('geo_vert_score')
        vid_edge_w = self.get_output('vid_edge_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        weighted_geo = tf.mul(geo_encoded, geo_vert_w)
        vid_edge_weighted = tf.mul(edge_factor_matched, vid_edge_w)
        return weighted_sub + weighted_obj + weighted_geo + tf.segment_sum(vid_edge_weighted, self.rel_fus_segment_inds)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_fus(edge_factor, vert_factor, reuse)

    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_3d_fus(vert_factor, edge_factor, reuse)



class fus_early_multi(dual_graph_vrd_fus):
    """
    Extension with multi-label classification
    """
    def _rel_pred(self, input_layer, layer_suffix='', reuse=False, new_var=False):
        layer_name = 'rel_score'+layer_suffix if new_var else 'rel_score'
        (self.feed(input_layer)
             .fc(self.data['num_predicates'], relu=False, name=layer_name,
                 reuse=reuse)
             .sigmoid(name='rel_prob'+layer_suffix)
             .argmax(1, name='rel_pred'+layer_suffix))

    def _sg_losses(self, ops={}, suffix=''):
        ops = self._frc_losses(ops, suffix)
        rel_score = self.get_output('rel_score'+suffix)
        #ops['loss_rel'+suffix] = losses.sparse_softmax(rel_score, self.data['predicates'], name='rel_loss'+suffix, ignore_bg=True)
        ops['loss_rel'+suffix] = losses.multi_label_rel(rel_score, self.data['predicates'], name='rel_loss'+suffix, ignore_bg=True)
        return ops
