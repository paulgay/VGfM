#!/usr/bin/env python

# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# Adapted from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import _init_paths
from fast_rcnn.test import test_net, eval_net
from fast_rcnn.visualize import viz_net2
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb, get_imdb_multi
import argparse
import pprint
import time, os, sys
import tensorflow as tf

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a scene graph generation network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--weights', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb',
                        help='dataset to test',
                        default='im_512.h5', type=str)
    parser.add_argument('--roidb', dest='roidb',
                        help='dataset to test',
                        default='VG', type=str)
    parser.add_argument('--rpndb', dest='rpndb',
                        help='dataset to test',
                        default='proposals.h5', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_dir',
                        default=None, type=str)
    parser.add_argument('--inference_iter', dest='inference_iter',
                        default=3, type=int)
    parser.add_argument('--test_size', dest='test_size',
                        default=1000, type=int)
    parser.add_argument('--test_mode', dest='test_mode',
                        default='fg', type=str)
    parser.add_argument('--load_score', dest='load_score',
                         default=None, type=str)
    parser.add_argument('--dump_file', dest='dump_file',
                             default=None, type=str)
    parser.add_argument('--write_rel_f', dest='write_rel_f',
                                 default=None, type=str)
    parser.add_argument('--multi_label', dest='multi_label', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str)
    parser.add_argument('--split', type=int, default=2)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.TEST.INFERENCE_ITER = args.inference_iter

    print('Using config:')
    pprint.pprint(cfg)

    cfg.GPU_ID = args.gpu_id

    config = tf.ConfigProto()
    config.allow_soft_placement=True

    if args.multi_label == 1:
        imdb = get_imdb_multi(args.roidb, args.imdb, args.rpndb, args.data_dir, split=args.split, num_im=args.test_size)
    else:
        imdb = get_imdb(args.roidb, args.imdb, args.rpndb, args.data_dir, split=args.split, num_im=args.test_size)
    if args.test_mode == 'viz_cls' or args.test_mode == 'viz_det':  # visualize result
        viz_net2(args.network_name, args.model, imdb, args.dump_file, args.test_mode)
    elif args.load_score is not None:
        eval_net(args.load_score, imdb, args.test_mode, args.write_rel_f)
    else:
        test_net(args.network_name, args.model, imdb, args.test_mode)
