#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
INFERENCE_ITER=$2
EXP_DIR=$3
GPU_ID=$4
datadir=$5
multi=$6


ROIDB=synth-SGG
RPNDB=proposals.h5
IMDB=imdb_875.h5
ITERS=50000



CFG_FILE=experiments/cfgs/sparse_graph.yml
PRETRAINED=data/pretrained/coco_vgg16_faster_rcnn_final.npy
TF_LOG=$EXP_DIR/tf_logs
rm -rf ${EXP_DIR}/logs/
rm -rf ${TF_LOG}
mkdir -p ${EXP_DIR}/logs
LOG="$EXP_DIR/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time  ./tools/train_net_vid.py --gpu $GPU_ID \
  --weights ${PRETRAINED} \
  --imdb ${IMDB} \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB} \
  --iters ${ITERS} \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --output ${EXP_DIR} \
  --tf_log ${TF_LOG} \
  --multi_label $multi \
  --data_dir $datadir
