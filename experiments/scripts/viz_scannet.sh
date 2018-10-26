set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=scannet
NUM_IM=-1
NET=$2 #dual_graph_vrd_final
INFERENCE_ITER=2
WEIGHT_FN=$1 #checkpoints/test/weights_999.ckpt
TEST_MODE=viz_cls
GPU_ID=$3
DUMP_FILE=$4
CFG_FILE=experiments/cfgs/sparse_graph.yml

case $DATASET in
    vg)
        ROIDB=VG-SGG
        RPNDB=proposals
        IMDB=imdb_1024
        ;;
    mini-vg)
        ROIDB=mini_VG-SGG
        RPNDB=mini_proposals
        IMDB=mini_imdb_1024
        ;;
    scannet)
        ROIDB=scannet-SGG
        RPNDB=proposals
        IMDB=imdb_1296
        ;;
    synth)
        ROIDB=synth-SGG
        RPNDB=proposals
        IMDB=imdb_875
        ;;
    *)
        echo "Wrong dataset"
        exit
        ;;
esac


export CUDA_VISIBLE_DEVICES=$GPU_ID

time ./tools/test_scannet.py --gpu $GPU_ID \
  --weights ${WEIGHT_FN} \
  --imdb ${IMDB}.h5 \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB}.h5 \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --test_size ${NUM_IM} \
  --test_mode ${TEST_MODE} \
  --dump_file ${DUMP_FILE}
