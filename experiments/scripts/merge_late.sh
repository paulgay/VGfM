set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=scannet
NUM_IM=-1
INFERENCE_ITER=2
TEST_MODE=all #pred_cls #viz_cls
SCORE_FILE=$1
REL_FILE=$2

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

time ./tools/merge_late.py --imdb ${IMDB}.h5 \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB}.h5 \
  --cfg ${CFG_FILE} \
  --inference_iter ${INFERENCE_ITER} \
  --test_size ${NUM_IM} \
  --test_mode ${TEST_MODE} \
  --load_score ${SCORE_FILE} \
  --write_rel_f ${REL_FILE}
