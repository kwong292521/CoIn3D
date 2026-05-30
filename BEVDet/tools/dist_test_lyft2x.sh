#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-8888}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MODEL_NAME=${MODEL_NAME:-"bevdet-r50-4d-depth-CoIn3D"}
EXP_TAG=${EXP_TAG:-"caronly"}
SRC_DATASET=${SRC_DATASET:-"lyft"}
SRC_TRAIN_CLS=${SRC_TRAIN_CLS:-"raw"}
# DST_EVAL_CLS=${SRC_TRAIN_CLS:-"general_3cls"}
# CHECKPOINT_PATH="logs/train/${MODEL_NAME}/${SRC_DATASET}_${EXP_TAG}/src_cls@${SRC_TRAIN_CLS}/epoch_24_ema.pth"
CHECKPOINT_PATH="ckpts/lyft.pth"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    --seed 0 \
    --launcher pytorch ${@:1} \
    --src_dataset $SRC_DATASET \
    --dst_dataset nuscenes \
    --src_train_cls $SRC_TRAIN_CLS \
    --dst_eval_cls raw \
    --model_name $MODEL_NAME \
    --checkpoint $CHECKPOINT_PATH \
    --exp_tag $EXP_TAG
