#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-8888}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# nuscene+waymo+lyft mix training
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --seed 0 \
    --launcher pytorch ${@:1} \
    --model_name bevdet-r50-4d-depth-CoIn3D \
    --src_dataset mix \
    --src_train_cls general_3cls \
    # --auto-resume


# nuscene only
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --seed 0 \
    --launcher pytorch ${@:1} \
    --model_name bevdet-r50-4d-depth-CoIn3D \
    --exp_tag 'caronly' \
    --src_dataset nuscenes \
    --src_train_cls general_3cls \
    # --auto-resume


# waymo only
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --seed 0 \
    --launcher pytorch ${@:1} \
    --model_name bevdet-r50-4d-depth-CoIn3D \
    --exp_tag 'caronly' \
    --src_dataset waymo \
    --src_train_cls general_3cls \
    # --auto-resume


# lyft only
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --seed 0 \
    --launcher pytorch ${@:1} \
    --model_name bevdet-r50-4d-depth-CoIn3D \
    --exp_tag 'caronly' \
    --src_dataset lyft \
    --src_train_cls raw \
    # --auto-resume
