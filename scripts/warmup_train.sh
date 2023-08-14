#!/bin/bash

### for sbd
PRETRAINED_MODEL="./pretrained_models/model_init_inst_sbd.pth"
python main.py \
  --dataset sbd \
  --cls-num 20 \
  --checkpoint-folder ./checkpoint/warm_up \
  --epochs 20 \
  --workers 8 \
  --lr 5e-8 \
  --save-freq 1 \
  --acc-steps 1 \
  --batch_size 8\
  --pretrained-model $PRETRAINED_MODEL | tee warmup_log.txt


## for cityscapes
#PRETRAINED_MODEL="./pretrained_models/model_init_cityscapes.pth"
#python main.py \
#  --dataset cityscapes \
#  --cls-num 19 \
#  --checkpoint-folder ./checkpoint/warm_up \
#  --epochs 150 \
#  --workers 16 \
#  --lr 2.5e-8 \
#  --save-freq 1 \
#  --acc-steps 1 \
#  --lr_steps 10000 20000 30000 40000 50000 \
#  --batch_size 8\
#  --pretrained-model $PRETRAINED_MODEL | tee warmup_log.txt