#!/bin/bash

### for sbd
PRETRAINED_CASENET="./release_model_params/sbd/casenet-s_sbd_retrain.pth"

python pnt_main.py \
  --dataset sbd \
  --cls-num 20 \
  --checkpoint-folder ./checkpoint/psl_train \
  --epochs 10 \
  --workers 8 \
  --lr 1e-6 \
  --acc-steps 4 \
  --batch-size 2 \
  --train-scheme stn \
  --stn-confin \
  --pretrained-casenet $PRETRAINED_CASENET | tee psl_log.txt

### for cityscapes
#PRETRAINED_CASENET="./release_model_params/cityscapes/casenet-s_cityscapes_retrain.pth"
#
#python pnt_main.py \
#  --dataset cityscapes \
#  --cls-num 19 \
#  --checkpoint-folder ./checkpoint/psl_train \
#  --epochs 80 \
#  --workers 12 \
#  --lr 1e-6 \
#  --acc-steps 4 \
#  --batch-size 2 \
#  --train-scheme stn \
#  --lr_steps 5000 10000 15000 20000 25000 \
#  --save_freq 5 \
#  --stn-confin \
#  --pretrained-casenet $PRETRAINED_CASENET | tee psl_log.txt
