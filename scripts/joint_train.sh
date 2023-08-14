#!/bin/bash

### for sbd
PRETRAINED_CASENET="./release_model_params/sbd/casenet-s_sbd_retrain.pth"
PRETRAINED_STN="./release_model_params/sbd/psl_sbd_model.pth"

python pnt_main.py \
  --dataset sbd \
  --cls-num 20 \
  --checkpoint-folder ./checkpoint/joint_train \
  --epochs 10 \
  --workers 8 \
  --lr 1e-9 \
  --acc-steps 4 \
  --batch-size 2 \
  --train-scheme casenet \
  --stn-confin \
  --pretrained-casenet $PRETRAINED_CASENET \
  --pretrained-stn $PRETRAINED_STN | tee joint_log.txt


### for cityscapes
#PRETRAINED_CASENET="./release_model_params/cityscapes/casenet-s_cityscapes_retrain.pth"
#PRETRAINED_STN="./release_model_params/cityscapes/psl_city_model.pth"
#
#python pnt_main.py \
#  --dataset cityscapes \
#  --cls-num 19 \
#  --checkpoint-folder ./checkpoint/joint_train \
#  --epochs 80 \
#  --workers 12 \
#  --lr 1e-9 \
#  --acc-steps 4 \
#  --batch-size 2 \
#  --train-scheme casenet \
#  --lr_steps 10000 15000 20000 25000 \
#  --save_freq 5 \
#  --stn-confin \
#  --pretrained-casenet $PRETRAINED_CASENET \
#  --pretrained-stn $PRETRAINED_STN | tee joint_log.txt