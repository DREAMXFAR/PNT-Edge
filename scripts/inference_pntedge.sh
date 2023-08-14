#!/bin/bash

### SBD
python get_results_for_pntedge.py \
-m ./release_model_params/sbd/checkpoint_7.pth \
-o output_dir/pntedge_ck7 \
-l /dat02/xwj/edge_detection/CASENet-master/sbd-preprocess/data_proc/txt_list/reanno_test.txt \
-d /dat02/xwj/edge_detection/CASENet-master/sbd-preprocess/data_proc/

# -d ./example/image/test/ \
#-l ./example/debug_test.txt
# -f 2008_000075.png



### Cityscapes
# python get_results_for_pntedge.py \
# --dataset cityscapes \
# --num-cls 19 \
# -m /path/to/model.pth\
# -d ./example/image/test
# -o output_dir/debug\
# -l /path/to/val.txt\
