#!/bin/bash

### SBD
python get_results_for_benchmark.py \
-m ./release_model_params/sbd/casenet-s_sbd_retrain.pth \
-d ./example/image/test/ \
-f 2008_000075.png \
-o output_dir/casenet_s
#-l ./example/debug_test.txt\


### Cityscapes
# python get_results_for_benchmark.py \
# --dataset cityscapes \
# --num-cls 19 \
# -m /path/to/model.pth\
# -d ./example/image/test
# -o output_dir/debug\
# -l /path/to/val.txt\


