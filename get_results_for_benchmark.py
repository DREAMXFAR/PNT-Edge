import os
import sys
import argparse

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from imageio import imwrite

import torch
from torch import sigmoid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_cityscapes_dataset import RGB2BGR, ToTorchFormatTensor
from prep_dataset.categories import sbd_categories, cityscapes_categories
from prep_dataset.colors import sbd_colors, cityscapes_colors

import utils.utils as utils


def center_crop(img, orig_size):
    (cur_w, cur_h, _) = img.shape
    orig_w, orig_h = orig_size
    cropped_img = img[(cur_h - orig_h) // 2: (cur_h + orig_h) // 2, (cur_w - orig_w) // 2: (cur_w + orig_w) // 2, :]
    assert cropped_img.shape[:-1] == (orig_h, orig_w)

    return cropped_img


if __name__ == "__main__":
    import os

    """
    ### SBD 
    python get_results_for_benchmark.py \
    -m /dat02/xwj/edge_detection/CASENet-master/checkpoint/ablation/sbd_inst_casenet-s_train_stnrefinedgt_conf1/checkpoint_10.pth\
    -o /dat02/xwj/edge_detection/CASENet-master/output/ablation_clean/sbd_inst_casenet-s_train_stnrefinedgt_conf1_ep10\
    -l /dat02/xwj/edge_detection/CASENet-master/sbd-preprocess/data_proc/txt_list/ablation_test.txt\
    -d /dat02/xwj/edge_detection/CASENet-master/sbd-preprocess/data_proc

    python get_results_for_benchmark.py \
    -m /dat02/xwj/edge_detection/CASENet-master/checkpoint/baseline/sbd_inst_casenet-s_wofreeze_trainval_lr5e8/checkpoint_18.pth\
    -d /dat02/xwj/edge_detection/seal-master/data/sbd-preprocess/data_orig/img/ \
    -f 2008_000003.jpg \
    -o output_dir/debug
    
    ### Cityscapes
    python get_results_for_benchmark.py \
    --dataset cityscapes --num-cls 19\
    -m /dat02/xwj/edge_detection/CASENet-master/checkpoint/cityscapes_inst_casenet-s_wofreeze_train_lr25e8_epoch100_woscale/checkpoint_20.pth\
    -o /dat02/xwj/edge_detection/CASENet-master/cityscapes_output/my_output/cityscapes_inst_casenet-s_wofreeze_train_lr25e8_epoch100_woscale_ep20\
    -l /dat02/xwj/edge_detection/CASENet-master/cityscapes-preprocess/data_proc/txt_list/val.txt\
    -d /dat02/xwj/edge_detection/CASENet-master/cityscapes-preprocess/data_proc

    /dat02/xwj/edge_detection/CASENet-master/pretrained_models/ac_cityscapes_models/pretrained_pytorch/model_casenet.pth\
    /dat02/xwj/edge_detection/CASENet-master/pretrained_models/seal_cityscapes_models/pretrained_pytorch/model_seal.pth\
    /dat02/xwj/edge_detection/CASENet-master/pretrained_models/seal_cityscapes_models/pretrained_pytorch/model_casenet-s.pth
    
    """
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--dataset', type=str, default='sbd')
    parser.add_argument('-m', '--model', type=str,
                        help="path to the pytorch(.pth) containing the trained weights")
    parser.add_argument('--num-cls', type=int, default=20)
    parser.add_argument('-l', '--image_list', type=str, default='',
                        help="list of image files to be tested")
    parser.add_argument('-f', '--image_file', type=str, default='',
                        help="a single image file to be tested")
    parser.add_argument('-d', '--image_dir', type=str, default='',
                        help="root folder of the image files in the list or the single image file")
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help="folder to store the test results")

    args = parser.parse_args(sys.argv[1:])


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    vis_on = False
    reverse_color = False

    # load input path
    if os.path.exists(args.image_list):
        with open(args.image_list) as f:
            ori_test_lst = [x.strip().split()[0] for x in f.readlines()]
            if args.image_dir != '':
                test_lst = [
                    args.image_dir + x if os.path.isabs(x)
                    else os.path.join(args.image_dir, x)
                    for x in ori_test_lst]
    else:
        image_file = os.path.join(args.image_dir, args.image_file)
        if os.path.exists(image_file):
            ori_test_list = [args.image_file]
            test_lst = [image_file]
        else:
            raise IOError('nothing to be tested!')

    # load net
    model = CASENet_resnet101(pretrained=False, num_classes=args.num_cls).cuda()
    model = model.eval()
    # cudnn.benchmark = True
    utils.load_pretrained_model(model, args.model)

    # Define normalization for data    
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])
    if args.dataset == 'sbd':
        crop_size = [512, 512]
    elif args.dataset == 'cityscapes':
        crop_size = [1024, 2048]
    else:
        raise Exception('Wrong crop size!')

    img_transform = transforms.Compose([
        RGB2BGR(roll=True),
        ToTorchFormatTensor(div=False),
        normalize,
    ])

    for idx_img in range(len(test_lst)):
        print('====> {}/{}: {}'.format(idx_img, len(test_lst), test_lst[idx_img]))
        img = Image.open(test_lst[idx_img]).convert('RGB')

        processed_img = img_transform(img).unsqueeze(0)  # 1 X 3 X H X W
        height = processed_img.size()[2]
        width = processed_img.size()[3]

        if crop_size[0] < height or crop_size[1] < width:
            raise ValueError("Input image size must be smaller than crop size!")
        pad_h = crop_size[0] - height
        pad_w = crop_size[1] - width
        padded_processed_img = F.pad(processed_img, (0, pad_w, 0, pad_h), "constant", 0).data

        processed_img_var = utils.check_gpu(0, padded_processed_img)

        score_feats5, score_fuse_feats = model(processed_img_var)  # 1 X 19 X CROP_SIZE X CROP_SIZE
        score_output = sigmoid(score_fuse_feats.transpose(1, 3).transpose(1, 2)).squeeze(0)[:height, :width, :]  # H X W X 19

        ### save boundary for evaluate
        for cls_idx in range(args.num_cls):
            # Convert binary prediction to uint8
            im_arr = (score_output[:, :, cls_idx].data.cpu().numpy()) * 255.0
            im_arr = im_arr.astype(np.uint8)

            if reverse_color:
                cls_idx = args.num_cls - 1 - cls_idx
            # Store value into img
            img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
            if not os.path.exists(os.path.join(args.output_dir, 'class_' + str(cls_idx+1))):
                os.makedirs(os.path.join(args.output_dir, 'class_' + str(cls_idx+1)))
            imwrite(os.path.join(args.output_dir, 'class_' + str(cls_idx+1), img_base_name_noext + '.png'), im_arr)
            # print('processed: ' + test_lst[idx_img])

        ### save colored prediction
        if args.dataset == "sbd":
            colors = sbd_colors
        elif args.dataset == "cityscapes":
            colors = cityscapes_colors
        else:
            raise Exception('Wrong dataset type!')

        # the color coding can be referred to SEAL-V3 paper
        score_output = score_output.data.cpu().numpy()
        rgb = np.zeros((score_output.shape[0], score_output.shape[1], 3))
        bdry_sum = np.zeros((score_output.shape[0], score_output.shape[1]))
        bdry_max = np.zeros((score_output.shape[0], score_output.shape[1]))
        for idx_cls in range(args.num_cls):
            score_pred = score_output[:, :, idx_cls]
            bdry_sum = bdry_sum + score_pred
            bdry_max = np.max(np.concatenate((score_pred[:, :, np.newaxis], bdry_max[:, :, np.newaxis]), axis=2), axis=2)

            if reverse_color:
                idx_cls = args.num_cls - 1 - idx_cls
            r = score_pred * (1 - colors[idx_cls][0] / 255.0)
            g = score_pred * (1 - colors[idx_cls][1] / 255.0)
            b = score_pred * (1 - colors[idx_cls][2] / 255.0)
            rgb[:, :, 0] += r
            rgb[:, :, 1] += g
            rgb[:, :, 2] += b

        bdry_sum[bdry_sum == 0] = 1

        bdry_vis = np.ones((score_output.shape[0], score_output.shape[1], 3)) * 255
        bdry_vis = (bdry_vis - 255 * rgb * bdry_max[:, :, np.newaxis] / (bdry_sum[:, :, np.newaxis])).astype(np.uint8)

        if vis_on:
            plt.figure()
            plt.imshow(bdry_vis)
            plt.show()

        # Store value into img
        img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
        save_dir_path = os.path.join(args.output_dir, 'pred_color')
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        imwrite(os.path.join(save_dir_path, img_base_name_noext + '.png'), bdry_vis)

        del score_feats5
        del score_fuse_feats
        del score_output
        del padded_processed_img
        del processed_img
        del processed_img_var
        torch.cuda.empty_cache()

    print('Done!')
