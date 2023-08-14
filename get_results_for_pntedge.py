import os
import sys
import argparse

import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import shutil
import h5py
from imageio import imwrite
from scipy.io import loadmat
import scipy

import torch
from torch import sigmoid
import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet_with_STN import CASENetSTN, SpaticalTransformer
from prep_dataset.prep_cityscapes_dataset import RGB2BGR, ToTorchFormatTensor
from prep_dataset.categories import sbd_categories, cityscapes_categories
from prep_dataset.colors import sbd_colors, cityscapes_colors

import utils.utils as utils
import neurite as ne


def center_crop(img, orig_size):
    (cur_w, cur_h, _) = img.shape
    orig_w, orig_h = orig_size
    cropped_img = img[(cur_h - orig_h) // 2: (cur_h + orig_h) // 2, (cur_w - orig_w) // 2: (cur_w + orig_w) // 2, :]
    assert cropped_img.shape[:-1] == (orig_h, orig_w)
    return cropped_img


def save_cls(score_output, num_cls, dir_prefix='class_', inverse_id=True):
    ### save boundary for evaluate
    for cls_idx in range(num_cls):
        if inverse_id:
            im_arr = (score_output[:, :, 19 - cls_idx].data.cpu().numpy()) * 255.0
        else:
            im_arr = (score_output[:, :, cls_idx].data.cpu().numpy()) * 255.0

        # Convert binary prediction to uint8
        im_arr = im_arr.astype(np.uint8)

        # Store value into img
        img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
        if not os.path.exists(os.path.join(args.output_dir, dir_prefix + str(cls_idx + 1))):
            os.makedirs(os.path.join(args.output_dir, dir_prefix + str(cls_idx + 1)))
        imwrite(os.path.join(args.output_dir, dir_prefix + str(cls_idx + 1), img_base_name_noext + '.png'), im_arr)


def save_colored_pred(score_output, dir_name='pred_color', vis=False, colors=sbd_colors, inverse_id=True):
    ### save colored prediction
    # the color coding can be referred to SEAL-V3 paper
    score_output = score_output.data.cpu().numpy()
    rgb = np.zeros((score_output.shape[0], score_output.shape[1], 3))
    bdry_sum = np.zeros((score_output.shape[0], score_output.shape[1]))
    bdry_max = np.zeros((score_output.shape[0], score_output.shape[1]))
    for idx_cls in range(num_cls):
        if inverse_id:
            score_pred = score_output[:, :, 19 - idx_cls]
        else:
            score_pred = score_output[:, :, idx_cls]

        bdry_sum = bdry_sum + score_pred
        bdry_max = np.max(np.concatenate((score_pred[:, :, np.newaxis], bdry_max[:, :, np.newaxis]), axis=2), axis=2)

        r = score_pred * (1 - colors[idx_cls][0] / 255.0)
        g = score_pred * (1 - colors[idx_cls][1] / 255.0)
        b = score_pred * (1 - colors[idx_cls][2] / 255.0)
        rgb[:, :, 0] += r
        rgb[:, :, 1] += g
        rgb[:, :, 2] += b

    bdry_sum[bdry_sum == 0] = 1

    bdry_vis = np.ones((score_output.shape[0], score_output.shape[1], 3)) * 255
    bdry_vis = (bdry_vis - 255 * rgb * bdry_max[:, :, np.newaxis] / (bdry_sum[:, :, np.newaxis])).astype(np.uint8)

    if vis:
        plt.figure()
        plt.imshow(bdry_vis)
        plt.show()

    # Store value into img
    img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
    save_dir_path = os.path.join(args.output_dir, dir_name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    imwrite(os.path.join(save_dir_path, img_base_name_noext + '.png'), bdry_vis)


def convert_mat_to_numpy(label_path, h, w, cls_num=20, dataset='sbd'):
    if dataset == 'sbd':
        mat_data = loadmat(label_path)['GTinst']["Boundaries"]
    elif dataset == 'cityscapes':
        mat_data = loadmat(label_path)['labelEdge']
    else:
        raise Exception("Wrong type!")
    np_data = np.zeros((h, w, cls_num))
    for i in range(cls_num):
        cur_cls_gt = mat_data[0][0][i][0].toarray()
        np_data[:, :, i] = cur_cls_gt

    return np_data


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--dataset', type=str, default='sbd')
    parser.add_argument('-m', '--model', type=str,
                        help="path to the pytorch(.pth) containing the trained weights")
    parser.add_argument('-l', '--image_list', type=str, default='',
                        help="list of image files to be tested")
    parser.add_argument('-f', '--image_file', type=str, default='',
                        help="a single image file to be tested")
    parser.add_argument('-d', '--image_dir', type=str, default='',
                        help="root folder of the image files in the list or the single image file")
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help="folder to store the test results")

    args = parser.parse_args(sys.argv[1:])

    ### Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    check_result = False
    stnconfin = True

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
    if args.dataset == 'sbd':
        num_cls = 20
        in_channels = 43
        crop_size = [512, 512]
        color = sbd_colors
        infer_on = False
        inverse_id = False
    elif args.dataset == 'cityscapes':
        num_cls = 19
        in_channels = 41
        crop_size = [1024, 2048]
        color = cityscapes_colors
        infer_on = True
        inverse_id =True
    else:
        raise Exception('Wrong crop size!')

    model = CASENetSTN(pretrained=False, num_classes=num_cls, field_size=crop_size, mode='bilinear', cuda=True, in_chn=in_channels).cuda()
    utils.load_pretrained_model(model, args.model, print_info=False)
    # cudnn.benchmark = True
    model = model.eval()

    # Define normalization for data    
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

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

        # ##############################################################################################################
        if args.dataset == 'sbd':
            mat_path = os.path.join(test_lst[idx_img].replace("data_proc//image/test", "gt_eval/gt_orig_raw/inst"))
            mat_path = mat_path.replace("png", "mat")
            np_data = convert_mat_to_numpy(mat_path, h=height, w=width, cls_num=20)
            gt_tensor = torch.from_numpy(np_data).transpose(2, 1).transpose(1, 0).unsqueeze(0)
            padded_processed_target = F.pad(gt_tensor, (0, pad_w, 0, pad_h), "constant", 0).data.cuda().float()

        elif check_result and args.dataset == 'cityscapes':
            mat_path = os.path.join(test_lst[idx_img].replace("leftImg8bit/", "gtFine/").replace("_leftImg8bit.png", "_gtFine_edge.mat"))
            np_data = convert_mat_to_numpy(mat_path, h=height, w=width, cls_num=19, dataset=args.dataset)
            gt_tensor = torch.from_numpy(np_data).transpose(2, 1).transpose(1, 0).unsqueeze(0)
            padded_processed_target = F.pad(gt_tensor, (0, pad_w, 0, pad_h), "constant", 0).data.cuda().float()
            del gt_tensor, np_data
        else:
            padded_processed_target = torch.zeros((1, 20, 512, 512)).cuda()

        # prediction
        score_feats5, score_fuse_feats, y_fused, flow_field = model(processed_img_var, padded_processed_target.float(), conf_on=stnconfin, inference_mode=infer_on)  # 1 X 19 X CROP_SIZE X CROP_SIZE

        # get sampler
        sampler = SpaticalTransformer(size=crop_size, mode='bilinear', cuda=True)
        score_output = sigmoid(score_fuse_feats.transpose(1, 3).transpose(1, 2)).squeeze(0)[:height, :width, :]  # H X W X 19
        if not y_fused is None:
            y_fused = y_fused.transpose(1, 3).transpose(1, 2).squeeze(0)[:height, :width, :]

        # ##############################################################################################################
        if check_result:
            if args.dataset == 'sbd':
                plt.figure()
                plt.subplot(221)
                plt.imshow(np.array(img))
                plt.subplot(222)
                plt.imshow(np.max(score_output.cpu().data.squeeze().numpy()[:height, :width, :], axis=2) > 0.1)
                plt.title('score output > 0.1')
                plt.subplot(223)

                tmp_score_fuse_feats = torch.sigmoid(score_fuse_feats)
                print(tmp_score_fuse_feats.shape)
                print(flow_field.shape)
                tmp_y_fused = sampler(tmp_score_fuse_feats, flow_field)
                tmp_y_fused = tmp_y_fused.transpose(1, 3).transpose(1, 2).squeeze(0)[:height, :width, :]
                vis_y_fused = np.max(tmp_y_fused.cpu().data.squeeze().numpy()[:height, :width, :], axis=2)

                plt.imshow(vis_y_fused, cmap='gray')  # [:, :height, :width]
                plt.title('y_fused')
                plt.subplot(224)

                plt.title('y - y_fused')
                plt.imshow(np.max(score_output.cpu().data.squeeze().numpy()[:height, :width, :], axis=2), cmap='tab20')
                plt.colorbar()
                plt.show()
            elif args.dataset == 'cityscapes':
                plt.figure()
                plt.subplot(121)
                plt.imshow(np.array(img))
                plt.subplot(122)
                plt.imshow(np.max(score_output.cpu().data.squeeze().numpy()[:height, :width, :], axis=2))
                plt.show()

            if flow_field is not None:
                # visualize field
                flow_field = (flow_field.squeeze().data.cpu().numpy()[:, :height, :width]).transpose(1, 2, 0)
                x_shift = -flow_field[:, :, 0]
                y_shift = flow_field[:, :, 1]
                flow_field = np.concatenate([y_shift[:, :, np.newaxis], x_shift[:, :, np.newaxis]], axis=2)

                plt.figure()
                plt.subplot(121)
                plt.imshow(flow_field[:, :, 0])
                plt.title("flow_x")
                plt.subplot(122)
                plt.imshow(flow_field[:, :, 1])
                plt.subplot(122)
                plt.show()

                ### get clean gt
                if args.dataset == 'sbd':
                    mat_path = test_lst[idx_img].replace("png", "mat").replace("proc/image/test", "gt_eval/gt_orig_raw/inst")
                elif args.dataset == 'cityscapes':
                    mat_path = os.path.join(test_lst[idx_img].replace("jpg", "mat").replace("image/test", "gt_eval/gt_orig_raw/inst").replace("test", ""))
                else:
                    raise Exception("Wrong dataset name!")

                # visualize field
                np_data = convert_mat_to_numpy(mat_path, h=height, w=width, cls_num=num_cls)
                np_data_sc = np.max(np_data, axis=2)
                flow_field = flow_field * np_data_sc[:, :, np.newaxis]
                ne.plot.flow([flow_field[:, :, :]], width=10)

        ### save results
        save_cls(score_output, num_cls, dir_prefix='class_', inverse_id=inverse_id)
        save_colored_pred(score_output, dir_name='pred_color', inverse_id=inverse_id)
        # save_cls(y_fused, num_cls, dir_prefix='class_')
        save_colored_pred(y_fused, dir_name='pred_color_y')

        del score_feats5
        del score_fuse_feats
        del score_output
        del padded_processed_img
        del processed_img
        del processed_img_var
        torch.cuda.empty_cache()

    print('Done!')
