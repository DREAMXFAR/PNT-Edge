# -*- coding:utf-8 -*-
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import glob


def draw_overlap(img, anno, color=(0, 1, 0)):
    draw_img = np.array(img).copy()  # (h, w, 3)
    template = np.stack((anno * color[0], anno * color[1], anno * color[2]), axis=2).astype(np.float)
    draw_img = np.clip((draw_img + template), 0, 255) / 255
    return draw_img


def gen_sbd_canny():
    # root path
    image_root = r'./sbd-preprocess/data_proc/image/train/scale_1'
    save_root = r'./sbd-preprocess/data_proc/my_labels/canny/train'

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    # images name list
    img_list = sorted(os.listdir(image_root))

    for scale_factor in [0.5, 0.75, 1, 1.25, 1.5]:
        save_dir_path = os.path.join(save_root, 'scale_{}'.format(scale_factor))
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)

        for idx, aimg_name in tqdm(enumerate(img_list)):
            # read image
            aimg_path = os.path.join(image_root, aimg_name)

            tmp_img = cv2.imread(aimg_path, 0)
            height, width = tmp_img.shape
            # resize
            resized_w = int(np.ceil(width * scale_factor))
            resized_h = int(np.ceil(height * scale_factor))

            tmp_img = cv2.resize(tmp_img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

            tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 1.5)
            edges = cv2.Canny(tmp_img, 25, 150, apertureSize=3, L2gradient=True)

            save_path = os.path.join(save_dir_path, aimg_name)
            cv2.imwrite(save_path, edges)

            # if idx >= 0:
            #     break


def gen_cityscapes_canny():
    # root path
    dir_name = "train"
    image_root = r'./seal-master/data/cityscapes-preprocess/data_proc/leftImg8bit/{}'.format(dir_name)
    save_root = r'./seal-master/data/cityscapes-preprocess/data_proc/canny/{}'.format(dir_name)

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for acity in os.listdir(image_root):
        save_dir_path = os.path.join(save_root, acity)
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)

        # images name list
        img_list = sorted(os.listdir(os.path.join(image_root, acity)))  # [:-1]

        for idx, aimg_name in tqdm(enumerate(img_list)):
            # read image
            aimg_path = os.path.join(image_root, acity, aimg_name)

            tmp_img = cv2.imread(aimg_path, 0)
            height, width = tmp_img.shape
            # resize
            resized_w = int(np.ceil(width * 1))
            resized_h = int(np.ceil(height * 1))

            tmp_img = cv2.resize(tmp_img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

            tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 2.0)
            edges = cv2.Canny(tmp_img, 10, 100, apertureSize=3, L2gradient=True)

            save_path = os.path.join(save_dir_path, aimg_name)
            cv2.imwrite(save_path, edges)


if __name__ == "__main__":
    # sbd
    gen_sbd_canny()

    # cityscapes
    gen_cityscapes_canny()



