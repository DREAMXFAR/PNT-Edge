import os
import numpy as np
import time
import PIL
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms as transforms

import sys
sys.path.append("../")

from dataloader.sbd_data import SBDTrainData, SBDValData, SBDTrainDataWithCanny

import config


class RGB2BGR(object):
    """
    Since we use pretrained model from Caffe, need to be consistent with Caffe model.
    Transform RGB to BGR.
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        if img.mode == 'L':
            return np.concatenate([np.expand_dims(img, 2)], axis=2) 
        elif img.mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(img)[:, :, ::-1]], axis=2)
            else:
                return np.concatenate([np.array(img)], axis=2)


class ToTorchFormatTensor(object):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] or [0, 255]. 
    """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        return img.float().div(255) if self.div else img.float()


def get_dataloader(args):
    # Define data files path.
    ### orig trainval
    root_img_folder = r"./example/"
    root_label_folder = r"./example/"
    train_anno_txt = r"./example/debug_single.txt"
    val_anno_txt = r"./example/debug_test.txt"

    # 472 for sbd and cityscapes in seal
    input_size = 472  # 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ])
    train_label_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = SBDTrainData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        input_size,
        thinpb=args.thingt,
        cls_num=args.cls_num,
        img_transform=transforms.Compose([
                        train_augmentation,
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform=transforms.Compose([
                        transforms.ToPILImage(),
                        train_label_augmentation,
                        transforms.ToTensor(),
                        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  # by xwj, set false to debug
        num_workers=args.workers, pin_memory=True)

    # ##################################################################################################################
    val_crop_size = 512  # in sbd, height <= 500, width <= 500

    val_augmentation = transforms.CenterCrop(val_crop_size)
    val_label_augmentation = transforms.CenterCrop(val_crop_size)

    val_dataset = SBDValData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        input_size,
        cls_num=args.cls_num,
        img_transform=transforms.Compose([
                        val_augmentation,
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform=transforms.Compose([
                        transforms.ToPILImage(),
                        val_label_augmentation,
                        transforms.ToTensor(),
                        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1), shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_my_dataloader(args, thingt, batch_size=None):
    ### orig trainval
    root_img_folder = r"./example/"
    root_label_folder = r"./example/"
    train_anno_txt = "./example/debug_single.txt"
    val_anno_txt = "./example/debug_test.txt"

    # 472 for sbd and cityscapes in seal
    input_size = 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ])
    train_label_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size), pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = SBDTrainDataWithCanny(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        input_size,
        thinpb=thingt,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        train_augmentation,
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform=transforms.Compose([
                        transforms.ToPILImage(),
                        train_label_augmentation,
                        transforms.ToTensor(),
                        ]))

    if batch_size is None:
        batch_size = args.batch_size
    else:
        batch_size = batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  # by xwj, set false to debug
        num_workers=args.workers, pin_memory=True)

    val_loader = None

    return train_loader, val_loader


def center_crop(img, orig_size):
    (cur_w, cur_h, _) = img.shape
    orig_w, orig_h = orig_size
    cropped_img = img[(cur_h - orig_h)//2: (cur_h + orig_h)//2, (cur_w - orig_w)//2: (cur_w + orig_w)//2, :]

    assert cropped_img.shape[:-1] == (orig_h, orig_w)
    return cropped_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import cv2
    from dataloader.sbd_data import same_seeds
    from myCode.edge_process.bwmorph import bwmorph_thin

    np.random.seed(55)
    same_seeds(11)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    args = config.get_args()
    args.batch_size = 1
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target, canny_gt) in enumerate(train_loader):
        print("target.size: {} -- image.size: {} -- canny_gt.size: {}".format(target.size(), img.size(), canny_gt.size()))
        np_img = np.transpose(img.squeeze().numpy()/255.0, [1, 2, 0])
        # cropped_img = center_crop(np_img, target["orig_size"])

        plt.figure()
        plt.subplot(131)
        plt.imshow(np_img)

        plt.subplot(132)
        plt.imshow(np.max(target.squeeze().numpy()[:, :, :], axis=2))

        plt.subplot(133)
        target_raw = np.max(target.squeeze().numpy()[:, :, :], axis=2)
        target_thin = bwmorph_thin(target_raw, None)
        print(np.unique(target_thin))
        plt.imshow(target_thin)
        plt.title('thin')
        plt.show()

        if i > -1:
            break
