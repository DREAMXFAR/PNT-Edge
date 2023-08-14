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

from dataloader.cityscapes_data import CityscapesData, CityscapesDataWithCanny

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
            img = torch.from_numpy(pic).permute(2, 0, 1)  # .contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        return img.float().div(255) if self.div else img.float()


def get_dataloader(args):
    # Define data files path.
    root_img_folder = "../seal-master/data/cityscapes-preprocess/data_proc/"
    root_label_folder = "../seal-master/data/cityscapes-preprocess/data_proc/"
    train_anno_txt = "../seal-master/data/cityscapes-preprocess/data_proc/txt_list/train.txt"
    val_anno_txt = "../seal-master/data/cityscapes-preprocess/data_proc/txt_list/val_pairs.txt"

    input_size = 472  # 472 for sbd and cityscapes in seal
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ])
    train_label_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        input_size,
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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        input_size,
        cls_num=args.cls_num,
        img_transform=transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        RGB2BGR(roll=True),
                        ToTorchFormatTensor(div=False),
                        normalize,
                        ]),
        label_transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader


def get_my_dataloader(args, thingt, batch_size=None):
    # Define data files path.
    root_img_folder = "../seal-master/data/cityscapes-preprocess/data_proc/"
    root_label_folder = "../seal-master/data/cityscapes-preprocess/data_proc/"
    train_anno_txt = "../seal-master/data/cityscapes-preprocess/data_proc/txt_list/train.txt"
    val_anno_txt = "../seal-master/data/cityscapes-preprocess/data_proc/txt_list/val_pairs.txt"

    # 472 for sbd and cityscapes in seal
    input_size = 472
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

    train_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ])
    train_label_augmentation = transforms.Compose([
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = CityscapesDataWithCanny(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        input_size,
        thinpb=thingt,
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

    if batch_size is None:
        batch_size = args.batch_size
    else:
        batch_size = batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  # by xwj, set false to debug
        num_workers=args.workers, pin_memory=True)

    val_loader = None

    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader.sbd_data import same_seeds

    np.random.seed(55)
    same_seeds(11)

    args = config.get_args()
    args.num_cls = 19
    args.batch_size = 1
    train_loader, val_loader = get_my_dataloader(args, thingt=False)
    for i, (img, target, ref_gt) in enumerate(train_loader):
        print("image.size():{0}".format(img.size()))
        print("target.size():{0}".format(target.size()))
        print("ref_gt.size():{0}".format(ref_gt.size()))

        plt.figure()
        plt.subplot(131)
        plt.imshow(np.transpose(img.squeeze().numpy()/255.0, [1, 2, 0]))
        plt.subplot(132)
        plt.imshow(target.squeeze().numpy()[:, :, 0])
        plt.subplot(133)
        plt.imshow(np.max(ref_gt.squeeze().numpy()[:, :, :], axis=0))
        plt.show()

        if i > 0:
            break
