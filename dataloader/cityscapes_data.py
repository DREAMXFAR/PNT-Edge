import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import scipy.io as sio
from myCode.edge_process.bwmorph import bwmorph_thin

global_seed = 2147483647


# fix random seed
def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():  # GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def convert_mat_to_numpy(label_path, h, w, cls_num=19):
    mat_data = sio.loadmat(label_path)['labelEdge']
    np_data = np.zeros((h, w, cls_num))
    for i in range(cls_num):
        cur_cls_gt = mat_data[i][0].toarray()
        np_data[:, :, i] = cur_cls_gt

    return np_data


class CityscapesData(data.Dataset):
    def __init__(self, img_folder, label_folder, anno_txt, input_size, cls_num, img_transform, label_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.input_size = input_size
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        self.idx2name_dict = {}
        self.ids = []
        f = open(anno_txt, 'r')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0][1:]  # add [1:] by xwj
            label_name = row_data[1][1:]  # add [1:] by xwj
            self.idx2name_dict[cnt] = {}
            self.idx2name_dict[cnt]['img'] = img_name
            self.idx2name_dict[cnt]['label'] = label_name
            self.ids.append(cnt)
            cnt += 1

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, label_name)

        # Set the same random seed for img and label transform
        seed = np.random.randint(global_seed)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB')  # W X H
        w, h = img.size

        same_seeds(seed)  # to keep same transformation for image and label
        processed_img = self.img_transform(img)  # 3 X H X W

        np_data = convert_mat_to_numpy(label_path, h, w, cls_num=19)

        label_data = []
        num_cls = np_data.shape[2]
        for k in range(num_cls):
            if np_data[:, :, num_cls-1-k].sum() > 0:  # The order is reversed to be consistent with class name idx in official.
                # Before transform, set random seed same as img transform, to keep consistent!
                same_seeds(seed)  # add by xwj
                label_tensor = self.label_transform(torch.from_numpy(np_data[:, :, num_cls-1-k]).unsqueeze(0).float())
            else:  # ALL zeros, don't need transform, maybe a bit faster?..
                label_tensor = torch.zeros(1, self.input_size, self.input_size).float()
            label_data.append(label_tensor.squeeze(0).long())

        label_data = torch.stack(label_data).transpose(0, 1).transpose(1, 2)  # N X H X W -> H X W X N

        return processed_img, label_data, 0
        # processed_img: 3 X 472(H) X 472(W)
        # label tensor: 472(H) X 472(W) X 19

    def __len__(self):
        return len(self.ids)


class CityscapesDataWithCanny(data.Dataset):
    def __init__(self, img_folder, label_folder, anno_txt, input_size, cls_num, img_transform, label_transform, thinpb=False):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.canny_folder = r'./seal-master/data/cityscapes-preprocess/data_proc/canny/'
        self.input_size = input_size
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.thinpb = thinpb

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        self.idx2name_dict = {}
        self.ids = []
        f = open(anno_txt, 'r')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0][1:]  # add [1:] by xwj
            label_name = row_data[1][1:]  # add [1:] by xwj
            self.idx2name_dict[cnt] = {}
            self.idx2name_dict[cnt]['img'] = img_name
            self.idx2name_dict[cnt]['label'] = label_name
            self.idx2name_dict[cnt]['canny'] = label_name.replace('gtFine/', self.canny_folder).replace('mat', 'png').replace("_gtFine_edge", "_leftImg8bit")
            self.ids.append(cnt)

            cnt += 1

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, label_name)
        # print("==>", img_name)

        # Set the same random seed for img and label transform
        seed = np.random.randint(global_seed)  # 2147483647

        # Load img into tensor
        img = Image.open(img_path).convert('RGB')  # W X H
        w, h = img.size

        same_seeds(seed)  # add by xwj
        processed_img = self.img_transform(img)  # 3 X H X W

        # ### from .mat
        np_data = convert_mat_to_numpy(label_path, h, w, cls_num=19)

        label_data = []
        num_cls = np_data.shape[2]
        for k in range(num_cls):
            # Before transform, set random seed same as img transform, to keep consistent!
            same_seeds(seed)
            if np_data[:, :, k].sum() > 0:  # The order is consistent with class name idx in official. n
                if self.thinpb:
                    # cur_data = erosion(np_data[:, :, k], square(3))
                    cur_data = bwmorph_thin(np_data[:, :, k])
                else:
                    cur_data = np_data[:, :, k]
                label_tensor = self.label_transform(torch.from_numpy(cur_data).unsqueeze(0).float())
            else:
                label_tensor = torch.zeros(1, w, h).float()
                label_tensor = self.label_transform(label_tensor)

            label_data.append(label_tensor.squeeze(0).float())

        # del label_tensor
        label_data = torch.stack(label_data).transpose(0, 1).transpose(1, 2)  # N X H X W -> H X W X N

        # generate canny gt online
        canny_name = self.idx2name_dict[index]['canny']
        canny_gt = np.array(Image.open(canny_name).convert('L'))
        canny_data = []
        for k in range(num_cls):
            same_seeds(seed)
            if np_data[:, :, k].sum() > 0:
                canny_gt_tensor = self.label_transform(canny_gt)
            else:
                canny_gt_tensor = torch.zeros(1, w, h).float()
                canny_gt_tensor = self.label_transform(canny_gt_tensor)

            canny_data.append(canny_gt_tensor.squeeze(0).float())

        canny_data = torch.stack(canny_data)  # .transpose(0, 1).transpose(1, 2)

        return processed_img, label_data, canny_data
        # processed_img: 3 X 472(H) X 472(W)
        # label tensor: 472(H) X 472(W) X 20
        # canny data: 20 x 472 x 472

    def __len__(self):
        return len(self.ids)
