import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as nnf
import math
import torchvision.models as models
import sys
sys.path.append("../")

import numpy as np
import utils.utils as utils
import os

from modules.CASENet import CASENet_resnet101, gen_mapping_layer_name

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

import matplotlib.pyplot as plt


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel_size=7, sigma=2, k=3, padding=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k = k
        # kernel = np.array([[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
        #                     [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        #                     [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
        #                     [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        #                     [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]])

        kernel = self.creat_gauss_kernel(kernel_size=self.kernel_size, sigma=self.sigma, k=self.k)

        kernel = np.repeat(kernel[np.newaxis, np.newaxis, :, :], self.channels, axis=0)
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        return x

    def creat_gauss_kernel(self, kernel_size=3, sigma=1, k=1):
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return gauss


class LocalizationNet(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(LocalizationNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        BN_momentum = 0.1
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)
        channel_num = 32  # 64

        self.ConvEn11 = nn.Conv2d(self.in_chn, channel_num, kernel_size=(3, 3), padding=(1, 1))
        self.BNEn11 = nn.BatchNorm2d(channel_num, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(channel_num, channel_num, kernel_size=(3, 3), padding=(1, 1))
        self.BNEn12 = nn.BatchNorm2d(channel_num, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(channel_num, channel_num*2, kernel_size=(3, 3), padding=(1, 1))
        self.BNEn21 = nn.BatchNorm2d(channel_num*2, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(channel_num*2, channel_num*2, kernel_size=(3, 3), padding=(1, 1))
        self.BNEn22 = nn.BatchNorm2d(channel_num*2, momentum=BN_momentum)

        self.MaxDe = nn.ConvTranspose2d(channel_num*2, channel_num*2, (4, 4), stride=(2, 2))

        self.ConvDe12 = nn.Conv2d(channel_num*2, channel_num, kernel_size=(3, 3), padding=(1, 1))
        self.BNDe12 = nn.BatchNorm2d(channel_num, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(channel_num, self.out_chn, kernel_size=(3, 3), padding=(1, 1))
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):
        size0 = x.size()
        x = nnf.relu(self.BNEn11(self.ConvEn11(x)))
        x = nnf.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        # Stage 2
        x = nnf.relu(self.BNEn21(self.ConvEn21(x)))
        x = nnf.relu(self.BNEn22(self.ConvEn22(x)))

        # DECODE LAYERS
        # Stage 1
        offset = 1
        x = self.MaxDe(x)
        x = x[:, :, offset:-offset, offset:-offset]
        x = nnf.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        return x


class SpaticalTransformer(nn.Module):
    def __init__(self, size, mode='bilinear', cuda=False, gauss_blur=False):
        super(SpaticalTransformer, self).__init__()
        # interpolation mode
        self.mode = mode
        self.cuda = cuda

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        self.grid = torch.unsqueeze(grid, 0).float()

        self.gauss_blur = gauss_blur
        if self.gauss_blur:
            self.gauss = GaussianBlurConv(channels=20)

        if self.cuda:
            self.grid = self.grid.cuda()

    def forward(self, x, flow):
        # grid shape: [2, 472. 472]
        # flow shape: [2, 472. 472]
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] before resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)  # [1, c, w, h] -> [1, h, w, c]
        new_locs = new_locs[..., [1, 0]]  # change x and y

        # sample from x to sampled_x, if delta_x > 0, sampled_x will move to left, delta_y > 0, sample_x will move to up
        if self.gauss_blur:
            x = self.gauss(x)

        sampled_x = nnf.grid_sample(x, new_locs, align_corners=True, mode=self.mode)
        return sampled_x


class STN(nn.Module):
    def __init__(self, field_size=(472, 472), mode='bilinear', in_chn=23, cuda=False):
        super(STN, self).__init__()

        self.localization = LocalizationNet(in_chn=in_chn, out_chn=2)
        self.spatial_transformer = SpaticalTransformer(size=field_size, mode=mode, cuda=cuda)

        for m in self.localization.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, score_fused, target=None, conf_pred=False):
        # preprocessing
        x_norm = (x - x.min()) / (x.max() - x.min())
        sigmoid_pred = torch.sigmoid(score_fused)

        if conf_pred:
            sigmoid_pred = (sigmoid_pred > 0.1).float()  # 0.05

        # concat input
        if target is None:
            x_with_pred = torch.cat([x_norm, sigmoid_pred], dim=1)
        else:
            x_with_pred = torch.cat([x_norm, sigmoid_pred, target], dim=1)

        flow_field = self.localization(x_with_pred)
        # narrow thr output into [0, 20]
        flow_field = 10 * (torch.sigmoid(flow_field) - 0.5) / 0.5   # 10 for sbd  # 3 for cityscapes

        # return flow_field, flow_field
        y_score_fused = self.spatial_transformer(sigmoid_pred, flow_field)
        return y_score_fused, flow_field


class CASENetSTN(nn.Module):
    def __init__(self, pretrained=False, num_classes=20, field_size=(472, 472), in_chn=23, mode='bilinear', cuda=False, detach_x=True):
        super(CASENetSTN, self).__init__()
        self.detach_x = detach_x

        # backbone network
        self.casenet = CASENet_resnet101(pretrained, num_classes)
        self.stn = STN(field_size=field_size, in_chn=in_chn, mode=mode, cuda=cuda)

        ### initialization
        for m in self.stn.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # print(self.localization[4].weight.mean())

    def forward(self, x, target=None, conf_on=False, inference_mode=False):
        # get original prediction
        score_feats5, score_fused = self.casenet(x)  # not with sigmoid, x is the input image
        if inference_mode:
            return score_feats5, score_fused, None, None

        if self.detach_x:
            y_score_fused, flow_field = self.stn(x.detach(), score_fused.detach(), target, conf_pred=conf_on)
        else:
            y_score_fused, flow_field = self.stn(x, score_fused, target, conf_pred=conf_on)

        return score_feats5, score_fused, y_score_fused, flow_field


if __name__ == "__main__":
    import random
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    input_data = torch.rand(1, 3, 472, 472)
    input_var = Variable(input_data)

    ### compute flops
    from ptflops.flops_counter import get_model_complexity_info

    model = CASENetSTN(pretrained=False, num_classes=20, field_size=(472, 472), mode='bilinear', cuda=False, detach_x=True, in_chn=43)
    flops, params = get_model_complexity_info(model, (3, 472, 472), as_strings=True, print_per_layer_stat=True)

    print('#' * 20)
    print('Flops: {}'.format(flops))
    print('Params: {}'.format(params))






