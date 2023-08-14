import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import disk, dilation, square
import re

import torch
from torch import sigmoid
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as nnf
from torch.autograd import Variable

from train_val.losses import Grad2D, MSE, Sparse2D
from modules.CASENet_with_STN import SpaticalTransformer
from myCode.edge_process.distmap import gen_shift, min_dist_mapping, min_dist_mapping_tflow

import neurite as ne

import sys
sys.path.append("../")

# Local imports
import utils.utils as utils
from utils.utils import AverageMeter


def get_model_policy(model, mode='casenet'):
    if mode == 'casenet':
        lr_list = [10, 20, 0, 0, 1]

        score_feats_conv_weight = []
        score_feats_conv_bias = []
        other_pts = []

        for m in model.casenet.named_modules():
            if m[0] != '' and m[0] != 'module':
                # print(mode, ' ==> ', m[0])
                if re.search('^casenet$', m[0]) is not None or re.search('^stn.localization$', m[0]) is not None or re.search('^stn$', m[0]) is not None:
                    continue
                elif ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                    ps = list(m[1].parameters())
                    score_feats_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        score_feats_conv_bias.append(ps[1])
                    print("Totally new layer:{0}".format(m[0]))
                else:  # For all the other module that is not totally new layer.
                    ps = list(m[1].parameters())
                    other_pts.extend(ps)
        return [
            {'params': score_feats_conv_weight, 'lr_mult': lr_list[0], 'name': 'score_conv_weight'},  # 10
            {'params': score_feats_conv_bias, 'lr_mult': lr_list[1], 'name': 'score_conv_bias'},  # 20
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': lr_list[4], 'name': 'other'},  # 1
        ]

    elif mode == 'stn':
        lr_list = [0, 0, 10, 20, 1]

        localization_conv_weight = []
        localization_conv_bias = []
        other_pts = []

        for m in model.stn.named_modules():
            if m[0] != '' and m[0] != 'module':
                # print(mode, ' ==> ', m[0])
                if re.search('^casenet$', m[0]) is not None or re.search('^localization$', m[0]) is not None or re.search('^stn$', m[0]) is not None:
                    continue
                elif ('localization.' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                    ps = list(m[1].parameters())
                    localization_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        localization_conv_bias.append(ps[1])
                    print("Totally new layer:{0}".format(m[0]))
                else:  # For all the other module that is not totally new layer.
                    ps = list(m[1].parameters())
                    other_pts.extend(ps)

        return [
            {'params': localization_conv_weight, 'lr_mult': lr_list[2], 'name': 'localization_conv_weight'},  # 0
            {'params': localization_conv_bias, 'lr_mult': lr_list[3], 'name': 'localization_conv_bias'},  # 0
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': lr_list[4], 'name': 'other'},  # 1
        ]
    else:
        raise Exception('Wrong mode!')


def estimate_local_complexity(conf_samples, ref_gt):
    ref_gt = torch.max(ref_gt, dim=1)[0]
    mean_filter = nn.AvgPool2d(kernel_size=(15, 15), stride=1, padding=(7, 7))
    C = mean_filter(ref_gt) * (conf_samples > 0)
    C_norm = C / (C.max() + 1e-10)

    return C_norm


def generate_dist_map(target, flow_field):
    D = flow_field[:, 0, :, :].square() + flow_field[:, 1, :, :].square()
    D = D * target
    D_norm = D / (D.max() + 1e-10)

    return D_norm


def train(args, train_loader, model, optimizer, epoch, curr_lr, global_step,
          accumulation_steps=1, distributed=False, mode='casenet'):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    unmapped_feats5_losses = AverageMeter()
    unmapped_fusion_losses = AverageMeter()

    smooth_losses = AverageMeter()
    mse_losses = AverageMeter()
    spvsd_flow_losses = AverageMeter()
    unspvsd_flow_losses = AverageMeter()
    sparse_losses = AverageMeter()
    reg_losses = AverageMeter()

    total_losses = AverageMeter()

    # switch to eval mode to make BN unchanged.
    if mode == 'casenet':
        model.casenet.eval()  # this is because the STN is trained with this.
        model.stn.eval()
    elif mode == 'stn':
        model.casenet.eval()
        model.stn.train()
    else:
        raise Exception("Wrong mode!!!")

    # initialize optimizer
    optimizer.zero_grad()

    # loss function
    loss_function = nnf.binary_cross_entropy  # _with_logits
    coefficient = 1.0 / (args.batch_size * accumulation_steps)

    end = time.time()

    for i, (img, target, refer_gt) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input for Image CNN.
        img_var = utils.check_gpu(0, img)  # BS X 3 X H X W
        b, c, w, h = img_var.shape

        target_var = utils.check_gpu(0, target)  # BS X H X W X NUM_CLASSES
        target_var = target_var.transpose(1, 3).transpose(2, 3)  # BS x NUM_CLASSES x H x W

        # the official model have different output channel according to different classes
        if not "pretrained_pytorch" in args.pretrained_casenet:
            target_var = torch.flip(target_var, dims=[1])
            refer_gt = torch.flip(refer_gt, dims=[1])

        bs = img.size()[0] * accumulation_steps

        ### compute weighted CE
        if args.weighted_loss:
            cur_weight = edge_weight(target_var, mode=args.loss_mode)
        else:
            cur_weight = None

        ### forward
        score_feats5, fused_feats, _, flow_field = model(img_var, target_var, conf_on=args.stn_confin)
        score_feats5_sigmoid = torch.sigmoid(score_feats5).clamp(min=1e-10, max=1)
        fused_feats_sigmoid = torch.sigmoid(fused_feats).clamp(min=1e-10, max=1)

        ### visualize field gt
        visualize_flow = False
        if visualize_flow:
            vis_flow = (flow_field.squeeze().data.cpu().numpy()).transpose(1, 2, 0)
            x_shift = -vis_flow[:, :, 0]
            y_shift = vis_flow[:, :, 1]
            vis_flow = np.concatenate([y_shift[:, :, np.newaxis], x_shift[:, :, np.newaxis]], axis=2)
            # valid field
            valid_field = ((f(target_var)).squeeze()[:, :, np.newaxis]) * vis_flow
            ne.plot.flow([valid_field[50:200, 100:300, :]], width=10)

        ### sampler functions
        sampler = SpaticalTransformer(size=(472, 472), mode='bilinear', cuda=True)

        if mode == 'casenet':
            y_score_feats5_sigmoid = sampler(score_feats5_sigmoid, flow_field)
            y_fused_feats_sigmoid = sampler(fused_feats_sigmoid, flow_field)

            unmapped_x = gen_unmapped_flow(flow_field, b, w, h, cls_num=args.cls_num, vis=False)
            zero_map = torch.zeros_like(unmapped_x).float().cuda()

            lambda_1 = 1  # 0.1
            unmapped_feats5_loss = lambda_1 * coefficient * loss_function(score_feats5_sigmoid.float() * (1 - unmapped_x), zero_map.float(), weight=cur_weight, reduction="sum")
            unmapped_fused_feats_loss = lambda_1 * coefficient * loss_function(fused_feats_sigmoid.float() * (1 - unmapped_x), zero_map.float(), weight=cur_weight, reduction="sum")

            ### compute loss
            feats5_loss = coefficient * loss_function(y_score_feats5_sigmoid.float(), target_var.float(), weight=cur_weight, reduction="sum")
            fused_feats_loss = coefficient * loss_function(y_fused_feats_sigmoid.float(), target_var.float(), weight=cur_weight, reduction="sum")

            loss = feats5_loss + fused_feats_loss + unmapped_feats5_loss + unmapped_fused_feats_loss

        elif mode == 'stn':
            confident_pred = fused_feats_sigmoid.detach().float() > 0.1  # 0.05

            ### local regularization
            refer_gt_var = utils.check_gpu(0, refer_gt)
            sc_confident_pred = torch.max(confident_pred, dim=1, keepdim=True)[0]
            sc_target_var = torch.max(target_var, dim=1, keepdim=True)[0]
            C_norm = estimate_local_complexity(sc_confident_pred, refer_gt_var)  # local complexity estimation
            y_C_norm = sampler(C_norm, flow_field)
            D_norm = generate_dist_map(sc_target_var, flow_field)  # compute shift-dist D, and normalize

            ### compute gt_flow and mapped target for confident pred
            mapped_target_mask, flow_gt = gen_mapped_target_with_flow(confident_pred, target_var, vis=False)  # flow defined on target image

            ### y_target as the target of prediction
            y_pred = sampler(fused_feats_sigmoid.detach().float(), flow_field)  # for predoverthr

            ### loss computation
            ## 1. sparcity loss
            sparse_loss = torch.tensor(0).cuda()
            ## 2. smooth loss
            smooth_loss = torch.tensor(0).cuda()
            ## 3. similarity loss
            # supervised part
            # spvsd_flow_loss = torch.tensor(0).cuda()
            spvsd_flow_loss = coefficient * MSE().loss(flow_field * mapped_target_mask, flow_gt * mapped_target_mask, mode='sum') * 0.01
            # unsupervised part
            # unspvsd_flow_loss = coefficient * MSE().loss(y_pred, target_var.float(), mode='sum')
            unspvsd_flow_loss = coefficient * loss_function(y_pred.float(), target_var.float(), weight=None, reduction='sum')
            mse_loss = spvsd_flow_loss + unspvsd_flow_loss
            ## 4. regularization loss
            reg_loss = coefficient * nn.MSELoss(reduction="sum")(D_norm.float().clamp(min=1e-10, max=0.99), y_C_norm.float().clamp(min=1e-10, max=0.99)) * 3

            loss = smooth_loss + mse_loss + sparse_loss + reg_loss
        else:
            raise Exception('Wrong mode!')

        # loss backward
        loss.backward()

        # clear memory
        del img_var
        del target_var
        del score_feats5
        del fused_feats
        del flow_field
        torch.cuda.empty_cache()

        # increase batch size by factor of accumulation steps (Gradient accumulation) for training with limited memory
        if (i + 1) % accumulation_steps == 0:
            if mode == 'casenet':
                feats5_losses.update(feats5_loss.data, bs)
                fusion_losses.update(fused_feats_loss.data, bs)
                unmapped_feats5_losses.update(unmapped_feats5_loss.data, bs)
                unmapped_fusion_losses.update(unmapped_fused_feats_loss.data, bs)

            elif mode == 'stn':
                smooth_losses.update(smooth_loss.data, bs)
                mse_losses.update(mse_loss.data, bs)
                spvsd_flow_losses.update(spvsd_flow_loss.data, bs)
                unspvsd_flow_losses.update(unspvsd_flow_loss.data, bs)
                sparse_losses.update(sparse_loss.data, bs)
                reg_losses.update(reg_loss.data, bs)

            total_losses.update(loss.data, bs)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                if mode == 'casenet':
                    print('Epoch: [casenet] [{0}][{1}/{2}] \t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Total Loss {total_loss.val:.3f} ({total_loss.avg:.3f})\t'
                          'feat5_loss {feats5_losses.val:.3f} ({feats5_losses.avg:.3f})\t'
                          'fused_loss {fusion_losses.val:.3f} ({fusion_losses.avg:.3f})\t'
                          'unmapped_feat5_loss {unmapped_feats5_losses.val:.3f} ({unmapped_feats5_losses.avg:.3f})\t'
                          'unmapped_fused_loss {unmapped_fusion_losses.val:.3f} ({unmapped_fusion_losses.avg:.3f})\t'
                          'lr {learning_rate:.10f}\t'
                          .format(epoch, int((i + 1) / accumulation_steps), int(len(train_loader) / accumulation_steps),
                                  batch_time=batch_time,
                                  data_time=data_time, total_loss=total_losses, feats5_losses=feats5_losses,
                                  unmapped_feats5_losses=unmapped_feats5_losses, unmapped_fusion_losses=unmapped_fusion_losses,
                                  fusion_losses=fusion_losses, learning_rate=curr_lr))

                elif mode == 'stn':
                    print('Epoch: [stn] [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Total Loss {total_loss.val:.3f} ({total_loss.avg:.3f})\t'
                          'smooth_loss {smooth_losses.val:.8f} ({smooth_losses.avg:.8f})\t'
                          'mse_loss {mse_losses.val:.8f} ({mse_losses.avg:.8f})\t'
                          'spvsd_loss {spvsd_flow_losses.val:.8f} ({spvsd_flow_losses.avg:.8f})\t'
                          'unspvsd_loss {unspvsd_flow_losses.val:.8f} ({unspvsd_flow_losses.avg:.8f})\t'
                          'sparse_loss {sparse_losses.val:.8f} ({sparse_losses.avg:.8f})\t'
                          'reg_loss {reg_losses.val:.11f} ({reg_losses.avg:.11f})\t'
                          'lr {learning_rate:.10f}\t'
                          .format(epoch, int((i + 1) / accumulation_steps), int(len(train_loader) / accumulation_steps),
                                  batch_time=batch_time,
                                  data_time=data_time, total_loss=total_losses, smooth_losses=smooth_losses, mse_losses=mse_losses,
                                  spvsd_flow_losses=spvsd_flow_losses, unspvsd_flow_losses=unspvsd_flow_losses,
                                  sparse_losses=sparse_losses, reg_losses=reg_losses, learning_rate=curr_lr))

        # debug
        break

    # clear memory
    del cur_weight
    if mode == 'casenet':
        del feats5_loss
        del fused_feats_loss
        del feats5_losses
        del fusion_losses
        del total_losses
    elif mode == 'stn':
        del smooth_loss
        del smooth_losses
        del mse_loss
        del mse_losses
        del spvsd_flow_loss
        del spvsd_flow_losses
        del unspvsd_flow_loss
        del unspvsd_flow_losses
        del sparse_loss
        del sparse_losses
        del reg_loss
        del reg_losses
        del total_losses

    torch.cuda.empty_cache()

    return global_step


def vis_field(flow_field):
    # visualize field
    flow_gt = (flow_field.squeeze().data.cpu().numpy()).transpose(1, 2, 0)
    x_shift = flow_gt[:, :, 0]
    y_shift = flow_gt[:, :, 1]
    flow_gt = np.concatenate([y_shift[:, :, np.newaxis], -x_shift[:, :, np.newaxis]], axis=2)  # add minus for vis
    ne.plot.flow([flow_gt], width=10)


def gen_unmapped_flow(flow, b, w, h, cls_num=20, vis=False):
    x_map = torch.zeros([b, 1, w, h]).cuda().float()

    for i in range(b):
        flow_x = (flow[i, 0, :, :] + 0.5).int()
        flow_y = (flow[i, 1, :, :] + 0.5).int()

        grid_x, grid_y = torch.meshgrid([torch.arange(0, w), torch.arange(0, h)])
        mapped_y = grid_y.unsqueeze(0).cuda() + flow_y
        mapped_y = mapped_y.clamp(min=0, max=(w-1))
        mapped_x = grid_x.unsqueeze(0).cuda() + flow_x
        mapped_x = mapped_x.clamp(min=0, max=(h-1))

        x_map[i, 0, mapped_x, mapped_y] = 1

    if vis:
        plt.figure()
        plt.imshow(x_map[0, :, :, :].cpu().squeeze().numpy(), cmap='gray')
        plt.show()

    x_map = x_map.repeat(1, cls_num, 1, 1)

    return x_map


def gen_mapped_target_with_flow(conf_pred, target, vis=False):
    # preprocessing
    single_channel_confident_pred = torch.max(conf_pred, dim=1)[0].cpu().squeeze().numpy().astype(np.uint8)
    if len(single_channel_confident_pred.shape) == 2:
        single_channel_confident_pred = torch.max(conf_pred, dim=1)[0].cpu().squeeze().unsqueeze(0).numpy().astype(np.uint8)

    tmp_target = torch.max(target.float(), dim=1)[0].data.cpu().numpy().astype(np.uint8)

    flow_gt_list = []
    mapped_target_list = []
    for c in range(target.size()[0]):
        mapped_target, x_shift, y_shift = min_dist_mapping_tflow(single_channel_confident_pred[c, :, :], tmp_target[c, :, :])
        if vis:
            plt.figure()
            plt.subplot(131)
            plt.imshow(single_channel_confident_pred[0, :, :])
            plt.title('y')
            plt.subplot(132)
            plt.imshow(mapped_target)
            plt.title('gt')
            plt.subplot(133)
            plt.imshow(single_channel_confident_pred[0, :, :] - mapped_target)
            plt.title('y - gt')
            plt.show()
        cur_mapped_target = torch.from_numpy(mapped_target[np.newaxis, np.newaxis, :, :]).cuda()
        mapped_target_list.append(cur_mapped_target)
        cur_flow_gt = torch.cat(
            [torch.from_numpy(sh[np.newaxis, np.newaxis, :, :]) for sh in [x_shift, y_shift]], dim=1).cuda()
        flow_gt_list.append(cur_flow_gt)

    flow_gt = torch.cat(flow_gt_list, dim=0)
    mapped_target = torch.cat(mapped_target_list, dim=0)

    return mapped_target, flow_gt


def f(x):
    return np.max(x.data.cpu().squeeze().numpy(), axis=0)


def edge_weight(target, balance=1.0, mode=None, pred=None):
    """ 2022-10-06 by xwj
    Args:
        args: configuration
        target: groundtruth
        pred: network prediction
        balance: balance edge and no-edge

    Returns: weights of input size
    """
    n, c, h, w = target.size()

    if mode is None:
        return None

    elif mode == "strict":
        weights = np.zeros((n, c, h, w)).astype(np.float)
        for i in range(n):
            t = target[i, :, :, :].cpu().data.numpy()
            pos = (t == 1).sum()
            neg = h * w - pos
            weights[i, t == 0] = pos * 1. / (h * w)
            weights[i, t == 1] = neg * 1. / (h * w)
        weights = torch.Tensor(weights)
    elif mode == 'relax':
        weights = np.zeros((n, c, h, w)).astype(np.float)
        for i in range(n):
            t = target[i, :, :, :].cpu().data.numpy()
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            valid = neg + pos
            weights[i, t == 1] = neg * 1. / valid
            weights[i, t == 0] = pos * balance / valid
        weights = torch.Tensor(weights)
    else:
        raise Exception("Edge-weight mode ({}) not exists!".format(mode))

    weights = weights.cuda()

    return weights


def freeze_bn(model, distributed=False):
    """
    Override the default train() to freeze the BN parameters
    !!! if freeze_bn(), the network will not converge
    """
    if distributed:
        contain_bn_layers = [model.module.bn_conv1, model.module.res2, model.module.res3, model.module.res4,
                             model.module.res5]
    else:
        contain_bn_layers = [model.bn_conv1, model.res2, model.res3, model.res4, model.res5]

    print("===> Freezing Mean/Var of BatchNorm2D.")

    for each_block in contain_bn_layers:
        for m in each_block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


if __name__ == "__main__":
    # For settings
    import os
    import config
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import cv2
    from dataloader.sbd_data import same_seeds
    from prep_dataset.prep_sbd_dataset import get_dataloader
    from model_play import WeightedMultiLabelSigmoidLoss
    import torch.nn.functional as nnf

    # env settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    np.random.seed(666)
    same_seeds(666)

    # configuration
    args = config.get_args()

    # dataloader
    args.batch_size = 1
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target) in enumerate(train_loader):
        # print("{} -- image.size():{}".format(target["img_name"], img.size()))
        print("image.size:{}, target.size:{}".format(img.size(), target.size()))
        print("target range: {} ~ {}, unique value: {}".format(torch.min(target), torch.max(target),torch.unique(target)))

        np_img = np.transpose(img.squeeze().numpy() / 255.0, [1, 2, 0])
        print("(ensure the input is same) image mean: ", np.mean(np_img))

        pred = torch.ones((1, 20, 472, 472)).cuda()
        target = target.cuda()

        ### original loss implementation
        loss = WeightedMultiLabelSigmoidLoss(pred, target, weighted=True)
        print('====> WeightedMultiLabelSigmoidLoss = {}'.format(loss))

        ### update loss implementation
        loss_function = nnf.binary_cross_entropy_with_logits
        target = target.transpose(1, 3).transpose(2, 3)
        cur_weight = edge_weight(target, mode='strict')
        check_loss = loss_function(pred.float(), target.float(), weight=cur_weight, reduction="sum")

        print('====> checked loss = {}'.format(check_loss))
        break




