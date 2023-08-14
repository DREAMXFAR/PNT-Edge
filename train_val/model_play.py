import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import sigmoid
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import sys
sys.path.append("../")

# Local imports
import utils.utils as utils
from utils.utils import AverageMeter


def train(args, train_loader, model, optimizer, epoch, curr_lr, win_feats5, win_fusion, viz, global_step,
          accumulation_steps, distributed=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to eval mode to make BN unchanged.
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (img, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Input for Image CNN.
        img_var = utils.check_gpu(0, img)  # BS X 3 X H X W
        target_var = utils.check_gpu(0, target)  # BS X H X W X NUM_CLASSES

        bs = img.size()[0] * accumulation_steps

        score_feats5, fused_feats = model(img_var)  # BS X NUM_CLASSES X 472 X 472

        if args.weighted_loss:
            # weighted loss
            feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, target_var, weighted=True)
            fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, target_var, weighted=True)
        else:
            # unweighted loss
            feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, target_var, weighted=False, manual_weight=args.manual_weight)
            fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, target_var, weighted=False, manual_weight=args.manual_weight)

        loss = feats5_loss + fused_feats_loss
        loss.backward()

        # clear memory
        del img_var
        del target_var
        del score_feats5
        torch.cuda.empty_cache()

        # increase batch size by factor of accumulation steps (Gradient accumulation) for training with limited memory
        if (i+1) % accumulation_steps == 0:
            feats5_losses.update(feats5_loss.data, bs)
            fusion_losses.update(fused_feats_loss.data, bs)
            total_losses.update(loss.data, bs)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i+1) % args.print_freq == 0):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Total Loss {total_loss.val:.11f} ({total_loss.avg:.11f})\t'
                      'lr {learning_rate:.15f}\t'
                      .format(epoch, int((i+1)/accumulation_steps), int(len(train_loader)/accumulation_steps), batch_time=batch_time,
                       data_time=data_time, total_loss=total_losses, learning_rate=curr_lr))

        # TODO: for debug
        break

    del feats5_loss
    del fused_feats_loss
    del feats5_losses
    del fusion_losses
    del total_losses
    torch.cuda.empty_cache()
    return global_step


def validate(args, val_loader, model, epoch, win_feats5, win_fusion, viz, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    total_losses = AverageMeter()
    
    # switch to train mode
    model.eval()
    torch.no_grad()

    end = time.time()
    for i, (img, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Input for Image CNN.
        img_var = utils.check_gpu(0, img)  # BS X 3 X H X W
        target_var = utils.check_gpu(0, target)  # BS X H X W X NUM_CLASSES
        
        bs = img.size()[0]

        score_feats5, fused_feats = model(img_var) # BS X NUM_CLASSES X 472 X 472
       
        feats5_loss = WeightedMultiLabelSigmoidLoss(score_feats5, target_var) 
        fused_feats_loss = WeightedMultiLabelSigmoidLoss(fused_feats, target_var) 
        loss = feats5_loss + fused_feats_loss
             
        feats5_losses.update(feats5_loss.data, bs)
        fusion_losses.update(fused_feats_loss.data, bs)
        total_losses.update(loss.data, bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # clear memory
        del img_var
        del target_var
        del score_feats5
        del fused_feats_loss
        del feats5_loss
        torch.cuda.empty_cache()

        if (i % args.print_freq == 0):
            # print("\n")
            print('validation ==> Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                  .format(epoch, i, len(val_loader), batch_time=batch_time,
                   data_time=data_time, total_loss=total_losses))
    
    return fusion_losses.avg 


def WeightedMultiLabelSigmoidLoss(model_output, target, weighted=True, manual_weight=False):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES 
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = utils.check_gpu(0, target.sum(dim=1).sum(dim=1).sum(dim=1).float().data)  # BS * 1
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1]*target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1]*target.size()[2]-weight_sum.data) / float(target.size()[1]*target.size()[2]))
    coefficient = utils.check_gpu(0, torch.tensor(1.0).cuda() / float(target.size()[1]*target.size()[2]))
    one_sigmoid_out = sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    target = target.transpose(1, 3).transpose(2, 3).float()  # BS X NUM_CLASSES X H X W

    if weighted:
        loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*target*torch.log(one_sigmoid_out.clamp(min=1e-10)) - \
                edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*(1-target)*torch.log(zero_sigmoid_out.clamp(min=1e-10))
    elif manual_weight:
        weight_1 = 0.9
        weight_2 = 0.1
        loss = weight_1 * (-target * torch.log(one_sigmoid_out.clamp(min=1e-10)) - weight_2 * (1 - target) * torch.log(zero_sigmoid_out.clamp(min=1e-10)))
    else:
        loss = (-target * torch.log(one_sigmoid_out.clamp(min=1e-10)) - (1 - target) * torch.log(zero_sigmoid_out.clamp(min=1e-10))) # * coefficient

    return loss.mean(dim=0).sum()


def freeze_bn(model, distributed=False):
    """
    Override the default train() to freeze the BN parameters
    !!! if freeze_bn(), the network will not converge
    """
    if distributed:
        contain_bn_layers = [model.module.bn_conv1, model.module.res2, model.module.res3, model.module.res4, model.module.res5]
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

    import torch.nn.functional as nnf

    # env settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        print("target range: {} ~ {}, unique value: {}".format(torch.min(target), torch.max(target), torch.unique(target)))

        np_img = np.transpose(img.squeeze().numpy() / 255.0, [1, 2, 0])
        print("(ensure the input is same) image mean: ", np.mean(np_img))

        pred = torch.zeros((1, 20, 472, 472)).cuda()

        plt.figure()
        plt.imshow(np.max(target[0, :, :, :].numpy(), axis=2) * 255, cmap='gray')
        plt.show()

        target = target.cuda()

        ### original loss implementation
        loss = WeightedMultiLabelSigmoidLoss(pred, target, weighted=True)
        print('==> WeightedMultiLabelSigmoidLoss = {}'.format(loss))

        ### update loss implementation
        loss_function = nnf.binary_cross_entropy_with_logits
        # print("before: ", target.shape)
        target = target.transpose(1, 3).transpose(2, 3)
        # print("after: ", target.shape)
        cur_weight = edge_weight(target, mode='relax')
        check_loss = loss_function(pred.float(), target.float(), weight=cur_weight, reduction="sum")

        print('==> checked loss = {}'.format(check_loss))
        print('==> loss sum = {}, shape: {}'.format(check_loss.mean(dim=0).sum(), check_loss.shape))


        break




