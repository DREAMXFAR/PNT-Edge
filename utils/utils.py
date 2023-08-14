import os
import re
import numpy as np
import torch
from torch.autograd import Variable


def get_model_policy(model, with_stn=False):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    other_pts = []

    if not with_stn:
        for m in model.named_modules():
            if m[0] != '' and m[0] != 'module':
                if ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                    ps = list(m[1].parameters())
                    score_feats_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        score_feats_conv_bias.append(ps[1])
                    # print("Totally new layer:{0}".format(m[0]))
                else:  # For all the other module that is not totally new layer.
                    ps = list(m[1].parameters())
                    other_pts.extend(ps)

        return [
                {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
                {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'},
                {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 1, 'name': 'other'},
        ]
    else:
        localization_conv_weight = []
        localization_conv_bias = []

        localization_bn_weight = []
        localization_bn_bias = []

        for m in model.named_modules():
            if m[0] != '' and m[0] != 'module':
                if re.search('^casenet$', m[0]) is not None or re.search('^localization$', m[0]) is not None:
                    continue
                elif ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                    ps = list(m[1].parameters())
                    score_feats_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        score_feats_conv_bias.append(ps[1])
                    # print("Totally new layer:{0}".format(m[0]))
                elif ('localization.' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                    ps = list(m[1].parameters())
                    localization_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        localization_conv_bias.append(ps[1])
                    # print("Totally new layer:{0}".format(m[0]))
                elif ('localization.' in m[0]) and isinstance(m[1], torch.nn.BatchNorm2d):
                    ps = list(m[1].parameters())
                    localization_bn_weight.append(ps[0])
                    if len(ps) == 2:
                        localization_bn_bias.append(ps[1])
                    # print("Totally new layer:{0}".format(m[0]))
                else:  # For all the other module that is not totally new layer.
                    ps = list(m[1].parameters())
                    other_pts.extend(ps)

        return [
            {'params': score_feats_conv_weight, 'lr_mult': 0, 'name': 'score_conv_weight'},  # 10
            {'params': score_feats_conv_bias, 'lr_mult': 0, 'name': 'score_conv_bias'},  # 20
            {'params': localization_conv_weight, 'lr_mult': 10, 'name': 'localization_conv_weight'},
            {'params': localization_conv_bias, 'lr_mult': 20, 'name': 'localization_conv_bias'},
            {'params': localization_bn_weight, 'lr_mult': 10, 'name': 'localization_bn_weight'},
            {'params': localization_bn_bias, 'lr_mult': 20, 'name': 'localization_bn_bias'},
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 0, 'name': 'other'},  # 1
        ]


def check_gpu(gpu, *args):
    """Move data in *args to GPU?
        gpu: options.gpu (None, or 0, 1, .. gpu index)
    """
    if gpu == None:
        if isinstance(args[0], dict):
            d = args[0]
            #print(d.keys())
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key])
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        # it's a list of arguments
        if len(args) > 1:
            return [Variable(a) for a in args]
        else:  # single argument, don't make a list
            return Variable(args[0])

    else:
        if isinstance(args[0], dict):
            d = args[0]
            #print(d.keys())
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key].cuda(gpu))
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        # it's a list of arguments
        if len(args) > 1:
            return [Variable(a.cuda(gpu)) for a in args]
        else:  # single argument, don't make a list
            return Variable(args[0].cuda(gpu))


def load_pretrained_model(model, pretrained_model_path, print_info=False, mode=None):
    # Load trained model.
    trained_model = torch.load(pretrained_model_path)
    if 'casenet_state_dict' in trained_model.keys():
        pretrained_dict = trained_model['casenet_state_dict']
    else:
        pretrained_dict = trained_model['state_dict']
        if mode == 'stn' and 'stn.localization.ConvEn21.weight' in pretrained_dict.keys():
            pretrained_dict = {k.replace("stn.", ""): v for k, v in pretrained_dict.items()}
        if mode == 'casenet' and 'casenet.bn_conv1.weight' in pretrained_dict.keys():
            pretrained_dict = {k.replace("casenet.", ""): v for k, v in pretrained_dict.items()}
    removed_prefix_pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    # Judge pretrained model and current model are multi-gpu or not
    multi_gpu_pretrained = False
    multi_gpu_current_model = False
    for k, v in pretrained_dict.items():
        if "module" in k:
            multi_gpu_pretrained = True
            break
    for k, v in model_dict.items():
        if "module" in k:
            multi_gpu_current_model = True
            break
    
    # Different ways to deal with diff cases
    if multi_gpu_pretrained and multi_gpu_current_model:
        updated_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    elif multi_gpu_pretrained and not multi_gpu_current_model:
        updated_pretrained_dict = {k: v for k, v in removed_prefix_pretrained_dict.items() if k in model_dict}
    elif not multi_gpu_pretrained and multi_gpu_current_model:
        updated_pretrained_dict = {}
        for current_k in model_dict:
            removed_prefix_k = current_k.replace("module.", "")
            updated_pretrained_dict[current_k] = pretrained_dict[removed_prefix_k]
    else:
        updated_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    if print_info:  # added by xwj
        for k in updated_pretrained_dict.keys():
            print(" ==> {0} is loaded successfully".format(k))

    # 2. overwrite entries in the existing state dict
    model_dict.update(updated_pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def load_official_pretrained_model(model, pretrained_model_path):
    # Load trained model.
    trained_model = torch.load(pretrained_model_path)
    model_dict = model.state_dict()
    
    for k in trained_model.keys():
        print("{0} exists in ori pretrained model".format(k))
    for k in model_dict.keys():
        print("{0} exists in ori model".format(k))

    updated_pretrained_dict = {k: v for k, v in trained_model.items() if (k in model_dict)}

    for k in updated_pretrained_dict.keys():
        print("{0} is loaded successfully".format(k))
    
    # overwrite entries in the existing state dict
    model_dict.update(updated_pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)


def save_checkpoint(state, epoch, folder, filename='min_loss_checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, filename)
    torch.save(state, filename)


def adjust_learning_rate(ori_lr, args, optimizer, global_step, lr_steps):
    decay = 0.1 ** (sum(global_step >= np.array(lr_steps)))
    lr = ori_lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:
            param_group['weight_decay'] = decay * param_group['decay_mult']

    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

