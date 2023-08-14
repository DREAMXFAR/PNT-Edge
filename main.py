import os
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# Local imports
import utils.utils as utils

# For data loader
import prep_dataset.prep_cityscapes_dataset as prep_cityscapes_dataset
import prep_dataset.prep_sbd_dataset as prep_sbd_dataset

# For model
from modules.CASENet import CASENet_resnet101

# For training and validation
import train_val.model_play as model_play

# For settings
import config


args = config.get_args()


def main():
    global args
    print("config:{0}".format(args))

    checkpoint_dir = args.checkpoint_folder

    global_step = 0
    min_val_loss = 999999999

    if args.dataset == "cityscapes":
        train_loader, val_loader = prep_cityscapes_dataset.get_dataloader(args)
        assert args.cls_num == 19
    elif args.dataset == "sbd":
        train_loader, val_loader = prep_sbd_dataset.get_dataloader(args)
        assert args.cls_num == 20

    model = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)

    if args.multigpu:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    policies = get_model_policy(model)  # Set the lr_mult=10 of new layer
    optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.pretrained_model:
        utils.load_pretrained_model(model, args.pretrained_model)

    if args.resume_model:
        checkpoint = torch.load(args.resume_model)
        args.start_epoch = checkpoint['epoch']+1
        min_val_loss = checkpoint['min_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        curr_lr = utils.adjust_learning_rate(args.lr, args, optimizer, global_step, args.lr_steps)

        global_step = model_play.train(args, train_loader, model, optimizer, epoch, curr_lr, None, None, None,
                                       global_step, accumulation_steps=1, distributed=args.multigpu)
        torch.cuda.empty_cache()

        # Always store current model to avoid process crashed by accident.
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': min_val_loss,
        }, epoch, folder=checkpoint_dir, filename="cur_checkpoint.pth.tar")

        # save every global step
        print('==> global step: ', global_step)
        if (epoch+1) % args.save_freq == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir, filename="checkpoint_{}.pth".format(epoch+1))


def get_model_policy(model):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    other_pts = []
    for m in model.named_modules():
        if m[0] != '' and m[0] != 'module':
            if ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                ps = list(m[1].parameters())
                score_feats_conv_weight.append(ps[0])
                if len(ps) == 2:
                    score_feats_conv_bias.append(ps[1])
                print("Totally new layer:{0}".format(m[0]))
            else:  # For all the other module that is not totally new layer.
                ps = list(m[1].parameters())
                other_pts.extend(ps)

    return [
        {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
        {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'},
        {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 1, 'name': 'other'},
    ]


if __name__ == '__main__':
    import os
    import numpy as np
    from dataloader.sbd_data import same_seeds

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set random seed only for debug
    np.random.seed(555)
    same_seeds(666)

    main()

