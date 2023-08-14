import os
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# For local imports
import utils.utils as utils

# For data loader
import prep_dataset.prep_cityscapes_dataset as prep_cityscapes_dataset
import prep_dataset.prep_sbd_dataset as prep_sbd_dataset

# For model
from modules.CASENet_with_STN import CASENetSTN

# For training and validation
import train_val.pntedge_model_play as model_play

# For settings
import pnt_config as config
args = config.get_args()


def main():
    global args
    print("config:{0}".format(args))

    checkpoint_dir = args.checkpoint_folder

    min_val_loss = 999999999

    # dataloader settings
    if args.dataset == "cityscapes":
        assert args.cls_num == 19
        casenet_train_loader, val_loader = prep_cityscapes_dataset.get_my_dataloader(args, thingt=False, batch_size=args.batch_size)
        stn_train_loader, val_loader = prep_cityscapes_dataset.get_my_dataloader(args, thingt=False, batch_size=args.batch_size)
        in_channel = 41
    elif args.dataset == "sbd":
        assert args.cls_num == 20
        casenet_train_loader, val_loader = prep_sbd_dataset.get_my_dataloader(args, thingt=False, batch_size=args.batch_size)
        stn_train_loader, val_loader = prep_sbd_dataset.get_my_dataloader(args, thingt=False, batch_size=args.batch_size)
        in_channel = 43
    else:
        raise Exception("Wrong dataset mode!")

    # model settings
    model = CASENetSTN(pretrained=False, num_classes=args.cls_num, field_size=(472, 472), mode='bilinear', cuda=True, detach_x=True, in_chn=in_channel)

    # multi-gpu settings
    if args.multigpu:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    cudnn.benchmark = True

    # load pretrained model
    if not args.pretrained_casenet is None:
        utils.load_pretrained_model(model.casenet, args.pretrained_casenet, print_info=False, mode='casenet')
    if not args.pretrained_stn is None:
        utils.load_pretrained_model(model.stn, args.pretrained_stn, print_info=False, mode='stn')

    # set parameter updata policy
    casenet_policies = model_play.get_model_policy(model, mode='casenet')
    casenet_optimizer = torch.optim.SGD(casenet_policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    stn_policies = model_play.get_model_policy(model, mode='stn')
    stn_optimizer = torch.optim.SGD(stn_policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # resume model
    if args.resume_model:
        print('==> resume model: {}'.format(args.resume_model))
        checkpoint = torch.load(args.resume_model)
        min_val_loss = checkpoint['min_loss']
        model.load_state_dict(checkpoint['state_dict'])
        casenet_optimizer.load_state_dict(checkpoint['casenet_optimizer'])
        stn_optimizer.load_state_dict(checkpoint['stn_optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    # training scheme settings
    scheme_dict = {
        'casenet': [1, 0, ],
        'stn': [0, 1, ],
    }
    casenet_on, stn_on = scheme_dict[args.train_scheme]

    # outer loop
    global_step = 0

    for epoch in range(args.start_epoch, args.epochs):
        cur_step = global_step

        ### ==> train casenet ------------------------------------------------------------------------------------------
        if casenet_on:
            torch.cuda.empty_cache()

            curr_lr = utils.adjust_learning_rate(args.lr, args, casenet_optimizer, cur_step, args.lr_steps)
            global_step_casenet = model_play.train(args, casenet_train_loader, model, casenet_optimizer, epoch, curr_lr, cur_step,
                                                   accumulation_steps=args.acc_steps, distributed=args.multigpu, mode='casenet')
            # Always store current model to avoid process crashed by accident.
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'casenet_optimizer': casenet_optimizer.state_dict(),
                'stn_optimizer': stn_optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir, filename="curr_checkpoint.pth.tar")

            global_step = global_step_casenet

        ### ==> train stn ----------------------------------------------------------------------------------------------
        if stn_on:
            torch.cuda.empty_cache()

            curr_lr = utils.adjust_learning_rate(args.lr, args, stn_optimizer, cur_step, args.lr_steps)
            global_step_stn = model_play.train(args, stn_train_loader, model, stn_optimizer, epoch, curr_lr, cur_step,
                                               accumulation_steps=args.acc_steps, distributed=args.multigpu, mode='stn')
            # update step info
            # assert (global_step_stn == global_step_casenet)
            global_step = global_step_stn

            # Always store current model to avoid process crashed by accident.
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'casenet_optimizer': casenet_optimizer.state_dict(),
                'stn_optimizer': stn_optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir, filename="curr_checkpoint.pth.tar")

        # save every global step
        print('==> global step: ', global_step)
        if (epoch+1) % args.save_freq == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'casenet_optimizer': casenet_optimizer.state_dict(),
                'stn_optimizer': stn_optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir, filename="checkpoint_{}.pth".format(epoch+1))


if __name__ == '__main__':
    import os
    from dataloader.sbd_data import same_seeds

    # set gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # set random seed only for debug. Annotate it when training
    np.random.seed(554)
    same_seeds(666)

    # run
    main()


