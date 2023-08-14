import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Edge Detection Training')
    
    parser.add_argument('--checkpoint-folder', metavar='DIR',
                        help='path to checkpoint dir',
                        default='./checkpoint')

    parser.add_argument('--dataset', help='dataset to train, sbd or cityscapes. ', type=str,
                        default='sbd')
    
    # store_true stores False if no argument passed
    parser.add_argument('--multigpu', action='store_true', 
                        help='use multiple GPUs')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run (default: 150)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--cls-num', default=19, type=int, metavar='N',
                        help='The number of classes (default: 19 for Cityscapes)')

    parser.add_argument('--lr-steps', default=[20000, 40000, 60000, 80000], type=int, nargs="+",
                        metavar='LRSteps', help='iterations to decay learning rate by 10')
    parser.add_argument('--acc-steps', default=1, type=int, metavar='AccSteps',
                        help='accumulation steps for Gradient accumulation for training with limited memory')
    
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--lr', default=1e-7, type=float, metavar='L',
                        help='lr ')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--resume-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')

    parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: None)')

    parser.add_argument('--weighted-loss', action='store_true',
                        help='whether to use weighted loss function (default: True)')

    parser.add_argument('--save-freq', default=1, type=int,
                        metavar='N', help='save .pth model frequency (default: 1)')

    parser.add_argument('--stn', action='store_true',
                        help='if true, use stn to predict transition. ')

    parser.add_argument('--thingt', action='store_true',
                        help='if true, thin noisy gt.')

    parser.add_argument('--scale', default=None, type=float,
                        help='stn scale factor')

    parser.add_argument('--manual-weight', action='store_true')

    args = parser.parse_args()

    return args


