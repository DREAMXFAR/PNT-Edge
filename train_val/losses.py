import torch
import torch.nn.functional as F
import numpy as np
import math


class MSE:
    """
    Mean squared error loss.
    """
    def loss(self, y_true, y_pred, mode='sum'):
        if mode == 'sum':
            return torch.sum((y_true - y_pred) ** 2)
        elif mode == 'mean':
            return torch.mean((y_true - y_pred) ** 2)
        else:
            raise Exception('Mode {} not exists!'.format(mode))


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad2D:
    """
    2-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred, mode='mean'):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        if mode == 'mean':
            d = torch.mean(dx) + torch.mean(dy)
        elif mode == 'sum':
            d = torch.sum(dx) + torch.sum(dy)
        else:
            raise Exception('Mode {} not exists!'.format(mode))

        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Sparse2D:
    """
    2-D gradient loss.
    """
    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, flow, mode='mean'):
        # loss = torch.abs(flow)
        loss = flow * flow * 0.5

        if mode == 'mean':
            loss = torch.mean(loss)
        elif mode == 'sum':
            loss = torch.sum(loss)
        else:
            raise Exception('Mode {} not exists!'.format(mode))

        if self.loss_mult is not None:
            loss *= self.loss_mult
        return loss


class LocalConsistency:
    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult
        self.mean_pool = torch.nn.AvgPool2d(kernel_size=(5, 5), stride=1, padding=2)

    def loss(self, pred, mode='mean', return_mean=False):
        local_mean = self.mean_pool(pred)

        if mode == 'mean':
            loss_function = torch.nn.MSELoss(reduction='mean')
        elif mode == 'sum':
            loss_function = torch.nn.MSELoss(reduction='sum')
        else:
            raise Exception('Mode {} not exists!'.format(mode))

        loss = loss_function(pred, local_mean)

        if self.loss_mult is not None:
            loss *= self.loss_mult

        if return_mean:
            return loss, local_mean
        else:
            return loss


if __name__ == "__main__":
    a = np.array(
        [[1, 2, 1, 0, 0],
         [0, 1, 2, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         ]
    )

    a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float()

    loss_function = LocalConsistency()
    loss = loss_function.loss(a)

    print(loss)




