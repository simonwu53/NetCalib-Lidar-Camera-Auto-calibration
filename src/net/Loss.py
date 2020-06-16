import torch.nn as nn
import logging

# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Loss')


class Loss(nn.Module):
    def __init__(self, mode='L1', reduction='sum', alpha=0.7, beta=0.3):
        super(Loss, self).__init__()
        if mode == 'L1':
            self.loss_r = nn.L1Loss(reduction=reduction)
            self.loss_t = nn.L1Loss(reduction=reduction)
            self.alpha = alpha
            self.beta = beta
        else:
            raise NotImplementedError('Other Loss functions are not implemented currently.')
        return

    def forward(self, pred, target):
        rot, trans = pred
        gt_r, gt_t = target
        loss_r = self.loss_r(rot, gt_r)*self.alpha
        loss_t = self.loss_t(trans, gt_t)*self.beta
        return loss_r+loss_t
