import torch.nn as nn
import torch
import logging
import numpy as np

# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('SPPLayer')


class SPPLayer(nn.Module):
    def __init__(self, pyramid=(1, 4, 16), mode='max'):
        """
        Generate fixed length representation regardless of image dimensions
        Based on the paper "Spatial Pyramid Pooling in Deep Convolutional Networks
        for Visual Recognition" (https://arxiv.org/pdf/1406.4729.pdf)

        :param pyramid: Sequence[int], Number of pools to split each input feature map into.
                        Each element must be a perfect square in order to equally divide the
                        pools across the feature map. Default corresponds to the original
                        paper's implementation
        :param    mode: str, Specifies the type of pooling, either max or average
        """
        super(SPPLayer, self).__init__()
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool2d
        elif mode == 'average':
            pool_func = nn.AdaptiveAvgPool2d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")

        self.pools = []
        for p in pyramid:
            side_length = np.sqrt(p)
            if not side_length.is_integer():
                raise ValueError('Pyramid size %f can not be divided to equal side length!' % p)
            self.pools.append(pool_func(int(side_length)))
        return

    def forward(self, x):
        """
        Pool feature maps at different bin levels and concatenate

        :param x: tensor, Arbitrarily shaped spatial and channel dimensions extracted from any
                  generic convolutional architecture. Shape ``(N, C, H, W)``
        :return pooled: tensor, Concatenation of all pools with shape ``(N, C, sum(num_pools))``
        """
        if x.dim() != 4:
            raise ValueError('Expected 4D input of (N, C, H, W)')
        N, C = x.size(0), x.size(1)
        pooled = [p(x).view(N, C, -1) for p in self.pools]
        return torch.cat(pooled, dim=2)
