import torch.nn as nn
import logging

# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('ConvLayer')


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kerner_size, stride=1, padding=0, batch_norm=False, activation='relu', squeeze=None):
        super(ConvLayer, self).__init__()
        # conv layer
        self.conv = self.create_conv(in_c=in_c, out_c=out_c, kerner_size=kerner_size, stride=stride, padding=padding)

        # batch normalization
        if batch_norm:
            self.batch_norm = True
            self.bn = self.create_bn(num_features=out_c)
        else:
            self.batch_norm = False

        # activation
        if activation == 'relu':
            self.activate = nn.ReLU()
        else:
            raise NotImplementedError("Other activation function is not supported currently!")

        # squeeze
        self.squeeze = squeeze
        return

    def forward(self, x):
        # convolution
        x = self.conv(x)

        # batch normalization
        if self.batch_norm:
            x = self.bn(x)

        # activation
        x = self.activate(x)

        # return activation
        if self.squeeze is not None:
            return x.squeeze(dim=self.squeeze)
        else:
            return x

    def create_conv(self, in_c, out_c, kerner_size, stride=1, padding=0):
        return NotImplementedError('Use subclassed layer to complete this function.')

    def create_bn(self, num_features):
        return NotImplementedError('Use subclassed layer to complete this function.')


class Conv1d(ConvLayer):
    def __init__(self, in_c, out_c, kerner_size, stride=1, padding=0, batch_norm=False, activation='relu', squeeze=None):
        super(Conv1d, self).__init__(in_c, out_c, kerner_size, stride, padding, batch_norm, activation, squeeze)
        return

    def create_conv(self, in_c, out_c, kerner_size, stride=1, padding=0):
        return nn.Conv1d(in_c, out_c, kernel_size=kerner_size, stride=stride, padding=padding)

    def create_bn(self, num_features):
        return nn.BatchNorm1d(num_features=num_features)


class Conv2d(ConvLayer):
    def __init__(self, in_c, out_c, kerner_size, stride=1, padding=0, batch_norm=False, activation='relu', squeeze=None):
        super(Conv2d, self).__init__(in_c, out_c, kerner_size, stride, padding, batch_norm, activation, squeeze)
        return

    def create_conv(self, in_c, out_c, kerner_size, stride=1, padding=0):
        return nn.Conv2d(in_c, out_c, kernel_size=kerner_size, stride=stride, padding=padding)

    def create_bn(self, num_features):
        return nn.BatchNorm2d(num_features=num_features)


class Conv3d(ConvLayer):
    def __init__(self, in_c, out_c, kerner_size, stride=1, padding=0, batch_norm=False, activation='relu', squeeze=None):
        super(Conv3d, self).__init__(in_c, out_c, kerner_size, stride, padding, batch_norm, activation, squeeze)
        return

    def create_conv(self, in_c, out_c, kerner_size, stride=1, padding=0):
        return nn.Conv3d(in_c, out_c, kernel_size=kerner_size, stride=stride, padding=padding)

    def create_bn(self, num_features):
        return nn.BatchNorm3d(num_features=num_features)
