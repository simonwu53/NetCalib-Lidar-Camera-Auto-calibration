import torch
import torch.nn as nn
import logging
from .Convolution import Conv2d
from .SpatialPyramidPooling import SPPLayer


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('ClibNet')


def get_model(model=2):
    if model == 1:
        return CalibNet()
    else:
        raise NotImplementedError("Model not implemented.")


def disparity_matching_module():
    # this module is built for Lidar & Cameras disparity maps respectively
    model = nn.Sequential(
        # layer 1, conv+NIN, out 32, 128, 384
        Conv2d(in_c=1, out_c=32, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=32, out_c=32, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # layer 2, out 128, 64, 192
        Conv2d(in_c=32, out_c=128, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # layer 3, out 128, 32, 96
        Conv2d(in_c=128, out_c=128, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=128, out_c=128, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),
    )
    return model


def global_regression_module():
    # this module is built for concatenated lidar & camera tensor
    model = nn.Sequential(
        # layer 1, conv+NIN, out N, 256, 16, 48
        Conv2d(in_c=256, out_c=512, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=512, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # layer 2, out N, 512, 8, 24
        Conv2d(in_c=256, out_c=512, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=512, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=512, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=512, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # layer 3, out N, 64, 4, 12
        Conv2d(in_c=512, out_c=1024, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=1024, out_c=1024, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=1024, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=512, kerner_size=3, stride=1, padding=1, batch_norm=True, activation='relu'),
        Conv2d(in_c=512, out_c=256, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        Conv2d(in_c=256, out_c=64, kerner_size=1, stride=1, padding=0, batch_norm=True, activation='relu'),
        nn.MaxPool2d(kernel_size=2),

        # layer 4, spp-layer, out N, 64, 21
        SPPLayer(pyramid=(1, 4, 16), mode='max'),
    )
    return model


def output_layer(in_size, mid_size=128):
    model = nn.Sequential(
        nn.Linear(in_size, mid_size),  # (1, 128)
        nn.ReLU(),
        nn.Linear(mid_size, 3)  # (1, 3)
    )
    return model


class CalibNet(nn.Module):
    def __init__(self):
        super(CalibNet, self).__init__()
        self.disp_matching_L = disparity_matching_module()
        self.disp_matching_C = disparity_matching_module()
        self.global_regression = global_regression_module()
        self.rot_output = output_layer(672)
        self.trans_output = output_layer(672)
        return

    def forward(self, disp_l, disp_c):
        # # input shape (1, 1, 256, 768), N == Batch size == consecutive frames
        # disparity matching, out 1, 128, 32, 120
        disp_matched_l = self.disp_matching_L(disp_l)
        disp_matched_c = self.disp_matching_C(disp_c)

        # concatenate outputs at channels dim, out 1, 256, 32, 120
        concat = torch.cat([disp_matched_c, disp_matched_l], dim=1)

        # global regression, out 1, 64, 21
        regressed = self.global_regression(concat).squeeze(0)

        # reshape to two tensors
        rot_regress = regressed[:32, :].view(1, -1)  # 1, 32*21
        trans_regress = regressed[32:, :].view(1, -1)  # 1, 32*21

        # output, out 1, 3
        rot = self.rot_output(rot_regress)
        trans = self.trans_output(trans_regress)

        return rot, trans  # out (1, 3), (1, 3)


if __name__ == '__main__':
    pass
