import torch
import numpy as np


# setting LOG
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Net-Utils')


def count_parameters(model):
    """Calculate number of total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def isRotationMatrix(R):
    """R*R^{-1}=I"""
    return torch.norm(torch.eye(3, dtype=R.dtype)-R.mm(R.t())) < 1e-6


def cuda_memory_usage():
    LOG.warning('Cuda memory usage: %d' % torch.cuda.memory_allocated())
    return


def gen_fake_data(N_sample, channel, H, W):  # (1398, 1, 374, 1238)
    lidar_data = np.random.rand(N_sample, channel, H, W).astype(np.float32)
    cam_data = np.random.rand(N_sample, channel, H, W).astype(np.float32)
    return cam_data, lidar_data


def calc_crop_bbox(orig_shape, crop_shape):
    """
    Calculate the bounding box of the cropping image

    @param orig_shape: Tuple[int], a tuple of original shape values, (H1, W1)
    @param crop_shape: Tuple[int], a tuple of cropped shape values, (H2, W2), H2<H1, W2<W1
    @return: the bounding box boundaries, Hmin, Hmax, Wmin, Wmax
    """
    H, W = orig_shape
    h, w = crop_shape
    assert h < H and w < W, 'Crop size must be smaller than the original size!'
    hmin = H // 2 - (h // 2)
    wmin = W // 2 - (w // 2)
    return hmin, hmin+h, wmin, wmin+w


def center_crop(arr, crop_size):
    """
    Center crop an Numpy image

    @param arr: ndarray, image with shape (H,W) and (C,H,W), channel first.
    @param crop_size: Tuple[int], the target size/shape of the output
    @return: cropped image with crop_size
    """
    if len(arr.shape) == 2:
        H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 2
    elif len(arr.shape) == 3:
        C, H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 3
    elif len(arr.shape) == 4:
        B, C, H, W = arr.shape  # h -> y-axis, w -> x-axis
        dim = 4
    else:
        raise ValueError('The input array shape is not supported. Supported shape (H,W), (C,H,W), (B,C,H,W).')

    hmin, hmax, wmin, wmax = calc_crop_bbox((H, W), crop_size)

    if dim == 2:
        return arr[hmin:hmax, wmin:wmax]
    elif dim == 3:
        return arr[:, hmin:hmax, wmin:wmax]
    else:
        return arr[:, :, hmin:hmax, wmin:wmax]
