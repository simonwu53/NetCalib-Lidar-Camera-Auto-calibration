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
    assert h < H and w < W, \
        'Crop size must be smaller than the original size! \n' \
        'Original shape ({o[0]},{o[1]}), Crop shape ({c[0]},{c[1]})'.format(o=orig_shape, c=crop_shape)
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


def calib_np_to_phi(M, no_exception=False):
    """
    Transform calibration matrix to calibration vector (Numpy version)
    M_{calib} = | R  T |   4*4 matrix
                | 0  1 |

    :param M: 4*4 calibration matrix
    :param no_exception: bool, do not raise error there contains a wrong rotation matrix

    :return: calibration vector [r_x,r_y,r_z,t_x,t_y,t_z]
    """
    if M.shape[0] != 4 or M.shape[1] != 4:
        raise ValueError("A calibration (transformation) matrix must be a matrix of shape (4, 4)!")

    translation = np.transpose(M[:3, 3])
    orientation = rotation_matrix_to_angle(M[:3, :3], no_exception=no_exception)
    calib_vec = np.zeros(6, np.float32)
    calib_vec[:3] = orientation
    calib_vec[3:] = translation
    return calib_vec


def rotation_matrix_to_angle(R, no_exception=False):
    """
    Convert a 3*3 rotation matrix to roll, pitch, yaw
    :param R: 3*3 rotation matrix
    :param no_exception: bool, do not raise error when a wrong matrix comes
    :return: roll, pitch, yaw in radians
    """
    if not isRotationMatrix_np(R):
        if not no_exception:
            raise ValueError('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
        else:
            # LOG.warning('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
            return 0, 0, 0

    pitch = -np.arcsin(R[2, 0])

    if R[2,0] == 1:
        yaw = 0
        roll = np.arctan2(-R[0,1], -R[0,2])
    elif R[2,0] == -1:
        yaw = 0
        roll = np.arctan2(R[0,1], R[0,2])
    else:
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])

    return roll, pitch, yaw


def isRotationMatrix_np(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.matmul(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
