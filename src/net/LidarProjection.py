import torch
import torch.utils.data
from config import *
from .utils import calc_crop_bbox
import numpy as np


# setting LOG
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Net-Projection')


MAT_T_BOTTOM = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=DEVICE)


def inv_transform(T):
    """
    The inverse matrix of a transformation matrix (from pose 2 -> pose 1)
    :param T: transformation (calibration) matrix, | R  T |   4*4 matrix
                                                   | 0  1 |
    :return: inversed transformation matrix, shape (4,4)
    """
    # rotation
    rot = T[:3, :3]
    # translation
    trans = T[:3, 3]
    # assemble inverse transformation matrix
    rt = rot.t()
    tt = -rt.mm(trans.expand(1, 3).t())  # -R^{-1}*t
    return torch.cat([torch.cat([rt, tt], axis=1), MAT_T_BOTTOM], axis=0)


def inv_transform_vectorized(T):
    """
    The inverse matrix of a transformation matrix (from pose 2 -> pose 1), vectorized version
    :param T: transformation (calibration) matrix, | R  T |   N*4*4 matrix
                                                   | 0  1 |
    :return: inversed transformation matrix, shape (N, 4, 4)
    """
    # rotation
    rot = T[:, :3, :3].transpose(1, 2)  # (N, 3, 3)
    # translation
    trans = T[:, :3, 3]
    # re-assemble inverse tranformation
    Tinv = torch.zeros((T.shape[0], 4, 4), dtype=torch.float32, device=DEVICE, requires_grad=False)
    Tinv[:, :3, :3] = rot
    Tinv[:, :3, 3] = -rot.bmm(trans.unsqueeze(2)).squeeze()  # # -R^{-1}*t
    Tinv[:, 3, 3] = 1
    return Tinv


def phi_to_transformation_matrix(phi, device=DEVICE, requires_grad=False):
    """
    Transform calibration vector to calibration matrix (Numpy version)
    \theta_{calib} = [r_x,r_y,r_z,t_x,t_y,t_z]^T -> \phi_{calib} 4*4 matrix

    :param phi: calibration PyTorch Tensor (length 6, shape (6,)), which is an output from calibration network
    :param device: 'cuda' or 'cpu', specify where to put the new Tensor. Works with "requires_grad=False" only.
    :param requires_grad: bool, if True, keep tracking the gradients of the input Tensor, will ignore "device" param.

    :return: transformation matrix from Lidar coordinates to camera's frame
    """
    # split rotation & translation values
    rot, trans = phi[:3], phi[3:]
    # get rotation matrix
    rot_mat = angle_to_rotation_matrix(rot, device=device, requires_grad=requires_grad)

    if not requires_grad:
        # create transformation matrix
        T = torch.zeros(4, 4, dtype=torch.float32, device=DEVICE, requires_grad=False)

        T[:3, :3] = rot_mat
        T[:3, 3] = trans
        T[3, 3] = 1
        return T
    else:
        device = phi.device
        dtype = phi.dtype
        bot = torch.tensor([[0, 0, 0, 1]], dtype=dtype, device=device)
        return torch.cat([torch.cat([rot_mat, trans.unsqueeze(1)], dim=1), bot], dim=0)


def phi_to_transformation_matrix_vectorized(phi):
    """
    Transform calibration vector to calibration matrix (Numpy version)
    \theta_{calib} = [r_x,r_y,r_z,t_x,t_y,t_z]^T -> \phi_{calib} 4*4 matrix

    :param phi: calibration PyTorch Tensor (length 3 shape (3,), roll-pitch-yaw or x-y-z), output from calibration network

    :return: transformation matrix from Lidar coordinates to camera's frame
    """
    # split rotation & translation values
    rot, trans = phi[:, :3], phi[:, 3:]

    # get rotation matrix
    rot_mat = angle_to_rotation_matrix_vectorized(rot)

    # create transformation matrix
    T = torch.zeros((phi.shape[0], 4, 4), dtype=torch.float32, device=DEVICE, requires_grad=False)

    T[:, :3, :3] = rot_mat
    T[:, :3, 3] = trans.unsqueeze(0)
    T[:, 3, 3] = 1
    return T


def angle_to_rotation_matrix(rot, device=DEVICE, requires_grad=False):
    """
    Transform vector of Euler angles to rotation matrix
    ref: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    :param rot: euler angle PyTorch Tensor (length 3 shape (3,), roll-pitch-yaw or x-y-z)
    :param device: 'cuda' or 'cpu', specify where to put the new Tensor. Works with "requires_grad=False" only.
    :param requires_grad: bool, if True, keep tracking the gradients of the input Tensor, will ignore "device" param.
    :return: 3*3 rotation matrix
    """
    u, v, w = rot
    s_u, c_u = u.sin(), u.cos()
    s_v, c_v = v.sin(), v.cos()
    s_w, c_w = w.sin(), w.cos()

    if not requires_grad:
        return torch.tensor([
            [c_v*c_w, s_u*s_v*c_w-c_u*s_w, s_u*s_w+c_u*s_v*c_w],
            [c_v*s_w, c_u*c_w+s_u*s_v*s_w, c_u*s_v*s_w-s_u*c_w],
            [-s_v, s_u*c_v, c_u*c_v]
        ], dtype=torch.float32, device=device, requires_grad=False)

    else:
        # keep tracking the gradients, devices, and dtype
        return torch.stack([torch.stack([c_v*c_w, s_u*s_v*c_w-c_u*s_w, s_u*s_w+c_u*s_v*c_w]),
                            torch.stack([c_v*s_w, c_u*c_w+s_u*s_v*s_w, c_u*s_v*s_w-s_u*c_w]),
                            torch.stack([-s_v, s_u*c_v, c_u*c_v])])


def angle_to_rotation_matrix_vectorized(rot):
    """
    Transform vector of Euler angles to rotation matrix (vectorized version)
    ref: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    :param rot: euler angle PyTorch Tensor (shape (N, 3), each column represents roll-pitch-yaw or x-y-z respectively)
    :return: N*3*3 rotation matrix
    """
    # get batch of roll, pitch, yaw
    u, v, w = rot[:, 0], rot[:, 1], rot[:, 2]

    # calculate the intermediate values
    s_u, c_u = u.sin(), u.cos()
    s_v, c_v = v.sin(), v.cos()
    s_w, c_w = w.sin(), w.cos()
    a00 = c_v*c_w
    a01 = s_u*s_v*c_w-c_u*s_w
    a02 = s_u*s_w+c_u*s_v*c_w
    a10 = c_v*s_w
    a11 = c_u*c_w+s_u*s_v*s_w
    a12 = c_u*s_v*s_w-s_u*c_w
    a20 = -s_v
    a21 = s_u*c_v
    a22 = c_u*c_v

    row1 = torch.cat([a00.unsqueeze(1), a01.unsqueeze(1), a02.unsqueeze(1)], 1)
    row2 = torch.cat([a10.unsqueeze(1), a11.unsqueeze(1), a12.unsqueeze(1)], 1)
    row3 = torch.cat([a20.unsqueeze(1), a21.unsqueeze(1), a22.unsqueeze(1)], 1)

    return torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], 1)


def proj_lidar(scan, T, P0, resolution, R0=None, crop=None, only_pts=False):
    """
    Project Lidar point cloud onto image plane by a given transformation matrix T

    :param scan: point cloud scan in pytorch Tensor, shape [N_points, 4]
    :param T: transformation matrix, pytorch Tensor, shape (4, 4)
    :param P0: intrinsic cam P matrix (left), pytorch Tensor, used in standard projection
    :param R0: Left Rectification matrix (3*3 rotation matrix), pytorch Tensor
    :param resolution: image origin resolution, shape (H,W).
    :param crop: Optional[Tuple[int]], crop size in tuple (H, W)
    :param only_pts: bool, if True, return 2d index in camera's view, and corresponding depth values from Lidar
    :return: projected image shape Tensor
    """
    if crop is None:
        xmin, ymin = 0, 0
        xmax, ymax = resolution[1], resolution[0]
        new_shape = (resolution[0], resolution[1])
    else:
        ymin, ymax, xmin, xmax = calc_crop_bbox(resolution, crop)
        new_shape = (ymax-ymin, xmax-xmin)

    # Reflectance > 0
    pts3d = scan[scan[:, 3] > 0, :]
    pts3d[:, 3] = 1

    # project
    if R0 is None:
        pts3d_cam = T.mm(pts3d.t())
    else:
        pts3d_cam = R0.mm(T.mm(pts3d.t()))

    # Before projecting, keep only points with z>0
    # (points that are in front of the camera).
    idx = pts3d_cam[2, :] > 0
    pts2d_cam = P0.mm(pts3d_cam[:, idx])

    # get projected 2d & 3d points
    pts3d = pts3d[idx]
    pts2d = pts2d_cam / pts2d_cam[2, :]

    # keep points projected in the image plane
    pts2d = pts2d.t().round().type(torch.long)
    mask = (xmin < pts2d[:, 0]) & (pts2d[:, 0] < xmax) & \
           (ymin < pts2d[:, 1]) & (pts2d[:, 1] < ymax)
    pts2d = pts2d[mask][:, :2]  # keep only coordinates
    pts3d = pts3d[mask][:, 0]  # keep only x values of scan (depth)

    if crop is not None:
        pts2d[:, 0] = pts2d[:, 0] - xmin  # shift coordinates for assigning values
        pts2d[:, 1] = pts2d[:, 1] - ymin

    if only_pts:
        return pts2d, pts3d

    # create depth map
    return draw_depth_map(new_shape, pts2d, pts3d)  # disparity: bs*fu/pts3d


def draw_depth_map(shape, idx, val):
    """
    create an empty image, put values into desired coordinates

    :param shape: (H, W), image shape to be created
    :param idx: shape (N, 2), dtype long, a series of coordinates, each of them is (h, w) index on the image
    :param val: values to assign on the image
    :return: the interpreted image
    """
    depth = torch.zeros(shape, dtype=torch.float32, device=DEVICE, requires_grad=False)
    depth[idx[:, 1], idx[:, 0]] = val
    return depth


def proj_lidar_new(pc, T, P0, resolution, R0=None, crop=None, min_dist=2.0,
                   return_idx_depth=False, move_idx=True):
    """
    Project Lidar point cloud onto image plane by a given transformation matrix T

    :param pc: point cloud scan in pytorch Tensor, shape [N_points, 4]
    :param T: transformation matrix, pytorch Tensor, shape (4, 4)
    :param P0: intrinsic cam P matrix (left), pytorch Tensor, used in standard projection
    :param R0: Left Rectification matrix (3*3 rotation matrix), pytorch Tensor
    :param resolution: image origin resolution, shape (H,W).
    :param crop: Optional[Tuple[int]], crop size in tuple (H, W)
    :param min_dist: minimum distance to exclude a Lidar point
                     (to avoid get a point detected from cameras reflections), float
    :param return_idx_depth: bool, if True, return 2d index in camera's view, and corresponding depth values from Lidar
    :param move_idx: bool, if True, the calculated 2d coordinates will be shifted when cropping is used
                           (to make the coordinates start from 0)
    :return: projected image shape Tensor
    """
    # Reflectance > 0
    pts3d = pc[pc[:, 3] > 0, :]
    pts3d[:, 3] = 1

    # transform from lidar's frame to reference camera's frame
    if R0 is None:
        pts3d_cam = T.mm(pts3d.t())  # in "pykitti" lib, R (rectification) is added in T (transformation)
    else:
        pts3d_cam = R0.mm(T.mm(pts3d.t()))

    # extract the depth
    # (x axis in Lidar's frame, but z axis in reference camera's frame after transformation)
    depths = pts3d_cam[2, :]

    # project camera intrinsic matrix and normalize on third dimension to get coordinates
    pts2d_cam = P0.mm(pts3d_cam)
    pts2d = pts2d_cam / pts2d_cam[2, :]

    # calculate the camera view's new boundaries if cropping
    if crop is None:
        xmin, ymin = 0, 0
        xmax, ymax = resolution[1], resolution[0]
        new_shape = resolution
    else:
        ymin, ymax, xmin, xmax = calc_crop_bbox(resolution, crop)
        new_shape = (ymax-ymin, xmax-xmin)

    # sift points and only keep points in the camera view and keep 1 pixel margin
    mask = torch.ones(depths.shape[0], dtype=torch.bool, device=DEVICE)
    mask = torch.logical_and(mask, depths > min_dist)
    mask = torch.logical_and(mask, pts2d[0, :] > xmin+1)
    mask = torch.logical_and(mask, pts2d[0, :] < xmax-1)
    mask = torch.logical_and(mask, pts2d[1, :] > ymin+1)
    mask = torch.logical_and(mask, pts2d[1, :] < ymax-1)
    pts2d = pts2d[:, mask]
    depths = depths[mask]
    # transpose the coordinates back to normal shape (that I prefer) and convert to int (for slicing)
    pts2d = pts2d.t()
    pts2d = pts2d.round().type(torch.long)

    # return unshifted 2D coordinates
    if return_idx_depth and not move_idx:
        return pts2d, depths

    # move (shift) the 2d coordinates to start from 0
    # when you want to get the value from the image that has already been cropped,
    # if you want to get the value from full image, no coordinates movement needed.
    if crop is not None:
        pts2d[:, 0] = pts2d[:, 0] - xmin  # shift coordinates for assigning values
        pts2d[:, 1] = pts2d[:, 1] - ymin

    # return shifted 2D coordinates
    if return_idx_depth and move_idx:
        return pts2d, depths

    # draw depth map
    return draw_depth_map(new_shape, pts2d, depths)


def lidar_depth_maps_check(dataset, cam=0, crop=None, proj_func=proj_lidar, ):
    """
    Create lidar depth map tensor for all lidar scans

    :param dataset: pykitti.raw (or other data loader has the same api) object
    :param proj_func: project function
    :param cam: camera number, default use grayscale left camera for projection
    :param crop: Optional[Tuple[int]], crop size in tuple (H, W)
    :return: depth maps in one tensor, shape (frames, H, W)
    """
    # get output shape
    out_shape = np.array(dataset.get_cam0(0)).shape
    num_frames = len(dataset)

    # create output
    maps = torch.zeros((num_frames,)+out_shape, device=DEVICE, requires_grad=False)

    # get calibration parameters
    if cam == 0:
        T = torch.from_numpy(dataset.calib.T_cam0_velo).float().cuda()
        P0 = torch.from_numpy(dataset.calib.P_rect_00).float().cuda()
    elif cam == 1:
        T = torch.from_numpy(dataset.calib.T_cam1_velo).float().cuda()
        P0 = torch.from_numpy(dataset.calib.P_rect_10).float().cuda()
    elif cam == 2:
        T = torch.from_numpy(dataset.calib.T_cam2_velo).float().cuda()
        P0 = torch.from_numpy(dataset.calib.P_rect_20).float().cuda()
    elif cam == 3:
        T = torch.from_numpy(dataset.calib.T_cam3_velo).float().cuda()
        P0 = torch.from_numpy(dataset.calib.P_rect_30).float().cuda()
    else:
        LOG.error('Could not determine the camera.')
        return None

    # compute depth maps for lidar
    for i, scan in enumerate(dataset.velo):
        lidar = torch.from_numpy(scan).float().cuda()
        maps[i] = proj_func(lidar, T, P0, out_shape, crop=crop)

    return maps
