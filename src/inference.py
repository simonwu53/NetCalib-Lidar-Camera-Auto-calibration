import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import normalize
import cv2
from tqdm import tqdm
import logging
import os
import time
from config import *
from net import LidarProjection, Loss
from train import loss_batch, proj_module


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('test')


def num_to_rgb(val, max_val=83, bgr=True, ignore_zeros=False):
    """
    map a list of values to RGB heat map
    :param val: list of values, List[int or float]
    :param max_val: float, maximum of given values
    :param bgr: if True, return BGR values instead of RGB
    :param ignore_zeros: if True, ignore mapping zeros
    :return: mapped color values, shape (N, 3), uint8
    """
    if np.any(val > max_val):
        max_val = np.max(val)
        # LOG.warning("[num_to_rgb] val %.2f is greater than max_val %.2f." % (np.max(val), max_val))
    if np.any(np.logical_or(val < 0, max_val < 0)):
        raise ValueError("arguments may not be negative")

    # available colors (RGB)
    colors = np.zeros((1024, 3), dtype=np.uint8)
    colors[:256, 0] = 255
    colors[:256, 1] = np.arange(256)
    colors[256:512, 0] = np.arange(255, -1, -1)
    colors[256:512, 1] = 255
    colors[512:768, 1] = 255
    colors[512:768, 2] = np.arange(256)
    colors[768:1024, 1] = np.arange(255, -1, -1)
    colors[768:1024, 2] = 255

    valScaled = np.floor(map_to_range(val, 0, max_val, 0, 1023)).astype(np.int32)

    mask = val == 0 if ignore_zeros else np.zeros(valScaled.shape[0], dtype=np.bool)

    new_val = colors[valScaled]
    new_val[mask] = 0

    if not bgr:
        # rgb
        return new_val
    else:
        # bgr
        new_val[:, [0, 1, 2]] = new_val[:, [2, 1, 0]]  # swap channels
        return new_val


def map_to_range(val, old_min, old_max, new_min, new_max):
    """
    map values from a range to another
    :param val: single int or float value or a list of values
    :param old_min: old values' minimum
    :param old_max: old values' maximum
    :param new_min: new values' minimum
    :param new_max: new values' maximum
    :return: mapped (float) values
    """
    old_range = old_max - old_min
    new_range = new_max - new_min
    valScaled = (val - old_min) / old_range  # to 0-1 range
    return new_min + (valScaled * new_range)  # to new range


def mix_layers(base, layer):
    """
    mix projected LiDAR scan with background image
    :param base: ndarray, background image
    :param layer: ndarray, projected LiDAR scan, should be the same size as base
    :return: mixed image
    """
    bg = base.copy()
    mask = layer!=0
    bg[mask] = layer[mask]
    return bg


def run(model, dataset, loader, config):
    # folders for training & results
    log_dir = config.get('log_dir', None)
    save_dir = config.get('save_dir', None)
    save_root = config.get('save_root', None)

    # initialize tensorboard
    LOG.warning('Adding tensorboard writer...')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir, max_queue=20, flush_secs=60, filename_suffix='-test')

    # check point status
    global_i = config.get('global_step', 0)  # record global steps
    trained_epoch = config.get('epoch', 0)  # record trained epochs
    iterative = config.get('iter', False)
    valid_loss = 0.0  # testing status
    phi = torch.zeros((1, 6), dtype=torch.float32, requires_grad=False).cuda() if iterative else None
    delay = 0  # player control
    test_err = None
    empty_err = torch.eye(4).unsqueeze(0).float().cuda()

    # calculate statistics
    loss_stat = np.zeros((len(loader), 6), dtype=np.float64)

    # create loss function
    criteria = Loss.Loss(mode='L1', reduction='sum', alpha=0.7, beta=0.3)

    LOG.warning('Loaded checkpoint: Global iteration %d, Trained epoch %d' % (global_i, trained_epoch))
    LOG.warning('Starting testing, iterative mode: %s; dataset length: %d.' % ('True' if config['iter'] else 'False',
                                                                               len(loader)))

    # create view window
    winScale = config.get('win_scale', 2.5)
    cv2.namedWindow('Stereo Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Depth', int(CROP[1]*winScale), int(CROP[0]*winScale))
    cv2.namedWindow('Lidar Depth with Error', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lidar Depth with Error', int(CROP[1]*winScale), int(CROP[0]*winScale))
    cv2.namedWindow('Lidar Depth Fixed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lidar Depth Fixed', int(CROP[1]*winScale), int(CROP[0]*winScale))
    cv2.namedWindow('Lidar Depth GT', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lidar Depth GT', int(CROP[1]*winScale), int(CROP[0]*winScale))
    # cv2.resizeWindow('Stereo Depth', 800, 800)

    # iterating over dataset
    LOG.warning('Evaluation on val_data...')
    model.eval()
    pbar = tqdm(loader)
    with torch.no_grad():
        for i, data in enumerate(pbar):
            t1 = time.time()
            # unpack data
            cam_raw, lidar_pc, T, P0, rot, trans, shape, cam2 = \
                data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), \
                data[4].cuda(), data[5].cuda(), data[6][0].cuda(), data[7][0].cuda()

            # standardize depth input
            cam_depth = normalize(cam_raw, mean=(CAM_MEAN,), std=(CAM_STD,)).unsqueeze(0)  # add batch dim

            # if iterative mode, use static error
            if iterative:
                if test_err is None:
                    test_err = torch.cat((rot, trans), dim=1)
                    gt = dataset.standardize_err((rot, trans))
                    T_err = LidarProjection.phi_to_transformation_matrix_vectorized(test_err)

            # otherwise use random error in every loop
            else:
                test_err = torch.cat((rot, trans), dim=1)
                gt = dataset.standardize_err((rot, trans))
                T_err = LidarProjection.phi_to_transformation_matrix_vectorized(test_err)

            # project lidar point cloud
            lidar_depth, lidar_depth_err = proj_module(lidar_pc, T, T_err, P0, shape, phi=phi, standardize=True)
            # model inference
            out = model(lidar_depth, cam_depth)
            # loss metric
            loss = loss_batch(out, gt, criteria)
            # running loss
            valid_loss += loss.item()
            t2 = time.time()-t1

            # update model guessed error
            pred_destandardized = torch.cat(dataset.destandardize_err(out), dim=1)
            if iterative:
                phi = pred_destandardized

            # visualization
            lidar_depth_fixed = proj_module(lidar_pc, T, T_err, P0, shape,
                                            phi=pred_destandardized,
                                            standardize=False)
            lidar_depth_gt = proj_module(lidar_pc, T, empty_err, P0, shape,
                                         standardize=False)
            vis_cam = cam_raw[0].cpu().numpy()
            vis_cam_rgb = num_to_rgb(vis_cam.flatten(), ignore_zeros=True).reshape(CROP+(3,))
            vis_lidar_err = lidar_depth_err.cpu().numpy()  # error projection
            vis_lidar_err_rgb = num_to_rgb(vis_lidar_err.flatten(), ignore_zeros=True).reshape(CROP+(3,))
            vis_lidar_fix = lidar_depth_fixed.cpu().numpy()
            vis_lidar_fix_rgb = num_to_rgb(vis_lidar_fix.flatten(), ignore_zeros=True).reshape(CROP+(3,))
            vis_lidar_gt = lidar_depth_gt.cpu().numpy()
            vis_lidar_gt_rgb = num_to_rgb(vis_lidar_gt.flatten(), ignore_zeros=True).reshape(CROP+(3,))

            cam_base = cv2.cvtColor(cam2.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            vis_cam_mix = mix_layers(cam_base, vis_cam_rgb)
            vis_lidar_err_mix = mix_layers(cam_base, vis_lidar_err_rgb)
            vis_lidar_fix_mix = mix_layers(cam_base, vis_lidar_fix_rgb)
            vis_lidar_gt_mix = mix_layers(cam_base, vis_lidar_gt_rgb)

            cv2.imshow('Stereo Depth', vis_cam_mix)
            cv2.imshow('Lidar Depth with Error', vis_lidar_err_mix)
            cv2.imshow('Lidar Depth Fixed', vis_lidar_fix_mix)
            cv2.imshow('Lidar Depth GT', vis_lidar_gt_mix)

            # write statistics to tensorboard
            a, b, c, x, y, z = (pred_destandardized-test_err).cpu().numpy()[0]
            a, b, c, x, y, z = np.rad2deg(a), np.rad2deg(b), np.rad2deg(c), x*100, y*100, z*100
            test_err = test_err.cpu().numpy()
            writer.add_scalar('Test/Loss', loss.item(), i)
            writer.add_scalar('Test/Loss/avg', valid_loss / (i+1), i)
            writer.add_scalar('Test/Loss/Roll', a, i)
            writer.add_scalar('Test/Loss/Pitch', b, i)
            writer.add_scalar('Test/Loss/Yaw', c, i)
            writer.add_scalar('Test/Loss/X', x, i)
            writer.add_scalar('Test/Loss/Y', y, i)
            writer.add_scalar('Test/Loss/Z', z, i)
            writer.add_scalar('Test/Error/Roll', np.rad2deg(test_err[0][0]), i)
            writer.add_scalar('Test/Error/Pitch', np.rad2deg(test_err[0][1]), i)
            writer.add_scalar('Test/Error/Yaw', np.rad2deg(test_err[0][2]), i)
            writer.add_scalar('Test/Error/X', test_err[0][3]*100, i)
            writer.add_scalar('Test/Error/Y', test_err[0][4]*100, i)
            writer.add_scalar('Test/Error/Z', test_err[0][5]*100, i)
            writer.flush()

            # collect statistics
            loss_stat[i] = (a, b, c, x, y, z)
            pbar.set_description("Loss: Row{:6.2f} Pitch{:6.2f} Yaw{:6.2f} X{:7.2f} Y{:7.2f} Z{:7.2f} speed {:6.4f}s/it".format(a, b, c, x, y, z, t2))

            # wait key
            key = cv2.waitKey(delay)
            if key == ord('q') or key == 27:  # Q or ESC
                break
            if key == ord(' '):  # play & pause
                delay = 50 if delay == 0 else 0
            if key == ord('s'):  # save screenshots
                for frame, label in zip((vis_cam_mix, vis_lidar_err_mix, vis_lidar_fix_mix, vis_lidar_gt_mix),
                                        ('stereo', 'lidar-err', 'lidar-fix', 'lidar-gt')):
                    cv2.imwrite(os.path.join(save_root, 'test-%d-%s.png' % (i+1, label)), frame)

        # compute validation results
        valid_res = valid_loss / len(loader)
        LOG.warning('Testing running average loss: %.4f' % valid_res)

    if TENSORBOARD:
        writer.close()

    print()
    LOG.warning("[Statistics]")
    LOG.warning("Mean: Row{:6.2f} Pitch{:6.2f} Yaw{:6.2f} X{:7.2f} Y{:7.2f} Z{:7.2f}".format(*np.mean(loss_stat, axis=0).tolist()))
    LOG.warning("Min: Row{:6.2f} Pitch{:6.2f} Yaw{:6.2f} X{:7.2f} Y{:7.2f} Z{:7.2f}".format(*np.min(loss_stat, axis=0).tolist()))
    LOG.warning("Max: Row{:6.2f} Pitch{:6.2f} Yaw{:6.2f} X{:7.2f} Y{:7.2f} Z{:7.2f}".format(*np.max(loss_stat, axis=0).tolist()))
    LOG.warning("STD: Row{:6.2f} Pitch{:6.2f} Yaw{:6.2f} X{:7.2f} Y{:7.2f} Z{:7.2f}".format(*np.std(loss_stat, axis=0).tolist()))
    return
