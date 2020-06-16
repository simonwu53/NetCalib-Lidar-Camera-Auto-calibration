import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import logging
import os
import shutil
from config import *
from net import LidarProjection, Loss

# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('train')


def save_checkpoint(model, optmizer, epoch, global_step, save_path):
    """
    Save the trained model checkpoint with a given name

    :param model: pytorch model to save
    :param optmizer: optimizer to save
    :param epoch: current epoch value
    :param global_step: current global step for tensorboard
    :param save_path: path to save the model
    """
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optmizer.state_dict(),
    }, save_path)
    return


def load_checkpoint(model, path, optimizer=None):
    """
    Load a pre-trained model on GPU for training or evaluation

    :param model: pytorch model object to load trained parameters
    :param optimizer: optimizer object used in the last training
    :param path: path to the saved checkpoint
    """
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return epoch, global_step


def proj_module(scans, T, T_err, P0, shape, phi=None, standardize=True):
    if phi is not None:
        # phi is the guess from model (error)
        T_phi = LidarProjection.phi_to_transformation_matrix_vectorized(phi)
        # T_fix is the inverse of the error transformation (phi)
        T_fix = LidarProjection.inv_transform_vectorized(T_phi)
        # compose the final transformation matrix: T_fix*T_err_generated*T_gt
        T_composed = T_fix.bmm(T_err.bmm(T))
    else:
        T_composed = T_err.bmm(T)
    lidar_depth = LidarProjection.proj_lidar(scans[0], T_composed[0], P0[0], shape, crop=CROP)

    if standardize:
        lidar_depth_norm = normalize(lidar_depth.unsqueeze(0), mean=(LIDAR_MEAN,), std=(LIDAR_STD,))  # adds batch dim
        return lidar_depth_norm.unsqueeze(0), lidar_depth  # adds channel dimensions
    else:
        return lidar_depth  # shape (H, W)


def loss_batch(logits, gt, loss_func, opt=None):
    loss = loss_func(logits, gt)

    if opt is not None:
        # auto-calculate gradients
        loss.backward()
        # apply gradients
        opt.step()
        # zero the parameter gradients
        opt.zero_grad()
    return loss


def write_summary(writer, raw_errors, losses, step, mode, others=None):
    a, b, c, x, y, z = raw_errors
    la, lb, lc, lx, ly, lz = losses
    writer.add_scalar('%s/Loss/Roll' % mode, la, step)
    writer.add_scalar('%s/Loss/Pitch' % mode, lb, step)
    writer.add_scalar('%s/Loss/Yaw' % mode, lc, step)
    writer.add_scalar('%s/Loss/X' % mode, lx, step)
    writer.add_scalar('%s/Loss/Y' % mode, ly, step)
    writer.add_scalar('%s/Loss/Z' % mode, lz, step)
    writer.add_scalar('%s/Error/Roll' % mode, a, step)
    writer.add_scalar('%s/Error/Pitch' % mode, b, step)
    writer.add_scalar('%s/Error/Yaw' % mode, c, step)
    writer.add_scalar('%s/Error/X' % mode, x, step)
    writer.add_scalar('%s/Error/Y' % mode, y, step)
    writer.add_scalar('%s/Error/Z' % mode, z, step)
    if others is not None:
        for tag in others:
            writer.add_scalar(tag, others[tag], step)
    writer.flush()
    return


def run(model, optimizer, datasets, loaders, config):
    # folders for training & results
    log_dir = config.get('log_dir', LOG_DIR)
    save_dir = config.get('save_dir', SAVE_DIR)
    save_root = config.get('save_root', SAVE_ROOT)
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # save configs
    shutil.copy2('./config.py', save_root)
    shutil.copy2('./net/CalibrationNet.py', save_root)

    # training params
    graph_loaded = False  # tensorboard graph status
    global_i = config.get('global_step', 0)  # record global steps
    global_j = 0
    trained_epoch = config.get('epoch', 0)  # record trained epochs
    best_val = 10000  # record the best validation loss
    train_dataset, val_dataset = datasets  # unpack datasets
    train_loader, val_loader = loaders  # unpack the loaders

    # config tensor board
    if TENSORBOARD:
        LOG.warning('Adding tensorboard writer...')
        if global_i != 0:
            writer = SummaryWriter(log_dir=log_dir, purge_step=global_i, max_queue=10,
                                   flush_secs=120, filename_suffix='-train')
        else:
            writer = SummaryWriter(log_dir=log_dir, max_queue=10, flush_secs=120, filename_suffix='-train')

    # create loss function
    criteria = Loss.Loss(mode='L1', reduction='sum', alpha=A, beta=B)

    # start training
    LOG.warning('Starting training, epoch size: %d, epoch number: %d.' % (len(train_loader), EPOCHS))
    for epoch in range(trained_epoch, trained_epoch+EPOCHS):
        running_loss = 0.0
        valid_loss = 0.0
        pbar = tqdm(train_loader)

        # iterating over dataset
        model.train()
        for i, data in enumerate(pbar):
            # unpack data
            # 0. cam_raw (1, H, W)
            # 1. Lidar (1, N, 4)
            # 2. T (1, 4, 4)
            # 3. P0 (1, 3, 4)
            # 4. rot (1, 3)
            # 5. trans (1, 3)
            # 6. shape (H', W')
            cam_raw, lidar_pc, T, P0, rot, trans, shape = \
                data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), \
                data[4].cuda(), data[5].cuda(), data[6][0].cuda()

            # standardize input
            cam_depth = normalize(cam_raw, mean=(CAM_MEAN,), std=(CAM_STD,)).unsqueeze(0)  # add batch dim
            # standardize output
            gt = train_dataset.standardize_err((rot, trans))

            # forward + backward + optimize
            with torch.no_grad():
                T_err = LidarProjection.phi_to_transformation_matrix_vectorized(torch.cat((rot, trans), dim=1))
                lidar_depth, lidar_depth_err = proj_module(lidar_pc, T, T_err, P0, shape, standardize=True)
            out = model(lidar_depth, cam_depth)
            loss = loss_batch(out, gt, criteria, optimizer)

            # collect statistics
            running_loss += loss.item()
            pbar.set_description("Epoch %d, train running loss: %.4f" % (epoch+1, running_loss/(i+1)))

            # write to tensorboard
            if i % TB_SCALAR_UPDATE == 0 and TENSORBOARD:
                with torch.no_grad():
                    errors, model_pred = torch.cat((rot, trans), dim=1)[0], torch.cat(train_dataset.destandardize_err(out), dim=1)[0]
                    write_summary(writer, errors, torch.abs(model_pred-errors).cpu().numpy(),
                                  global_i, 'Train', {'Train/Loss': loss.item()})

            # exit function if profiling
            if i % TB_STATUS_UPDATE == 0 and TENSORBOARD:  # something fired periodically
                if not graph_loaded:
                    writer.add_graph(model, (lidar_depth, cam_depth))
                    graph_loaded = True
                lidar_depth_fixed = proj_module(lidar_pc, T, T_err, P0, shape,
                                                phi=torch.cat(train_dataset.destandardize_err(out), dim=1),
                                                standardize=False)
                writer.add_image('Train/Camera Depth', cam_raw[0], global_i, dataformats='HW')
                writer.add_image('Train/Lidar Depth Err', lidar_depth_err, global_i, dataformats='HW')
                writer.add_image('Train/Lidar Depth Fix', lidar_depth_fixed, global_i, dataformats='HW')
                writer.flush()

            # update global step
            global_i += 1

            # validate model
            if global_i % VALID_INTERVAL == 0 and global_i != 0:
                # validation
                model.eval()
                with torch.no_grad():
                    for j, data in enumerate(val_loader):
                        # unpack data
                        cam_raw, lidar_pc, T, P0, rot, trans, shape = \
                            data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), \
                            data[4].cuda(), data[5].cuda(), data[6][0].cuda()

                        # standardize depth input
                        cam_depth = normalize(cam_raw, mean=(CAM_MEAN,), std=(CAM_STD,)).unsqueeze(0)  # add batch dim
                        # standardize output
                        gt = val_dataset.standardize_err((rot, trans))
                        # project lidar point cloud
                        T_err = LidarProjection.phi_to_transformation_matrix_vectorized(torch.cat((rot, trans), dim=1))
                        lidar_depth, lidar_depth_err = proj_module(lidar_pc, T, T_err, P0, shape, standardize=True)
                        # model inference
                        out = model(lidar_depth, cam_depth)
                        # loss metric
                        loss = loss_batch(out, gt, criteria)
                        # running loss
                        valid_loss += loss.item()
                        pbar.set_description("(%d/%d) Valid running loss: %.4f" % (j+1, len(val_loader), valid_loss/(j+1)))

                        global_j += 1
                        # update tensorboard
                        # write to TensorBoard
                        if TENSORBOARD and j % TB_SCALAR_UPDATE == 0:
                            # add imgs
                            if j == 0:
                                lidar_depth_fixed = proj_module(lidar_pc, T, T_err, P0, shape,
                                                                phi=torch.cat(val_dataset.destandardize_err(out), dim=1),
                                                                standardize=False)
                                writer.add_image('Validation/Camera Depth', cam_raw[0], global_j, dataformats='HW')
                                writer.add_image('Validation/Lidar Depth Err', lidar_depth_err, global_j,
                                                 dataformats='HW')
                                writer.add_image('Validation/Lidar Depth Fix', lidar_depth_fixed, global_j,
                                                 dataformats='HW')
                            # add scalars
                            errors, model_pred = torch.cat((rot, trans), dim=1)[0], \
                                                 torch.cat(val_dataset.destandardize_err(out), dim=1)[0]
                            write_summary(writer, errors, torch.abs(model_pred - errors).cpu().numpy(), global_j,
                                          'Validation', {'Validation/Loss': loss.item()})
                            writer.flush()

                    # compute validation results
                    valid_res = valid_loss / len(val_loader)

                    if valid_res < best_val:
                        best_val = valid_res
                        # save the best validation model
                        save_checkpoint(model, optimizer, (epoch + 1), global_i,
                                        join(save_dir, 'Epoch%dloss%.4f' % (epoch + 1, best_val) + '.tar'))
                # convert back to training mode
                model.train()

    if TENSORBOARD:
        writer.close()

    # save final model
    save_checkpoint(model, optimizer, trained_epoch+EPOCHS, global_i,
                    join(save_dir, 'Epoch%dloss%.4f' % (trained_epoch+EPOCHS, valid_res) + '.tar'))  # last validation loss
    return


if __name__ == '__main__':
    pass
