from typing import Mapping
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import os
import pykitti2
from net.RandomError import ErrorGenerator
from config import *
from net.utils import center_crop


# setting LOG
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Net-DataLoader')


def get_dataloader(mode: str, err_dist: str, err_config: Mapping):
    """
    create dataloader from a pytorch dataset for training.

    :param mode: select from 'train' or 'val', which uses different base path for the dataset.
    :param err_dist: select from 'uniform' or 'normal', distribution used to generate random errors
    :param err_config: a dictionary contains two keys: 'R' and 'T'
                       if dist is 'uniform', 'R' is the range for rotation [a, b) where a+b=0 (symmetric), the same for 'T' as translation
                       if dist is 'normal', 'R' is a tuple which specify (mean, std) value pair for rotation, the same for 'T' as translation
    :return: pytorch DataLoader
    """
    LOG.warning('Creating %s Data Loader.' % mode)
    if PIN_MEMORY:
        LOG.warning('Pin memory flag enabled.')
    if DROP_LAST:
        LOG.warning('Last uncomplete batch will be dropped.')
    if SHUFFLE:
        LOG.warning('Batch samples will be shuffled.')
    LOG.warning('Batch size set to %d.' % BATCHSIZE)
    LOG.warning('Number of workers for data loader: %d' % NUM_WORKERS)

    if mode == 'train':
        # create dataset
        dataset = TrainDataSet(mode='train', err_dist=err_dist, err_config=err_config)

        # Prepare batches
        loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, drop_last=DROP_LAST, timeout=2)
    elif mode == 'val':
        # create dataset
        dataset = TrainDataSet(mode='val', err_dist=err_dist, err_config=err_config)

        # Prepare batches
        loader = DataLoader(dataset, batch_size=1, shuffle=SHUFFLE, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, drop_last=DROP_LAST, timeout=2)
    elif mode == 'test':
        # create dataset
        dataset = TestDataSet(mode='test', err_dist=err_dist, err_config=err_config)

        # Prepare batches
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, timeout=2)
    else:
        LOG.error('Unknown mode for load_model: %s' % mode)
        return

    LOG.warning('Total samples in the data loader: %d' % len(dataset))
    return dataset, loader


def read_depth(img, sparse_val=0):
    """
    Convert a PIL image (mode="I") to the depth map (np.ndarray)

    :param img: PIL image object
    :param sparse_val: value to encode sparsity with
    :return: depth map in np.ndarray
    """
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(img, dtype=np.float32)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)
    depth = depth_png / 256
    if sparse_val > 0:
        depth[depth_png == 0] = sparse_val
    return depth  # shape (H, W)


class TrainDataSet(Dataset):
    def __init__(self, mode: str, err_dist: str, err_config: Mapping):
        """
        Train Dataset used by the DataLoader during the training

        :param mode: select from 'train' or 'val', which uses different base path.
        :param err_dist: select from 'uniform' or 'normal', distribution used to generate random errors
        :param err_config: a dictionary contains two keys: 'R' and 'T'
                           if dist is 'uniform', 'R' is the range for rotation [a, b) where a+b=0 (symmetric), the same for 'T' as translation
                           if dist is 'normal', 'R' is a tuple which specify (mean, std) value pair for rotation, the same for 'T' as translation
        """
        # error generator
        self.errgen = ErrorGenerator(err_dist, err_config)
        # self.errgen = None

        # store all data loaders
        self.loader = {}
        self.len = 0  # total training frames (imgs)
        self.sparse_val = SPARSE_VAL
        self.max_lidar_quantity = 130000  # maximum points in a lidar scan
        self.crop = CROP
        if mode == 'train':
            kitti_base = KITTI_TRAIN_BASE
        elif mode == 'val':
            kitti_base = KITTI_VAL_BASE
        elif mode == 'test':
            kitti_base = KITTI_TEST_BASE
        else:
            LOG.error('Unknown mode when initialize TrainDataSet: %s' % mode)
            exit(0)
        self.mode = mode

        # get dates in training folder
        dates = os.listdir(kitti_base)
        dates_str = ', '.join(dates)
        LOG.warning("Found following dates in '%s' folder: %s" % (mode, dates_str))
        # get drives in date folder
        for date in dates:
            drives = os.listdir(os.path.join(kitti_base, date))
            T = None  # each date has different calibration matrix
            P0 = None
            for drive in drives:
                if drive.endswith('sync'):
                    loader = pykitti2.raw(kitti_base, date, drive[17:21])
                    num_depths = loader.get_depth_len()

                    if T is None:
                        T = loader.calib.T_cam2_velo.astype(np.float32)
                        P0 = loader.calib.P_rect_20.astype(np.float32)

                    # register loader in dict and its ground truth calibrations
                    self.loader[range(self.len, self.len+num_depths)] = (loader, T, P0)
                    self.len += num_depths  # accumulate length
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        depth, lidar, T, P0, rot, trans, shape = None, None, None, None, None, None, None
        # lidar point cloud container
        lidar_np = np.zeros((self.max_lidar_quantity, 4), dtype=np.float32)

        # find idx in which data loader
        for k in self.loader:
            if idx in k:
                # load data
                loader, T, P0 = self.loader[k]
                # get real idx
                idx = idx-k.start

                # get depth map
                raw_depth = read_depth(loader.get_cam2d(idx), self.sparse_val)  # (H,W)
                shape = np.array(raw_depth.shape)  # (H, W)
                depth = center_crop(raw_depth, CROP)

                # get lidar point cloud
                pc = loader.get_velo(idx+5)
                lidar_np[:pc.shape[0]] = pc
                lidar = torch.from_numpy(lidar_np).float()
                T = torch.from_numpy(T).float()
                P0 = torch.from_numpy(P0).float()

                # generate randomized error
                rot, trans = self.errgen.gen_err()

                break  # avoid unnecessary loops

        if depth is None:
            LOG.error('Could not find the depth data by idx %d!!' % idx)
            raise ValueError('Could not find the depth data by idx.')
        return depth, lidar, T, P0, rot, trans, shape

    def destandardize_err(self, err):
        return self.errgen.destandardize_err(err)

    def standardize_err(self, err):
        return self.errgen.standardize_err(err)


class TestDataSet(TrainDataSet):
    def __init__(self, mode: str, err_dist: str, err_config: Mapping, crop=True):
        super().__init__(mode, err_dist, err_config)
        self.crop = crop
        self.current_pointer = 0  # current selected index (not global, for selected loader only)
        self.current_loader = None  # current selected loader
        return

    def __getitem__(self, idx):
        depth, lidar, T, P0, rot, trans, shape, cam2 = None, None, None, None, None, None, None, None
        # lidar point cloud container
        lidar = np.zeros((self.max_lidar_quantity, 4), dtype=np.float32)

        # find idx in which data loader
        for k in self.loader:
            if idx in k:
                # load data
                loader, T, P0 = self.loader[k]
                # get real idx
                idx = idx - k.start
                # store pointer
                self.current_pointer = idx
                self.current_loader = loader

                # get color cam
                if self.crop:
                    cam2 = center_crop(np.array(loader.get_cam2(idx + 5), dtype=np.uint8).transpose(2, 0, 1), CROP)
                else:
                    cam2 = np.array(loader.get_cam2(idx + 5), dtype=np.uint8).transpose(2, 0, 1)

                # get depth map
                raw_depth = read_depth(loader.get_cam2d(idx), self.sparse_val)  # (H,W)
                shape = np.array(raw_depth.shape)  # (H, W)
                if self.crop:
                    depth = center_crop(raw_depth, CROP)
                else:
                    depth = raw_depth

                # get lidar point cloud
                pc = loader.get_velo(idx + 5)
                lidar[:pc.shape[0]] = pc

                # PATCH 11.11.2020: remove torch conversion for better compatibility
                #                   PyTorch will do this after loaded data automatically
                # lidar = torch.from_numpy(lidar).float()
                # T = torch.from_numpy(T).float()
                # P0 = torch.from_numpy(P0).float()
                # END OF PATCH

                # generate randomized error
                rot, trans = self.errgen.gen_err()

                break  # avoid unnecessary loops

        if depth is None:
            LOG.error('Could not find the depth data by idx %d!!' % idx)
            raise ValueError('Could not find the depth data by idx.')
        return depth, lidar, T, P0, rot, trans, shape, cam2


class DataStat(IterableDataset):
    def __init__(self):
        """
        This Dataset is used for calculate mean and standard deviation of the target dataset.
        This class provides an iterable, hence batch_size must be set to 1, set shuffle to False, set worker to 0
        Calculation happened on GPU.
        """
        self.len = 0

        # searching all training folders to get total length
        dates = os.listdir(KITTI_TRAIN_BASE)
        for date in dates:
            drives = os.listdir(os.path.join(KITTI_TRAIN_BASE, date))

            for drive in drives:
                if drive.endswith('sync'):
                    num_depths = pykitti2.raw(KITTI_TRAIN_BASE, date, drive[17:21]).get_depth_len()
                    self.len += num_depths
        return

    def __len__(self):
        return self.len

    def __iter__(self):
        # get dates in training folder
        dates = os.listdir(KITTI_TRAIN_BASE)

        # get drives in date folder
        for date in dates:
            drives = os.listdir(os.path.join(KITTI_TRAIN_BASE, date))
            T = None  # each date has different calibration matrix
            P0 = None
            for drive in drives:
                if drive.endswith('sync'):
                    loader = pykitti2.raw(KITTI_TRAIN_BASE, date, drive[17:21])
                    num_depths = loader.get_depth_len()

                    if T is None:
                        T = loader.calib.T_cam2_velo.astype(np.float32)
                        P0 = loader.calib.P_rect_20.astype(np.float32)

                    # yield dataset
                    for idx in range(num_depths):
                        # Cam: shape (1, H, W), (H,W)=CROP;  # Lidar: shape (N, 4)
                        depth = loader.get_cam2d(idx)
                        shape = np.array(depth.size[::-1])
                        yield center_crop(read_depth(depth), CROP), loader.get_velo(idx+5), T, P0, shape

# notes
# https://github.com/JiaxiongQ/DeepLiDAR/blob/master/dataloader/trainLoader.py
# https://github.com/wvangansbeke/Sparse-Depth-Completion/blob/master/Datasets/dataloader.py # delete 8 files
