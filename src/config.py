from datetime import datetime
from os.path import join
import numpy as np
""" System configuration parameters """

# signals to handle
SIGDICT = {2: 'SIGINT', 15: 'SIGTERM'}

# Set global save path
PROJECT_BASE = '/home/user/NetCalib-Lidar-Camera-Auto-calibration/'
DATETIME = datetime.now().strftime("%H%M_%d%m%Y")
SAVE_ROOT = join(PROJECT_BASE, 'results/', DATETIME)
LOG_DIR = join(SAVE_ROOT, 'log/')
SAVE_DIR = join(SAVE_ROOT, 'ckpt')

# 1. Train Data
KITTI_TRAIN_BASE = '/home/user/datasets/KITTI/train/'
KITTI_VAL_BASE = '/home/user/datasets/KITTI/val/'
KITTI_TEST_BASE = '/home/user/datasets/KITTI/test/'
LIDAR_MEAN = 0.8119
LIDAR_STD = 4.0786
CAM_MEAN = 4.1367
CAM_STD = 9.5892

# 2. Neural Network Training
# 2.1 Err choose
ERR_GROUP = {
    1: {'T': 1.5, 'R': np.deg2rad(20)},  # unit: T-meters, R-degrees
    2: {'T': 1.0, 'R': np.deg2rad(10)},
    3: {'T': 0.5, 'R': np.deg2rad(5)},
    4: {'T': 0.2, 'R': np.deg2rad(2)},
    5: {'T': 0.1, 'R': np.deg2rad(1)},
}

# 2.2 Torch Dataset Loader
PIN_MEMORY = True
DROP_LAST = True
DEVICE = 'cuda'
BATCHSIZE = 1
SHUFFLE = True
NUM_WORKERS = 4
SPARSE_VAL = 0.0
CROP = (256, 768)  # (256, 960) (370, 1200)

# 2.3 Training loop
EPOCHS = 15
LR = 1e-3
L2 = 0.0
OPTIMIZER = 'rmsprop'
PLATEAU = 0
TB_SCALAR_UPDATE = 100
TB_STATUS_UPDATE = 5000  # loops interval to show training process information or record info in tensorboard
VALID_INTERVAL = 20000
TENSORBOARD = True

# 2.4 LOSS
A = 1.0
B = 1.0
# -----------------------------
