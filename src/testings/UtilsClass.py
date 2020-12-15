import random
import numpy as np
from DataLoader import TestDataSet
from testings.DataTypes import INT, FLOAT, DataSlice, DataLoader, Ndarray, Resolution, ERR_GROUP
from testings.UtilsFuncs import calib_np_to_phi, map_range
from typing import Optional, Callable, Tuple
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('UtilsClass')


# Decorator for checking data is loaded
def _checkDataLoaded(f: Callable) -> Callable:
    def checkDataWrapper(self, *args, **kwargs):
        if self.dataset is None:
            LOG.error("Dataset is not loaded. Use loadData() to load data first!")
            raise AssertionError("Dataset is not loaded. Use loadData() to load data first!")
        # call normal function
        return f(self, *args, **kwargs)

    return checkDataWrapper


# Class Tester
class Loader(object):

    def __init__(self):
        self.dataset: Optional[TestDataSet] = None
        self.currentSlice: DataSlice = DataSlice()
        self.currentLoader: Optional[DataLoader] = None
        self.currentLoaderIndex: int = 0
        self.pointer: int = 0
        self.min_idx: int = 0
        self.max_idx: int = 0
        return

    def loadData(self, mode: str = "test"):
        """Load dataset"""
        # cfg = {'iter': 1000, 'show': True, 'log_interval': 10, 'preview_interval': 200, 'train_samples': 10}
        # 'save_root': "/home/shanwu/Projects/autocalibration_project/results/2039_21102020/",}
        self.dataset = TestDataSet(mode=mode, err_dist="uniform", err_config=ERR_GROUP[4], crop=False)
        self.max_idx = len(self.dataset)
        LOG.debug("Data loaded.")
        return

    @_checkDataLoaded
    def getSlice(self, idx: Optional[int] = None) -> DataSlice:
        """
        Get a sample from the dataset, dataset must be loaded
        :param idx: int, optional, index of the sample
        :return: A DataSlice sample
        """
        if idx is None or not (idx is not None and 0 <= idx < self.max_idx):
            LOG.warning("Random idx generated in getSlice().")
            idx = random.randint(self.min_idx, self.max_idx)

        # set pointer
        self.pointer = idx

        # load data
        d_stereo, p_lidar, T, P0, rot, trans, shape, cam2 = self.dataset[idx]
        p_lidar = p_lidar[p_lidar[:, 3] > 0, :]
        p_lidar[:, 3] = 1
        self.currentLoader = self.dataset.current_loader
        self.currentLoaderIndex = self.dataset.current_pointer

        # structured data
        rot, trans = rot.numpy(), trans.numpy()
        self.currentSlice = DataSlice(d_stereo, p_lidar, T, P0, rot, trans, shape, cam2)
        return self.currentSlice

    def __len__(self):
        return len(self.dataset)


class CameraInfo(object):
    def __init__(self, projection_matrix: Ndarray, resolution: Optional[Resolution] = None):
        if not self.validation_check(projection_matrix):
            raise ValueError("Input is not a projection matrix.")
        # Projection/camera matrix (for stereo camera system, KITTI has four)
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        self.P: Ndarray = projection_matrix
        self.fx: FLOAT = projection_matrix[0, 0]
        self.fy: FLOAT = projection_matrix[1, 1]
        self.cx: FLOAT = projection_matrix[0, 2]
        self.cy: FLOAT = projection_matrix[1, 2]
        self.tx: FLOAT = projection_matrix[0, 3]
        self.ty: FLOAT = projection_matrix[1, 3]
        self.Pinv: Ndarray = self.calc_inverse_projection_matrix(projection_matrix)
        self.resolution: Optional[Resolution] = resolution
        return

    @staticmethod
    def calc_inverse_projection_matrix(projection_matrix: Ndarray) -> Optional[Ndarray]:
        if projection_matrix.shape == (3, 4):
            return np.linalg.inv(np.vstack([projection_matrix,
                                            np.array([0, 0, 0, 1], dtype=np.float32)[np.newaxis, :]]))
        elif projection_matrix.shape == (4, 4):
            return np.linalg.inv(projection_matrix)
        else:
            LOG.error("Projection matrix must be a shape of (3, 4) or (4, 4)!")
            return

    @staticmethod
    def validation_check(projection_matrix) -> bool:
        if projection_matrix.shape == (3, 4):
            if not np.all(projection_matrix[2, :3] == np.array([0, 0, 1])):
                LOG.error("Input is not a projection matrix.")
                return False
            else:
                return True
        elif projection_matrix.shape == (4, 4):
            if not np.all(projection_matrix[3] == np.array([0, 0, 0, 1])):
                LOG.error("Input is not a projection matrix.")
                return False
            else:
                return True
        else:
            LOG.error("Projection matrix must be a shape of (3, 4) or (4, 4)!")
            return False


class Projector(object):

    def __init__(self, cam: CameraInfo):
        self.cam: CameraInfo = cam
        return

    @staticmethod
    def transform(pc: Ndarray, transform: Ndarray) -> Ndarray:
        return (transform @ pc.T).T

    def convert_to_uvd(self, pc: Ndarray, transform: Optional[Ndarray] = None, min_dist: FLOAT = 2.0,
                       return_raw_3d: bool = False) -> Tuple[Ndarray, Optional[Ndarray]]:
        """
        Convert point cloud to uvd pixels
        :param pc: shape (N, 4), columns are (x, y, z, 1)
        :param transform: shape (4, 4), optional, transformation from lidar frame to camera frame, change origin
        :param min_dist: float, minimum valid distance for lidar depth
        :param return_raw_3d: bool, if True, return corresponding raw point cloud along with the uvd pixels.
        :return: Tuple[uvd, Optional[raw3d]]
        """
        # transform to camera's coordinate system
        if transform is not None:
            pc = self.transform(pc, transform)

        # extract the depth
        # (x axis in Lidar's frame, but z axis in reference camera's frame after transformation)
        depths = pc[:, 2]

        # project camera intrinsic matrix and normalize on third dimension to get coordinates
        pts2d_cam = self.cam.P @ pc.T
        pts2d = pts2d_cam / pts2d_cam[2, :]


        # sift points and only keep points in the camera view and keep 1 pixel margin
        mask = np.ones(depths.shape[0], dtype=np.bool)
        mask = np.logical_and(mask, depths > min_dist)
        if self.cam.resolution is not None:  # sift points based on camera's view
            resolution = self.cam.resolution
            xmin, ymin, xmax, ymax = 0, 0, resolution[1], resolution[0]
            mask = np.logical_and(mask, pts2d[0, :] > xmin + 1)
            mask = np.logical_and(mask, pts2d[0, :] < xmax - 1)
            mask = np.logical_and(mask, pts2d[1, :] > ymin + 1)
            mask = np.logical_and(mask, pts2d[1, :] < ymax - 1)
        pts2d = pts2d[:, mask]
        depths = depths[mask]

        # transpose the coordinates back to normal shape (that I prefer) and convert to int (for slicing)
        pts2d = pts2d.T[:, :2]
        pts2d = pts2d.round()

        # if return raw corresponding 3D points
        if return_raw_3d:
            # return (u, v, 1, 1/d), (x, y, z)
            return np.concatenate([pts2d, np.ones(pts2d.shape[0], dtype=np.float32)[:, np.newaxis],
                                   1 / depths[:, np.newaxis]], axis=1), \
                   pc[mask, :]

        # return (u, v, 1, 1/d), None
        return np.concatenate([pts2d, np.ones(pts2d.shape[0], dtype=np.float32)[:, np.newaxis],
                               1 / depths[:, np.newaxis]], axis=1), \
               None

    def convert_from_uvd(self, uvd: Ndarray) -> Ndarray:
        """
        Convert depth pixels to point cloud
        :param uvd: shape (N, 4), columns are u, v, 1, 1/z, where u=x direction idx, v=y direction idx, z=depth
        :return: Numpy point cloud, shape (N, 4)
        """
        # the third column is 1
        assert np.all(uvd[:, 2]), LOG.error("The format of an uvd point is (u, v, 1, 1/z)!")
        # shape (N, 4)
        assert uvd.shape[1] == 4, LOG.error("The shape of an uvd set must be (N, 4)!")
        pts3d = self.cam.Pinv @ uvd.T
        return (pts3d / pts3d[3, :]).T

    def test_conversion(self, pc: Ndarray, transform: Ndarray) -> FLOAT:
        """Testing the conversion from point cloud to and from depth map"""
        print("Testing the conversion from point cloud -> 2D depth pixel -> point cloud (all in camera frame)")
        uvd, pts3d = self.convert_to_uvd(pc, transform, return_raw_3d=True)
        pts3d_new = self.convert_from_uvd(uvd)
        error = np.sum(pts3d-pts3d_new, axis=1).mean()
        print("Conversion error is {:6.4f}".format(error))
        return error

    def depth_map_to_3d(self, depth_map: Ndarray) -> Ndarray:
        # formatting data to uvd
        h, w = np.where(depth_map != 0)
        # (u, v, 1, 1/d)
        uvd = np.concatenate([w[np.newaxis, :],
                              h[np.newaxis, :],
                              np.ones(w.shape[0])[np.newaxis, :],
                              1/depth_map[h, w][np.newaxis, :]]).T

        # call convert_from_uvd()
        return self.convert_from_uvd(uvd)

    def convert_to_depth_map(self, pc: Ndarray, transform: Optional[Ndarray] = None,
                             min_dist: FLOAT = 2.0, base: Optional[Ndarray] = None,
                             color_range: bool = True) -> Ndarray:
        uvd, _ = self.convert_to_uvd(pc, transform=transform, min_dist=min_dist)
        uv = uvd[:, :2].round().astype(np.int32)
        d = 1 / uvd[:, 3]

        if color_range:
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
            assert d.min() >= min_dist
            d = colors[np.floor(map_range(d, min_dist, np.max(d), 0, 1023)).astype(np.int32)]

        if base is None:
            base = np.zeros(tuple(self.cam.resolution)+(3,), dtype=np.uint8)
        else:
            base = base.copy()

        base[uv[:, 1], uv[:, 0]] = d

        return base


class Evaluator(object):

    def __init__(self):
        self.running_loss: FLOAT = 0
        self.n_samples_logged: INT = 0
        self.src_calib_matrix: Optional[Ndarray] = None

    def add_source_calib_matrix(self, source: Ndarray) -> None:
        assert source.shape == (4, 4), "The size of calibration matrix must be a shape of (4, 4)."
        self.src_calib_matrix = source
        return

    def compare_calib_matrix(self, target: Ndarray, source: Optional[Ndarray] = None,
                             show: bool = True) -> Tuple[Ndarray, Ndarray]:
        assert target.shape == (4, 4), "The size of calibration matrix must be a shape of (4, 4)."
        if self.src_calib_matrix is None and source is None:
            LOG.error("Source calibration matrix is not loaded.")
            raise ValueError("Source calibration matrix is not loaded.")
        elif source is None:
            source = self.src_calib_matrix

        # convert to vector: roll, pitch, yaw, x, y, z
        calib_gt = calib_np_to_phi(source)
        rot_gt = np.rad2deg(calib_gt[:3])
        trans_gt = calib_gt[3:]
        calib_op = calib_np_to_phi(target)
        rot_op = np.rad2deg(calib_op[:3])
        trans_op = calib_op[3:]

        # get absolute differences
        diff_rot = rot_gt - rot_op
        diff_trans = trans_gt - trans_op

        if show:
            print("Source calibration (GT):")
            print(source)
            print("Target calibration (assessed):")
            print(target)
            diagnose_str = "The rotation errors are: {r}\nThe translation errors are: {t}"
            print(diagnose_str.format(r=diff_rot, t=diff_trans))
        return diff_rot, diff_trans

    def calib_matrix_to_vector(self, source: Ndarray) -> Ndarray:
        assert source.shape == (4, 4), "The size of calibration matrix must be a shape of (4, 4)."
        return calib_np_to_phi(source)


if __name__ == "__main__":
    pass
