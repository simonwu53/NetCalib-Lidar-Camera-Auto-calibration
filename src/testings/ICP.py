import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import time
from testings.UtilsClass import Loader, CameraInfo, Projector, Evaluator
from testings.UtilsFuncs import gen_errors, assemble_calib_matrix, numpy_list_to_pointcloud, show_pointcloud
from testings.DataTypes import ERR_GROUP, DataSlice, Ndarray, Color, FLOAT, INT, ErrOption, ProgressBar
from typing import Tuple, Optional
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('ICP')


def get_dataset(mode: str = "test") -> Loader:
    # create dataset
    dataset = Loader()
    # load dataset
    dataset.loadData(mode=mode)
    return dataset


def init(dataset: Loader, idx: Optional[int] = None,
         err_type: ErrOption = 4) -> Tuple[DataSlice, CameraInfo, Projector, Ndarray, Ndarray, Ndarray, Ndarray]:
    # get sample data for testing
    if idx is not None:
        sample = dataset.getSlice(idx=idx)
    else:
        sample = dataset.getSlice()

    # create camera
    cam = CameraInfo(sample.P0, sample.shape)
    # create projector
    proj = Projector(cam)

    # re-project depth map back to 3D space
    pts3d_cam = proj.depth_map_to_3d(sample.depth_map)

    # generate artificial errors
    err_rot, err_trans = gen_errors(ERR_GROUP[err_type])
    # transform errors to matrix
    m_err = assemble_calib_matrix(err_rot, err_trans)
    LOG.debug("Artificial rotation error: {}".format(err_rot))
    LOG.debug("Artificial translation error: {}".format(err_trans))
    LOG.debug("Artificial error matrix:\n {}".format(m_err))
    # initialize calibration matrix (with errors)
    m_calib = m_err @ sample.T

    # select lidar points appear in cam frame only (transform to cam coordinate first)
    # Note: Errors are also applied in "m_calib", hence pts3d_lidar and pts3d_cam are not aligned at this point.
    #       The errors are supposed to be found in the ICP algorithm.
    _, pts3d_lidar = proj.convert_to_uvd(sample.lidar_points, m_calib, return_raw_3d=True)
    return sample, cam, proj, pts3d_lidar, pts3d_cam, m_calib, m_err


def icp_registration(pts3d_lidar: Ndarray, pts3d_cam: Ndarray,
                     m_init: Optional[Ndarray] = np.eye(4, dtype=np.float32),
                     src_color: Color = (1, 0.706, 0),  # orange, lidar pc
                     target_color: Color = (0, 0.651, 0.929),  # blue, cam pc
                     distance_threshold: FLOAT = 17.0,
                     max_converge_iteration: INT = 500000,
                     show: bool = False,
                     desc_bar: Optional[ProgressBar] = None) -> Tuple[Ndarray, FLOAT]:

    # convert point clouds to open3d format
    pts3d_cam_pc, pts3d_lidar_pc = numpy_list_to_pointcloud([pts3d_cam, pts3d_lidar])
    # add colors
    pts3d_cam_pc.paint_uniform_color(target_color)
    pts3d_lidar_pc.paint_uniform_color(src_color)
    if show:
        show_pointcloud([pts3d_lidar_pc, pts3d_cam_pc], window_name="Before ICP")

    # do ICP optimization
    t1 = time.time()
    # pts3d_lidar_pc.estimate_normals(fast_normal_computation=False)
    # pts3d_cam_pc.estimate_normals(fast_normal_computation=False)
    reg_p2p = o3d.pipelines.registration.\
        registration_icp(source=pts3d_lidar_pc, target=pts3d_cam_pc,
                         max_correspondence_distance=distance_threshold,
                         init=m_init,
                         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_converge_iteration))

    # evaluation
    elapsed: FLOAT = time.time() - t1
    evaluation_p2p = o3d.pipelines.registration.evaluate_registration(pts3d_lidar_pc, pts3d_cam_pc,
                                                                      distance_threshold, reg_p2p.transformation)
    if desc_bar is not None:
        desc_bar.set_description("ICP fitness: {}; RMSE: {}; time{:6.4f}".format(evaluation_p2p.fitness,
                                                                                 evaluation_p2p.inlier_rmse, elapsed))
    elif show:
        print(evaluation_p2p)
        print("Time per iteration: {:6.4f}".format(elapsed))

    # visualize result
    if show:
        pts3d_lidar_pc.transform(reg_p2p.transformation)
        show_pointcloud([pts3d_lidar_pc, pts3d_cam_pc], window_name="After ICP")
    # return the transformation for fixing the errors, ideally should be m_err^(-1),
    # which means: reg_p2p.transformation @ m_err @ calib_gt == calib_gt
    return reg_p2p.transformation, elapsed


# Usage example
def run(mode: str = "test", n_samples: int = 258, save: Optional[str] = None) -> Tuple[Ndarray, Ndarray, Ndarray]:
    # get dataset
    dataset = get_dataset(mode=mode)
    # generate random index of samples
    if len(dataset) > n_samples:
        idx = np.random.choice(len(dataset), n_samples, replace=False)
    elif len(dataset) < n_samples:
        idx = np.random.choice(len(dataset), n_samples, replace=True)
    else:
        idx = np.arange(len(dataset))

    # initialize variables
    pbar = tqdm(range(n_samples))
    evaluator = Evaluator()
    res_errors = np.zeros((n_samples, 6), dtype=np.float32)
    res_transformation = np.zeros((n_samples, 3, 4, 4), dtype=np.float32)  # (samples, error/calib/calib_gt, 4, 4)
    res_times = np.zeros(n_samples, dtype=np.float64)
    # start testing
    for i in pbar:
        # get testing data
        sample, cam, proj, pts3d_lidar, pts3d_cam, m_calib, m_err = init(dataset, idx=idx[i])
        # ICP
        m_icp, t = icp_registration(pts3d_lidar, pts3d_cam, show=False, desc_bar=pbar)
        # test error
        diff_rot, diff_trans = evaluator.compare_calib_matrix(np.linalg.inv(m_icp), m_err, show=False)
        pbar.set_description("Error: rot [{rot[0]:7.4f},{rot[1]:7.4f},{rot[2]:7.4f}]; "
                             "trans [{trans[0]:7.4f},{trans[1]:7.4f},{trans[2]:7.4f}]".format(rot=diff_rot,
                                                                                              trans=diff_trans))
        # record result
        res_errors[i] = np.concatenate([diff_rot, diff_trans])
        res_transformation[i, 0] = m_err
        res_transformation[i, 1] = m_icp
        res_transformation[i, 2] = sample.T
        res_times[i] = t

    if save is not None:
        p = Path(save)
        if p.parent.exists():
            np.savez_compressed(save, err=res_errors, transformation=res_transformation, time=res_times)
        else:
            LOG.warning("Path does not exist! Saving aborted.")
    mean_errors = res_errors.mean(axis=0)
    std_errors = res_errors.std(axis=0)
    print("Rotation error: mean [{mean[0]:7.4f},{mean[1]:7.4f},{mean[2]:7.4f}]; "
          "std [{std[0]:7.4f},{std[1]:7.4f},{std[2]:7.4f}]".format(mean=mean_errors[:3], std=std_errors[:3]))
    print("Translation error: mean [{mean[0]:7.4f},{mean[1]:7.4f},{mean[2]:7.4f}]; "
          "std [{std[0]:7.4f},{std[1]:7.4f},{std[2]:7.4f}]".format(mean=mean_errors[3:], std=std_errors[3:]))
    print("Average time per iteration: {:6.4f}".format(res_times.mean()))
    return res_errors, res_transformation, res_times


def run_once(dataset: Loader, idx: Optional[int] = None, err_type: ErrOption = 4) -> Tuple[Ndarray, Ndarray, FLOAT]:
    sample, cam, proj, pts3d_lidar, pts3d_cam, m_calib, m_err = init(dataset, idx=idx, err_type=err_type)
    m_icp, elapsed = icp_registration(pts3d_lidar, pts3d_cam, show=True)
    eva = Evaluator()
    r, t = eva.compare_calib_matrix(m_icp, m_err, show=True)
    return r, t, elapsed
