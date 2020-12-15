import numpy as np
import open3d as o3d
from os.path import isfile
from typing import Optional, List, Mapping, Tuple, Union
from testings.DataTypes import INT, Ndarray, FLOAT, RGB, PointCloud, Visualizer, PLOTHANDLER, \
    BASIC_COLOR_STRING, FONT_RC
import matplotlib.pyplot as plt
import logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('UtilsFuncs')
# update plt params
plt.rcParams.update(FONT_RC)


def numpy_to_pointcloud(pc: Ndarray) -> PointCloud:
    pcd = o3d.geometry.PointCloud()
    if pc.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(pc)
    elif pc.shape[1] == 4:
        pc = pc[pc[:, 3] > 0, :][:, :3]
        pcd.points = o3d.utility.Vector3dVector(pc)
    else:
        LOG.error("The shape of point cloud must be (N, 3) or (N, 4)!")
        raise ValueError("The shape of point cloud must be (N, 3) or (N, 4)!")
    return pcd


def numpy_list_to_pointcloud(pc: List[Ndarray]) -> List[PointCloud]:
    return [numpy_to_pointcloud(p) for p in pc]


def show_pointcloud(pc: List[PointCloud],
                    window_name: str = 'PointCloud',
                    width: INT = 1920,
                    height: INT = 1080,
                    view: Optional[str] = None,
                    show_axis: bool = False,
                    bg_color: RGB = (0, 0, 0),
                    point_size: FLOAT = 3.0,
                    add_random_color: bool = False,
                    ) -> None:
    # check data
    if add_random_color:
        for p in pc:
            p.paint_uniform_color(np.random.randint(0, 255, 3).astype(np.float32) / 255)

    # initialize visualizer class
    vis: Visualizer = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)

    # add PointClouds to the scene
    for geometry in pc:
        vis.add_geometry(geometry)

    # set visualozer options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.show_coordinate_frame = show_axis
    opt.background_color = bg_color

    # get visualizer control
    ctr = vis.get_view_control()
    # set view point
    if view is not None and isfile(view):
        if not isfile(view):
            LOG.error("Could not find the view file! Path: {}".format(view))
        else:
            parameters = o3d.io.read_pinhole_camera_parameters(view)
            ctr.convert_from_pinhole_camera_parameters(parameters)

    # run visualization
    vis.run()  # block
    vis.destroy_window()
    return


def error_plot(mean: Ndarray, std: Ndarray, minimum: Ndarray, maximum: Ndarray,
               color: BASIC_COLOR_STRING = "k", ecolor: BASIC_COLOR_STRING = "gray",
               xticks: Optional[List[str]] = None, ax: Optional[PLOTHANDLER] = None,
               title: Optional[str] = None, ylabel: Optional[str] = None,
               legend: Optional[str] = None, scale: Optional[int] = None,
               show: bool = True) -> Optional[PLOTHANDLER]:
    # validate inputs
    assert mean.shape == std.shape == minimum.shape == maximum.shape, \
        "Mean array, std array, min/max array should have the same size."
    if scale is not None:
        mean *= scale
        std *= scale
        minimum *= scale
        maximum *= scale
    # create axe handler
    if ax is None:
        ax: PLOTHANDLER = plt.subplot(111)
    # plot error bar
    arr = np.zeros_like(mean)
    ax.errorbar(np.arange(mean.shape[0]), mean, std, fmt="o"+color, lw=3)
    ax.errorbar(np.arange(mean.shape[0]), mean, [arr-minimum, maximum-arr],
                fmt="."+color, ecolor=ecolor, lw=1, label=legend)
    # finalize the plot
    ax.set_xlim(-1, mean.shape[0])
    ax.legend()
    if xticks is not None:
        assert len(xticks) == mean.shape[0], "XTick labels should have the same size as mean array."
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # visualize
    if show:
        plt.show()
        return
    else:
        return ax


def save_pointcloud(pc: PointCloud, fname: str = "./testings/archives/point_cloud.ply") -> None:
    o3d.io.write_point_cloud(fname, pc)
    return


def gen_errors(cfg: Mapping[str, FLOAT]) -> Tuple[Ndarray, Ndarray]:
    r_range = cfg.get("R", np.deg2rad(2))
    t_range = cfg.get("T", 0.2)
    return np.random.uniform(-r_range, r_range, 3).astype(np.float32), \
           np.random.uniform(-t_range, t_range, 3).astype(np.float32)


def assemble_calib_matrix(rotation: Tuple[FLOAT, FLOAT, FLOAT], trans: Tuple[FLOAT, FLOAT, FLOAT]) -> Ndarray:
    m_rot = angle_to_rotation_matrix(rotation)
    m_calib = np.zeros((4, 4), dtype=np.float32)
    m_calib[:3, :3] = m_rot
    m_calib[:3, 3] = trans
    m_calib[3, 3] = 1
    return m_calib


def angle_to_rotation_matrix(rotation: Tuple[FLOAT, FLOAT, FLOAT]) -> Ndarray:

    u, v, w = rotation
    s_u, c_u = np.sin(u), np.cos(u)
    s_v, c_v = np.sin(v), np.cos(v)
    s_w, c_w = np.sin(w), np.cos(w)

    return np.array([
        [c_v*c_w, s_u*s_v*c_w-c_u*s_w, s_u*s_w+c_u*s_v*c_w],
        [c_v*s_w, c_u*c_w+s_u*s_v*s_w, c_u*s_v*s_w-s_u*c_w],
        [-s_v, s_u*c_v, c_u*c_v]
    ], dtype=np.float32)


def calib_np_to_phi(M: Ndarray, no_exception: bool = False) -> Ndarray:
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


def rotation_matrix_to_angle(R: Ndarray, no_exception: bool = False) -> Tuple[FLOAT, FLOAT, FLOAT]:
    """
    Convert a 3*3 rotation matrix to roll, pitch, yaw
    :param R: 3*3 rotation matrix
    :param no_exception: bool, do not raise error when a wrong matrix comes
    :return: roll, pitch, yaw in radians
    """
    if not isRotationMatrix(R):
        if not no_exception:
            raise ValueError('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
        else:
            LOG.warning('This is not a rotation matrix! Hence can not convert to row, pitch, yaw!')
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


def isRotationMatrix(R: Ndarray) -> bool:
    Rt = np.transpose(R)
    shouldBeIdentity = np.matmul(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def map_range(val: Union[FLOAT, Ndarray],
                 old_min: Union[INT, FLOAT], old_max: Union[INT, FLOAT],
                 new_min: Union[INT, FLOAT], new_max: Union[INT, FLOAT]):
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
