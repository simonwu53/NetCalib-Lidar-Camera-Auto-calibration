import numpy as np
import open3d as o3d
from tqdm import tqdm
from pykitti2.raw import raw
from typing import Optional, NamedTuple, Mapping, Union, Tuple, Final, Literal
from matplotlib.axes._subplots import Axes


""" ##########################
Define Data Types
########################## """
ErrTypes = Mapping[int, Mapping[str, float]]
ErrOption = Literal[1, 2, 3, 4, 5]
DataLoader = raw
Ndarray = np.ndarray
FLOAT = Union[float, np.float32, np.float64]
INT = Union[int, np.int32, np.int64]
RGB = Tuple[INT, INT, INT]
Color = Tuple[FLOAT, FLOAT, FLOAT]
Point3D = Tuple[FLOAT, FLOAT, FLOAT]
Resolution = Tuple[INT, INT]
PointCloud = o3d.geometry.PointCloud
Visualizer = o3d.visualization.Visualizer
ProgressBar = tqdm
PLOTHANDLER = Axes
BASIC_COLOR_STRING = Literal["b", "g", "r", "c", "m", "y", "k", "w", "gray"]

# Constants
ERR_GROUP: Final[ErrTypes] = {
    1: {'T': 1.5, 'R': np.deg2rad(20)},  # unit: T-meters, R-degrees
    2: {'T': 1.0, 'R': np.deg2rad(10)},
    3: {'T': 0.5, 'R': np.deg2rad(5)},
    4: {'T': 0.2, 'R': np.deg2rad(2)},
    5: {'T': 0.1, 'R': np.deg2rad(1)},
}

FONT_RC = {"font.size": 18}


class DataSlice(NamedTuple):
    depth_map: Optional[Ndarray] = None
    lidar_points: Optional[Ndarray] = None
    T: Optional[Ndarray] = None
    P0: Optional[Ndarray] = None
    err_rot: Optional[Ndarray] = None
    err_trans: Optional[Ndarray] = None
    shape: Optional[Ndarray] = None
    cam2_img: Optional[Ndarray] = None
