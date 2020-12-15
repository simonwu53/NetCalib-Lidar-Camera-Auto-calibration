from typing import Mapping, Sequence
from torch.distributions import uniform, normal
import numpy as np
import logging


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('ErrorGenerator')


class ErrorGenerator:
    def __init__(self, dist: str, config: Mapping):
        """
        Generate a set of random rotation and translation error based on a specified distribution

        :param dist: 'uniform' or 'normal', distributions where the random samples are picked from
        :param config: type=dict, keys=['R', 'T'], which give the configuration of the distribution,
                       default values are given in the code
        """
        self.error = None

        if dist == 'uniform':
            self.dist = 'uniform'
            R_range = config.get('R', np.deg2rad(2))  # rotation error upto 2 degrees by default
            T_range = config.get('T', 0.2)  # translation error upto 20 cm by default
            LOG.warning('Rotation error range (degrees): [-%.2f, %.2f]' % (np.rad2deg(R_range), np.rad2deg(R_range)))
            LOG.warning('Translation error range (meters): [-%.3f, %.3f]' % (T_range, T_range))
            self.R_dist = uniform.Uniform(-R_range, R_range)
            self.T_dist = uniform.Uniform(-T_range, T_range)

            # statistics
            self.R_mean, self.T_mean = 0, 0  # (a+b)/2
            self.R_std = 2*R_range/np.sqrt(12)  # (b-a)/sqrt(12)
            self.T_std = 2*T_range/np.sqrt(12)
        elif dist == 'normal':
            self.dist = 'normal'
            R_params = config.get('R', (np.deg2rad(2), np.deg2rad(0.1)))  # rotation error at 2 degrees as mean, 0.1 degree for std.
            T_params = config.get('T', (0.2, 0.01))  # translation error at 20cm as mean, 1cm for std.
            LOG.warning('Rotation error mean (degrees): -%.2f, std: %.2f' % (np.rad2deg(R_params[0]), np.rad2deg(R_params[1])))
            LOG.warning('Translation error mean (meters): -%.3f, std: %.3f' % (T_params[0], T_params[1]))
            self.R_dist = normal.Normal(*R_params)
            self.T_dist = normal.Normal(*T_params)

            # statistics
            self.R_mean, self.R_std = R_params
            self.T_mean, self.T_std = T_params
        else:
            LOG.error('Unknown distribution given: %s' % dist)
        return

    def gen_err(self):
        """
        :return: Generate a set of random errors, shape (6,)
        """
        rot = self.R_dist.sample((3,))  # sample three values for roll, pitch, yaw
        trans = self.T_dist.sample((3,))  # sample for translation error
        return rot, trans

    def standardize_err(self, err: Sequence):
        """
        Standardize the error with given distribution
        :param err: error tensor, shape (6,)
        :return: standardized error with the same shape
        """
        rot, trans = err
        return (rot-self.R_mean)/self.R_std, (trans-self.T_mean)/self.T_std

    def destandardize_err(self, err: Sequence):
        """
        De-standardize the error with given distribution
        :param err: error tensor, shape (6,)
        :return: de-standardized error with the same shape
        """
        rot, trans = err
        return rot*self.R_std+self.R_mean, trans*self.T_std+self.T_mean
