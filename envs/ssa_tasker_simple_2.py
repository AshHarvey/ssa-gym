import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from filterpy.kalman import MerweScaledSigmaPoints as SigmasPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise as Q_noise_fn
from poliastro.bodies import Earth
from poliastro.core.elements import rv2coe
from envs.transformations import arcsec2rad, deg2rad, lla2ecef, gcrs2irts_matrix_a,gcrs2irts_matrix_b, get_eops, ecef2aer, ecef2lla
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import robust_cholesky
from envs.results import observations as obs_fn, error, error_failed, plot_delta_sigma, plot_rewards, plot_nees
from envs.results import plot_histogram, plot_orbit_vis, plot_regimes, reward_proportional_trinary_true
from envs.results import moving_average_plot, bound_plot, map_plot
import gym
from gym.utils import seeding
from copy import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from statsmodels.tsa.stattools import acf
from . import env_config

"""
Unit Abbreviations: 
arc minutes: '
arc seconds: "
degrees: deg
kilometers: km
latitude: lat
longitude: lon
meters: m
radians: rad
seconds: s
unitless: u
"""

class SSA_Tasker_Env(gym.Env):
    """
    The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """
    # metadata = {'render.modes': ['human']}
    def __init__(self, config=env_config):
        # super(SSA_Tasker_Env, self).__init__()
        s = time.time()
        self.runtime = {'__init__': 0, 'reset': 0, 'step': 0, 'step prep': 0, 'propagate next true state': 0,
                        'perform predictions': 0, 'update with observation': 0, 'Observations and Reward': 0,
                        'filter_error': 0, 'visible_objects': 0, 'object_visibility': 0, 'anees': 0,
                        'failed_filters': 0, 'plot_sigma_delta': 0, 'plot_rewards': 0, 'plot_anees': 0,
                        'plot_actions': 0, 'all_true_obs': 0, 'plot_visibility': 0}
        """Simulation configuration"""
        self.t_0 = config['t_0']                # time at start of simulation
        self.dt = config['time_step']           # length of time steps [s]
        self.n = config['steps']                # max run steps
        self.m = config['rso_count']            # number of Resident Space Object (RSO) to include in the simulation
        self.obs_limit = np.radians(config['obs_limit'])  # don't observe objects below this elevation [rad]

        # configuration parameters for RSOs; sma: Semi-major axis [m], ecc: Eccentricity [u], inc: Inclination (rad),
        # raan: Right ascension of the ascending node (rad), argp: Argument of perigee (rad), nu: True anomaly (rad)

        self.orbits = config['orbits']              # orbits to sample from
        self.obs_lla = np.array(config['observer']) * [deg2rad, deg2rad, 1]  # Observer coordinates - lat, lon, height (deg, deg, m)
        self.obs_itrs = lla2ecef(self.obs_lla)      # Observer coordinates in ITRS (m)
        self.update_interval = config['update_interval']  # how often an observation should be taken
        self.i = 0
        """Filter configuration"""
        # standard deviation of noise added to observations [rad, rad, m]
        self.obs_type = config['obs_type']
        if self.obs_type == 'aer':
            self.z_sigma = config['z_sigma'] * np.array([arcsec2rad, arcsec2rad, 1])
        elif self.obs_type == 'xyz':
            self.z_sigma = config['z_sigma']
        else:
            print('Invalid Observation Type: ' + str(config['obs_type']))
            exit()
        # standard deviation of noise added to initial state estimates; [m, m, m, m/s, m/s, m/s]
        self.x_sigma = np.array(config['x_sigma'])
        self.Q = Q_noise_fn(dim=2, dt=self.dt, var=config['q_sigma'] ** 2, block_size=3, order_by_dim=False)
        self.eops = get_eops()
        self.fx = config['fx']
        self.hx = config['hx']
        self.mean_z = config['mean_z']
        self.residual_z = config['residual_z']
        self.msqrt = config['msqrt']
        self.alpha, self.beta, self.kappa = config['alpha'], config['beta'], config['kappa']  # sigma point configuration parameters
        """Prep arrays"""
        # variables for the filter
        x_dim = int(6)
        z_dim = int(3)
        if config['P_0'] is None:
            self.P_0 = np.copy(np.diag(self.x_sigma ** 2))  # Assumed covariance of the estimates at simulation start
        else:
            self.P_0 = np.copy(config['P_0'])
        if config['R'] is None:
            self.R = np.diag(self.z_sigma ** 2)  # Noise added to the filter during observation updates
        else:
            self.R = np.copy(config['R'])
        self.x_true = np.empty(shape=(self.n, self.m, x_dim))  # means for all objects at each time step
        self.x_filter = np.empty(shape=(self.n, self.m, x_dim))  # means for all objects at each time step
        self.P_filter = np.empty(shape=(self.n, self.m, x_dim, x_dim))  # covariances for all objects at each time step
        self.obs = np.empty(shape=(self.n, self.m, x_dim * 2))  # observations for all objects at each time step
        self.time = [self.t_0 + (timedelta(seconds=self.dt) * i) for i in range(self.n)]  # time for all time steps
        self.trans_matrix = gcrs2irts_matrix_b(self.time, self.eops)  # used for celestial to terrestrial
        self.z_noise = np.empty(shape=(self.n, self.m, z_dim))  # array to contain the noise added to each observation
        self.z_true = np.empty(shape=(self.n, self.m, z_dim))  # array to contain the observations which are made
        self.y = np.empty(shape=(self.n, self.m, z_dim))  # array to contain the innovation of each observation
        self.S = np.empty(shape=(self.n, self.m, z_dim, z_dim))  # array to contain the innovation covariance
        self.x_noise = np.empty(shape=(self.m, x_dim))  # array to contain the noise added to each RSO
        self.filters = []  # creates a list for ukfs
        # variables for environment performance
        self.delta_pos = np.empty((self.n, self.m))  # euclidean distance between true and filter mean position elements
        self.delta_vel = np.empty((self.n, self.m))  # euclidean distance between true and filter mean velocity elements
        self.sigma_pos = np.empty((self.n, self.m))  # euclidean magnitude of diagonal position elements of covariance
        self.sigma_vel = np.empty((self.n, self.m))  # euclidean magnitude of diagonal velocity elements of covariance
        self.scores = np.empty((self.n, self.m))  # score of each RSO at each time step
        self.rewards = np.empty(self.n)  # reward for each time step
        self.failed_filters_id = []  # prep list for failed states
        self.failed_filters_msg = ["None"] * self.m  # prep list for failure messages
        self.actions = np.empty(self.n, dtype=int)  # prep variable for keeping track of all previous actions
        self.obs_taken = np.empty(self.n,
                                  dtype=bool)  # prep variable for keeping track of which obs were actually taken
        self.x_failed = np.array([1e20, 1e20, 1e20, 1e12, 1e12, 1e12])  # failed filter will be set to this value
        self.P_failed = np.diag([1e20, 1e20, 1e20, 1e12, 1e12, 1e12])  # failed filter will be set to this value
        self.nees = np.empty((self.n, self.m))  # used for normalized estimation error squared (NEES) and its average
        self.visibility = []  # used to store a log of visible objects at each time step
        self.sigmas_h = np.empty((self.n, x_dim * 2 + 1, z_dim))  # used to store sigmas points used in updates
        """Define Gym spaces"""
        self.action_space = gym.spaces.Discrete(self.m)  # the action is choosing which RSO to look at
        self.observation_space = gym.spaces.Box(low=np.tile(-np.inf, (self.m, 12)), high=np.tile(np.inf, (self.m, 12)),
                                                dtype=np.float64)  # the obs is x [6] and diag(P) [6] for each RSO [m]
        """Initial reset and seed calls"""
        self.np_random = None
        self.init_seed = self.seed()
        self.reset()
        e = time.time()
        self.runtime['__init__'] += e - s

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_seed = seed
        return [seed]

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        :return:  observation (object): the initial observation of the
            space.
        """
        s = time.time()
        """reset filter"""
        # Clear history
        self.x_true[:], self.x_filter[:], self.P_filter[:], self.obs[:], self.sigmas_h[:] = [0] * 5
        self.z_true[:], self.y[:], self.S[:] = np.nan, np.nan, np.nan
        """initialize RSO"""
        self.filters = []
        for j in range(self.m):
            self.x_true[0][j] = self.orbits[self.np_random.randint(low=0, high=self.orbits.shape[0]), :]
            self.x_noise[j] = self.np_random.normal(size=6) * self.x_sigma
            self.x_filter[0][j] = np.copy(self.x_true[0][j] + self.x_noise[j])
            self.P_filter[0][j] = np.copy(self.P_0)
            self.filters.append(UKF(dim_x=6, dim_z=3, dt=self.dt, fx=self.fx, hx=self.hx,
                                    points=SigmasPoints(n=6, alpha=self.alpha, beta=self.beta, kappa=self.kappa,
                                                        sqrt_method=self.msqrt),
                                    z_mean_fn=self.mean_z, residual_z=self.residual_z, sqrt_fn=self.msqrt))
            self.filters[j].x = np.copy(self.x_filter[0][j])
            self.filters[j].P = np.copy(self.P_filter[0][j])  # initial uncertainty
            self.filters[j].R = np.copy(self.R)  # uncertainty of each observation
            self.filters[j].Q = np.copy(self.Q)  # uncertainty of each prediction
        for i in range(self.n):
            for j in range(self.m):
                self.z_noise[i, j] = self.np_random.normal(size=3) * self.z_sigma
        """Reset variables for tracking environment performance"""
        # blank out all tracking variables
        self.scores[:], self.delta_pos[:], self.delta_vel[:], self.sigma_pos[:], self.sigma_vel[:] = [np.nan] * 5
        self.actions[:], self.obs_taken[:], self.failed_filters_id, self.visibility = np.nan, False, [], []
        self.failed_filters_msg = ["None"] * self.m  # reset list for failure messages
        """Observations and Reward"""
        self.obs[0] = obs_fn(self.x_filter[0], self.P_filter[0])  # set initial observation based on x and P
        self.delta_pos[0], self.delta_vel[0], self.sigma_pos[0], self.sigma_vel[0] = error(self.x_true[0],
                                                                                           self.obs[0])  # initial error
        self.rewards[:] = 0
        self.i = 0  # sets initial time step
        e = time.time()
        self.runtime['reset'] += e - s
        return self.obs[0]

    def step(self, a):

        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        :param a:
            action (object): an action provided by the environment
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        """
        step_s = time.time()
        s = time.time()
        assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a))  # check for valid action
        self.i += 1  # increments current step
        self.actions[self.i] = np.copy(a)  # record current action
        e = time.time()
        self.runtime['step prep'] += e - s
        """propagate next true state"""
        s = time.time()
        for j in range(self.m):
            self.x_true[self.i][j] = self.fx(self.x_true[self.i - 1][j], self.dt)
        e = time.time()
        self.runtime['propagate next true state'] += e - s
        """perform predictions"""
        s = time.time()
        for j in range(self.m):
            if not (j in self.failed_filters_id):
                try:
                    self.filters[j].predict()
                    if np.any(np.isnan(self.filters[j].x)):
                        self.filter_error(object_id=j, activity='predict', error_type=', predict returned nan. ')
                except ValueError:
                    self.filter_error(object_id=j, activity='predict', error_type=', ValueError. ')
                except np.linalg.LinAlgError:
                    self.filter_error(object_id=j, activity='predict', error_type=', LinAlgError. ')
                except:
                    self.filter_error(object_id=j, activity='predict', error_type=', Unknown. ')
            self.x_filter[self.i, j] = np.copy(self.filters[j].x)  # update filter mean history
            self.P_filter[self.i, j] = np.copy(self.filters[j].P)  # update covariance mean history
        e = time.time()
        self.runtime['perform predictions'] += e - s
        """update with observation"""
        s = time.time()
        if (self.i % self.update_interval) == 0:
            if not (a in self.failed_filters_id):
                hx_kwargs = {"trans_matrix": self.trans_matrix[self.i],
                             "observer_itrs": self.obs_itrs,
                             "observer_lla": self.obs_lla,
                             "time": self.time[self.i]}
                self.z_true[self.i, a] = self.hx(self.x_true[self.i][a], **hx_kwargs)
                if self.object_visible([a])[0]:
                    try:
                        self.filters[a].update(self.z_true[self.i, a] + self.z_noise[self.i, a], **hx_kwargs)
                        self.y[self.i, a] = np.copy(self.filters[a].y)
                        self.S[self.i, a] = np.copy(self.filters[a].S)
                        self.sigmas_h[self.i] = np.copy(self.filters[a].sigmas_h)
                        self.obs_taken[self.i] = True
                        if np.any(np.isnan(self.filters[a].x)):
                            self.filter_error(object_id=a, activity='update', error_type=', update returned nan. ')
                    except ValueError:
                        self.filter_error(object_id=a, activity='update', error_type=', ValueError. ')
                    except np.linalg.LinAlgError:
                        self.filter_error(object_id=a, activity='update', error_type=', LinAlgError. ')
                    except:
                        self.filter_error(object_id=a, activity='update', error_type=', Unknown. ')
                    self.x_filter[self.i, a] = np.copy(self.filters[a].x)  # update filter mean history
                    self.P_filter[self.i, a] = np.copy(self.filters[a].P)  # update covariance mean history
        e = time.time()
        self.runtime['update with observation'] += e - s
        """Observations and Reward"""
        s = time.time()
        self.obs[self.i] = obs_fn(self.x_filter[self.i], self.P_filter[self.i])
        tmp = error(self.x_true[self.i], self.obs[self.i])
        self.delta_pos[self.i], self.delta_vel[self.i], self.sigma_pos[self.i], self.sigma_vel[self.i] = tmp
        self.rewards[self.i] = reward_proportional_trinary_true(self.delta_pos[self.i])
        done = False
        if self.i + 1 >= self.n:
            done = True
        e = time.time()

        self.runtime['Observations and Reward'] += e - s
        step_e = time.time()
        self.runtime['step'] += step_e - step_s
        return self.obs[self.i], self.rewards[self.i], done, {}  # observations, reward, and done

    def filter_error(self, object_id, activity, error_type):
        s = time.time()
        self.filters[object_id].x = np.copy(self.x_failed)
        self.filters[object_id].P = np.copy(self.P_failed)
        # self.x_filter[self.i - 1, object_id] = np.copy(self.x_failed)
        # self.P_filter[self.i - 1, object_id] = np.copy(self.P_failed)
        msg = ["".join(['Object ', str(object_id), ' failed on ', activity, ' step ', str(self.i), error_type,
                        str(np.round(error_failed(state=self.x_true[self.i - 1, object_id],
                                                  x=self.x_filter[self.i - 1, object_id],
                                                  P=np.diag(self.P_filter[self.i - 1, object_id])), 2))])]
        self.failed_filters_msg[object_id] = copy(msg)  # record error message
        self.failed_filters_id.append(object_id)  # add filter it list of failed filters
        e = time.time()
        self.runtime['filter_error'] += e - s

    @property
    def visible_objects(self):
        s = time.time()
        RSO_ID = [j for j in range(self.m)]
        viz = np.where(self.object_visible(RSO_ID))[0]
        e = time.time()
        self.runtime['visible_objects'] += e - s
        return viz

    def object_visible(self, RSO_ID=[]):
        if not RSO_ID:
            print('RSO ID expected, but not supplied')
            return RSO_ID
        x_itrs = np.array([self.trans_matrix[self.i] @ self.x_true[self.i, j, :3] for j in RSO_ID])
        el = np.array([ecef2aer(self.obs_lla, x, self.obs_itrs)[1] for x in x_itrs])
        viz_bool = el >= self.obs_limit
        return viz_bool

    @property
    def object_visibility(self):
        s = time.time()
        x_itrs = np.array([self.trans_matrix[self.i] @ self.x_true[self.i, j, :3] for j in range(self.m)])
        el = np.array([ecef2aer(self.obs_lla, x, self.obs_itrs)[1] for x in x_itrs])
        viz = el >= self.obs_limit
        e = time.time()
        self.runtime['object_visibility'] += e - s
        return viz

    @property
    def anees(self):
        """
        Desc:  based on 3.1 in https://pdfs.semanticscholar.org/1c1f/6c864789630d8cd37d5342f67ad8d480f077.pdf
        :return: the average normalized estimation error squared (ANEES)
        """
        s = time.time()

        delta = self.x_true - self.x_filter
        for i in range(self.n):
            for j in range(self.m):
                self.nees[i, j] = delta[i, j] @ np.linalg.inv(self.P_filter[i, j]) @ delta[i, j]
        e = time.time()
        self.runtime['anees'] += e - s
        return np.mean(self.nees)

    def failed_filters(self):
        s = time.time()
        if not self.failed_filters_id:
            print("No failed Objects")
        else:
            for rso_id in self.failed_filters_id:
                print(self.failed_filters_msg[rso_id])
        e = time.time()
        self.runtime['failed_filters'] += e - s


########################################################################################################################
#    Plots derived from results.py
########################################################################################################################

    def plot_sigma_delta(self, style=None, yscale='log', objects=np.array([]), ylim='max', title='default',
                         save_path='default', display=True):
        """
        :param objects: For displaying specific objects
        :return: plot using result.sigma_delta_plot function
        """
        s = time.time()
        if title == 'default':
            title = 'Filter performance for ' + str(self.m) + ' RSOs, seed = ' + str(self.init_seed)
        if save_path == 'default':
            save_path = str(self.m) + 'RSO_' + str(self.init_seed) + 'seed_sigmas_delta_plot' + '.svg'
        if objects.shape[0] == 0:
            plot_delta_sigma(sigma_pos=self.sigma_pos, sigma_vel=self.sigma_vel, delta_pos=self.delta_pos,
                             delta_vel=self.delta_vel, dt=self.dt, t_0=self.t_0, style=style, yscale=yscale, ylim=ylim,
                             title=title, save_path=save_path, display=display)
        else:
            plot_delta_sigma(sigma_pos=self.sigma_pos[:, objects], sigma_vel=self.sigma_vel[:, objects],
                             delta_pos=self.delta_pos[:, objects], delta_vel=self.delta_vel[:, objects], dt=self.dt,
                             t_0=self.t_0, style=style, yscale=yscale, ylim=ylim, title=title, save_path=save_path,
                             display=display)
        e = time.time()
        self.runtime['plot_sigma_delta'] += e - s

    def plot_rewards(self, style=None, yscale='linear'):
        s = time.time()
        plot_rewards(rewards=self.rewards, dt=self.dt, t_0=self.t_0, style=style, yscale=yscale)
        e = time.time()
        self.runtime['plot_rewards'] += e - s

    def plot_anees(self, axis=None, title='default', save_path='default', display=True, yscale='linear'):
        s = time.time()
        _ = self.anees
        if title == 'default':
            title = 'Filter performance for ' + str(self.m) + ' RSOs, seed = ' + str(self.init_seed)
        if save_path == 'default':
            save_path = str(self.m) + 'RSO_' + str(self.init_seed) + 'seed_anees_plot_axis_' + str(axis) + '.svg'
        plot_nees(self.nees, self.dt, self.t_0, style=None, yscale=yscale, axis=axis, title=title,
                  save_path=save_path, display=display)
        e = time.time()
        self.runtime['plot_anees'] += e - s

    def plot_actions(self, axis=0, title='default', save_path='default', display=True):
        s = time.time()
        if title == 'default':
            title = 'Frequency of observation for RSO (ID), ' + str(self.m) + ' RSOs, seed = ' + str(self.init_seed)
        if save_path == 'default':
            save_path = str(self.m) + 'RSO_' + str(self.init_seed) + 'seed_action_plot.svg'
        plot_histogram(self.actions[1:], bins='int', style=None, title=title, xlabel='RSO ID', save_path=save_path,
                       display=display)
        e = time.time()
        self.runtime['plot_actions'] += e - s

    @property
    def z_true_all(self):
        s = time.time()
        z = []
        for i in range(self.n):
            for j in range(self.m):
                z.append(self.hx(self.x_true[i, j, :3], self.trans_matrix[i], self.obs_lla, self.obs_itrs))
        z = np.array(z)
        z = np.reshape(z, (self.n, self.m, 3))
        e = time.time()
        self.runtime['all_true_obs'] += e - s
        return z

    @property
    def aer_true_all(self):
        s = time.time()
        aer = []
        for i in range(self.n):
            for j in range(self.m):
                aer.append(hx(self.x_true[i, j, :3], self.trans_matrix[i], self.obs_lla, self.obs_itrs))
        aer = np.array(aer)
        aer = np.reshape(aer, (self.n, self.m, 3))
        e = time.time()
        self.runtime['all_true_obs'] += e - s
        return aer

    def plot_visibility(self, save_path=None, display=True):
        s = time.time()
        visibility = self.aer_true_all[:, :, 1].T > self.obs_limit
        xlabel = 'Time Step (' + str(self.dt) + ' seconds per)'
        title = 'Visibility Plot (white = visible); elevation > ' + str(
            np.round(np.degrees(self.obs_limit), 0)) + ' degrees'
        plot_orbit_vis(visibility, title, xlabel, display=display, save_path=save_path)
        e = time.time()
        self.runtime['plot_visibility'] += e - s

    def plot_regimes(self, save_path=None, display=True):
        s = time.time()
        lla = np.array([ecef2lla(x[:3] @ self.trans_matrix[0]) for x in self.x_true[0]])

        coes = np.array(
            [rv2coe(k=Earth.k.to_value(u.km ** 3 / u.s ** 2), r=x[:3] / 1000, v=x[3:] / 1000) for x in self.x_true[0]])

        x = lla[:, 2] / 1000
        y = coes[:, 1]

        plot_regimes(np.column_stack((x, y)), save_path=save_path, display=display)
        e = time.time()
        self.runtime['plot_visibility'] += e - s

    def plot_NIS(self, save_path=None, display=True):
        NIS = []
        for i in range(1, self.n):
            NIS.append(self.y[i, self.actions[i]] @
                       np.linalg.inv(self.S[i, int(self.actions[i])]) @
                       self.y[i, int(self.actions[i])])
        title = 'Normalized Innovation Squared (NIS) for Observation at Each Time Step'
        xlabel = 'Time Step'
        ylabel = '$NIS = (z_{obs}^t-z_{pred}^t)(S^t)^{-1}(z_{obs}^t-z_{pred}^t)$ for i = [0, ' + str(
            self.n) + '), j = [0, ' + str(self.m) + ')'
        llabel = 'NIS'
        moving_average_plot(np.array(NIS), n=20, alpha=0.05, dof=len(self.z_true[0, 0]), style=None, title=title,
                            xlabel=xlabel, ylabel=ylabel, llabel=llabel, save_path=save_path, display=display)


    def plot_innovation_bounds(self, ID=None, save_path=None, display=True):
        if ID is None:
            innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
            st_dev = np.sqrt([np.diag(self.S[i, self.actions[i]]) for i in range(1, self.n)])
        else:
            innovation = np.empty((self.n - 1, len(self.z_true[0, 0])))
            st_dev = np.empty((self.n - 1, len(self.z_true[0, 0])))
            innovation[:], st_dev[:] = np.nan, np.nan
            for i in range(1, self.n):
                if self.actions[i] == ID:
                    innovation[i - 1] = self.y[i, self.actions[i]]
                    st_dev[i - 1] = np.sqrt(np.diag(self.S[i, self.actions[i]]))
        title = 'Innovation and Innovation Standard Deviation Bounds'
        xlabel = 'Time Step'
        if self.obs_type == 'xyz':
            ylabel = ['x (meters)', 'y (meters)', 'z (meters)']
            sharey = True
        if self.obs_type == 'aer':
            ylabel = ['Azimuth (radians)', 'Elevation (radians)', 'distance (meters)']
            sharey = False
        bound_plot(innovation, st_dev, style=None, title=title, xlabel=xlabel, ylabel=ylabel, yscale='linear',
                   sharey=sharey,
                   save_path=save_path, display=display)

    def innovation_bounds(self):
        innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
        st_dev = np.sqrt([np.diag(self.S[i, self.actions[i]]) for i in range(1, self.n)])
        frac_sigma_bound = np.mean((innovation < st_dev) * (innovation > -st_dev), axis=0)
        frac_two_sigma_bound = np.mean((innovation < 2 * st_dev) * (innovation > -2 * st_dev), axis=0)
        data = np.round(np.stack((frac_sigma_bound, frac_two_sigma_bound)) * 100, 2)
        if self.obs_type == 'xyz':
            df = pd.DataFrame(data, index=['Sigma', 'Two Sigmas'], columns=['x (meters)', 'y (meters)', 'z (meters)'])
        if self.obs_type == 'aer':
            df = pd.DataFrame(data, index=['Sigma', 'Two Sigmas'],
                              columns=['Azimuth (radians)', 'Elevation (radians)', 'distance (meters)'])
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(111)

        ax.table(cellText=df.values,
                 rowLabels=df.index,
                 colLabels=df.columns,
                 loc="center"
                 )
        ax.set_title("Innovation Standard Deviation Bounds (Percent)")

        ax.axis("off")

    def plot_map(self, objects=np.array([]), timesteps='All'):
        """
        :param objects:  plot specific objects to locate different orbits (GEO, MEO , LEO etc)
        :param timesteps: All steps
        :return: Map plot for all orbits showing uncertainty vs true
        """

        if timesteps == 'All':
            timesteps = [0, len(self.x_filter)]
        if objects.shape[0] == 0:
            objects = [0, len(self.x_filter[0])]
            x_filter = self.x_filter[timesteps[0]:timesteps[1], objects[0]:objects[1]]
            x_true = self.x_true[timesteps[0]:timesteps[1], objects[0]:objects[1]]
        else:
            x_filter = self.x_filter[timesteps[0]:timesteps[1], objects[:]]
            x_true = self.x_true[timesteps[0]:timesteps[1], objects[:]]
        map_plot(x_filter, x_true, self.trans_matrix, self.obs_lla)

    @property
    def innovation(self):
        innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
        RSO_ID = np.arange(0, self.m)
        indexes_not = [np.where(((self.actions[1:] == i)==False)*(self.obs_taken[1:])) for i in RSO_ID]
        innovations = [np.copy(innovation) for i in RSO_ID]
        for i in range(3):
            for j in range(len(RSO_ID)):
                np.put(innovations[j][:, i], np.array(indexes_not[j]), np.nan)
            np.put(innovation[:, i], np.where((self.obs_taken[1:]) == False), np.nan)
        return innovation, innovations

    @property
    def autocorrelation(self):
        innovation, innovations = self.innovation
        autocorrelation = []
        for i in range(0, 3):
            autocorrelation.append(acf(x=innovation[:, i], missing='conservative', fft=False))
        autocorrelation = np.array(autocorrelation)
        autocorrelations = []
        for j in range(len(innovations)):
            aa = []
            for i in range(0, 3):
                aa.append(acf(x=innovations[j][:, i], missing='conservative', fft=False))
            autocorrelations.append(np.array(aa))
        return autocorrelation, autocorrelations

    def plot_autocorrelation(self, RSO_ID=None, save_path=None, display=True):
        autocorrelation, autocorrelations = self.autocorrelation
        if RSO_ID is None:
            RSO_ID = np.arange(0, self.m)
        autocorrelations = [autocorrelations[j] for j in RSO_ID]
        if self.obs_type == 'xyz':
            ylabels = ['x', 'y', 'z']
        elif self.obs_type == 'aer':
            ylabels = ['Azimuth', 'Elevation', 'Range']
        else:
            ylabels = [None, None, None]
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        fig.suptitle('Autocorrelation of the Innovation \n lines for overall, colored `x` for individual objects')
        ci = 2/np.sqrt(self.n)
        for i in range(0, 3):
            axs[i].bar(x=np.arange(0, len(autocorrelation[i])), height=autocorrelation[i], width=0.3)
            axs[i].set_ylabel(ylabels[i])
            #axs[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
            axs[i].axhline(y=0, color='black', linestyle='-')
            axs[i].axhline(y=-ci, color='black', linestyle='--', alpha=0.3)
            axs[i].axhline(y=ci, color='black', linestyle='--', alpha=0.3)
            for j in range(len(RSO_ID)):
                axs[i].scatter(x=np.arange(0, len(autocorrelations[j][i])), y=autocorrelations[j][i], marker='x')
        axs[-1].set_xlabel('Observations Included (- for prior, + for subsequent)')
        if save_path is not None:
            plt.savefig(save_path, dpi=300, format='svg')
        if display:
            plt.show()
        else:
            plt.close()

    def fitness_test(self):
        """
        Source: http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf
        Test 1 - Innovation magnitude bound test
        Test 2 - Normalized innovations squared χ2 test
        Test 3 - Innovation whiteness (autocorrelation) test
        Test 4 - Normalized estimation error squared χ2 test
        """
        innovation, innovations = self.innovation

        if self.obs_type == 'xyz':
            clabels = ['x', 'y', 'z']
        elif self.obs_type == 'aer':
            clabels = ['Azimuth', 'Elevation', 'Range']

        autocorr, autocorrs = self.autocorrelation
        autocorr_contained = []
        for i in range(autocorr.shape[0]):
            autocorr_contained.append(np.mean(
                (autocorr[i] < 2/np.sqrt(self.n)) * (autocorr[i] > -2/np.sqrt(self.n))))
        overall_innovation_acorr = np.copy(autocorr_contained)

        rso_innovation_acorr = []
        for a in autocorrs:
            autocorr_contained = []
            for i in range(a.shape[0]):
                autocorr_contained.append(np.mean((a[i] < 2/np.sqrt(np.sum(self.actions==i))) * (
                            a[i] > -2/np.sqrt(np.sum(self.actions==i)))))
            rso_innovation_acorr.append(np.copy(autocorr_contained))
        rso_innovation_acorr_stat = [np.max(rso_innovation_acorr, axis=0), np.min(rso_innovation_acorr, axis=0)]

        innovation_acorr_stats = np.round(np.vstack((overall_innovation_acorr, rso_innovation_acorr_stat)) * 100, 2)

        innovation_autocorrelation_test = pd.DataFrame(innovation_acorr_stats,
                                                       index=['Test 3a: Acorr overall 2σ',
                                                              'Test 3b: Acorr per object max 2σ',
                                                              'Test 3c: Acorr per object min 2σ'],
                                                       columns=clabels)

        inn_st_dev = np.sqrt([np.diag(self.S[i, self.actions[i]]) for i in range(1, self.n)])
        innovation_nonan = np.array([innovation[~np.isnan(innovation[:, i]), i] for i in range(len(innovation[0]))]).T
        inn_st_dev_nonan = np.array([inn_st_dev[~np.isnan(inn_st_dev[:, i]), i] for i in range(len(inn_st_dev[0]))]).T
        inn_frac_sigma_bound = np.mean((innovation_nonan < inn_st_dev_nonan) * (innovation_nonan > -inn_st_dev_nonan), axis=0)
        inn_frac_two_sigma_bound = np.mean((innovation_nonan < 2 * inn_st_dev_nonan) * (innovation_nonan > -2 * inn_st_dev_nonan), axis=0)
        inn_frac_bound = np.round(np.stack((inn_frac_sigma_bound, inn_frac_two_sigma_bound)) * 100, 2)
        innovation_bound_test = pd.DataFrame(inn_frac_bound,
                                             index=['Test 1a: Innovation 1σ bound',
                                                    'Test 1b: Innovation 2σ bound'],
                                             columns=clabels)

        NIS = []
        for i in range(1, self.n):
            NIS.append(self.y[i, self.actions[i]] @
                       np.linalg.inv(self.S[i, int(self.actions[i])]) @
                       self.y[i, int(self.actions[i])])

        NIS = np.array(NIS)
        NIS = NIS[~np.isnan(NIS)]
        alpha = 0.05
        ci = [alpha / 2, 1 - alpha / 2]
        nis_cr_points = stats.chi2.ppf(ci, df=len(self.y[0, 0]))
        points_contained = np.mean((NIS > nis_cr_points[0]) * (NIS < nis_cr_points[1]))
        normalized_innovation_squared_test = pd.DataFrame(np.round(points_contained * 100, 2),
                                                          index=['Test 2: NIS χ2'],
                                                          columns=['95% CI'])

        nees_cr_points = stats.chi2.ppf(ci, df=6)
        _ = self.anees
        nees_points_contained = np.round(np.mean((self.nees > nees_cr_points[0]) * (self.nees < nees_cr_points[1])), 4)
        normalized_estimation_error_squared_test = pd.DataFrame(np.round(nees_points_contained * 100, 2),
                                                                index=['Test 4: NEES χ2'],
                                                                columns=['95% CI'])

        Tests = innovation_bound_test.append(normalized_innovation_squared_test.append(
            innovation_autocorrelation_test.append(normalized_estimation_error_squared_test)))
        Tests = Tests.replace(np.nan, '', regex=True)
        return Tests

    @property
    def innovation_dw_test(self):
        """
        concept source: https://www.statsmodels.org/stable/_modules/statsmodels/stats/stattools.html#durbin_watson
        Calculates the Durbin-Watson statistic
         Notes
        -----
        The null hypothesis of the test is that there is no serial correlation.
        The Durbin-Watson test statistics is defined as:

        .. math::

           \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

        The test statistic is approximately equal to 2*(1-r) where ``r`` is the
        sample autocorrelation of the residuals. Thus, for r == 0, indicating no
        serial correlation, the test statistic equals 2. This statistic will
        always be between 0 and 4. The closer to 0 the statistic, the more
        evidence for positive serial correlation. The closer to 4, the more
        evidence for negative serial correlation.

        :return: autocorrelation test - Durbin-Watson Statistic for Innovation
        """

        innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
        innovations = []
        indexes = [np.where(self.actions[:] == i) for i in range(0, self.m)]
        for index in indexes:
            ys = []
            for i in index:
                ys.append(self.y[i, self.actions[i]])
            innovations.append(np.copy(np.array(ys)))
        # Durbin-Watson statistic for all obs
        diff_innovation = np.diff(innovation, 1, axis=0)
        dw_innovation = np.round(np.sum(diff_innovation ** 2, axis=0) / np.sum(innovation ** 2, axis=0), 3)
        dw_innovations = []
        # Durbin-Watson statistic for min-max per obj
        for inn in innovations:
            diff_inn = np.diff(inn[0], 1, axis=0)
            dw_innovations.append(np.sum(diff_inn ** 2, axis=0) / np.sum(inn[0] ** 2, axis=0))
        dw_innovations = np.array(dw_innovations)
        dw_innovations_stat = np.round(np.stack([np.min(dw_innovations, axis=0), np.max(dw_innovations, axis=0)]), 3)
        if self.obs_type == 'xyz':
            clabels = ['x', 'y', 'z']
        elif self.obs_type == 'aer':
            clabels = ['Azimuth', 'Elevation', 'Range']
        dw_autocorr_test = pd.DataFrame(data=np.vstack([dw_innovation, dw_innovations_stat]), columns=clabels,
                                        index=['Durbin-Watson Statistic for All Obs',
                                               'Durbin-Watson Statistic for Min per Obj',
                                               'Durbin-Watson Statistic for Max per Obj'])
        dw_autocorr_test.name = 'Durbin-Watson Statistic for Innovation'
        return dw_autocorr_test

