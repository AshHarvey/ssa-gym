import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from filterpy.kalman import MerweScaledSigmaPoints as SigmasPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise as Q_noise_fn
from poliastro.bodies import Earth
from poliastro.core.elements import rv2coe
from envs.transformations import arcsec2rad, deg2rad, lla2ecef, gcrs2irts_matrix_b, get_eops, ecef2aer, ecef2lla
from envs.dynamics import fx_xyz_farnocchia, hx_aer_erfa, mean_z_uvw, residual_z_aer, robust_cholesky
from envs.results import observations as obs_fn, error, error_failed, plot_delta_sigma, plot_rewards, plot_nees
from envs.results import plot_histogram, plot_orbit_vis, plot_regimes, reward_proportional_trinary_true
from envs.results import moving_average_plot, bound_plot
import gym
from gym.utils import seeding
from copy import copy
import time
import pandas as pd
import matplotlib.pyplot as plt


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

sample_orbits = np.load('envs/sample_orbits.npy')


class SSA_Tasker_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    RE = Earth.R_mean.to_value(u.m) # radius of earth

    def __init__(self, steps=2880, rso_count=50, time_step=30.0, t_0=datetime(2020, 5, 4, 0, 0, 0),
                 obs_limit=15, observer=(38.828198, -77.305352, 20.0), x_sigma=(1000, 1000, 1000, 10, 10, 10),
                 z_sigma=(1, 1, 1000), q_sigma=0.001, P_0=None, R=None,  update_interval=1, orbits=sample_orbits,
                 fx=fx_xyz_farnocchia, hx=hx_aer_erfa, alpha=0.001, beta=2., kappa=3-6, mean_z=mean_z_uvw,
                 residual_z=residual_z_aer, msqrt=robust_cholesky, obs_type='aer'):
        super(SSA_Tasker_Env, self).__init__()
        s = time.time()
        self.runtime = {'__init__': 0, 'reset': 0, 'step': 0, 'step prep': 0, 'propagate next true state': 0,
                        'perform predictions': 0, 'update with observation': 0, 'Observations and Reward': 0,
                        'filter_error': 0, 'visible_objects': 0, 'object_visibility': 0, 'anees': 0,
                        'failed_filters': 0, 'plot_sigma_delta': 0, 'plot_rewards': 0, 'plot_anees': 0,
                        'plot_actions': 0, 'all_true_obs': 0, 'plot_visibility': 0}
        """Simulation configuration"""
        self.t_0 = t_0 # time at start of simulation
        self.dt = time_step # length of time steps [s]
        self.n = steps # max run steps
        self.m = rso_count # number of Resident Space Object (RSO) to include in the simulation
        self.obs_limit = np.radians(obs_limit) # don't observe objects below this elevation [rad]
        # configuration parameters for RSOs; sma: Semi-major axis [m], ecc: Eccentricity [u], inc: Inclination (rad),
        # raan: Right ascension of the ascending node (rad), argp: Argument of perigee (rad), nu: True anomaly (rad)
        self.orbits = orbits # orbits to sample from
        self.obs_lla = np.array(observer)*[deg2rad, deg2rad, 1] # lat, lon, height (deg, deg, m)
        self.obs_itrs = lla2ecef(self.obs_lla) # ITRS (m)
        self.update_interval = update_interval # how often an observation should be taken
        self.i = 0
        """Filter configuration"""
        # standard deviation of noise added to observations [rad, rad, m]
        self.obs_type = obs_type
        if self.obs_type=='aer':
            self.z_sigma = z_sigma * np.array([arcsec2rad, arcsec2rad, 1])
        elif self.obs_type=='xyz':
            self.z_sigma = z_sigma
        else:
            print('Invalid Observation Type: ' + str(obs_type))
            exit()
        # standard deviation of noise added to initial state estimates; [m, m, m, m/s, m/s, m/s]
        self.x_sigma = np.array(x_sigma)
        self.Q = Q_noise_fn(dim=2, dt=self.dt, var=q_sigma**2, block_size=3, order_by_dim=False)
        self.eops = get_eops()
        self.fx = fx
        self.hx = hx
        self.mean_z = mean_z
        self.residual_z = residual_z
        self.msqrt = msqrt
        self.alpha, self.beta, self.kappa = alpha, beta, kappa # sigma point configuration parameters
        """Prep arrays"""
        # variables for the filter
        x_dim = 6
        z_dim = 3
        if P_0 is None:
            self.P_0 = np.copy(np.diag(self.x_sigma**2)) # Assumed covariance of the estimates at simulation start
        else:
            self.P_0 = np.copy(P_0)
        if R is None:
            self.R = np.diag(self.z_sigma**2) # Noise added to the filter during observation updates
        else:
            self.R = np.copy(R)
        self.x_true = np.empty(shape=(self.n, self.m, x_dim)) # means for all objects at each time step
        self.x_filter = np.empty(shape=(self.n, self.m, x_dim)) # means for all objects at each time step
        self.P_filter = np.empty(shape=(self.n, self.m, x_dim, x_dim)) # covariances for all objects at each time step
        self.obs = np.empty(shape=(self.n, self.m, x_dim * 2)) # observations for all objects at each time step
        self.time = [self.t_0 + timedelta(seconds=self.dt)*i for i in range(self.n)] # time for all time steps
        self.trans_matrix = gcrs2irts_matrix_b(self.time, self.eops) # used for celestial to terrestrial
        self.z_noise = np.empty(shape=(self.n, self.m, z_dim)) # array to contain the noise added to each observation
        self.z_true = np.empty(shape=(self.n, self.m, z_dim)) # array to contain the observations which are made
        self.y = np.empty(shape=(self.n, self.m, z_dim)) # array to contain the innovation of each observation
        self.S = np.empty(shape=(self.n, self.m, z_dim, z_dim)) # array to contain the innovation covariance
        self.x_noise = np.empty(shape=(self.m, x_dim)) # array to contain the noise added to each RSO
        self.filters = [] # creates a list for ukfs
        # variables for environment performance
        self.delta_pos = np.empty((self.n, self.m)) # euclidean distance between true and filter mean position elements
        self.delta_vel = np.empty((self.n, self.m)) # euclidean distance between true and filter mean velocity elements
        self.sigma_pos = np.empty((self.n, self.m)) # euclidean magnitude of diagonal position elements of covariance
        self.sigma_vel = np.empty((self.n, self.m)) # euclidean magnitude of diagonal velocity elements of covariance
        self.scores = np.empty((self.n, self.m)) # score of each RSO at each time step
        self.rewards = np.empty(self.n) # reward for each time step
        self.failed_filters_id = [] # prep list for failed states
        self.failed_filters_msg = ["None"]*self.m # prep list for failure messages
        self.actions = np.empty(self.n, dtype=int) # prep variable for keeping track of all previous actions
        self.x_failed = np.array([1e20, 1e20, 1e20, 1e12, 1e12, 1e12]) # failed filter will be set to this value
        self.P_failed = np.diag([1e10, 1e10, 1e10, 1e5, 1e5, 1e5]) # failed filter will be set to this value
        self.nees = np.empty((self.n, self.m)) # used for normalized estimation error squared (NEES) and its average
        self.visibility = [] # used to store a log of visible objects at each time step
        self.sigmas_h = np.empty((self.n, x_dim*2+1, z_dim)) # used to store sigmas points used in updates
        """Define Gym spaces"""
        self.action_space = gym.spaces.Discrete(self.m) # the action is choosing which RSO to look at
        self.observation_space = gym.spaces.Box(low=np.tile(-np.inf, (self.m, 12)), high=np.tile(np.inf, (self.m, 12)),
                                                dtype=np.float64) # the obs is x [6] and diag(P) [6] for each RSO [m]
        """Initial reset and seed calls"""
        self.np_random = None
        self.init_seed = self.seed()
        self.reset()
        e = time.time()
        self.runtime['__init__'] += e-s

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_seed = seed
        return [seed]
    
    def reset(self):
        s = time.time()
        """reset filter"""
         # Clear history
        self.x_true[:], self.x_filter[:], self.P_filter[:], self.obs[:], self.sigmas_h[:] = [0]*5
        self.z_true[:], self.y[:], self.S[:] = np.nan, np.nan, np.nan
        """initialize RSO"""
        self.filters = []
        for j in range(self.m):
            self.x_true[0][j] = self.orbits[self.np_random.randint(low=0, high=self.orbits.shape[0]), :]
            self.x_noise[j] = self.np_random.normal(size=6)*self.x_sigma
            self.x_filter[0][j] = np.copy(self.x_true[0][j] + self.x_noise[j])
            self.P_filter[0][j] = np.copy(self.P_0)
            self.filters.append(UKF(dim_x=6, dim_z=3, dt=self.dt, fx=self.fx, hx=self.hx,
                                    points=SigmasPoints(n=6, alpha=self.alpha, beta=self.beta, kappa=self.kappa,
                                                        sqrt_method=self.msqrt),
                                    z_mean_fn=self.mean_z, residual_z=self.residual_z, sqrt_fn=self.msqrt))
            self.filters[j].x = np.copy(self.x_filter[0][j])
            self.filters[j].P = np.copy(self.P_filter[0][j]) # initial uncertainty
            self.filters[j].R = np.copy(self.R) # uncertainty of each observation
            self.filters[j].Q = np.copy(self.Q) # uncertainty of each prediction
        for i in range(self.n):
            for j in range(self.m):
                self.z_noise[i, j] = self.np_random.normal(size=3)*self.z_sigma
        """Reset variables for tracking environment performance"""
        # blank out all tracking variables
        self.scores[:], self.delta_pos[:], self.delta_vel[:], self.sigma_pos[:], self.sigma_vel[:] = [np.nan]*5
        self.actions[:], self.failed_filters_id, self.visibility = np.nan, [], []
        self.failed_filters_msg = ["None"]*self.m # reset list for failure messages
        """Observations and Reward"""
        self.obs[0] = obs_fn(self.x_filter[0], self.P_filter[0]) # set initial observation based on x and P
        self.delta_pos[0], self.delta_vel[0], self.sigma_pos[0], self.sigma_vel[0] = error(self.x_true[0],
                                                                                           self.obs[0]) # initial error
        self.rewards[:] = 0
        self.i = 0 # sets initial time step
        e = time.time()
        self.runtime['reset'] += e-s
        return self.obs[0]
    
    def step(self, a):
        step_s = time.time()
        s = time.time()
        assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a)) # check for valid action
        self.i += 1 # increments current step
        self.actions[self.i] = np.copy(a) # record current action
        e = time.time()
        self.runtime['step prep'] += e-s
        """propagate next true state"""
        s = time.time()
        for j in range(self.m):
            self.x_true[self.i][j] = self.fx(self.x_true[self.i-1][j], self.dt)
        e = time.time()
        self.runtime['propagate next true state'] += e-s
        """perform predictions"""
        s = time.time()
        for j in range(self.m):
            if not(j in self.failed_filters_id):
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
            self.x_filter[self.i, j] = np.copy(self.filters[j].x) # update filter mean history
            self.P_filter[self.i, j] = np.copy(self.filters[j].P) # update covariance mean history
        e = time.time()
        self.runtime['perform predictions'] += e-s
        """update with observation"""
        s = time.time()
        if self.i % self.update_interval == 0:
            if not(a in self.failed_filters_id):
                hx_kwargs = {"trans_matrix": self.trans_matrix[self.i],
                             "observer_itrs": self.obs_itrs,
                             "observer_lla": self.obs_lla,
                             "time": self.time[self.i]}
                self.z_true[self.i, a] = self.hx(self.x_true[self.i][a], **hx_kwargs)
                if ecef2aer(self.obs_lla, self.x_true[self.i][a][:3], self.obs_itrs)[1] >= self.obs_limit:
                    try:
                        self.filters[a].update(self.z_true[self.i, a] + self.z_noise[self.i, a], **hx_kwargs)
                        self.y[self.i, a] = np.copy(self.filters[a].y)
                        self.S[self.i, a] = np.copy(self.filters[a].S)
                        self.sigmas_h[self.i] = np.copy(self.filters[a].sigmas_h)
                        if np.any(np.isnan(self.filters[a].x)):
                            self.filter_error(object_id=a, activity='update', error_type=', update returned nan. ')
                    except ValueError:
                        self.filter_error(object_id=a, activity='update', error_type=', ValueError. ')
                    except np.linalg.LinAlgError:
                        self.filter_error(object_id=a, activity='update', error_type=', LinAlgError. ')
                    except:
                        self.filter_error(object_id=a, activity='update', error_type=', Unknown. ')
                    self.x_filter[self.i, a] = np.copy(self.filters[a].x) # update filter mean history
                    self.P_filter[self.i, a] = np.copy(self.filters[a].P) # update covariance mean history
        e = time.time()
        self.runtime['update with observation'] += e-s
        """Observations and Reward"""
        s = time.time()
        self.obs[self.i] = obs_fn(self.x_filter[self.i], self.P_filter[self.i])
        tmp = error(self.x_true[self.i], self.obs[self.i])
        self.delta_pos[self.i], self.delta_vel[self.i],  self.sigma_pos[self.i],  self.sigma_vel[self.i] = tmp
        self.rewards[self.i] = reward_proportional_trinary_true(self.delta_pos[self.i])
        done = False
        if self.i + 1 >= self.n:
            done = True
        e = time.time()
        self.runtime['Observations and Reward'] += e-s
        step_e = time.time()
        self.runtime['step'] += step_e-step_s
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
        self.failed_filters_msg[object_id] = copy(msg) # record error message
        self.failed_filters_id.append(object_id) # add filter it list of failed filters
        e = time.time()
        self.runtime['filter_error'] += e-s

    @property
    def visible_objects(self):
        s = time.time()
        x_itrs = self.x_true[self.i, :, :3]@self.trans_matrix[self.i]
        el = np.array([ecef2aer(self.obs_lla, x, self.obs_itrs)[1] for x in x_itrs])
        viz = np.where(el >= self.obs_limit)[0]
        e = time.time()
        self.runtime['visible_objects'] += e-s
        return viz

    @property
    def object_visibility(self):
        s = time.time()
        x_itrs = self.x_true[self.i, :, :3]@self.trans_matrix[self.i]
        el = np.array([ecef2aer(self.obs_lla, x, self.obs_itrs)[1] for x in x_itrs])
        viz = el >= self.obs_limit
        e = time.time()
        self.runtime['object_visibility'] += e-s
        return viz


    @property
    def anees(self):
        s = time.time()
        # returns the average normalized estimation error squared (ANEES)
        # based on 3.1 in https://pdfs.semanticscholar.org/1c1f/6c864789630d8cd37d5342f67ad8d480f077.pdf
        delta = self.x_true - self.x_filter
        for i in range(self.n):
            for j in range(self.m):
                self.nees[i, j] = delta[i, j] @ np.linalg.inv(self.P_filter[i, j]) @ delta[i, j]
        e = time.time()
        self.runtime['anees'] += e-s
        return np.mean(self.nees)

    def failed_filters(self):
        s = time.time()
        if not self.failed_filters_id:
            print("No failed Objects")
        else:
            for rso_id in self.failed_filters_id:
                print(self.failed_filters_msg[rso_id])
        e = time.time()
        self.runtime['failed_filters'] += e-s

    def plot_sigma_delta(self, style=None, yscale='log', objects=np.array([]), ylim='max', title='default', save_path='default', display=True):
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
        self.runtime['plot_sigma_delta'] += e-s

    def plot_rewards(self, style=None, yscale='linear'):
        s = time.time()
        plot_rewards(rewards=self.rewards, dt=self.dt, t_0=self.t_0, style=style, yscale=yscale)
        e = time.time()
        self.runtime['plot_rewards'] += e-s

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
        self.runtime['plot_anees'] += e-s

    def plot_actions(self, axis=0, title='default', save_path='default', display=True):
        s = time.time()
        if title == 'default':
            title = 'Frequency of observation for RSO (ID), ' + str(self.m) + ' RSOs, seed = ' + str(self.init_seed)
        if save_path == 'default':
            save_path = str(self.m) + 'RSO_' + str(self.init_seed) + 'seed_action_plot.svg'
        plot_histogram(self.actions, bins=self.m, style=None, title=title, xlabel='RSO ID', save_path=save_path,
                       display=display)
        e = time.time()
        self.runtime['plot_actions'] += e-s

    @property
    def all_true_obs(self):
        s = time.time()
        aer = []
        for i in range(self.n):
            for j in range(self.m):
                aer.append(self.hx(self.x_true[i, j, :3], self.trans_matrix[j], self.obs_lla, self.obs_itrs))
        aer = np.array(aer)
        aer = np.reshape(aer, (self.n, self.m, 3))
        e = time.time()
        self.runtime['all_true_obs'] += e-s
        return aer

    def plot_visibility(self, save_path=None, display=True):
        s = time.time()
        observations = self.all_true_obs
        plot_orbit_vis(observations, self.obs_limit, self.dt, display=display, save_path=save_path)
        e = time.time()
        self.runtime['plot_visibility'] += e-s

    def plot_regimes(self, save_path=None, display=True):
        s = time.time()
        lla = np.array([ecef2lla(x[:3]@self.trans_matrix[0]) for x in self.x_true[0]])

        coes = np.array([rv2coe(k=Earth.k.to_value(u.km**3/u.s**2), r=x[:3]/1000, v=x[3:]/1000) for x in self.x_true[0]])

        x = lla[:, 2]/1000
        y = coes[:, 1]

        plot_regimes(np.column_stack((x, y)), save_path=save_path, display=display)
        e = time.time()
        self.runtime['plot_visibility'] += e-s

    def plot_NIS(self, save_path=None, display=True):
        NIS = []
        for i in range(1, self.n):
            NIS.append(self.y[i, self.actions[i]] @
                       np.linalg.inv(self.S[i, int(self.actions[i])]) @
                       self.y[i, int(self.actions[i])])
        title = 'Normalized Innovation Squared (NIS) for Observation at Each Time Step'
        xlabel = 'Time Step'
        ylabel = '$NIS = (z_{obs}^t-z_{pred}^t)(S^t)^{-1}(z_{obs}^t-z_{pred}^t)$'
        llabel = 'NIS'
        moving_average_plot(np.array(NIS), n=20, alpha=0.05, dof=len(self.z_true[0, 0]), style=None, title=title,
                            xlabel=xlabel, ylabel=ylabel, llabel=llabel, save_path=save_path, display=display)

    def plot_innovation_bounds(self, save_path=None, display=True):
        innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
        st_dev = np.sqrt([np.diag(self.S[i, self.actions[i]]) for i in range(1, self.n)])

        title = 'Innovation and Innovation Standard Deviation Bounds'
        xlabel = 'Time Step'
        if self.obs_type == 'xyz':
            ylabel = ['x (meters)', 'y (meters)', 'z (meters)']
            sharey = True
        if self.obs_type == 'aer':
            ylabel = ['Azimuth (radians)', 'Elevation (radians)', 'distance (meters)']
            sharey = False
        bound_plot(innovation, st_dev, style=None, title=title, xlabel=xlabel, ylabel=ylabel, yscale='linear', sharey=sharey,
                   save_path=save_path, display=display)

    @property
    def innovation_bounds(self):
        innovation = np.array([self.y[i, self.actions[i]] for i in range(1, self.n)])
        st_dev = np.sqrt([np.diag(self.S[i, self.actions[i]]) for i in range(1, self.n)])
        frac_sigma_bound = np.mean((innovation < st_dev)*(innovation > -st_dev), axis=0)
        frac_two_sigma_bound = np.mean((innovation < 2*st_dev)*(innovation > -2*st_dev), axis=0)
        data = np.round(np.stack((frac_sigma_bound, frac_two_sigma_bound))*100, 2)
        if self.obs_type == 'xyz':
            df = pd.DataFrame(data, index=['Sigma', 'Two Sigmas'], columns=['x (meters)', 'y (meters)', 'z (meters)'])
        if self.obs_type == 'aer':
            df = pd.DataFrame(data, index=['Sigma', 'Two Sigmas'], columns=['Azimuth (radians)', 'Elevation (radians)', 'distance (meters)'])
        fig = plt.figure(figsize = (8, 2))
        ax = fig.add_subplot(111)

        ax.table(cellText=df.values,
                  rowLabels=df.index,
                  colLabels=df.columns,
                  loc="center"
                 )
        ax.set_title("Innovation Standard Deviation Bounds (Percent)")

        ax.axis("off")

