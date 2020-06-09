import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from filterpy.kalman import MerweScaledSigmaPoints as SigmasPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise as Q_noise_fn
from poliastro.bodies import Earth
from envs.transformations import arcsec2rad, deg2rad, lla2itrs, gcrs2irts_matrix_b, get_eops
from envs.dynamics import init_state_vec, fx_xyz_markley as fx, hx_aer_kwargs as hx
from envs.results import observations as obs_fn, error, error_failed, plot_delta_sigma
import gym
from gym.utils import seeding
from copy import copy

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
    metadata = {'render.modes': ['human']}
    RE = Earth.R_mean.to_value(u.m) # radius of earth

    def __init__(self, steps=480, rso_count=50, time_step=30.0, t_0=datetime(2020, 3, 15, 0, 0, 0),
                 obs_limit=15, observer=(38.828198, -77.305352, 20.0), x_sigma=(1000, 1000, 1000, 10, 10, 10),
                 z_sigma=(1, 1, 1000), q_sigma=0.001, update_interval=1, sma=((RE + 400000), 42164000),
                 ecc=(0.001, 0.3), inc=(0, 180), raan=(0, 360), argp=(0, 360), nu=(0, 360)):
        super(SSA_Tasker_Env, self).__init__()
        """Simulation configuration"""
        self.t_0 = t_0 # time at start of simulation
        self.dt = time_step # length of time steps [s]
        self.n = steps # max run steps
        self.m = rso_count # number of Resident Space Object (RSO) to include in the simulation
        self.obs_limit = np.radians(obs_limit) # don't observe objects below this elevation [rad]
        # configuration parameters for RSOs; sma: Semi-major axis [m], ecc: Eccentricity [u], inc: Inclination (rad),
        # raan: Right ascension of the ascending node (rad), argp: Argument of perigee (rad), nu: True anomaly (rad)
        self.sma, self.ecc, self.inc, self.raan, self.argp, self.nu = sma, ecc, inc, raan, argp, nu # orbit sample para
        self.obs_itrs = lla2itrs(np.array(observer)*[deg2rad, deg2rad, 1]) # lat, lon, height (deg, deg, m) to ITRS (m)
        self.update_interval = update_interval # how often an observation should be taken
        self.i = 0
        """Filter configuration"""
        # standard deviation of noise added to observations [rad, rad, m]
        self.z_sigma = z_sigma * np.array([arcsec2rad, arcsec2rad, 1])
        # standard deviation of noise added to initial state estimates; [m, m, m, m/s, m/s, m/s]
        self.x_sigma = np.array(x_sigma)
        self.Q = Q_noise_fn(dim=2, dt=self.dt, var=q_sigma**2, block_size=3, order_by_dim=False)
        self.eops = get_eops()
        """Prep arrays"""
        # variables for the filter
        x_dim = 6
        z_dim = 3
        self.P_0 = np.copy(np.diag(self.x_sigma**2)) # Assumed covariance of the estimates at simulation start
        self.R = np.diag(self.z_sigma**2) # Noise added to the filter during observation updates
        self.x_true = np.empty(shape=(self.n, self.m, x_dim)) # means for all objects at each time step
        self.x_filter = np.empty(shape=(self.n, self.m, x_dim)) # means for all objects at each time step
        self.P_filter = np.empty(shape=(self.n, self.m, x_dim, x_dim)) # covariances for all objects at each time step
        self.obs = np.empty(shape=(self.n, self.m, x_dim * 2)) # observations for all objects at each time step
        self.trans_matrix = [gcrs2irts_matrix_b(self.t_0 + timedelta(seconds=self.dt)*i,
                                                self.eops) for i in range(self.n)] # used for celestial to terrestrial
        self.z_noise = np.empty(shape=(self.n, z_dim)) # array to contain the noise added to each observation
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
        self.actions = np.empty(self.n) # prep variable for keeping track of all previous actions
        self.x_failed = np.repeat(1e50, 6) # this is the value any failed filter will be set to
        self.P_failed = np.tile(1e50, (6, 6)) # this is the value any failed filter will be set to
        """Define Gym spaces"""
        self.action_space = gym.spaces.Discrete(self.m) # the action is choosing which RSO to look at
        self.observation_space = gym.spaces.Box(low=np.tile(-np.inf, (self.m, 12)), high=np.tile(np.inf, (self.m, 12)),
                                                dtype=np.float64) # the obs is x [6] and diag(P) [6] for each RSO [m]
        """Initial reset and seed calls"""
        self.np_random = None
        self.init_seed = self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        """reset filter"""
        self.P_0 = np.copy(np.diag(self.x_sigma**2)) # Assumed covariance of the estimates at simulation start
        self.R = np.diag(self.z_sigma**2) # Noise added to the filter during observation updates
        self.x_true[:], self.x_filter[:], self.P_filter[:], self.obs[:] = 0, 0, 0, 0 # zero out filter history
        """initialize RSO"""
        self.filters = []
        for j in range(self.m):
            self.x_true[0][j] = init_state_vec(sma=self.sma, ecc=self.ecc, inc=self.inc, raan=self.raan, argp=self.argp,
                                               nu=self.nu, random_state=self.np_random)
            self.x_noise[j] = self.np_random.normal(size=6)*self.x_sigma
            self.x_filter[0][j] = np.copy(self.x_true[0][j] + self.x_noise[j])
            self.P_filter[0][j] = np.copy(self.P_0)
            self.filters.append(UKF(dim_x=6, dim_z=3, dt=self.dt, fx=fx, hx=hx,
                                    points=SigmasPoints(n=6, alpha=0.001, beta=2., kappa=3-6)))
            self.filters[j].x = np.copy(self.x_filter[0][j])
            self.filters[j].P = np.copy(self.P_filter[0][j]) # initial uncertainty
            self.filters[j].R = np.copy(self.R) # uncertainty of each observation
            self.filters[j].Q = np.copy(self.Q) # uncertainty of each prediction
        for i in range(self.n):
            self.z_noise[i] = self.np_random.normal(size=3)*self.z_sigma
        """Reset variables for tracking environment performance"""
        self.delta_pos[:], self.delta_vel[:], self.sigma_pos[:], self.sigma_vel[:] = np.nan, np.nan, np.nan, np.nan
        self.scores[:], self.actions[:], self.failed_filters_id = np.nan, np.nan, [] # blank out all tracking variables
        self.failed_filters_msg = ["None"]*self.m # reset list for failure messages
        """Observations and Reward"""
        self.obs[0] = obs_fn(self.x_filter[0], self.P_filter[0]) # set initial observation based on x and P
        self.delta_pos[0], self.delta_vel[0], self.sigma_pos[0], self.sigma_vel[0] = error(self.x_true[0],
                                                                                           self.obs[0]) # initial error
        self.rewards[:] = 0
        self.i = 0 # sets initial time step
        return self.obs[0]
    
    def step(self, a):
        assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a)) # check for valid action
        self.i += 1 # increments current step
        self.actions[self.i] = np.copy(a) # record current action
        """propagate next true state"""
        for j in range(self.m):
            self.x_true[self.i][j] = fx(self.x_true[self.i-1][j], self.dt)
        """perform predictions"""
        for j in range(self.m):
            try:
                if not(j in self.failed_filters_id):
                    self.filters[j].predict()
            except np.linalg.LinAlgError:
                self.filters[j].x = np.copy(self.x_failed)
                self.filters[j].P = np.copy(self.P_failed)
                msg = ["".join(['Failed on prediction step ', str(self.i-1), ', sigma points not positive definite. ',
                                str(np.round(error_failed(state=self.x_true[self.i-1, j], x=self.x_filter[self.i-1, j],
                                                          P=np.diag(self.P_filter[self.i-1, j])), 2))])]
                self.failed_filters_msg[j] = copy(msg) # record error message
                self.failed_filters_id.append(j) # add filter it list of failed filters
            self.x_filter[self.i, j] = np.copy(self.filters[j].x) # update filter mean history
            self.P_filter[self.i, j] = np.copy(self.filters[j].P) # update covariance mean history
        """update with observation"""
        if self.i % self.update_interval == 0:
            try:
                if not(a in self.failed_filters_id):
                    hx_kwargs = {"trans_matrix": self.trans_matrix[self.i], "observer_itrs": self.obs_itrs}
                    obs_true = hx(self.x_true[self.i][a], **hx_kwargs)
                    self.filters[a].update(obs_true + self.z_noise[self.i], **hx_kwargs)
            except np.linalg.LinAlgError:
                self.filters[a].x = np.copy(self.x_failed)
                self.filters[a].P = np.copy(self.P_failed)
                msg = ["".join(['Failed on update step ', str(self.i-1), ', sigma points not positive definite. ',
                                str(np.round(error_failed(state=self.x_true[self.i-1, a], x=self.x_filter[self.i-1, a],
                                                          P=np.diag(self.P_filter[self.i-1, a])), 2))])]
                self.failed_filters_msg[a] = copy(msg) # record error message
                self.failed_filters_id.append(a) # add filter it list of failed filters
            self.x_filter[self.i, a] = np.copy(self.filters[a].x) # update filter mean history
            self.P_filter[self.i, a] = np.copy(self.filters[a].P) # update covariance mean history
        """Observations and Reward"""
        self.obs[self.i] = obs_fn(self.x_filter[self.i], self.P_filter[self.i])
        tmp = error(self.x_true[self.i], self.obs[self.i])
        self.delta_pos[self.i], self.delta_vel[self.i],  self.sigma_pos[self.i],  self.sigma_vel[self.i] = tmp
        self.rewards[self.i] = -np.max(self.delta_pos[self.i])
        done = False
        if self.i + 1 >= self.n:
            done = True
        return self.obs[self.i], self.rewards[self.i], done, {}  # observations, reward, and done

    def plot_sigma_delta(self, style=None, yscale='log'):
        plot_delta_sigma(sigma_pos=self.sigma_pos, sigma_vel=self.sigma_vel, delta_pos=self.delta_pos,
                         delta_vel=self.delta_vel, dt=self.dt, t_0=self.t_0, style=style, yscale=yscale)
