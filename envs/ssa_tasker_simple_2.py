import numpy as np
from datetime import datetime, timedelta
from astropy import time, units as u
from filterpy.kalman import MerweScaledSigmaPoints as SigmasPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise as Q_noise_fn
from poliastro.bodies import Earth
from envs.transformations import arcsec2rad, deg2rad, lla2itrs, gcrs2irts_matrix_b, get_eops
from envs.dynamics import init_state_vec, fx_xyz_markley as fx, hx_aer_kwargs as hx
from envs.results import observations as obs_fn, error, error_failed
import gym

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
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        """Simulation configuration"""
        self.t_0 = datetime(2020, 3, 15, 0, 0, 0) # time at start of simulation
        self.dt = 30.0 # legnth of time steps [s]
        self.n = 480 # max run steps
        self.m = 50 # number of Resident Space Object (RSO) to include in the simulation
        self.obs_limit = np.radians(15) # don't observe objects below this elevation [rad]
        # configuration parameters for RSOs; a: Semi-major axis [m], ecc: Eccentricity [u], inc: Inclination (rad),
        # rann: Right ascension of the ascending node (rad), argp: Argument of perigee (rad), nu: True anomaly (rad)
        self.rso_kernal = {"a": ((Earth.R_mean.to_value(u.m) + 400000), 42164000), "ecc": (0.001, 0.3), "inc": (0, 180),
                           "raan": (0, 360), "argp": (0, 360), "nu": (0, 360)}
        self.observer = (38.828198, -77.305352, 20.0) # lat (deg), lon (deg), height (meters)
        """Filter configuration"""
        self.update_interval = 1 # only make observations every update_interval steps
        # standard deviation of noise added to observations [rad, rad, m]
        self.z_sigma = np.array([1 * arcsec2rad, 1 * arcsec2rad, 1000])
        # standard deviation of noise added to initial state estimates; [m, m, m, m/s, m/s, m/s]
        self.x_sigma = np.array([1000, 1000, 1000, 10, 10, 10])
        self.Q = Q_noise_fn(dim=2, dt=self.dt, var=0.0001**2, block_size=3, order_by_dim=False)
        self.eops = get_eops()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random = gym.utils.seeding.create_seed(seed)
        return [self.np_random]
    
    def reset(self):
        x_dim = 6
        """reset filter"""
        self.P_0 = np.copy(np.diag(self.x_sigma**2)) # Assumed covariance of the estimates at simulation start
        self.R = np.diag(self.z_sigma**2) # Noise added to the filter during observation updates
        self.x_true = np.zeros(shape=(self.n, self.m, x_dim)) # means for all objects at each timestep
        self.x_filter = np.zeros(shape=(self.n, self.m, x_dim)) # means for all objects at each timestep
        self.P_filter = np.zeros(shape=(self.n, self.m, x_dim, x_dim)) # covariances for all objects at each timestep
        self.observations = np.zeros(shape=(self.n, self.m, x_dim*2)) # observations for all objects at each timestep
        self.obs_itrs = lla2itrs(np.array(self.observer)*[deg2rad, deg2rad, 1]) # Set observation location in ITRS
        # build transformation matrices
        self.trans_matrix = [gcrs2irts_matrix_b(self.t_0 + timedelta(seconds=self.dt)*i, self.eops) for i in range(self.n)]
        self.z_noise = np.array([np.random.normal(size=3)*self.z_sigma for i in range(self.n)])
        self.x_noise = np.array([np.random.normal(size=6)*self.x_sigma for i in range(self.n)])
        """initialize RSO"""
        self.filters = []
        for j in range(self.m):
            self.x_true[0][j] = init_state_vec(config=self.rso_kernal)
            self.x_filter[0][j] = np.copy(self.x_true[0][j] + self.x_noise[j])
            self.P_filter[0][j] = np.copy(self.P_0)
            self.filters.append(UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=self.dt, fx=fx, hx=hx,
                                                      points=SigmasPoints(n=6, alpha=0.001, beta=2., kappa=3-6)))
            self.filters[-1].x = np.copy(self.x_filter[0][j])
            self.filters[-1].P = np.copy(self.P_filter[0][j]) # initial uncertainty
            self.filters[-1].R = np.copy(self.R) # 1 standard
            self.filters[-1].Q = np.copy(self.Q)
        """Create variable for tracking environment performance"""
        self.delta_pos = np.empty((self.n,self.m))*np.nan
        self.delta_vel = np.empty((self.n,self.m))*np.nan
        self.sigma_pos = np.empty((self.n,self.m))*np.nan
        self.sigma_vel = np.empty((self.n,self.m))*np.nan
        self.scores = np.empty((self.n,self.m))*np.nan
        self.failed_filters_id = [] # prep list for failed states
        self.failed_filters_msg = ["None"]*self.m # prep list for failure messages
        self.actions = np.empty(self.n)*np.nan # prep variable for keeping track of all previous actions
        self.actions[0] = np.nan
        self.x_failed = np.repeat(1e50, 6)
        self.P_failed = np.tile(1e50, (6, 6))
        """Define Gym spaces"""
        self.action_space = gym.spaces.Discrete(self.m)
        self.observation_space = gym.spaces.Box(low=np.tile(-np.inf, (self.m, 12)), high=np.tile(np.inf, (self.m, 12)),
                                                dtype=np.float64)
        """Observations and Reward"""
        self.observations = np.zeros((self.n, self.m, 12))
        self.observations[0] = obs_fn(self.x_filter[0], self.P_filter[0])
        self.delta_pos[0], self.delta_vel[0], self.sigma_pos[0], self.sigma_vel[0] = error(self.x_true[0],
                                                                                              self.observations[0])
        self.i = 0
        self.rewards = np.zeros(self.n)
        return self.observations[0]
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action)) # check for valid action
        self.i += 1 # increments current step
        self.actions[self.i] = np.copy(action) # record current action
        """propagate next true state"""
        for j in range(self.m):
            self.x_true[self.i][j] = fx(self.x_true[self.i-1][j], self.dt)
        """perform predictions"""
        for j in range(self.m):
            try:
                if not(j in self.failed_filters_id):
                    self.filters[j].predict()
            except:
                self.filters[j].x = np.copy(self.x_failed)
                self.filters[j].P = np.copy(self.P_failed)
                self.failed_filters_msg[j] = ["".join(['Failed on prediction step ', str(self.i-1),
                                                       ', exception error. Errors: ',
                                                       str(np.round(error_failed(state=self.x_true[self.i-1, j],
                                                                                 x=self.x_filter[self.i-1, j],
                                                                                 P=np.diag(self.P_filter[self.i-1, j])), 2))])]
                self.failed_filters_id.append(j)
            self.x_filter[self.i, j] = np.copy(self.filters[j].x) # update filter mean history
            self.P_filter[self.i, j] = np.copy(self.filters[j].P) # update covariance mean history
        """update with observation"""
        if (self.i % self.update_interval==0):
            try:
                if not(action in self.failed_filters_id):
                    hx_kwargs = {"trans_matrix": self.trans_matrix[self.i], "observer_itrs": self.obs_itrs}
                    obs_true = hx(self.states[self.i][action], **hx_kwargs)
                    self.filters[action].update(obs_true + self.z_noise[self.i])
            except:
                self.filters[j].x = np.copy(self.x_failed)
                self.filters[j].P = np.copy(self.P_failed)
                self.failed_filters_msg[j] = ["".join(['Failed on update step ', str(self.i-1),
                                                       ', exception error. Errors: ',
                                                       str(np.round(error_failed(state=self.x_true[self.i-1, j],
                                                                                 x=self.x_filter[self.i-1, j],
                                                                                 P=np.diag(self.P_filter[self.i-1, j])), 2))])]
            self.x_filter[self.i, action] = np.copy(self.filters[action].x) # update filter mean history
            self.P_filter[self.i, action] = np.copy(self.filters[action].P) # update covariance mean history
        """Observations and Reward"""
        self.observations[self.i] = obs_fn(self.x_filter[self.i], self.P_filter[self.i])
        tmp = error(self.x_true[self.i], self.observations[self.i])
        self.delta_pos[self.i], self.delta_pos[self.i],  self.sigma_pos[self.i],  self.sigma_vel[self.i] = tmp
        self.rewards[self.i] = -np.max(self.delta_pos[self.i])
        done = False
        if self.i + 1 >= self.n:
            done = True
        return self.observations[self.i], self.rewards[self.i], done, {}  # observations, reward, and done


"""
    def errors(self):
        return errors(self.states, self.filters_x, self.filters_P)
    def plot(self, title):
        plot_results(self.states, self.filters_x, self.filters_P, self.t, title)
"""
