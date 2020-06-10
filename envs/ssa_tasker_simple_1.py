import numpy as np

from scipy.spatial import distance # used to calculate distance errors in 3D

from numba import jit

from datetime import datetime, timedelta

#from astropy import coordinates as coord
from astropy import units as u
from astropy import time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate
from poliastro.frames import Planes
from poliastro.core.elements import coe2rv
from poliastro.core.propagation import markley

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

#from astropy.time import Time
import matplotlib.pyplot as plt # used for all plots
import seaborn as sns # used for color pallettes

#%% imports for gym

import gym
from gym import spaces
from gym.utils import seeding

#%% Sim Functions

@jit
def init_state_vec(seed=None):
    if not(seed==None):
        np.random.seed(seed)
    k = 398600.4418 # (km^3 / s^2) - Standard gravitational parameter
    a = np.random.uniform((6378.1366 + 200),42164) # (km) – Semi-major axis.
    ecc = np.random.uniform(0.001,0.3) # (Unitless) – Eccentricity.
    inc = np.radians(np.random.uniform(0,180)) # (rad) – Inclination
    raan = np.radians(np.random.uniform(0,360)) # (rad) – Right ascension of the ascending node.
    argp = np.radians(np.random.uniform(0,360)) # (rad) – Argument of the pericenter.
    nu = np.radians(np.random.uniform(0,360)) # (rad) – True anomaly.
    p = a*(1-ecc**2) # (km) - Semi-latus rectum or parameter
    return np.concatenate(coe2rv(k,p,ecc,inc,raan,argp,nu))

@jit
def fx(x,dt):
    try:
        x2 = np.concatenate(markley(398600.4418,x[:3],x[3:],dt)) #  # (km^3 / s^2), (km), (km/s), (s)
        return x2
    except:
        return x

@jit
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    return x[:3]

@jit
def init_state_obj(seed=None):
    if not(seed==None):
        np.random.seed(seed)
    t_init = time.Time(datetime(2020, 3, 15, 0, 0, 0), scale='utc')
    a = (Earth.R.value + 200000)*u.m # (Quantity) – Semi-major axis.
    ecc = np.random.uniform(0.001,0.3)*u.dimensionless_unscaled # (Quantity) – Eccentricity.
    inc = np.random.uniform(0,180)*u.degree # (Quantity) – Inclination
    raan = np.random.uniform(0,360)*u.degree # (Quantity) – Right ascension of the ascending node.
    argp = np.random.uniform(0,360)*u.degree # (Quantity) – Argument of the pericenter.
    nu = np.random.uniform(0,360)*u.degree # (Quantity) – True anomaly.
    epoch = t_init # (Time, optional) – Epoch, default to J2000.
    plane = Planes.EARTH_EQUATOR # Fundamental plane of the frame.
    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch, plane=plane)

def initial_filter(x, dt, P_0, R, Q):
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(n=6, alpha=0.001, beta=2., kappa=3-6)
    # https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html#filterpy.kalman.MerweScaledSigmaPoints
    # de
    kf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=dt, fx=fx, hx=hx, points=points)
    # https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
    
    kf.x = x
    kf.P = P_0 # initial uncertainty
    kf.R = R # 1 standard
    kf.Q = Q
    return kf

'''
def score(P, dt=30.0):
    score = np.power(np.multiply(np.linalg.det(P),dt**6),1/12)
    return score

def score(P, dt=30.0):
    score = np.linalg.det(P)
    return score


def score(P):
    score = np.linalg.det(P[:3,:3])
    return score


def score(states,filters_x,filters_P):
    score_t = []
    for i in range(len(states)):
        score_t.append(distance.euclidean(states[i][:3],filters_x[i][:3]))
    reward = -max(score_t)
    return reward
'''

def score(states,filters_x,filters_P):
    score_t = []
    for P in filters_P:
        score_t.append(np.trace(P))
    reward = -max(score_t)
    return reward

'''
@jit
def score(P):
    score = P[0,0] * ( P[1,1] * P[2,2] - P[1,2] * P[2,1] ) - P[0,1] * ( P[1,0] * P[2,2] - P[1,2] * P[2,0] ) + P[0,3] * ( P[1,0] * P[2,1] - P[1,1] * P[2,0] )
    score = np.power(score,1/6)

@jit
def score(P, dt):
    diag = np.diag(P) # The main diagonal is always positive.
    position_error_sq = np.sum(diag[:3])
    velocity_error_sq = np.sum(diag[3:])
    position_error = np.sqrt(position_error_sq)
    velocity_error = np.sqrt(velocity_error_sq)
    score = position_error + velocity_error*30
    return score

@jit 
def score(P):
    return np.trace(P)
'''

def errors(states, filters_x, filters_P):
    filter_error_position = [[] for i in range(len(states[0]))]
    filter_error_velocity = [[] for i in range(len(states[0]))]
    errors_position = [[] for i in range(len(states[0]))]
    errors_velocity = [[] for i in range(len(states[0]))]
    pv = [[] for i in range(len(states[0]))]
    for i in range(len(states)): # Timesteps dimension
        for j in range(len(states[0])): # RSO dimension
            pv[j].append(states[i][j])
            errors_position[j].append(distance.euclidean(filters_x[i][j][:3],
                                     pv[j][i][:3]))
            errors_velocity[j].append(distance.euclidean(filters_x[i][j][3:],
                                     pv[j][i][3:]))
            filter_error_position[j].append(
                np.linalg.det(np.array(filters_P[i][j])[:3,:3]))
            filter_error_velocity[j].append(
                np.linalg.det(np.array(filters_P[i][j])[3:,3:]))
    return errors_position, errors_velocity, filter_error_position, filter_error_velocity

def fx_obj(state, dt):
    return propagate(state, time.TimeDelta(dt*u.s), method = markley)

def to_states(state,t):
    return Orbit.from_vectors(Earth,state[:3]*u.km,state[3:]*u.km/u.s,t,Planes.EARTH_EQUATOR)

@jit
def obs(filters):
    obs = []
    for ukf in filters:
        obs.append(np.append(ukf.x,np.diag(ukf.P)))

    obs = np.array(obs)
    
    return np.nan_to_num(obs,nan=99999, posinf=99999)

def plot_results(states, filters_x, filters_P, t, title=None):
    # obtain number for initial error plots
    errors_position, errors_velocity, filter_error_position, filter_error_velocity = errors(states, filters_x, filters_P)
    #create a figure
    #plt.subplots_adjust(top=0.9)
    fig = plt.figure(figsize=(24,12),dpi=200)
    if not(title==None):
        fig.suptitle(title)   
    ax = fig.add_subplot(121)
    #plot to first axes
    palette = sns.color_palette("hls",len(filter_error_position))
    for j in range(len(filter_error_position)):
        ax.plot(t,filter_error_position[j],color=palette[j],label="Magnitude of Filter's Position $\sigma$")
        ax.plot(t,filter_error_velocity[j],color=palette[j],label="Magnitude of Filter's Velocity $\sigma$")

    ax.set_ylabel('Error in meters (log scale)')
    ax.set_yscale('log')
    #create twin axes
    '''
    ax2=ax.twinx()
    #plot to twin axes
    for j in range(len(obs_steps)):
        ax2.scatter(obs_steps[j], np.degrees(Elevation[j]), s=80, alpha=0.7, color=palette[j], zorder=3, marker='.', label='Observations'); 
    ax2.set_ylabel('Elevation in Degrees')
    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = [copy.copy(h1[1])]
    handles.append(copy.copy(h1[1]))
    for handle in handles:
        color = (0,0,0)
        handle._color = color
    handles.append(copy.copy(h2[0]))
    handles[-1]._edgecolors = np.asarray([[0,0,0,0.3]])
    handles[-1]._facecolors = np.asarray([[0,0,0,0.3]])
    labels = l1[:2]
    labels += [l2[0]]
    ax.legend(handles=handles, labels=labels, loc='upper left')
    '''
    
    ax = fig.add_subplot(122)
    #plot to first axes
    for j in range(len(errors_position)):
        ax.plot(t,errors_position[j],color=palette[j],label="Magnitude of Position Error")
        ax.plot(t,errors_velocity[j],color=palette[j],label="Magnitude of Velocity Error")
    ax.set_ylabel('Error in meters (log scale)')
    ax.set_yscale('log')
    #create twin axes
    '''
    ax2=ax.twinx()
    #plot to twin axes
    for j in range(len(obs_steps)):
        ax2.scatter(obs_steps[j], np.degrees(Elevation[j]), s=80, alpha=0.3, color=palette[j], zorder=3, marker='.', label='Observations'); 
    ax2.set_ylabel('Elevation in Degrees')
    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = [copy.copy(h1[1])]
    handles.append(copy.copy(h1[1]))
    for handle in handles:
        color = (0,0,0)
        handle._color = color
    handles.append(copy.copy(h2[0]))
    handles[-1]._edgecolors = np.asarray([[0,0,0,0.3]])
    handles[-1]._facecolors = np.asarray([[0,0,0,0.3]])
    labels = l1[:2]
    labels += [l2[0]]
    ax.legend(handles=handles, labels=labels, loc='upper left')
    '''
    plt.show()
    
class SSA_Tasker_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        #%% Sim Constants
        self.initial_time = datetime(2020, 3, 15, 0, 0, 0)
        self.dt = 30.0 # in seconds
        self.run_length = 480 # max run steps
        self.update_interval = 1 # only make observations every update_interval steps
        self.obs_noise = np.repeat(100,3)
        # obs_limit = np.radians(15)
        self.RSO_Count = 50
        self.P_0 = np.diag([900,900,900,0.0005,0.0005,0.0005])
        self.ini_noise = np.sqrt(np.diag(self.P_0))
        self.R = np.eye(3)*self.obs_noise # 1 standard
        self.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.000001**2, block_size=3, order_by_dim=False)
        
        self.seed()
        self.reset()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.x_failed = np.array([0,0,0,0,0,0])
        self.P_failed = np.reshape(np.array(np.repeat(9999,36)),(6,6))


    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def reset(self):
        # initialize RSO
        self.steps = list(range(self.run_length+1))
        # get times for all propagation steps
        # t [timestep index]
        self.t = [self.initial_time + timedelta(seconds=self.dt)]
        self.t_obj = [time.Time(self.t[-1], scale='utc')]
        for i in self.steps:
            self.t.append(self.t[-1] + +timedelta(seconds=self.dt))
            self.t_obj.append(time.Time(self.t[-1], scale='utc'))
        
        # get initial states in vector form [RSO ID index][timestep]
        self.states = [[]]
        for i in range(self.RSO_Count):
            self.states[-1].append(init_state_vec(seed=np.random.randint(0,99999999)))
        # get initial states in object form [RSO ID index][timestep]
        self.states_obj = [[to_states(state,self.t_obj[0]) for state in self.states[-1]]]
        # get initial filters [RSO ID index]
        self.filters = [initial_filter(state+np.random.normal(scale=self.ini_noise),
                                       self.dt, self.P_0, self.R, self.Q) for state in self.states[0]]
        # get initial filter means [RSO ID index][timestep]
        self.filters_x = [[ukf.x for ukf in self.filters]]
        # get initial filter means [RSO ID index][timestep]
        self.filters_P = [[ukf.P for ukf in self.filters]]
        # get initial reward [timestep]
        self.scores = [score(self.states[-1], self.filters_x[-1], self.filters_P[-1])]
        # record time of the initial state [timestep]
        self.t = [self.t[0]]
        self.failed_filters = [] # prep list for failed states
        self.last_action = None # prep variable for keeping track of last action
        
        # Gym spaces
        
        self.action_space = spaces.Discrete(self.RSO_Count)
        self.low = np.tile(-np.inf,(self.RSO_Count,12))
        self.high = np.tile(np.inf,(self.RSO_Count,12))
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        
        # observations
        
        self.observation = obs(self.filters)
        
        self.i = 0
        
        return self.observation
    
    def step(self, action):
        global t
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        self.t.append(self.t[-1] + timedelta(seconds=self.dt))
        # propagate next true state
        self.states.append([fx(state, self.dt) for state in self.states[-1]])
        # propagate next estimated states
        for j in range(self.RSO_Count):
            try:
                if not(j in self.failed_filters):
                    self.filters[j].predict()
            except:
                if j not in self.failed_filters:
                    self.filters[j].x = self.x_failed
                    self.filters[j].P = self.P_failed
                    print('RSO ',j,'failed on update step ',self.i)
                    print('Last Action: ',self.last_action,'; Current Action: ',action)
                    self.failed_filters.append(j)
        if (self.i % self.update_interval==0):
            noise = np.random.normal(loc=0,scale=np.sqrt(self.obs_noise),size=3)
            if not(action in self.failed_filters):
                self.filters[action].update(self.states[-1][action][:3]+noise)
                if np.any(np.isnan(self.filters[action].x)):
                    self.filters[action].x = self.x_failed
                    self.filters[action].P = self.P_failed
        
        # update filtered states history
        self.filters_x.append([list(ukf.x) for ukf in self.filters])
        self.filters_P.append([list(ukf.P) for ukf in self.filters])
        # update reward history
        self.last_action = action
        # update reward history
        #self.scores.append(-max([score(ukf.P) for ukf in self.filters])/score(self.P_0))
        self.scores.append(score(self.states[-1], self.filters_x[-1], self.filters_P[-1]))

        self.observation = obs(self.filters)
        self.i = self.i + 1
        done = False
        if self.i >= self.run_length:
            done = True
        reward = self.scores[-1]
        return self.observation, reward, done, {}  # observations, reward, and done
    
    def errors(self):
        return errors(self.states, self.filters_x, self.filters_P)
    def plot(self, title):
        plot_results(self.states, self.filters_x, self.filters_P, self.t, title)
