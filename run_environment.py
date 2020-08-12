import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import hx_xyz, mean_xyz, robust_cholesky
from agents import agent_naive_greedy, agent_naive_random, agent_visible_greedy, agent_visible_random, agent_pos_error_greedy
from envs.transformations import arcsec2rad

from envs.farnocchia import fx_xyz_farnocchia as fx

sample_orbits = np.load('/home/ash/PycharmProjects/ssa-gym/envs/1.5_hour_viz_20000_of_20000_sample_orbits_seed_0.npy')

config = {'steps': 480, 'rso_count': 10, 'time_step': 30., 't_0': datetime(2020, 5, 4, 0, 0, 0), 'obs_limit': 15,
          'observer': (38.828198, -77.305352, 20.0), 'update_interval': 1, 'obs_type': 'aer',  'z_sigma': (1, 1, 1e3),
          'x_sigma': tuple([1e5]*3+[1e2]*3), 'q_sigma': 0.000025, 'P_0': np.diag(([1e5**2]*3 + [1e2**2]*3)),
          'R': np.diag(([arcsec2rad**2]*2 + [1e3**2])), 'alpha': 0.0001, 'beta': 2., 'kappa': 3-6, 'fx': fx, 'hx': hx,
          'mean_z': mean_z, 'residual_z': residual_z, 'msqrt': robust_cholesky, 'orbits': sample_orbits}

env = gym.make('ssa_tasker_simple-v2', **{'config': config})
env.seed(0)
obs = env.reset()
agent = agent_visible_greedy

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = agent(obs, env)
        obs, reward, done, _ = env.step(action)

env.failed_filters()
failing_objects = np.argwhere(env.delta_pos[env.i, :] > 10e5).flatten()

if failing_objects.shape[0] == 0:
    env.plot_sigma_delta()
else:
    env.plot_sigma_delta(objects=failing_objects)


env.plot_rewards()
