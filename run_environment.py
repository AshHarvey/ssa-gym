import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia, fx_xyz_markley, fx_xyz_cowell
from agents import agent_naive_greedy, agent_naive_random, agent_visible_greedy, agent_visible_random, agent_pos_error_greedy
from envs.transformations import arcsec2rad

P_0 = np.diag((1000**2, 1000**2, 1000**2, 10**2, 10**2, 10**2))

R = np.diag((1*arcsec2rad**2, 1*arcsec2rad**2, 1000**2))

x_sigma = (0, 0, 0, 0, 0, 0) # (1000, 1000, 1000, 10, 10, 10)
z_sigma = (0, 0, 0) # (1, 1, 1000)

kwargs = {'steps': 2880, 'rso_count': 50, 'time_step': 30.0, 't_0': datetime(2020, 5, 4, 0, 0, 0),
          'obs_limit': 15, 'observer': (38.828198, -77.305352, 20.0), 'x_sigma': x_sigma,
          'z_sigma': z_sigma, 'q_sigma': 0.001, 'P_0': P_0, 'R': R, 'update_interval': 1,
          'orbits': np.load('envs/1.5_hour_viz_20000_of_20000_sample_orbits_seed_0.npy'), 'fx': fx_xyz_farnocchia}

env = gym.make('ssa_tasker_simple-v2', **kwargs)
env.seed(0)
obs = env.reset()
agent = [agent_pos_error_greedy, agent_naive_random, agent_visible_random, agent_naive_greedy, agent_visible_greedy][0]

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
