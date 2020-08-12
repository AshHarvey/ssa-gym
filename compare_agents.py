import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import hx_xyz, mean_xyz, robust_cholesky
from agents import agent_naive_greedy, agent_naive_random, agent_visible_greedy, agent_visible_random, agent_pos_error_greedy
from envs.transformations import arcsec2rad
from envs.results import plot_performance

from envs.farnocchia import fx_xyz_farnocchia as fx

sample_orbits = np.load('/home/ash/PycharmProjects/ssa-gym/envs/1.5_hour_viz_20000_of_20000_sample_orbits_seed_0.npy')

config = {'steps': 480, 'rso_count': 20, 'time_step': 30., 't_0': datetime(2020, 5, 4, 0, 0, 0), 'obs_limit': 15,
          'observer': (38.828198, -77.305352, 20.0), 'update_interval': 1, 'obs_type': 'aer',  'z_sigma': (1, 1, 1e3),
          'x_sigma': tuple([1e5]*3+[1e2]*3), 'q_sigma': 0.000025, 'P_0': np.diag(([1e5**2]*3 + [1e2**2]*3)),
          'R': np.diag(([arcsec2rad**2]*2 + [1e3**2])), 'alpha': 0.0001, 'beta': 2., 'kappa': 3-6, 'fx': fx, 'hx': hx,
          'mean_z': mean_z, 'residual_z': residual_z, 'msqrt': robust_cholesky, 'orbits': sample_orbits}

env = gym.make('ssa_tasker_simple-v2', **{'config': config})
episodes = 10
agents = [agent_naive_random, agent_visible_random, agent_naive_greedy, agent_visible_greedy, agent_pos_error_greedy]
agent_names = ['Random (Naive)', 'Random (Visible)', 'P Greedy (Naive)', 'P Greedy (Visible)', 'Position Error Greedy (Visible)']
seeds = np.random.randint(0, 2**32-1, episodes)
print('seeds: ', seeds)

agent_rewards = np.zeros((len(agents), episodes, env.n))
results_of_agents = []

for agent in agents:
    results_of_agent = []
    for j in tqdm(range(len(seeds)), desc=agent.__name__):
        env.seed(seed=int(seeds[j]))
        obs = env.reset()
        done = False
        while not done:
            action = agent(obs=obs, env=env)
            obs, reward, done, info = env.step(action)
        results_of_agent.append(np.copy(env.rewards))

    results_of_agents.append(np.copy(results_of_agent))

rewards = np.copy(results_of_agents)

plot_performance(rewards=rewards, dt=env.dt, t_0=env.t_0, names=agent_names, sigma=1)
