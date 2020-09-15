import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import hx_xyz, mean_xyz, robust_cholesky
from agents import agent_naive_greedy, agent_naive_random, agent_visible_greedy, agent_visible_random, agent_pos_error_greedy,agent_shannon,agent_visible_greedy_spoiled
from envs.transformations import arcsec2rad
from envs.results import plot_performance

from envs.farnocchia import fx_xyz_farnocchia as fx
import os

from envs import env_config

env_config['obs_limit'] = 15
env_config['rso_count'] = 40
env_config['steps'] = 2880
env_config['reward_type'] = 'trinary'
env_config['obs_returned'] = 'flatten'

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})


episodes = 10
agents = [agent_visible_greedy_spoiled,agent_shannon]
agent_names = [  'spoiled', 'shannon']
seeds = np.random.randint(0, 2**32-1, episodes, dtype='u8')
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
