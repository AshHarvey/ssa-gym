import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from agents import agent_naive_greedy, agent_naive_random, agent_visible_greedy, agent_visible_random, agent_pos_error_greedy
from envs.dynamics import fx_xyz_farnocchia, fx_xyz_markley, fx_xyz_cowell
from envs.results import plot_performance


kwargs = {'steps': 2880, 'rso_count': 50, 'time_step': 30.0, 't_0': datetime(2020, 5, 4, 0, 0, 0),
          'obs_limit': 15, 'observer': (38.828198, -77.305352, 20.0), 'x_sigma': (1000, 1000, 1000, 10, 10, 10),
          'z_sigma': (1, 1, 1000), 'q_sigma': 0.001, 'update_interval': 1,
          'fx': fx_xyz_farnocchia}
env = gym.make('ssa_tasker_simple-v2', **kwargs)

episodes = 1
agents = [agent_pos_error_greedy] # agent_naive_random, agent_visible_random, agent_naive_greedy, agent_visible_greedy
seeds = np.random.randint(0, 2**32-1, episodes)
print('seeds: ', seeds)

agent_rewards = np.zeros((episodes, env.n))
results = []

for agent in agents:
    runs = []
    result = []
    for j in tqdm(range(len(seeds)), desc=agent.__name__):
        env.seed(seed=int(seeds[j]))
        obs = env.reset()
        done = False
        while not done:
            action = agent(obs=obs, env=env)
            observation, reward, done, info = env.step(action)
        result.append(np.copy(env.rewards))
        env.plot_sigma_delta()
    
    results.append(np.copy(result))

rewards = np.copy(results)

plot_performance(rewards=rewards, dt=env.dt, t_0=env.t_0, sigma=3)


