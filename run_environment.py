import gym
import numpy as np
import envs.ssa_tasker_simple_2
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import RE

kwargs = {'steps': 2880, 'rso_count': 50, 'time_step': 30.0, 't_0': datetime(2020, 5, 4, 0, 0, 0),
          'obs_limit': 15, 'observer': (38.828198, -77.305352, 20.0), 'x_sigma': (1000, 1000, 1000, 10, 10, 10),
          'z_sigma': (1, 1, 1000), 'q_sigma': 0.001, 'update_interval': 1, 'orbits': np.load('envs/sample_orbits.npy')}
env = gym.make('ssa_tasker_simple-v2', **kwargs)

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
if not env.failed_filters_id:
    print("No failed Objects")
else:
    print(len(env.failed_filters_id), " Objects Failed: ", env.failed_filters_id)

env.plot_sigma_delta()
