import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia, fx_xyz_markley, fx_xyz_cowell


kwargs = {'steps': 2880, 'rso_count': 50, 'time_step': 30.0, 't_0': datetime(2020, 5, 4, 0, 0, 0),
          'obs_limit': 15, 'observer': (38.828198, -77.305352, 20.0), 'x_sigma': (1000, 1000, 1000, 10, 10, 10),
          'z_sigma': (1, 1, 1000), 'q_sigma': 0.001, 'update_interval': 1, 'orbits': np.load('envs/sample_orbits.npy'),
          'fx': fx_xyz_farnocchia}

env = gym.make('ssa_tasker_simple-v2', **kwargs)
env.seed(0)
_ = env.reset()

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

env.failed_filters

env.plot_sigma_delta()
env.plot_rewards()
