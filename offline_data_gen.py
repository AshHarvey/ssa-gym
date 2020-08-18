import gym
import numpy as np
import os

import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import robust_cholesky
from envs.transformations import arcsec2rad
from agents import agent_visible_random

sample_orbits = np.load('/home/ash/PycharmProjects/ssa-gym/envs/1.5_hour_viz_20000_of_20000_sample_orbits_seed_0.npy')

env_config = {'steps': 480, 'rso_count': 10, 'time_step': 30., 't_0': datetime(2020, 5, 4, 0, 0, 0), 'obs_limit': 15,
          'observer': (38.828198, -77.305352, 20.0), 'update_interval': 1, 'obs_type': 'aer',  'z_sigma': (1, 1, 1e3),
          'x_sigma': tuple([1e5]*3+[1e2]*3), 'q_sigma': 0.000025, 'P_0': np.diag(([1e5**2]*3 + [1e2**2]*3)),
          'R': np.diag(([arcsec2rad**2]*2 + [1e3**2])), 'alpha': 0.0001, 'beta': 2., 'kappa': 3-6, 'fx': fx, 'hx': hx,
          'mean_z': mean_z, 'residual_z': residual_z, 'msqrt': robust_cholesky, 'orbits': sample_orbits, }

if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray.utils.get_user_temp_dir(), "agent_visible_random_10RSOs-out"))

    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for eps_id in tqdm(range(2084)): #
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            action = agent_visible_random(obs=obs, env=env)
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())
