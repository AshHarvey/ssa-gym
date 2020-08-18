"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z
from envs.dynamics import robust_cholesky
from envs.transformations import arcsec2rad
from envs.farnocchia import fx_xyz_farnocchia as fx

sample_orbits = np.load('/home/ash/PycharmProjects/ssa-gym/envs/1.5_hour_viz_20000_of_20000_sample_orbits_seed_0.npy')

env_config = {'steps': 480, 'rso_count': 10, 'time_step': 30., 't_0': datetime(2020, 5, 4, 0, 0, 0), 'obs_limit': 15,
          'observer': (38.828198, -77.305352, 20.0), 'update_interval': 1, 'obs_type': 'aer',  'z_sigma': (1, 1, 1e3),
          'x_sigma': tuple([1e5]*3+[1e2]*3), 'q_sigma': 0.000025, 'P_0': np.diag(([1e5**2]*3 + [1e2**2]*3)),
          'R': np.diag(([arcsec2rad**2]*2 + [1e3**2])), 'alpha': 0.0001, 'beta': 2., 'kappa': 3-6, 'fx': fx, 'hx': hx,
          'mean_z': mean_z, 'residual_z': residual_z, 'msqrt': robust_cholesky, 'orbits': sample_orbits, }

from ray import tune
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.agents.dqn.apex import apex_execution_plan
from ray.rllib.utils import merge_dicts

import ray
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG, DQNTrainer
from ray.tune.logger import pretty_print
from copy import copy

if ray.is_initialized() is False:
    ray.init()
config = DEFAULT_CONFIG
config["optimizer"] = merge_dicts(DQN_CONFIG["optimizer"], {"max_weight_sync_delay": 400,
                                                            "num_replay_buffer_shards": 4,
                                                            "debug": False})
config["num_workers"] = 18
config["num_gpus"] = 2
config["n_step"] = 3
config["buffer_size"] = 2000000
config["n_step"] = 3
config["learning_starts"] = 50000
config["train_batch_size"] = 512
config["timesteps_per_iteration"] = 25000
config["target_network_update_freq"] = 500000
config["exploration_config"] = {"type": "PerWorkerEpsilonGreedy"}
config["worker_side_prioritization"] = True
# config["min_iter_time_s"] = 30
# config["training_intensity"] = None
# config["log_level"] = 'DEBUG'
config["env_config"] = env_config
trainer = DQNTrainer(config=config,
                     env=SSA_Tasker_Env)
# Can optionally call trainer.restore(path) to load a checkpoint.

checkpoints = []
result = {'timesteps_total': 0}
i = 0
while result['timesteps_total'] < 1e7:
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if result['training_iteration'] % 4 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        checkpoints.append(copy(checkpoint))

# path = '/home/ash/ray_results/DQN_SSA_Tasker_Env_2020-08-13_13-26-22s_wgpxq5/checkpoint_1/'
# trainer.restore(path)

# trainer.import_model("my_weights.h5")

'''
ray.shutdown()
'''
