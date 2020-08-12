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
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.env.simple_corridor import SimpleCorridor
# env_config = {"corridor_length": 5,}
# tune.run(PPOTrainer, config={"env": SimpleCorridor, "env_config": env_config})
tune.run(PPOTrainer, config={"env": SSA_Tasker_Env, "env_config": env_config})
