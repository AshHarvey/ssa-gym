import ray
from ray import tune
from envs import env_config
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env

ray.init()

env_config['rso_count'] = 10

analysis = tune.run(
    "DQN",
    stop={"timesteps_total": 400000},
    config={
        "env": SSA_Tasker_Env,
        "env_config": env_config,
        "num_gpus": 0,
        "num_workers": 1,
        "train_batch_size": tune.grid_search([32, 256]),
        "batch_mode": tune.grid_search(["truncate_episodes", "complete_episodes"]),
        #"double_q": tune.grid_search([True, False]),
        #"dueling": tune.grid_search([True, False]),
        #"noisy": tune.grid_search([True, False]),
        "prioritized_replay": tune.grid_search([True, False]),
        "lr": tune.grid_search([0.0005, 0.00001]),
        #"n_step": tune.grid_search([1, 4]), # not used in paper, found benefitial after the fact
        "target_network_update_freq": tune.grid_search([500, 5000]),
        'exploration_config': {'type': 'EpsilonGreedy',
                               'initial_epsilon': 1.0,
                               'final_epsilon': 0.02,
                               'epsilon_timesteps': 10000},

    },
)

analysis2 = tune.run(
    "DQN",
    stop={"timesteps_total": 400000},
    config={
        "env": SSA_Tasker_Env,
        "env_config": env_config,
        "num_gpus": 0,
        "num_workers": 1,
        "train_batch_size": tune.grid_search([32, 256]),
        "batch_mode": tune.grid_search(["truncate_episodes", "complete_episodes"]),
        #"double_q": tune.grid_search([True, False]),
        #"dueling": tune.grid_search([True, False]),
        #"noisy": tune.grid_search([True, False]),
        "prioritized_replay": tune.grid_search([True, False]),
        "lr": tune.grid_search([0.0005, 0.00001]),
        "n_step": tune.grid_search([1, 4]),
        "target_network_update_freq": tune.grid_search([500, 5000]),
        'exploration_config': {'type': 'EpsilonGreedy',
                               'initial_epsilon': 1.0,
                               'final_epsilon': 0.02,
                               'epsilon_timesteps': 250000},

    },
)
