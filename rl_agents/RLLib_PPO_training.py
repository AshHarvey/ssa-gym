import gym
import ray.utils
from ray.tune.logger import pretty_print
from envs import env_config
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env
import datetime
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
import numpy as np
import pickle

ray.init()

env_config['rso_count'] = 20

config = PPO_CONFIG.copy()
config['num_gpus'] = 1
config['num_workers'] = 4
# !---- found that the network design from Jones's work had little effect in training
# config['model']['fcnet_hiddens'] = [180, 95, 50] # 10 RSOs
# config['model']['fcnet_hiddens'] = [360, 180, 100] # 20 RSOs
# config['model']['fcnet_hiddens'] = [720, 380, 200] # 40 RSOs
config['gamma'] = 0.99 # gamma  (float) Discount factor
config['rollout_fragment_length'] = 32
if env_config['rso_count'] == 40:
    #config['model']['fcnet_hiddens'] = [512, 512] # 40 RSOs
    config['rollout_fragment_length'] = 128 # n_steps (int) The number of steps to run for each environment per update
# (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
config['entropy_coeff'] = 0.01 # ent_coef (float) Entropy coefficient for the loss calculation
config['lr'] = 0.00025 # learning_rate (float or callable) The learning rate, it can be a function
config['vf_loss_coeff'] = 0.5 # vf_coef (float) Value function coefficient for the loss calculation
config['grad_clip'] = 0.5 # max_grad_norm (float) The maximum value for the gradient clipping
config['lambda'] = 0.95 # lam (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
config['num_sgd_iter'] = 4 # nminibatches  (int) Number of training minibatches per update. For recurrent policies,
# the number of environments run in parallel should be a multiple of nminibatches.
config['clip_param'] = 0.2 # cliprange (float or callable) Clipping parameter, it can be a function
config['vf_clip_param'] = 0.1 # cliprange_vf = None? --  (float or callable) Clipping parameter for the value function,
# it can be a function. This is a parameter specific to the OpenAI implementation. If None is passed (default), then
# cliprange (that is used for the policy) will be used. IMPORTANT: this clipping depends on the reward scaling. To
# deactivate value function clipping (and recover the original PPO implementation), you have to pass a negative value
# (e.g. -1).
config['env_config'] = env_config
config['train_batch_size'] = 4000


# !------------ Train example agent

env = gym.make('ssa_tasker_simple-v2')
trainer = PPOTrainer(config=config, env=SSA_Tasker_Env)

result = {'timesteps_total': 0,
          'episodes_total': 0}
num_step_train = 1000000
episodes_train = 10000
episode_len_mean = []
episode_reward_mean = []
episode_reward_max = []
episode_reward_min = []
start = datetime.datetime.now()
num_steps_trained = []
clock_time = []
training_iteration = []
reward_moving_average = []
while result['timesteps_total'] < num_step_train:
#while result['episodes_total'] < episodes_train:
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    episode_reward_mean.append(result['episode_reward_mean'])
    episode_reward_max.append(result['episode_reward_max'])
    episode_reward_min.append(result['episode_reward_min'])
    episode_len_mean.append(result['episode_len_mean'])
    clock_time.append(str(datetime.datetime.now() - start))
    num_steps_trained.append(result['info']['num_steps_trained'])
    training_iteration.append(result['training_iteration'])
    len_moving_average = np.convolve(episode_len_mean, np.ones((20,))/20, mode='valid')
    reward_moving_average = np.convolve(episode_reward_mean, np.ones((20,))/20, mode='valid')
    print('Current     ::: Len:: Mean: ' + str(episode_len_mean[-1]) + '; Reward:: Mean: ' + str(episode_reward_mean[-1]) + ', Max: ' + str(episode_reward_max[-1]) + ', Min: ' + str(episode_reward_min[-1]))
    print('mAverage20  ::: Len:: Mean: ' + str(np.round(len_moving_average[-1], 1)) + '; Reward:: Mean: ' + str(np.round(reward_moving_average[-1], 1)))

    if result['training_iteration'] % 10 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        output = {'episode_len_mean': episode_len_mean,
                  'episode_reward_mean': episode_reward_mean,
                  'episode_reward_max': episode_reward_max,
                  'episode_reward_min': episode_reward_min,
                  'num_steps_trained': num_steps_trained,
                  'clock_time': clock_time,
                  'training_iteration': training_iteration,
                  'len_moving_average': len_moving_average,
                  'reward_moving_average': reward_moving_average}
        output_path = trainer._logdir + '/_running_results.pkl'
        with open(output_path, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Running Results Saved To: ' + output_path)

checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)
output = {'episode_len_mean': episode_len_mean,
          'episode_reward_mean': episode_reward_mean,
          'episode_reward_max': episode_reward_max,
          'episode_reward_min': episode_reward_min,
          'num_steps_trained': num_steps_trained,
          'clock_time': clock_time,
          'training_iteration': training_iteration,
          'len_moving_average': len_moving_average,
          'reward_moving_average': reward_moving_average}
output_path = trainer._logdir + '/_running_results.pkl'
with open(output_path, 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Final Results Saved To: ' + output_path)

"""
# PPO_SSA_Tasker_Env_2020-08-30_13-39-09negprf9v = 10 SSA Full Jones; 819.2K timesteps
# checkpoint saved at /home/ash/ray_results/PPO_SSA_Tasker_Env_2020-08-30_13-39-09negprf9v/checkpoint_200/checkpoint-200

# PPO_SSA_Tasker_Env_2020-08-30_16-30-06d2ygpmw5 = 20 SSA Full Jones; 409.6K timesteps
# checkpoint saved at /home/ash/ray_results/PPO_SSA_Tasker_Env_2020-08-30_16-30-06d2ygpmw5/checkpoint_100/checkpoint-100

# PPO_SSA_Tasker_Env_2020-08-30_16-02-24oje3enkl = 40 SSA Full Jones; 4.096M timesteps
# checkpoint saved at /home/ash/ray_results/PPO_SSA_Tasker_Env_2020-08-30_16-02-24oje3enkl/checkpoint_1000/checkpoint-1000

'/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-03_10-43-46ba4j85h9/checkpoint_1000/checkpoint-1000' - 20 RSOs, Default Net
'/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-02_21-45-134nxei6pc/checkpoint_200/checkpoint-200' - 10 RSOs, Default Net
'/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-02_21-41-01djunqm5w/checkpoint_200/checkpoint-200 - 40 RSOs, Default Net
"""
