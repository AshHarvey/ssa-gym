import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.marwil import DEFAULT_CONFIG, MARWILTrainer
import numpy as np
import datetime
from envs import env_config
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env
import pickle

ray.init()

env_config['rso_count'] = 40
agent = 'agent_visible_greedy_spoiled'

logdir = '/home/ash/ray_results/ssa_experiences/' + agent + '/' + str(env_config['rso_count']) + 'RSOs_jones_flatten_10000episodes/'

config = DEFAULT_CONFIG.copy()
config['evaluation_num_workers'] = 10
config['env_config'] = env_config
config['evaluation_interval'] = 10
config['train_batch_size'] = 10000
'''if env_config['rso_count'] == 40:
    config['model']['fcnet_hiddens'] = [512, 512]'''
config['evaluation_config'] = {'input': 'sampler'}
config['beta'] = 1 # 0
config['input'] = logdir


trainer_MARWIL = MARWILTrainer(config=config, env=SSA_Tasker_Env)
best_athlete = 480
episode_len_mean = []
episode_reward_mean = []
episode_reward_max = []
episode_reward_min = []
start = datetime.datetime.now()
num_steps_trained = []
clock_time = []
training_iteration = []
for i in range(5000):
    # Perform one iteration of training the policy with DQN from offline data
    result_MARWIL = trainer_MARWIL.train()
    print(pretty_print(result_MARWIL))

    if result_MARWIL['training_iteration'] % config['evaluation_interval'] == 0:
        #
        episode_reward_mean.append(result_MARWIL['evaluation']['episode_reward_mean'])
        episode_reward_max.append(result_MARWIL['evaluation']['episode_reward_max'])
        episode_reward_min.append(result_MARWIL['evaluation']['episode_reward_min'])
        episode_len_mean.append(result_MARWIL['evaluation']['episode_len_mean'])
        clock_time.append(str(datetime.datetime.now() - start))
        num_steps_trained.append(result_MARWIL['info']['num_steps_trained'])
        training_iteration.append(result_MARWIL['training_iteration'])
        len_moving_average = np.convolve(episode_len_mean, np.ones((20,))/20, mode='valid')
        reward_moving_average = np.convolve(episode_reward_mean, np.ones((20,))/20, mode='valid')
        print('Current     ::: Len:: Mean: ' + str(episode_len_mean[-1]) + '; Reward:: Mean: ' + str(episode_reward_mean[-1]) + ', Max: ' + str(episode_reward_max[-1]) + ', Min: ' + str(episode_reward_min[-1]))
        print('mAverage20  ::: Len:: Mean: ' + str(np.round(len_moving_average[-1],1)) + '; Reward:: Mean: ' + str(np.round(reward_moving_average[-1], 1)))
        #
        if result_MARWIL['training_iteration'] % config['evaluation_interval']*10 == 0:
            checkpoint = trainer_MARWIL.save()
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
        output_path = trainer_MARWIL._logdir + '/_running_results.pkl'
        with open(output_path, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Running Results Saved To: ' + output_path)

        if best_athlete > episode_len_mean[-1]:
            best_athlete = episode_len_mean[-1]
            best_athlete_checkpoint = trainer_MARWIL.save()
            print("checkpoint saved at", best_athlete_checkpoint)

checkpoint = trainer_MARWIL.save()
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
output_path = trainer_MARWIL._logdir + '/_running_results.pkl'
with open(output_path, 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Final Results Saved To: ' + output_path)

"""
'/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-08-31_15-46-49wpu8yesd/checkpoint_1000/checkpoint-1000' = 10 RSOs
'/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-08-31_15-49-54jmez0fy5/checkpoint_1000/checkpoint-1000' = 20 RSOs

'/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-08-31_18-30-47p8ac740j/checkpoint_5385/checkpoint-5385' = 40 RSO best athlete
'/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-08-31_18-30-47p8ac740j/checkpoint_6951/checkpoint-6951' = 40 RSO final

"""

"""
rerun:
10
checkpoint saved at /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-20-065a0phk5m/checkpoint_5000/checkpoint-5000
Final Results Saved To: /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-20-065a0phk5m/_running_results.pkl

20
checkpoint saved at /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-24-14dv96mkld/checkpoint_5000/checkpoint-5000
Final Results Saved To: /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-24-14dv96mkld/_running_results.pkl

40
checkpoint saved at /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-46-52wmigl7hj/checkpoint_10000/checkpoint-10000
Final Results Saved To: /home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-46-52wmigl7hj/_running_results.pkl

best_athlete at '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-46-52wmigl7hj/checkpoint_5890/checkpoint-5890'
"""
