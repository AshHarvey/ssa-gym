import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.marwil import DEFAULT_CONFIG as MARWIL_config, MARWILTrainer
from ray.rllib.agents.pg import DEFAULT_CONFIG as PG_config, PGTrainer
import numpy as np
import datetime
from envs import env_config
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env
import pickle

ray.init()

env_config['rso_count'] = 40
explore = False

pg_config = PG_config.copy()
pg_config['batch_mode'] = 'complete_episodes'
pg_config['train_batch_size'] = 4000
pg_config['num_workers'] = 4
pg_config['rollout_fragment_length'] = 32
pg_config['lr'] = 0.00001
pg_config['evaluation_interval'] = None
pg_config['postprocess_inputs'] = True
pg_config['env_config'] = env_config
if env_config['rso_count'] == 40:
    #pg_config['model']['fcnet_hiddens'] = [512, 512]
    pg_config['rollout_fragment_length'] = 124

pg_config['explore'] = explore

marwil_config = MARWIL_config.copy()
logdir = '/home/ash/ray_results/ssa_experiences/agent_visible_greedy_spoiled/' + str(env_config['rso_count']) + 'RSOs_jones_flatten_10000episodes/'
marwil_config['evaluation_num_workers'] = 1
marwil_config['env_config'] = env_config
marwil_config['evaluation_interval'] = 1
marwil_config['evaluation_config'] = {'input': 'sampler'}
marwil_config['beta'] = 1 # 0
marwil_config['input'] = logdir
'''if env_config['rso_count'] == 40:
    marwil_config['model']['fcnet_hiddens'] = [512, 512]'''

# !--- experiment
pg_config['lr'] = 1e-7
pg_config['train_batch_size'] = 4000
pg_config['num_workers'] = 8

MARWIL_trainer = MARWILTrainer(config=marwil_config, env=SSA_Tasker_Env)
PG_trainer = PGTrainer(config=pg_config, env=SSA_Tasker_Env)

if env_config['rso_count'] == 10:
    MARWIL_trainer.restore('/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-20-065a0phk5m/checkpoint_5000/checkpoint-5000') # 10 SSA Complete
elif env_config['rso_count'] == 20:
    MARWIL_trainer.restore('/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-24-14dv96mkld/checkpoint_5000/checkpoint-5000') # 20 SSA Complete
elif env_config['rso_count'] == 40:
    MARWIL_trainer.restore('/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-46-52wmigl7hj/checkpoint_10000/checkpoint-10000') # 40 SSA Complete
else:
    print(str(env_config['rso_count']) + ' is not a valid number of RSOs')

with MARWIL_trainer.get_policy()._sess.graph.as_default():
    with MARWIL_trainer.get_policy()._sess.as_default():
        MARWIL_trainer.get_policy().model.base_model.save_weights("/tmp/pgr/weights3.h5")

with PG_trainer.get_policy()._sess.graph.as_default():
    with PG_trainer.get_policy()._sess.as_default():
        PG_trainer.get_policy().model.base_model.load_weights("/tmp/pgr/weights3.h5")


result = {'timesteps_total': 0}
num_steps_train = 5000000
best_athlete = 480
episode_len_mean = []
episode_reward_mean = []
episode_reward_max = []
episode_reward_min = []
start = datetime.datetime.now()
num_steps_trained = []
clock_time = []
training_iteration = []
result['timesteps_total'] = 0
while result['timesteps_total'] < num_steps_train:
    result = PG_trainer.train()
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
    print('mAverage20  ::: Len:: Mean: ' + str(np.round(len_moving_average[-1],1)) + '; Reward:: Mean: ' + str(np.round(reward_moving_average[-1], 1)))

    if result['training_iteration'] % 50 == 0:
        checkpoint = PG_trainer.save()
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
        output_path = PG_trainer._logdir + '/_running_results.pkl'
        with open(output_path, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Running Results Saved To: ' + output_path)

    if best_athlete > episode_len_mean[-1]:
        best_athlete = episode_len_mean[-1]
        best_athlete_checkpoint = PG_trainer.save()
        print("checkpoint saved at", best_athlete_checkpoint)

checkpoint = PG_trainer.save()
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
output_path = PG_trainer._logdir + '/_running_results.pkl'
with open(output_path, 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Final Results Saved To: ' + output_path)


"""
marwil to pg, no explore 40, checkpoints:
# good mid '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_12-22-40ahd_vmcw/checkpoint_5546/checkpoint-5546'
# final  '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_12-22-40ahd_vmcw/checkpoint_5585/checkpoint-5585'

marwil to pg, with explore 40, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_12-23-50hm6es2gk/checkpoint_5585/checkpoint-5585'

marwil to pg, with explore 10, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-02_23-57-09p0l4rfzr/checkpoint_1200/checkpoint-1200'

marwil to pg, with explore 20, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_08-50-33tfhk838k/checkpoint_1200/checkpoint-1200'
"""

"""
explore=true
marwil to pg, with explore 10, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-02_23-57-09p0l4rfzr/checkpoint_1200/checkpoint-1200'

marwil to pg, with explore 20, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_08-50-33tfhk838k/checkpoint_1200/checkpoint-1200'

marwil to pg, with explore 40, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_12-23-50hm6es2gk/checkpoint_5585/checkpoint-5585'
"""


"""
explore=false

marwil to pg, with explore 10, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-02_23-57-09p0l4rfzr/checkpoint_1200/checkpoint-1200'

marwil to pg, with explore 20, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_08-50-33tfhk838k/checkpoint_1200/checkpoint-1200'

marwil to pg, with explore 40, checkpoint
final '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-03_12-23-50hm6es2gk/checkpoint_5585/checkpoint-5585'
"""
