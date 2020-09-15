import gym
import numpy as np
import pickle
import pandas as pd
import ray.utils
from envs import env_config
from envs.ssa_tasker_simple_2 import SSA_Tasker_Env
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.marwil import MARWILTrainer, DEFAULT_CONFIG as MARWIL_CONFIG
from ray.rllib.agents.pg import DEFAULT_CONFIG as PG_CONFIG, PGTrainer
from tqdm import tqdm
from agents import agent_visible_greedy, agent_naive_random, agent_visible_greedy_spoiled

ray.init()

env_config['rso_count'] = 40

if env_config['rso_count'] == 10:
    ppo_checkpoint = '/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-10_22-03-49ygs3324j/checkpoint_245/checkpoint-245' # 245 itr,  1m ts
    marwil_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-20-065a0phk5m/checkpoint_5000/checkpoint-5000' # 5k iter, 10.6m tst, 1.13m tss
    pgr_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-10_17-44-58h06z13i1/checkpoint_246/checkpoint-246' # 246 itr, 1m ts
    pgre_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-10_18-29-33rqw1nex6/checkpoint_243/checkpoint-243' # 243 itr, 1m ts
    olr_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_21-36-258x9lqzjx/checkpoint_6034/checkpoint-6034' # 6034 itr, 1m tss, 24.7m tst
elif env_config['rso_count'] == 20:
    ppo_checkpoint = '/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-11_08-02-10ilu71y1a/checkpoint_245/checkpoint-245' # 5k iter, 1m ts
    marwil_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-24-14dv96mkld/checkpoint_5000/checkpoint-5000' # 5k iter, 10.7m tst,  1.3m tss
    pgr_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-10_18-03-53zpjcvcc4/checkpoint_243/checkpoint-243' # 243 itr,  1m ts
    pgre_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-10_18-29-41pn18bn3b/checkpoint_242/checkpoint-242' # 1m ts
    olr_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_21-38-206txe7ejn/checkpoint_5338/checkpoint-5338' # 1043 iter, 2.65m tst,  1m tss
elif env_config['rso_count'] == 40:
    ppo_checkpoint = '/home/ash/ray_results/PPO_SSA_Tasker_Env_2020-09-11_00-05-07dx8fztn6/checkpoint_1086/checkpoint-1086' # 40 RSOs, Default Net
    marwil_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_08-46-52wmigl7hj/checkpoint_10000/checkpoint-10000' # 5k iter, 51.1m tst,  1.95m tss
    # marwil_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-10_13-22-418ax_ko9v/checkpoint_5000/checkpoint-5000' # 5k iter, 51.1m tst,  1.95m tss
    # marwil alt - '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-08-31_18-30-47p8ac740j/checkpoint_6951/checkpoint-6951'
    pgr_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-11_10-36-50y4j6c25k/checkpoint_1050/checkpoint-1050' # 1050 itr,  5m ts, lr=1e-7
    # alt pgr_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-11_10-06-37oaqca_e3/checkpoint_123/checkpoint-123' # 123 itr,  5m ts, batch=40k
    pgre_checkpoint = '/home/ash/ray_results/PG_SSA_Tasker_Env_2020-09-11_00-29-48owhst3w2/checkpoint_1080/checkpoint-1080' # 1080 itr,  5m ts
    olr_checkpoint = '/home/ash/ray_results/MARWIL_SSA_Tasker_Env_2020-09-11_03-12-01n322t23n/checkpoint_322/checkpoint-322' # 322 itr, 1m tss, 1.89 tst
else:
    print(str(env_config['rso_count']) + ' is not a valid number of RSOs')

ppo_config = PPO_CONFIG.copy()
# !---- found that the network design from Jones's work had little effect in training
# config['model']['fcnet_hiddens'] = [180, 95, 50] # 10 RSOs
# config['model']['fcnet_hiddens'] = [360, 180, 100] # 20 RSOs
# config['model']['fcnet_hiddens'] = [720, 380, 200] # 40 RSOs
ppo_config['gamma'] = 0.99 # gamma  (float) Discount factor
ppo_config['rollout_fragment_length'] = 128 # n_steps (int) The number of steps to run for each environment per update
# (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
ppo_config['entropy_coeff'] = 0.01 # ent_coef (float) Entropy coefficient for the loss calculation
ppo_config['lr'] = 0.00025 # learning_rate (float or callable) The learning rate, it can be a function
ppo_config['vf_loss_coeff'] = 0.5 # vf_coef (float) Value function coefficient for the loss calculation
ppo_config['grad_clip'] = 0.5 # max_grad_norm (float) The maximum value for the gradient clipping
ppo_config['lambda'] = 0.95 # lam (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
ppo_config['num_sgd_iter'] = 4 # nminibatches  (int) Number of training minibatches per update. For recurrent policies,
# the number of environments run in parallel should be a multiple of nminibatches.
ppo_config['num_workers'] = 4 # noptepochs (int) Number of epoch when optimizing the surrogate
ppo_config['clip_param'] = 0.2 # cliprange (float or callable) Clipping parameter, it can be a function
ppo_config['vf_clip_param'] = 1 # cliprange_vf = None? --  (float or callable) Clipping parameter for the value function,
# it can be a function. This is a parameter specific to the OpenAI implementation. If None is passed (default), then
# cliprange (that is used for the policy) will be used. IMPORTANT: this clipping depends on the reward scaling. To
# deactivate value function clipping (and recover the original PPO implementation), you have to pass a negative value
# (e.g. -1).
ppo_config['env_config'] = env_config
ppo_config['train_batch_size'] = 4000
ppo_config['explore'] = False

PPO_agent = PPOTrainer(config=ppo_config, env=SSA_Tasker_Env)
PPO_agent.restore(ppo_checkpoint)
PPO_agent.get_policy().config['explore'] = False

logdir = '/home/ash/ray_results/ssa_experiences/agent_visible_greedy_spoiled/' + str(env_config['rso_count']) + 'RSOs_jones_flatten_10000episodes/'

marwil_config = MARWIL_CONFIG.copy()
marwil_config['evaluation_num_workers'] = 1
marwil_config['env_config'] = env_config
marwil_config['evaluation_interval'] = 1
marwil_config['evaluation_config'] = {'input': 'sampler'}
marwil_config['beta'] = 1 # 0
marwil_config['input'] = logdir
marwil_config['env_config'] = env_config
marwil_config['explore'] = False

MARWIL_agent = MARWILTrainer(config=marwil_config, env=SSA_Tasker_Env)
MARWIL_agent.restore(marwil_checkpoint)
MARWIL_agent.get_policy().config['explore'] = False

pg_config = PG_CONFIG.copy()
pg_config['batch_mode'] = 'complete_episodes'
pg_config['train_batch_size'] = 2000
pg_config['lr'] = 0.0001
pg_config['evaluation_interval'] = None
pg_config['postprocess_inputs'] = True
pg_config['env_config'] = env_config
pg_config['explore'] = False

PGR_agent = PGTrainer(config=pg_config, env=SSA_Tasker_Env)
PGR_agent.restore(pgr_checkpoint)
PGR_agent.get_policy().config['explore'] = False

PGRE_agent = PGTrainer(config=pg_config, env=SSA_Tasker_Env)
PGRE_agent.restore(pgre_checkpoint)
PGRE_agent.get_policy().config['explore'] = False

OLR_agent = PGTrainer(config=pg_config, env=SSA_Tasker_Env)
OLR_agent.restore(olr_checkpoint)
OLR_agent.get_policy().config['explore'] = False

def ppo_agent(obs, env):
    return PPO_agent.compute_action(obs)

def marwil_agent(obs, env):
    return MARWIL_agent.compute_action(obs)

def pgr_agent(obs, env):
    return PGR_agent.compute_action(obs)

def pgre_agent(obs, env):
    return PGRE_agent.compute_action(obs)

def olr_agent(obs, env):
    return OLR_agent.compute_action(obs)

def greedy_agent(obs, env):
    return agent_visible_greedy(obs, env)

def random_agent(obs, env):
    return agent_naive_random(obs, env)

def greedy_spoiled_agent(obs, env):
    return agent_visible_greedy_spoiled(obs, env)


env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})

agents = [greedy_agent, greedy_spoiled_agent, random_agent, ppo_agent, marwil_agent, pgr_agent, pgre_agent, olr_agent]
# agents = [pgr_agent, pgre_agent]
# agents = [pgre_agent]
# agents = [ppo_agent]
agent_names = [agent.__name__ for agent in agents]
metric = ['time steps', 'reward']
index = list(range(20, 120))
results = {}
dfs = []

for agent in agents:
    name = agent.__name__
    rewards = []
    time_steps = []
    actions = []
    for eps in tqdm(range(100)):
        env.seed(seed=eps+20)
        obs = env.reset()
        reward = 0
        done = False
        ts = 0
        while not done:
            ts += 1
            a = agent(obs, env)
            actions.append(a)
            obs, r, done, _ = env.step(a)
            reward += r
        rewards.append(reward)
        time_steps.append(ts)
    results[name] = {'rewards': rewards,
                     'time_steps': time_steps}

    path = 'results_' + str(env_config['rso_count']) + '_rsos.pkl'
    with open(path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dfs.append(pd.DataFrame(np.array([time_steps, rewards]).T, index=index, columns=([[name, name], ['time_steps', 'rewards']])))

    data = pd.concat(dfs, axis=1)

data.to_csv('results_' + str(env_config['rso_count']) + '.csv')
