import gym
import envs.ssa_tasker_simple_1
from envs.ssa_tasker_simple_1 import score, errors, plot_results
import numpy as np
from tqdm import tqdm, tnrange # used for progress bars
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from stable_baselines import A2C

def greedy_agent(observation, env):
    try:
        n = list(range(env.RSO_Count))
        error = [np.sum(observation[nn][6:]) for nn in n]
        action = n[np.argmax(error)]
    except:
        action = env.action_space.sample()
    return action

def naive_random_agent(observation, env):
    action = env.action_space.sample()
    return action

def steps_to_time(step,dt=30):
    time = step*dt/60/60
    return time #in hours

model2 = A2C.load("a2c_tasker_ts16k")

def ac2_agent(observation, env, model2=model2):
    action, _states = model2.predict(observation)
    return action

env = gym.make('ssa_tasker_simple-v1')
run_length = 480
episodes = 10

env.run_length = run_length
env.RSO_Count = 50
seeds = np.random.randint(0,2**32-1,episodes)
print('seeds: ',seeds)
#agents = [greedy_agent, naive_random_agent, ac2_agent] # [greedy_agent, naive_random_agent, ac2_agent]
agents = [naive_random_agent, greedy_agent] # [greedy_agent, naive_random_agent, ac2_agent]
df = []
actionlist = []
for i in range(len(agents)):
    agent = agents[i]
    runs = []
    for j in tqdm(range(len(seeds)),desc=agent.__name__):
        #env.seed(2)
        np.random.seed(seeds[j])
        observation = env.reset()
        rewards = []
        
        for k in range(run_length):
            action = agent(observation, env)
            
            actionlist.append(action)
                
            observation, reward, done, info = env.step(action)
            rewards.append(-reward)
            '''if done:
                print("Episode finished after {} timesteps".format(i+1))
                break'''
        runs.append(rewards)
        if len(env.failed_filters) > 0:
            env.plot()
    
    results = []
    n = 0
    for run in runs:
        results.append(np.array(list(zip(run,list(range(len(run)))))))
        n += 1
        
    results = np.array(results)
    #results = results.reshape((run_length*episodes),2)
    results = results.reshape((sum([len(run) for run in runs])),2)
    results = results[:,[1,0]]
    results = results.tolist()
    for result in list(results):
        result.append(agent.__name__)
    
    df.append(pd.DataFrame(results, columns=['TimeStep','Reward (Trace{P})','Agent']))
    df[-1].to_csv(agent.__name__+"_output.csv")  

df = pd.concat(df,axis=0)

df.insert(2,'Time', df[['TimeStep']].apply(lambda x: datetime.timedelta(seconds=x[0]*env.dt),axis=1))
df.insert(2,'Hours', df[['TimeStep']].apply(lambda x: steps_to_time(x[0],env.dt),axis=1))


ax = sns.lineplot(x="Hours", y="Reward (Trace{P})", hue="Agent", data=df)
ax.set(yscale="log")
ax.legend(loc='upper left')
ax
plt.savefig('Figure-50-RSO-10-Eps-480-TS.pdf')

plt.show()

env.close()
