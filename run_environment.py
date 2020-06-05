import gym
import envs.ssa_tasker_simple_2

env = gym.make('ssa_tasker_simple-v2')
_ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
