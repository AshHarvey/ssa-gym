import gym

from stable_baselines import A2C
import envs.ssa_tasker_simple_1

model = A2C('MlpPolicy', 'ssa_tasker_simple-v1', verbose=1, tensorboard_log="./a2c_ssa_50_tensorboard/")
model.learn(total_timesteps=16000)


model.save("a2c_tasker_ts16k")


