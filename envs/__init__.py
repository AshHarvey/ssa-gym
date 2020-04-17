
from gym.envs.registration import register

register(
    id='ssa_tasker_simple-v1',
    entry_point='envs.ssa_tasker_simple_1:SSA_Tasker_Env',
)
