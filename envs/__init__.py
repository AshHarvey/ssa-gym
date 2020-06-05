
from gym.envs.registration import register

register(
    id='ssa_tasker_simple-v1',
    entry_point='envs.ssa_tasker_simple_1:SSA_Tasker_Env',
)

register(
    id='ssa_tasker_simple-v2',
    entry_point='envs.ssa_tasker_simple_2:SSA_Tasker_Env',
)
