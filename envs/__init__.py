
from gym.envs.registration import register

register(
    id='ssa_tasker_simple-v2',
    entry_point='envs.ssa_tasker_simple_2:SSA_Tasker_Env',
)

register(
    id='ssa_tasker_simple-v3',
    entry_point='envs.ssa_tasker_simple_3:SSA_Tasker_Env',
)
