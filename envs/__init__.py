
from gym.envs.registration import register


register(
    id='ssa_tasker-v0',
    entry_point='envs.ssa_tasker:SSA_Tasker_Env',
)

register(
    id='ssa_tasker-v1',
    entry_point='envs.ssa_tasker_1:SSA_Tasker_Env',
)


register(
    id='ssa_tasker-v3',
    entry_point='envs.ssa_tasker_3:SSA_Tasker_Env',
)


register(
    id='ssa_tasker-v4',
    entry_point='envs.ssa_tasker_4:SSA_Tasker_Env',
)

register(
    id='ssa_tasker_simple-v1',
    entry_point='envs.ssa_tasker_simple_1:SSA_Tasker_Env',
)

register(
    id='oe_grid-v4',
    entry_point='envs.oe_grid_4:EphemOEGrid4',
)
