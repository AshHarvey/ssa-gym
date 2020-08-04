import numpy as np
#from stable_baselines import A2C


def agent_naive_greedy(obs, env=None):
    trace = np.sum(obs[:, 6:], axis=1)
    return np.argmax(trace)
def agent_naive_random(obs=None, env=None):
    return env.action_space.sample()


def agent_visible_random(obs, env):
    visible = env.visible_objects
    if not np.any(env.visible_objects):
        return env.action_space.sample()
    return np.random.choice(visible)


def agent_visible_greedy(obs, env):
    visible = env.visible_objects
    if not np.any(env.visible_objects):
        return env.action_space.sample()
    visible_trace = np.sum(obs[visible, 6:], axis=1)
    visible_id = np.argmax(visible_trace)
    return visible[visible_id]


def agent_pos_error_greedy(obs, env):
    visible = env.visible_objects
    if not np.any(env.visible_objects):
        return env.action_space.sample()
    visible_positional_error = env.delta_pos[env.i, visible]
    visible_id = np.argmax(visible_positional_error)
    return visible[visible_id]


def agent_vel_error_greedy(obs, env):
    visible = env.visible_objects
    if not np.any(env.visible_objects):
        return env.action_space.sample()
    visible_velocity_error = env.delta_vel[env.i, visible]
    visible_id = np.argmax(visible_velocity_error)
    return visible[visible_id]


"""
model_a2c = A2C.load("a2c_tasker_ts16k") # loads the model to be used in agent_ac2


def agent_ac2(observation, env, model=model_a2c):
    action, _states = model.predict(observation)
    return action
    
"""
