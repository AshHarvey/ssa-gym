import numpy as np

########################################################################################################################
# Heuristic Agents used:
########################################################################################################################

def agent_naive_greedy(obs, env=None):
    trace = [np.trace(P) for P in env.P_filter[env.i]]
    return np.argmax(trace)


def agent_naive_random(obs=None, env=None):
    return env.action_space.sample()

def agent_shannon(obs,env):
    """
    :param env: environment
    :return: Shannon Information Gain
    """
    visible = env.visible_objects()
    if not np.any(visible):
        return env.action_space.sample()
    with np.errstate(divide='ignore', invalid='ignore'):
        calculate = [(np.log(np.linalg.det(P) /np.linalg.det(P_i))) for P, P_i in zip(env.P_filter[env.i ][visible], env.P_filter[env.i - 1 ][visible])]
    visible_id = np.argmax(calculate)
    return visible[visible_id]

def agent_visible_random(obs, env):
    visible = env.visible_objects()
    if not np.any(visible):
        return env.action_space.sample()
    return np.random.choice(visible)


def agent_visible_greedy(obs, env):
    visible = env.visible_objects()
    if not np.any(visible):
        return env.action_space.sample()
    #visible_trace = np.sum(obs[visible, 6:], axis=1)
    visible_trace = [np.trace(P) for P in env.P_filter[env.i][visible]]
    visible_id = np.argmax(visible_trace)
    return visible[visible_id]


def agent_visible_greedy_spoiled(obs, env, p=0.25):
    visible = env.visible_objects()
    random = env.action_space.sample()
    if not np.any(visible):
        return random
    #visible_trace = np.sum(obs[visible, 6:], axis=1)
    visible_trace = [np.trace(P) for P in env.P_filter[env.i][visible]]
    visible_id = np.argmax(visible_trace)
    greedy = visible[visible_id]
    return np.random.choice(a=[greedy, random], p=[1-p, p])


def agent_visible_greedy_aer(obs, env):
    visible = env.visible_objects()
    if not np.any(visible):
        return env.action_space.sample()
    visible_trace = obs.reshape(int(len(obs)/4), 4)[visible, 3]
    visible_id = np.argmax(visible_trace)
    return visible[visible_id]


def agent_pos_error_greedy(obs, env):
    visible = env.visible_objects()
    if not np.any(visible):
        return env.action_space.sample()
    visible_positional_error = env.delta_pos[env.i, visible]
    visible_id = np.argmax(visible_positional_error)
    return visible[visible_id]


def agent_vel_error_greedy(obs, env):
    visible = env.visible_objects()
    if not np.any(visible):
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
