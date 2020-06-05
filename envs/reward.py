import numpy as np
from numba import jit
from scipy.spatial import distance


@jit
def score_scaled_trace_P(P, dt):
    diag = np.diag(P) # The main diagonal is always positive.
    position_error_sq = np.sum(diag[:3])
    velocity_error_sq = np.sum(diag[3:])
    position_error = np.sqrt(position_error_sq)
    velocity_error = np.sqrt(velocity_error_sq)
    score = position_error + velocity_error*30
    return score


@jit
def score_trace_P(P):
    return np.trace(P)


def score_neg_max_trace_P(states,filters_x,filters_P):
    score_t = []
    for P in filters_P:
        score_t.append(np.trace(P))
    reward = -max(score_t)
    return reward


def score_scaled_det_P(P, dt=30.0):
    score = np.power(np.multiply(np.linalg.det(P), dt**6), 1/12)
    return score


def score_det_P(P, dt=30.0):
    score = np.linalg.det(P)
    return score


def score_det_pos_P(P):
    score = np.linalg.det(P[:3, :3])
    return score


def score_neg_max_pos_error(states,filters_x,filters_P):
    score_t = []
    for i in range(len(states)):
        score_t.append(distance.euclidean(states[i][:3], filters_x[i][:3]))
    reward = -max(score_t)
    return reward
