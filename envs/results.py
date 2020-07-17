from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import timedelta
import numpy as np
from scipy.special import erf


@njit
def errors(states, filters_x, filters_diag_P):
    n, m, d = states.shape # Timesteps dimension, RSO dimension, dimension of x
    sigma_position = np.zeros(shape=(n, m)) # magnitude of position uncertainty (meters)
    sigma_velocity = np.zeros(shape=(n, m)) # magnitude of velocity uncertainty (meters)
    delta_position = np.zeros(shape=(n, m)) # mean position - true position (meters)
    delta_velocity = np.zeros(shape=(n, m)) # mean velocity - true velocity (meters)
    for i in range(n): # Timesteps dimension
        delta_position[i, :] = dist3d(filters_x[i][:, :3], states[0][:, :3])
        delta_velocity[i, :] = dist3d(filters_x[i][:, 3:], states[0][:, 3:])
        sigma_position[i, :] = var3d(filters_diag_P[i][:, :3])
        sigma_velocity[i, :] = var3d(filters_diag_P[i][:, 3:])
    return delta_position, delta_velocity, sigma_position, sigma_velocity


@njit
def error(states, obs):
    delta_position = dist3d(obs[:, :3], states[:, :3]) # mean position - true position (meters)
    delta_velocity = dist3d(obs[:, 3:6], states[:, 3:]) # mean velocity - true velocity (meters)
    sigma_position = var3d(obs[:, 6:9]) # magnitude of position uncertainty (meters)
    sigma_velocity = var3d(obs[:, 9:]) # magnitude of velocity uncertainty (meters)
    return delta_position, delta_velocity, sigma_position, sigma_velocity


@njit
def error_failed(state, x, P):
    result = np.zeros(4)
    result[0] = np.sqrt(np.sum((x[:3]-state[:3])**2)) # mean position - true position (meters)
    result[1] = np.sqrt(np.sum((x[3:]-state[3:])**2)) # mean velocity - true velocity (meters)
    result[2] = np.sqrt(np.sum(np.diag(P)[:3])) # magnitude of position uncertainty (meters)
    result[3] = np.sqrt(np.sum(np.diag(P)[3:])) # magnitude of velocity uncertainty (meters)
    return result


@njit
def observations(filters_x, filters_P):
    n = len(filters_x)
    observation = np.zeros((n, 12))
    for i in range(n):
        observation[i, :6] = filters_x[i]
        observation[i, 6:] = np.diag(filters_P[i])
    return observation


@njit
def dist3d(u, v):
    dist = np.sqrt(np.sum((u-v)**2, axis=1))
    return dist


@njit
def var3d(u):
    dist = np.sqrt(np.sum(u, axis=1))
    return dist


def obs_filter_object_x_diagP(filters):
    obs = []
    for ukf in filters:
        obs.append(np.append(ukf.x,np.diag(ukf.P)))

    obs = np.array(obs)

    return np.nan_to_num(obs,nan=99999, posinf=99999)


def plot_delta_sigma(sigma_pos, sigma_vel, delta_pos, delta_vel, dt, t_0, style=None, yscale='log', ylim='max',
                     title=None, save_path=None, display=True):
    n, m = sigma_pos.shape
    t = [t_0 + timedelta(seconds=dt * i) for i in range(n)]

    # plot with errors and uncertainties
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)
    if ylim == 'max':
        pos_lims_max = 10**np.ceil(np.log10(np.nanmax([delta_pos, sigma_pos])))
        pos_lims_min = 10**np.floor(np.log10(np.min([np.nanmin([delta_pos, sigma_pos])])))
        pos_lims = (pos_lims_min, pos_lims_max)
    else:
        pos_lims_max = 1e6
        pos_lims_min = 1e-3
        pos_lims = (pos_lims_min, pos_lims_max)
    if ylim == 'max':
        vel_lims_max = 10**np.ceil(np.log10(np.nanmax([delta_vel, sigma_vel])))
        vel_lims_min = 10**np.floor(np.log10(np.min([np.nanmin([delta_vel, sigma_vel])])))
        vel_lims = (vel_lims_min, vel_lims_max)
    else:
        vel_lims_max = 1e4
        vel_lims_min = 1e-4
        vel_lims = (vel_lims_min, vel_lims_max)

    tim_lim = (t_0.toordinal(), t_0.toordinal()+n/(24*60*60/dt))

    # Uncertainty in Position
    plt.subplot(221)
    plt.plot(t, sigma_pos, linewidth=0.5)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(pos_lims)
    plt.ylabel('$\sqrt{\sum (P_{i,j})}$ where i = j')
    plt.xlim(tim_lim)
    plt.title(r'Position ($m$)')
    plt.grid(True)

    # Uncertainty in Velocity
    plt.subplot(222)
    plt.plot(t, sigma_vel, linewidth=0.5)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(vel_lims)
    plt.xlim(tim_lim)
    plt.title(r'Velocity ($\frac{m}{s}$)')
    plt.grid(True)

    # Error in Position
    plt.subplot(223)
    plt.plot(t, delta_pos, linewidth=0.5)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(pos_lims)
    plt.ylabel('$\sqrt{\sum{(x_i-\hat{x}_i)^2}}$')
    plt.xlim(tim_lim)
    plt.xlabel('Simulation Time (HH:MM)')
    plt.grid(True)

    # Error in Velocity
    plt.subplot(224)
    plt.plot(t, delta_vel, linewidth=0.5)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(vel_lims)
    plt.xlim(tim_lim)
    plt.xlabel('Simulation Time (HH:MM)')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_rewards(rewards, dt, t_0, style=None, yscale='symlog'):
    n = len(rewards)
    t = [t_0 + timedelta(seconds=dt * i) for i in range(n)]

    # plot with Reward over Time
    plt.figure()
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)
    tim_lim = (t_0.toordinal(), t_0.toordinal()+n/(24*60*60/dt))
    reward_lim = (0, 1)

    # Reward over Time
    plt.plot(t, rewards, linewidth=1)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(reward_lim)
    plt.xlim(tim_lim)
    plt.title('Reward over Time')
    plt.grid(True)

    plt.show()


def plot_performance(rewards, dt, t_0, sigma=1.5, style=None, yscale='linear'):
    o, m, n = rewards.shape # n: time steps, m: episodes, o: agents
    t = [t_0 + timedelta(seconds=dt * i) for i in range(n)]

    q = (100-erf(sigma/np.sqrt(2))*100, 50, erf(sigma/np.sqrt(2))*100)

    performance = np.percentile(a=rewards[:, :, :], q=q, axis=1) # 3, o, n

    # plot with Reward over Time
    plt.figure()
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)
    tim_lim = (t_0.toordinal(), t_0.toordinal()+n/(24*60*60/dt))

    for k in range(o):
        plt.fill_between(x=t, y1=performance[0, k, :], y2=performance[2, k, :], alpha=0.25)
        plt.plot(t, performance[1, k, :], linewidth=1)

    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.xlim(tim_lim)
    plt.title('Performance of Agents over Time')
    plt.grid(True)

    plt.show()


def plot_nees(nees, dt, t_0, style=None, yscale='symlog', axis=0, title=None, save_path=None, display=True):
    n, m = nees.shape

    if axis == 0:
        x = np.array(range(m))
    elif axis == 1:
        x = [t_0 + timedelta(seconds=dt * i) for i in range(n)]

    # plot with Reward over Time
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)

    if axis == 0:
        x_lim = (0, m-1)
    elif axis == 1:
        x_lim = (t_0.toordinal(), t_0.toordinal()+n/(24*60*60/dt))

    anees = np.mean(nees, axis=axis)

    y_lim = (np.min(anees), np.max(anees))

    # ANEES by RSO or Time
    if axis == 0:
        plt.bar(x, anees)
        plt.axhline(y=1, linewidth=1, color='k', xmin=x_lim[0], xmax=x_lim[1])
        plt.ylabel(r'$\frac{1}{n_x M}\sum_{i=1}^M((x_m^i-\hat{x}_m^i)^T(P_{k|k}^i)^{-1}(x_m^i-\hat{x}_m^i))$')
        plt.xlabel('RSO ID')
        plt.title('Averaged Normalized Estimation Error Squared (ANEES) by RSO')
    elif axis == 1:
        plt.plot(x, anees, linewidth=1)
        plt.ylabel(r'$\frac{1}{n_x M}\sum_{i=1}^M((x_m^i-\hat{x}_m^i)^T(P_{k|k}^i)^{-1}(x_m^i-\hat{x}_m^i))$')
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        plt.xlabel('Simulation Time (HH:MM)')
        plt.title('Averaged Normalized Estimation Error Squared (ANEES) over Time')
    plt.yscale(yscale)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.xlabel('Simulation Time (HH:MM)')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_histogram(values, bins=None, style=None, title='Histogram of Errors (%)', xlabel=r'$\sqrt{\sum{(x_i-\hat{x}_i)^2}}$',
                   save_path=None, display=True):
    values = values.flatten()
    n, = values.shape

    # plot with Reward over Time
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)
    if bins is 'default':
        bins = [0, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]

    plt.xlabel(xlabel)

    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=n))
    plt.hist(x=values, bins=bins)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_orbit_vis(observations, obs_limit, dt, display=True, save_path=None):
    fig = plt.figure()
    plt.ylabel('RSO ID')
    plt.xlabel('Time Step (' + str(dt) + ' seconds per)')
    plt.title('Visibility Plot (white = visible)')
    ax = fig.add_subplot(111)
    ax.imshow(observations[:, :, 1].T > obs_limit, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_regimes(xy, save_path=None, display=True):
    plt.figure()
    plt.scatter(x=xy[:, 0], y=xy[:, 1])
    for i, ((x, y),) in enumerate(zip(xy)):
        plt.text(x, y, i, ha="center", va="center")

    plt.ylabel('Eccentricity')
    plt.xlabel('Altitude at initial time step (kilometers)')
    plt.title('Orbital Regime of RSOs (by ID)')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


@njit
def reward_proportional_trinary_true(delta_pos):
    return np.mean(((delta_pos < 1e4)*1 + (delta_pos < 1e7)*1))/2
