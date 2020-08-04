from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import timedelta
import numpy as np
from scipy.special import erf
from scipy import stats
from envs.transformations import ecef2lla
import os
os.environ['PROJ_LIB'] = '/home/ash/anaconda3/envs/ssa-gym'
from mpl_toolkits.basemap import Basemap


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
    ax1 = plt.subplot(221)
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
    ax2 = plt.subplot(222, sharex=ax1)
    plt.plot(t, sigma_vel, linewidth=0.5)
    plt.yscale(yscale)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    plt.ylim(vel_lims)
    plt.xlim(tim_lim)
    plt.title(r'Velocity ($\frac{m}{s}$)')
    plt.grid(True)

    # Error in Position
    ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
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
    ax4 = plt.subplot(224, sharex=ax1, sharey=ax2)
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
    fig = plt.figure()
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
    plt.xlabel('Simulation Time (HH:MM)')
    plt.ylabel('Reward per Time Step (0 to 1)')
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


def plot_nees(nees, dt, t_0, n_mov_ave=20, alpha=0.05, style=None, yscale='symlog', axis=None, title=None, save_path=None, display=True):
    n, m = nees.shape
    if alpha is not None:
        ci = [alpha/2, 1-alpha/2]
        if axis == 0:
            cr = stats.chi2.ppf(ci, df=n*6)/n
        elif axis == 1 or axis is None:
            cr = stats.chi2.ppf(ci, df=n*m*6)/(n*m)
            cr_points = stats.chi2.ppf(ci, df=6)
        if n_mov_ave is not None:
            cr_bar = stats.chi2.ppf(ci, df=n_mov_ave*m*6)/(n_mov_ave*m)

    if axis == 0:
        x = np.array(range(m))
        anees = np.mean(nees, axis=0)
        x_lim = (0, m-1)
    elif axis == 1 or axis is None:
        x = [t_0 + timedelta(seconds=dt * i) for i in range(n)]
        x_lim = (t_0.toordinal(), t_0.toordinal()+n/(24*60*60/dt))

        anees = np.mean(nees, axis=1)
        if n_mov_ave is not None:
            anees_bar = np.copy(anees)
            anees_bar[:] = 0
            for i in range(len(anees)):
                anees_bar[i] = np.mean(anees[:i+1][-n_mov_ave:])

    # plot with Reward over Time
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)

    # ANEES by RSO or Time
    if axis is None:
        y_lim = (np.min(nees), np.max(nees))
        for i in range(m):
            plt.scatter(x=x, y=nees[:, i], s=2, alpha=0.5, marker='x')
        plt.hlines(y=np.mean(nees), xmin=x[0], xmax=x[-1], linewidth=1, color='black',
                   label='Overall ANEES = ' + str(np.round(np.mean(anees), 2)))
        legend_elements = [mpl.lines.Line2D([0], [0], marker='x', lw=0, color='black', label='NEES (Colored by RSO ID)',
                                            markerfacecolor='g', markersize=5)]
        if alpha is not None:
            points_contained = np.mean((nees > cr_points[0])*(nees < cr_points[1]))
            plt.hlines(y=cr_points[0], xmin=x[0], xmax=x[-1], color='red', linestyle='--',
                       label='Confidence Region (points), alpha = ' + str(alpha))
            plt.hlines(y=cr_points[1], xmin=x[0], xmax=x[-1], color='red', linestyle='--')
            label = 'Point Confidence Region, alpha = ' + str(alpha) + '; ' + str(np.round(points_contained*100, 2)) + '% contained'
            legend_elements.append(mpl.lines.Line2D([0], [0], color='red', lw=2, linestyle='--', label=label))

        legend_elements.append(mpl.lines.Line2D([0], [0], color='black', lw=2,
                                                label='Overall ANEES = ' + str(np.round(np.mean(anees), 2))))
        if alpha is not None:
            plt.hlines(y=cr[0], xmin=x[0], xmax=x[-1], color='black', linestyle='--',
                       label='Confidence Region (overall), alpha = ' + str(alpha))
            plt.hlines(y=cr[1], xmin=x[0], xmax=x[-1], color='black', linestyle='--')
            legend_elements.append(mpl.lines.Line2D([0], [0], color='black', lw=2, linestyle='--',
                                                    label='Overall Confidence Region, alpha = ' + str(alpha)))
        if n_mov_ave is not None:
            plt.plot(x, anees_bar, linewidth=1, color='blue', label='ANEES Moving Average, n = ' + str(n_mov_ave))
            legend_elements.append(mpl.lines.Line2D([0], [0], color='blue', lw=2,
                                                    label='ANEES Moving Average, n = ' + str(n_mov_ave)))
            if alpha is not None:
                plt.hlines(y=cr_bar[0], xmin=x[0], xmax=x[-1], color='blue', linestyle='--',
                           label='Confidence Region (moving average), alpha = ' + str(alpha))
                plt.hlines(y=cr_bar[1], xmin=x[0], xmax=x[-1], color='blue', linestyle='--')
                legend_elements.append(mpl.lines.Line2D([0], [0], color='blue', lw=2, linestyle='--',
                                                        label='Moving Average Confidence Region, alpha = ' + str(alpha)))
        plt.axhline(y=1, linewidth=1, color='k', xmin=x_lim[0], xmax=x_lim[1])
        plt.ylabel(r'$(x_i^j-\hat{x}_i^j)^T(P_{i}^j)^{-1}(x_i^j-\hat{x}_i^j)$ for i = [0, ' + str(n) + '), j = [0, ' + str(m) + ')')
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        plt.xlabel('Simulation Time (HH:MM)')
        plt.title('Normalized Estimation Error Squared (NEES) by RSO over Time')
        plt.legend(handles=legend_elements)
    if axis == 0:
        y_lim = (np.min(anees), np.max(anees))
        plt.bar(x, anees)
        plt.axhline(y=np.mean(anees), linewidth=1, color='black', xmin=x_lim[0], xmax=x_lim[1],
                    label='Overall ANEES = ' + str(np.round(np.mean(anees), 2)))
        legend_elements = [mpl.lines.Line2D([0], [0], color='blue', lw=2, label='ANEES by RSO'),
                           mpl.lines.Line2D([0], [0], color='black', lw=2, label='Overall ANEES = ' + str(np.round(np.mean(anees), 2)))]
        if alpha is not None:
            plt.hlines(y=cr[0], xmin=x[0], xmax=x[-1], color='red', linestyle='--', label='Confidence Region, alpha = ' + str(alpha))
            plt.hlines(y=cr[1], xmin=x[0], xmax=x[-1], color='red', linestyle='--')
            legend_elements.append(mpl.lines.Line2D([0], [0], color='red', lw=2, linestyle='--',
                                                    label='Confidence Region, alpha = ' + str(alpha)))
        plt.ylabel(r'$\frac{1}{n_x M}\sum_{i=1}^M((x_m^i-\hat{x}_m^i)^T(P_{k|k}^i)^{-1}(x_m^i-\hat{x}_m^i))$')
        plt.xlabel('RSO ID')
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        #plt.yticks(np.arange(y_lim[0], y_lim[1], (y_lim[1]-y_lim[0])/10))
        plt.title('Averaged Normalized Estimation Error Squared (ANEES) by RSO')
        plt.legend(handles=legend_elements)
    elif axis == 1:
        y_lim = (np.min(anees), np.max(anees))
        plt.plot(x, anees, linewidth=1)
        plt.ylabel(r'$\frac{1}{n_x M}\sum_{i=1}^M((x_m^i-\hat{x}_m^i)^T(P_{k|k}^i)^{-1}(x_m^i-\hat{x}_m^i))$')
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        plt.xlabel('Simulation Time (HH:MM)')
        plt.title('Averaged Normalized Estimation Error Squared (ANEES) over Time')
    plt.yscale(yscale)
    if yscale != 'linear':
        plt.ylim(y_lim)
    plt.xlim(x_lim)
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
    if bins is 'int':
        m = np.max(values)+1
        bins = np.arange(m+1) - 0.5

    plt.xlabel(xlabel)

    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=n))
    plt.hist(x=values, bins=bins)
    if bins is 'int':
        plt.xticks(range(m))
        plt.xlim([-1, m])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_orbit_vis(visibility, title, xlabel, display=True, save_path=None):
    fig = plt.figure()
    plt.ylabel('RSO ID')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.imshow(visibility, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def plot_regimes(xy, save_path=None, display=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, 40000), ylim=(0.0, 0.75))
    plt.scatter(x=xy[:, 0], y=xy[:, 1])
    for i, ((x, y),) in enumerate(zip(xy)):
        ax.annotate(i, xy=(x, y), xycoords='data', ha="center", va="center", xytext=(20, 20),
                    textcoords='offset pixels', arrowprops=dict(arrowstyle="-"))

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


def moving_average_plot(x, n=20, alpha=0.05, dof=1, style=None, title=None, xlabel=None, ylabel=None, llabel=None, save_path=None, display=True):
    fig = plt.figure()

    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)

    x_ma = np.ma.masked_array(x, np.isnan(x))

    if alpha is not None:
        ci = [alpha/2, 1-alpha/2]
        cr = stats.chi2.ppf(ci, df=dof*n)/n
        cr_points = stats.chi2.ppf(ci, df=dof)

    x_bar = np.copy(x_ma)
    x_bar[:] = 0
    for i in range(len(x_ma)):
        x_bar[i] = np.mean(x_ma[:i+1][-n:])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x=range(0, len(x_ma)), y=x_ma, color='red', marker='+', label=llabel)
    plt.plot(x_bar, color='blue', linestyle='-', linewidth=2, label=str(n) + ' step moving average of ' + llabel)
    if alpha is not None:
        contained = np.mean((x_bar > cr[0])*(x_bar < cr[1]))
        plt.plot(np.repeat(cr[1], len(x_bar)), color='blue', linestyle='--', linewidth=2,
                 label='Moving average confidence region, alpha = ' + str(alpha) + '; ' + str(np.round(contained*100, 2)) + '% contained')
        plt.plot(np.repeat(cr[0], len(x_bar)), color='blue', linestyle='--', linewidth=2)
        points_contained = np.mean((x[~np.isnan(x)] > cr_points[0])*(x[~np.isnan(x)] < cr_points[1]))
        plt.plot(np.repeat(cr_points[1], len(x_bar)), color='red', linestyle='--', linewidth=2,
                 label='Point confidence region, alpha = ' + str(alpha) + '; ' + str(np.round(points_contained*100, 2)) + '% contained')
        plt.plot(np.repeat(cr_points[0], len(x_bar)), color='red', linestyle='--', linewidth=2)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def bound_plot(y, st_dev, style=None, title=None, xlabel=None, ylabel=None, yscale='symlog', sharey=False, save_path=None, display=True):
    if sharey:
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(3, sharex=True)

    if title is not None:
        fig.suptitle(title)
    if style is None:
        style = 'seaborn-deep'
    mpl.style.use(style)

    if np.any(np.isnan(y)):
        y_ID = np.where((np.isnan(y[:, 0]) == False))[0]

        y_not_nan = []
        st_dev_not_nan = []
        for i in y_ID:
            y_not_nan.append(y[i, :])
            st_dev_not_nan.append(st_dev[i, :])

        y_not_nan = np.array(y_not_nan)
        st_dev_not_nan = np.array(st_dev_not_nan)
    else:
        y_ID = np.arange(1, len(y)+1)
        y_not_nan = y
        st_dev_not_nan = st_dev

    frac_sigma_bound = np.mean((y_not_nan < st_dev_not_nan)*(y_not_nan > -st_dev_not_nan), axis=0)
    frac_two_sigma_bound = np.mean((y_not_nan < 2*st_dev_not_nan)*(y_not_nan > -2*st_dev_not_nan), axis=0)

    axs[2].xlabel = xlabel

    for i in range(len(y[0])):
        axs[i].ylabel = ylabel[i]
        axs[i].scatter(x=y_ID, y=y_not_nan[:, i], color='red', marker='+', label=ylabel[i])
        axs[i].plot(y_ID, 2*st_dev_not_nan[:, i], color='blue', linestyle='-', linewidth=2,
                    label='$2\sigma$ bound, ' + str(np.round(frac_two_sigma_bound[i]*100, 2)) + '% contained')
        axs[i].plot(y_ID, st_dev_not_nan[:, i], color='blue', linestyle='--', linewidth=2,
                    label='$\sigma$ bound, ' + str(np.round(frac_sigma_bound[i]*100, 2)) + '% contained')
        axs[i].plot(y_ID, -st_dev_not_nan[:, i], color='blue', linestyle='--', linewidth=2)
        axs[i].plot(y_ID, -2*st_dev_not_nan[:, i], color='blue', linestyle='-', linewidth=2)
        axs[i].legend()
        axs[i].set_yscale(yscale)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='svg')
    if display:
        plt.show()
    else:
        plt.close()


def map_plot(x_filter, x_true, trans_matrix, observer):
    n, m, x_dim = x_filter.shape
    print(n)
    lat_filter = np.empty((n, m))
    lon_filter = np.empty((n, m))
    lat_true = np.empty((n, m))
    lon_true = np.empty((n, m))
    x_filter_itrs = np.empty((n, m, 3))
    x_true_itrs = np.empty((n, m, 3))
    observer = np.degrees(observer[0]), np.degrees(observer[1]), observer[2]  # lat (deg), lon (deg), height (meters)

    plt.figure(figsize=(30, 30))
    # add map to figure


    for i in range(n):
        for j in range(m):
            x_filter_itrs[i, j, :] = x_filter[i, j, :3] @ trans_matrix[i]
            lat_filter[i, j], lon_filter[i, j], _ = ecef2lla(x_filter_itrs[i, j])
            x_true_itrs[i, j, :] = x_true[i, j, :3] @ trans_matrix[i]
            lat_true[i, j], lon_true[i, j], _ = ecef2lla(x_true_itrs[i, j])

    lat_filter = np.degrees(lat_filter)
    lon_filter = np.degrees(lon_filter)
    lat_true = np.degrees(lat_true)
    lon_true = np.degrees(lon_true)
    print(lat_filter)
    my_map = Basemap(projection='cyl', lon_0=0, lat_0=0, resolution='l')
    my_map.drawmapboundary()  # fill_color='aqua')
    my_map.fillcontinents(color='#dadada', lake_color='white')
    my_map.drawmeridians(np.arange(-180, 180, 30), color='gray')
    my_map.drawparallels(np.arange(-90, 90, 30), color='gray');
    for j in range(m):
        x , y = my_map(lon_filter[:, j],lat_filter[:, j])
        my_map.scatter(x, y, marker='o', zorder=3, label='Predicted Location')
        my_map.scatter(lon_true[:, j], lat_true[:, j], marker='x', zorder=3, label='Actual Location')

    plt.annotate('Observation Station',
                 xy=(observer[1], observer[0]),
                 xycoords='data',
                 xytext=(observer[1] - 180, observer[0] + 40),
                 textcoords='offset points',
                 fontsize=24,
                 color='g',
                 arrowprops=dict(arrowstyle="fancy", color='g')
                 )

    plt.show()

def autocorr_naive_nan(x):
    N = len(x)
    return np.array([np.nanmean(x[iSh:] * x[:N-iSh]) for iSh in range(N)])
