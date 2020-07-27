import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from poliastro.bodies import Earth
from envs.transformations import deg2rad, lla2ecef, gcrs2irts_matrix_a, get_eops, ecef2lla
from envs.dynamics import init_state_vec, fx_xyz_markley as fx, hx_aer_erfa as hx
from tqdm import tqdm
from poliastro.core.elements import rv2coe
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import itertools

"""
Unit Abbreviations:
arc minutes: '
arc seconds: "
degrees: deg
kilometers: km
latitude: lat
longitude: lon
meters: m
radians: rad
seconds: s
unitless: u
"""

samples = 20000
step_size = 60*2.5 # seconds
max_gap = 1.5 # hours
duration = 4 # hours
obs_limit = np.radians(15)
first_window = 45 # minutes
n = int(np.ceil(duration*60*60/step_size))

observer = (38.828198, -77.305352, 20.0) # lat, lon, height (deg, deg, m)
obs_lla = np.array(observer)*[deg2rad, deg2rad, 1] # lat, lon, height (deg, deg, m)
obs_itrs = lla2ecef(obs_lla) # lat, lon, height (deg, deg, m) to ITRS (m)
RE = Earth.R_mean.to_value(u.m)
t_0 = datetime(2020, 5, 4, 0, 0, 0)
t_samples = [t_0 + timedelta(seconds=step_size)*i for i in range(n)]
eops = get_eops()
trans_matrix = gcrs2irts_matrix_a(t=t_samples, eop=eops)
hx_kwargs = [{"trans_matrix": trans_matrix[i], "observer_itrs": obs_itrs, "observer_lla": obs_lla} for i in range(n)]

candidates = []
seed = 0
random_state = np.random.RandomState(seed)

for i in tqdm(range(samples)):
    regime = random_state.choice(a=['LEO', 'MEO', 'GEO', 'Tundra', 'Molniya'], p=[1/3, 1/3, 1/9, 1/9, 1/9])
    accepted = False
    while not accepted:
        candidate = init_state_vec(orbits=[regime], random_state=random_state)
        x_gcrs = [fx(candidate, step_size*i) for i in range(n)]
        x_itrs = [x_gcrs[i][:3]@trans_matrix[i] for i in range(n)]
        x_lla = np.array([ecef2lla(x_itrs[i]) for i in range(n)])
        candidate_elevation = [hx(x_gcrs=x_gcrs[i][:3], **hx_kwargs[i])[1] for i in range(n)]
        visibility = candidate_elevation >= obs_limit
        gaps = [sum(1 for _ in group) for key, group in itertools.groupby((visibility-1)*-1) if key]
        if np.all(x_lla[:, 2] > 300*1000):
            if not gaps == []:
                if sum(visibility[0:int(first_window*60/step_size)]) > 0:
                    if np.max(gaps) < max_gap*60*60/step_size:
                        candidates.append(candidate)
                        accepted = True
            else:
                if np.all(visibility):
                    candidates.append(candidate)
                    accepted = True

arr = np.copy(candidates)

lla = np.array([ecef2lla(x) for x in arr[:, :3]@trans_matrix[0]])

coes = np.array([rv2coe(k=Earth.k.to_value(u.km**3/u.s**2), r=x[:3]/1000, v=x[3:]/1000) for x in arr])

x = lla[:, 2]/1000
y = coes[:, 1]

# We can set the number of bins with the `bins` kwarg
plt.hist(x, bins=20)
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=(300, 300), norm=LogNorm())
plt.ylim(0, 0.75)
ax.set_ylabel('Eccentricity')
ax.set_title('2D Histogram, ' + str(len(y)) + ' objects visible every \n' + str(max_gap) + ' hours from ' + str(samples) + ' orbits sampled')
ax.set_xlabel('Altitude at initial time step (kilometers)')
plt.savefig(str(max_gap) + '_hour_viz_' + str(len(y)) + '_of_' + str(samples) + '_sample_orbits_seed_' + str(seed) + '_Orbit_Para_Plot.svg')

np.save(str(max_gap) + '_hour_viz_' + str(len(y)) + '_of_' + str(samples) + '_sample_orbits_seed_' + str(seed) + '.npy', arr)

run = np.empty((duration*60*2, len(arr), 6))
dt = 30.0
n = duration*60*2
m = len(arr)
t_steps = [t_0 + timedelta(seconds=30)*i for i in range(n)]
trans_matrix = gcrs2irts_matrix_a(t=t_steps, eop=eops)
hx_kwargs = [{"trans_matrix": trans_matrix[i], "observer_itrs": obs_itrs, "observer_lla": obs_lla} for i in range(n)]
for i in tqdm(range(n)):
    for j in range(m):
        if i == 0:
            run[i, j, :] = np.copy(arr[j])
        else:
            run[i, j, :] = fx(run[i-1, j, :], 30.0)

run_obs = np.array([[hx(run[i, j, :3], **hx_kwargs[i]) for j in range(m)] for i in range(n)])

fig = plt.figure()
plt.ylabel('RSO ID')
plt.xlabel('Time Step (30 seconds per)')
plt.title('Visibility Plot (white = visible)')
ax = fig.add_subplot(111)
ax.imshow(run_obs[:, :, 1].T > obs_limit, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
plt.savefig(str(max_gap) + '_hour_viz_' + str(len(y)) + '_of_' + str(samples) + '_sample_orbits_seed_' + str(seed) + '_Viz_Window_Plot.svg')
