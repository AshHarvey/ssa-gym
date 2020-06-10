import numpy as np
from datetime import datetime, timedelta
from astropy import units as u
from poliastro.bodies import Earth
from envs.transformations import deg2rad, lla2itrs, gcrs2irts_matrix_b, get_eops
from envs.dynamics import init_state_vec, fx_xyz_markley as fx, hx_aer_kwargs as hx
from tqdm import tqdm


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

observer = (38.828198, -77.305352, 20.0) # lat, lon, height (deg, deg, m)
obs_itrs = lla2itrs(np.array(observer)*[deg2rad, deg2rad, 1]) # lat, lon, height (deg, deg, m) to ITRS (m)
RE = Earth.R_mean.to_value(u.m)
t_0 = datetime(2020, 5, 4, 0, 0, 0)
t_samples = [t_0 + timedelta(minutes=2.5)*i for i in range(576)]
eops = get_eops()
trans_matrix = gcrs2irts_matrix_b(t=t_samples, eop=eops)
hx_kwargs = [{"trans_matrix": trans_matrix[i], "observer_itrs": obs_itrs} for i in range(576)]

candidates = []
random_state = np.random.RandomState()

for i in tqdm(range(100000)):
    candidate = init_state_vec(sma=((RE + 400000), 42164000), ecc=(0.001, 0.3), inc=(0, 180), raan=(0, 360),
                               argp=(0, 360), nu=(0, 360), random_state=random_state)
    candidate_elevation = [hx(x_gcrs=fx(candidate, 60*2.5*i)[:3],**hx_kwargs[i])[1] for i in range(576)]
    visibility = candidate_elevation >= np.radians(15)
    min_vis = min(sum(visibility[:144]), sum(visibility[144:288]), sum(visibility[288:432]), sum(visibility[432:]))
    if min_vis > 0:
        candidates.append(candidate)

arr = np.copy(candidates)

np.save('sample_orbits.npy', arr)
