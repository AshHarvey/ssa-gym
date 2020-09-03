from astropy.coordinates import SkyCoord, ITRS, EarthLocation, AltAz
from astropy import units as u
import numpy as np
from datetime import datetime, timedelta
import time
from numpy.core._multiarray_umath import ndarray
from scipy.spatial import distance
from scipy.linalg import det

print("Running Test Cases...")

# !------------ Test 1 - GCRS to ITRS
from envs.transformations import gcrs2irts_matrix_a, get_eops, gcrs2irts_matrix_b

eop = get_eops()

xyz1: ndarray = np.array([1285410, -4797210, 3994830], dtype=np.float64) # [meters x 3]

t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)
object = SkyCoord(x=xyz1[0] * u.m, y=xyz1[1] * u.m, z=xyz1[2] * u.m, frame='gcrs',
               representation_type='cartesian', obstime=t)# just for astropy
itrs = object.transform_to(ITRS)

test1a_error = distance.euclidean(itrs.cartesian.xyz._to_value(u.m),
                                 gcrs2irts_matrix_a(t, eop) @ xyz1)
test1b_error = distance.euclidean(itrs.cartesian.xyz._to_value(u.m),
                                  gcrs2irts_matrix_b(t, eop) @ xyz1)
assert test1a_error < 25, print("Failed Test 1: GCRS to ITRS transformation")
print("Test 1a: GCRS to ITRS (a) error in meters: ", test1a_error)
print("Test 1b: GCRS to ITRS (b) error in meters: ", test1b_error)

# !------------ Test 2a - ECEF (ITRS) to LLA
from envs.transformations import ecef2lla
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m, y=xyz1[1]*u.m, z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lat, lon, height]
test2_error = lla - np.asarray(ecef2lla(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2a: ECEF (ITRS) to LLA transformation")
print("Test 2a: ECEF (ITRS) to LLA error in rads,rads,meters: ", test2_error)

# !------------ Test 3 - ECEF (ITRS) to Az, El, Range
from envs.transformations import ecef2aer, ecef2lla
xyz1 = np.array([1285410, -4797210, 3994830], dtype=np.float64) # [meters x 3]
xyz2 = np.array([1202990, -4824940, 3999870], dtype=np.float64) # [meters x 3]

observer = EarthLocation.from_geocentric(x=xyz1[0]*u.m,y=xyz1[1]*u.m, z=xyz1[2]*u.m)
target = SkyCoord(x=xyz2[0] * u.m, y=xyz2[1] * u.m, z=xyz2[2] * u.m, frame='itrs',
                  representation_type='cartesian', obstime=t) # just for astropy
AltAz_frame = AltAz(obstime=t, location=observer)
results = target.transform_to(AltAz_frame)

az1 = results.az.to_value(u.rad)
alt1 = results.alt.to_value(u.rad)
sr1 = results.distance.to_value(u.m)

aer = ecef2aer(ecef2lla(xyz1), xyz2, xyz1)

test3_error = [az1-aer[0], alt1-aer[1], sr1-aer[2]]

assert np.absolute(az1 - aer[0]) < 0.001, print("Failed Test 3a: ECEF (ITRS) to Az transformation")
assert np.absolute(alt1 - aer[1]) < 0.001, print("Failed Test 3b: ECEF (ITRS) to El transformation")
assert np.absolute(sr1 - aer[2]) < 0.001, print("Failed Test 3c: ECEF (ITRS) to to Srange transformation")
print("Test 3: ECEF (ITRS) to Az, El, Range error in rads,rads,meters: ", test3_error)

# !------------ Test 4 - Time to generate cel2ter transformation
t=datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)
t = [t]

start = time.time()
for i in range(2880):
    t.append(t[-1]+timedelta(seconds=30))
gcrs2irts_matrix_a(t, eop)
end = time.time()
print("Test 4a: Time to generate cel2ter transformation matrix with gcrs2irts_matrix_a for every 30 seconds for an entire day: ", end-start, " seconds")

start = time.time()
for i in range(2880):
    t.append(t[-1]+timedelta(seconds=30))
gcrs2irts_matrix_b(t, eop)
end = time.time()
print("Test 4b: Time to generate cel2ter transformation matrix with gcrs2irts_matrix_b for every 30 seconds for an entire day: ", end-start, " seconds")

# !------------ Test 5 - ITRS to GCRS SOFA cases
from envs.transformations import gcrs2irts_matrix_a, gcrs2irts_matrix_b

t = datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)

Cel2Ter94 = np.asarray([[+0.973104317712772, +0.230363826174782, -0.000703163477127],
                        [-0.230363800391868, +0.973104570648022, +0.000118545116892],
                        [+0.000711560100206, +0.000046626645796, +0.999999745754058]])

Cel2Ter00aCIO = np.asarray([[+0.973104317697512, +0.230363826239227, -0.000703163482268],
                           [-0.230363800456136, +0.973104570632777, +0.000118545366806],
                           [+0.000711560162777, +0.000046626403835, +0.999999745754024]])

Cel2Ter00aEB = np.asarray([[+0.973104317697618, +0.230363826238780, -0.000703163482352],
                           [-0.230363800455689, +0.973104570632883, +0.000118545366826],
                           [+0.000711560162864, +0.000046626403835, +0.999999745754024]])

Cel2Ter06aCA = np.asarray([[+0.973104317697535, +0.230363826239128, -0.000703163482198],
                           [-0.230363800456037, +0.973104570632801, +0.000118545366625],
                           [+0.000711560162668, +0.000046626403995, +0.999999745754024]])

Cel2Ter06aXY = np.asarray([[+0.973104317697536, +0.230363826239128, -0.000703163481769],
                           [-0.230363800456036, +0.973104570632801, +0.000118545368117],
                           [+0.000711560162594, +0.000046626402444, +0.999999745754024]])

print("Test 5a: Cel2Ter06aXY vs Cel2Ter94, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter94))
print("Test 5b: Cel2Ter06aXY vs Cel2Ter00aCIO, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter00aCIO))
print("Test 5c: Cel2Ter06aXY vs Cel2Ter00aEB, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter00aEB))
print("Test 5d: Cel2Ter06aXY vs Cel2Ter06aCA, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter06aCA))
print("Test 5e: Cel2Ter06aXY vs gcrs2irts_matrix, magnitude of error: ", det(Cel2Ter06aXY - gcrs2irts_matrix_a(t, eop)))
print("Test 5f: Cel2Ter06aXY vs utc2cel06acio, magnitude of error: ", det(Cel2Ter06aXY - gcrs2irts_matrix_b(t, eop)))

# !------------ Test 6 - Filter Prediction
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter
from envs.dynamics import fx_xyz_farnocchia as fx, hx_xyz as hx

# Configurable Settings:
dim_x = 6
dim_z = 3
dt = 30.0
qvar = 0.000001
obs_noise = np.repeat(100, 3)
x = np.array([34090858.3,  23944774.4,  6503066.82, -1983.785080, 2150.41744,  913.881611]) # [meters x 3, meters / second x3]
P = np.eye(6) * np.array([1000,  1000,  1000, 1, 1,  1])
alpha = 0.001
beta = 2.0
kappa = 3-6

# check weights
points = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.000001**2, block_size=3, order_by_dim=False)


# Configurable Settings:
Q = noise
ukf = UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points)
ukf.x = np.copy(x)
ukf.P = np.copy(P)
ukf.Q = np.copy(Q)
x_true = np.copy(x)

# check filter predict
for i in range(50):
    ukf.predict(dt)
    x_true = fx(x_true, dt)

test6a = np.sqrt(np.sum((ukf.x - x_true)[:3]**2))
test6b = np.sqrt(np.sum((ukf.x - x_true)[3:]**2))
assert test6a < 1.0e-0, print("Test 6a: Prediction off by more than 1 meter")
assert test6b < 1.0e-4, print("Test 6b: Prediction off by more than 1.0e-4 meter/second")

print("Test 6: step 50 errors: pos: ", test6a, " (meters), vel: ", test6b, " (meters / second)")

# !------------ Test 7 - Filter Updates
R = np.array([125, 125, 125])
z = x_true[:3]

ukf.update(z=z, R=R)

test7a = np.sqrt(np.sum((ukf.x - x_true)[:3]**2))
test7b = np.sqrt(np.sum((ukf.x - x_true)[3:]**2))
assert test7a < 1.0e-2, print("Test 7a: Prediction off by more than 1.0e-2 meter")
assert test7b < 1.0e-2, print("Test 7b: Prediction off by more than 1.0e-2 meter/second")

print("Test 7: step 50, 1 update errors: pos: ", test7a, " (meters), vel: ", test7b, " (meters / second)")
print("The update improves the location accuracy, but the velocity was not observed and the update degrades its estimate")

for i in range(50):
    ukf.predict(dt)
    x_true = fx(x_true, dt)

z = x_true[:3]

ukf.update(z=z, R=R)

test7c = np.sqrt(np.sum((ukf.x - x_true)[:3]**2))
test7d = np.sqrt(np.sum((ukf.x - x_true)[3:]**2))
assert test7c < 1.0e-2, print("Test 7c: Prediction off by more than 1.0e-2 meter")
assert test7d < 1.0e-2, print("Test 7d: Prediction off by more than 1.0e-2 meter/second")

print("Test 7: step 100, 2 update errors: pos: ", test7c, " (meters), vel: ", test7d, " (meters / second)")

# !------------ Test 8 - Deprecated
print("Test 8 Deprecated")

# !------------ Test 9 - Az El means and residuals
from envs.transformations import lla2ecef
from envs.dynamics import residual_z_aer as residual_z, mean_z_uvw as mean_z

residual_az_cases = [0, 0.001, 90.0, 180, 270.0, 359.99, 360]
residual_az_cases = np.radians(residual_az_cases)
residual_el_cases = [-90.00, -89.99, -0.999, 0, 0.999, 89.99, 90.00]
residual_el_cases = np.radians(residual_el_cases)
residual_sr_cases = [-1000.0001, -1, -0.0001, 0, 0.0001, 1, 1000.0001]

from itertools import permutations
residual_az_cases2 = list(permutations(residual_az_cases, 2))
residual_el_cases2 = list(permutations(residual_el_cases, 2))
residual_sr_cases2 = list(permutations(residual_sr_cases, 2))

diffs = []
for az, el, sr in zip(residual_az_cases2, residual_el_cases2, residual_sr_cases2):
    aer0 = np.asarray([az[0], el[0], sr[0]])
    aer1 = np.asarray([az[1], el[1], sr[1]])
    diff = residual_z(aer0, aer1)
    az = np.round(az, 4)
    el = np.round(el, 4)
    sr = np.round(sr, 4)
    # print(np.round(az[0],4), " - ", np.round(az[1],4), " = ", np.round(diff[0],4))
    # print(np.round(el[0],4), " - ", np.round(el[1],4), " = ", np.round(diff[1],4))
    # print(np.round(sr[0],4), " - ", np.round(sr[1],4), " = ", np.round(diff[2],4))
    diffs.append(diff)

diffs = np.array(diffs)
test9a1 = np.min(diffs, axis=0) == [-np.pi, -np.pi, -2000.0002]
test9a2 = np.max(diffs, axis=0) == [np.pi, np.pi, 2000.0002]

if np.all([test9a1, test9a2]):
    print("Test 9a: residuals_z successful")
else:
    print("Test 9a: residuals_z failed")

noise = np.random.normal(scale=5000, size=(13*2880, 3))
observer_lla = np.array([np.radians(0), np.radians(0), 0])
observer_itrs = lla2ecef(observer_lla)
sat_samples = lla2ecef(np.array([0, 0, 20000])) + noise
sat_mean = np.mean(sat_samples, axis=0)

obs_samples = np.array([ecef2aer(observer_lla, sat_sample, observer_itrs) for sat_sample in sat_samples])
obs_mean = ecef2aer(observer_lla, sat_mean, observer_itrs)
obs_mean_calc = mean_z(obs_samples, Wm=np.repeat(1/obs_samples.shape[0], obs_samples.shape[0]))

if np.all(np.subtract(obs_mean, obs_mean_calc) < 1e-7):
    print("Test 9b: mean_z test 2 successful")
else:
    print("Test 9b: mean_z test 2 failed")

# !------------ Test 10 - Az El Measurement Function
from envs.transformations import lla2ecef
from envs.dynamics import hx_aer_erfa as hx, hx_aer_astropy as hx2

x = np.array([3.40908583e+07, 2.39447744e+07, 6.50306682e+06, -1.98378508e+03, 2.15041744e+03, 9.13881611e+02])

observer_lat = np.radians(38.828198)
observer_lon = np.radians(-77.305352)
observer_alt = np.float64(20.0) # in meters
observer_lla = np.array([observer_lat, observer_lon, observer_alt])
observer_itrs = lla2ecef(observer_lla) # meter

t = datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)

trans_matrix = gcrs2irts_matrix_b(t, eop)

hx_args1 = (trans_matrix, observer_lla, observer_itrs)
hx_args2 = (t, observer_lla, None, observer_itrs)

z1 = hx(x, *hx_args1)

z2 = hx2(x, *hx_args2)

hx_error = z2 - z1

print("Test 10a: hx error in azimuth (arc seconds) = ", np.degrees(hx_error[0])*60*60)
print("Test 10b: hx error in elevation (arc seconds) = ", np.degrees(hx_error[1])*60*60)
print("Test 10c: hx error in slant range (meters) = ", hx_error[2])
# resolved Issue opened with AstroPy at https://github.com/astropy/astropy/issues/10407

# !------------ Test 11 - Az El Updates
from filterpy.kalman import MerweScaledSigmaPoints as MerweScaledSigmaPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UnscentedKalmanFilter
from envs.dynamics import residual_xyz as residual_x, residual_z_aer as residual_z, mean_z_enu as mean_z
from envs.dynamics import fx_xyz_markley as fx, hx_aer_erfa as hx
from envs.transformations import lla2ecef, gcrs2irts_matrix_a as gcrs2irts_matrix, get_eops
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial import distance

arcsec2rad = np.pi/648000

# Sim Configurable Setting:
max_step = 500
obs_interval = 10
observer = (38.828198, -77.305352, 20.0) # lat (deg), lon (deg), height (meters)
x_init = (34090858.3,  23944774.4,  6503066.82, -1983.785080, 2150.41744,  913.881611) # 3 x m, 3 x m/s
t_init = datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)
eop = get_eops()

# Filter Configurable Settings:
dim_x = 6
dim_z = 3
dt = 30.0
R = np.diag([1, 1, 1000])
Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.0001**2, block_size=3, order_by_dim=False)
P = np.diag([1000,  1000,  1000, 1, 1, 1])
alpha = 0.001
beta = 2.0
kappa = 3-6

# Derived Settings:
lambda_ = alpha**2 * (dim_x + kappa) - dim_x
points = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
x_true = np.copy(x_init)
t = [t_init]
step = [0]
observer_lla = np.array([np.radians(observer[0]), np.radians(observer[1]), observer[2]])
observer_itrs = lla2ecef(observer_lla) # meter

# FilterPy Configurable Settings:
ukf = UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points)
ukf.x = np.copy(x)
ukf.P = np.copy(P)
ukf.Q = np.copy(Q)
ukf.R = np.copy(R*[arcsec2rad, arcsec2rad, 1])
ukf.residual_z = residual_z
ukf.z_mean = mean_z

fault = False
# run the filter
for i in range(max_step):
    # step truth forward
    step.append(step[-1] + 1)
    t.append(t[-1] + timedelta(seconds=dt))
    x_true = fx(x_true, dt)
    # step FilterPy forward
    ukf.predict(dt)

    # Check if obs should be taken
    if step[-1] % obs_interval == 0:
        # get obs:
        trans_matrix = gcrs2irts_matrix(t[-1], eop)
        hx_kwargs = {"trans_matrix": trans_matrix, "observer_lla": observer_lla, "observer_itrs": observer_itrs}
        z_true = hx(x_true, **hx_kwargs)
        # update FilterPy
        ukf.update(z_true, **hx_kwargs)

test11a_error = [distance.euclidean(ukf.x[:3], x_true[:3]), distance.euclidean(ukf.x[3:], x_true[3:])]

print("Test 11a: step ", max_step, " error after ", int(max_step/obs_interval), " updates (FilterPy): ",
      np.round(test11a_error[0], 4), " meters, ", np.round(test11a_error[1], 4), " meters per second")

ukf.x = np.copy(x_init)
ukf.P = np.copy(P)

def time_filterpy_predict():
    global ukf
    ukf.predict()

import timeit

print("Test 11b: Time to complete 500 steps using FilterPy prediction: ", np.round(timeit.timeit(time_filterpy_predict, number=500), 4), " seconds")

# !------------ Test 12 - Az El Updates with Uncertainty
# Sim Configurable Setting:
max_step = 2880
obs_interval = 50
observer = (38.828198, -77.305352, 20.0) # lat (deg), lon (deg), height (meters)
x_init = (34090858.3,  23944774.4,  6503066.82, -1983.785080, 2150.41744,  913.881611) # 3 x m, 3 x m/s
t_init = datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)
eop = get_eops()

# Filter Configurable Settings:
dim_x = 6
dim_z = 3
dt = 30.0
R = np.diag([1, 1, 1000])
Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.0001**2, block_size=3, order_by_dim=False)
x = np.array([34090858.3,  23944774.4,  6503066.82, -1983.785080, 2150.41744,  913.881611])
P = np.diag([1000,  1000,  1000, 1, 1,  1])
alpha = 0.001
beta = 2.0
kappa = 3-6

# Derived Settings:
lambda_ = alpha**2 * (dim_x + kappa) - dim_x
points = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
x_true = np.copy(x)
x_filter = np.copy(x) + np.random.normal(0,np.sqrt(np.diag(P)))
t = [t_init]
step = [0]
observer_lla = np.array([np.radians(observer[0]), np.radians(observer[1]), observer[2]])
observer_itrs = lla2ecef(observer_lla)/1000 # meter -> kilometers

# FilterPy Configurable Settings:
ukf = UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points)
ukf.x = np.copy(x_filter)
ukf.P = np.copy(P)
ukf.Q = np.copy(Q)
ukf.R = np.copy(R)
ukf.residual_z = residual_z
ukf.z_mean = mean_z

fault = False
# run the filter
for i in range(max_step):
    # step truth forward
    step.append(step[-1] + 1)
    t.append(t[-1] + timedelta(seconds=dt))
    x_true = fx(x_true, dt)
    # step FilterPy forward
    ukf.predict(dt)
    # Check if obs should be taken
    if step[-1] % obs_interval == 0:
        # get obs:
        trans_matrix = gcrs2irts_matrix(t[-1], eop)
        hx_kwargs = {"trans_matrix": trans_matrix, "observer_lla": observer_lla, "observer_itrs": observer_itrs}
        z_true = hx(x_true, **hx_kwargs)
        z_noise = np.random.normal(0, np.sqrt(np.diag(R))) * [arcsec2rad, arcsec2rad, 1]
        z_filter = z_true + z_noise
        # update FilterPy
        ukf.update(z_filter, **hx_kwargs)


test12a_error = [distance.euclidean(ukf.x[:3],x_true[:3]), distance.euclidean(ukf.x[3:],x_true[3:])]
print("Test 12a: step ", max_step, " error after ", int(max_step/obs_interval), " updates (FilterPy with noise): ",
      np.round(test12a_error[0],4), " meters, ", np.round(test12a_error[1],4), " meters per second")
print("Test 12a: step ", max_step, " uncertainty after ", int(max_step/obs_interval), " updates (FilterPy with noise): ",
      np.round(np.sqrt(np.sum(np.diag(ukf.P)[:3])),4), " meters, ", np.round(np.sqrt(np.sum(np.diag(ukf.P)[3:])),4), " meters per second")

# !------------ Test 13 - FilterPy Az El Updates with Uncertainty and Multiple RSOs
print("Test 13 Deprecated")
# !------------ Test 14 - simple env v2 - 20 objects, no viz limits, xyz measurements

import gym
import numpy as np
from tqdm import tqdm
from envs.dynamics import hx_xyz, mean_xyz
from agents import agent_visible_random
from envs import env_config

env_config['rso_count'] = 20
env_config['obs_limit'] = -90
env_config['obs_type'] = 'xyz'
env_config['z_sigma'] = tuple([5e2]*3)
env_config['x_sigma'] = tuple([1e5]*3+[1e2]*3)
env_config['P_O'] = np.diag(([1e5**2]*3 + [1e2**2]*3))
env_config['R'] = np.diag([5e2**2]*3)
env_config['hx'] = hx_xyz
env_config['mean_z'] = mean_xyz
env_config['residual_z'] = np.subtract
env_config['obs_returned'] = '2darray'
env_config['reward_type'] = 'trinary'

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})
env.seed(1)
obs = env.reset()
agent = agent_visible_random

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = agent(obs, env)

        obs, reward, done, _ = env.step(action)

print('Test 14 mean reward: ' + str(np.round(np.mean(env.rewards), 4)))
print('Test 14 fitness tests: 20 objects, no viz limits, xyz measurements')
print(env.fitness_test()) # TODO: Test 4 NEES needs fixing

print("Test 15: Plots...")
env.plot_sigma_delta(save_path=None)
env.plot_visibility(save_path=None)
env.plot_anees(save_path=None) # TODO: needs fixing
env.plot_autocorrelation(save_path=None)
env.plot_map()
env.plot_actions(save_path=None)
env.plot_rewards()
env.plot_innovation_bounds(save_path=None)
env.plot_regimes(save_path=None)
env.plot_NIS(save_path=None)


# !------------ Test 16 - simple env v2 - 5 objects, 15 degree viz limits, xyz measurements
import gym
import numpy as np
from tqdm import tqdm
from envs.dynamics import hx_xyz, mean_xyz
from agents import agent_visible_random
from envs import env_config

env_config['rso_count'] = 20
env_config['obs_limit'] = 15
env_config['obs_type'] = 'xyz'
env_config['z_sigma'] = tuple([5e2]*3)
env_config['x_sigma'] = tuple([1e5]*3+[1e2]*3)
env_config['P_O'] = np.diag(([1e5**2]*3 + [1e2**2]*3))
env_config['R'] = np.diag([5e2**2]*3)
env_config['hx'] = hx_xyz
env_config['mean_z'] = mean_xyz
env_config['residual_z'] = np.subtract
env_config['obs_returned'] = '2darray'
env_config['reward_type'] = 'trinary'

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})
env.seed(1)
obs = env.reset()
agent = agent_visible_random

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = agent(obs, env)

        obs, reward, done, _ = env.step(action)

print('Test 16 mean reward: ' + str(np.round(np.mean(env.rewards), 4)))
print('Test 16 fitness tests: 20 objects, 15 degree viz limits, xyz measurements')
print(env.fitness_test()) # TODO: Test 4 NEES needs fixing

# !------------ Test 17 - simple env v2 - 5 objects, no viz limits, aer measurements
import gym
import numpy as np
from tqdm import tqdm
from agents import agent_visible_random
from envs import env_config
env_config['obs_returned'] = '2darray'
env_config['reward_type'] = 'trinary'

env_config['rso_count'] = 5
env_config['obs_limit'] = -90

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})
env.seed(1)
obs = env.reset()
agent = agent_visible_random

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = agent(obs, env)

        obs, reward, done, _ = env.step(action)

print('Test 17 mean reward: ' + str(np.round(np.mean(env.rewards), 4)))
print('Test 17 fitness tests: 5 objects, no viz limits, aer measurements')
print(env.fitness_test()) # TODO: Test 4 NEES needs fixing

# !------------ Test 18 - simple env v2 - 5 objects, 15 degree viz limits, aer measurements
import gym
import numpy as np
from tqdm import tqdm
from agents import agent_visible_random
from envs import env_config
env_config['obs_returned'] = '2darray'
env_config['reward_type'] = 'trinary'

env_config['rso_count'] = 5

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})
env.seed(1)
obs = env.reset()
agent = agent_visible_random

done = False
for i in tqdm(range(env.n)):
    if not done:
        action = agent(obs, env)

        obs, reward, done, _ = env.step(action)

print('Test 18 mean reward: ' + str(np.round(np.mean(env.rewards), 4)))
print('Test 18 fitness tests: 5 objects, 15 degree viz limits, aer measurements')
print(env.fitness_test()) # TODO: RuntimeWarning: Probably from a nan value in nees, may be related to above


# !------------ Test 19 - simple env v2 - 20 objects, 15 degree viz limits, aer measurements
import gym
import numpy as np
from tqdm import tqdm
from datetime import datetime
from envs.dynamics import fx_xyz_farnocchia as fx, hx_aer_erfa as hx, mean_z_uvw as mean_z, residual_z_aer as residual_z, robust_cholesky
from agents import agent_visible_greedy_aer, agent_visible_random, agent_visible_greedy, agent_visible_greedy_spoiled
from envs import env_config

env_config['rso_count'] = 20
env_config['steps'] = 480
env_config['obs_limit'] = 15
env_config['reward_type'] = 'trinary'
env_config['obs_returned'] = 'flatten'

env = gym.make('ssa_tasker_simple-v2', **{'config': env_config})
env.seed(0)
obs = env.reset()
agent = agent_visible_greedy # agent_visible_greedy # agent_visible_greedy_aer # agent_visible_greedy_aer # agent_visible_random

done = False
rewards = []
ts = 0
for i in tqdm(range(env.n)):
    if not done:
        ts += 1
        action = agent(obs, env)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

env.plot_sigma_delta()

print('Test 19 mean reward: ' + str(np.round(np.mean(env.rewards), 4)))
print('Test 19 fitness tests: 20 objects, 15 degree viz limits, aer measurements')
print(env.fitness_test())

print("Done")
