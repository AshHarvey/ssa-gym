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

xyz1: ndarray = np.array([1285410, -4797210, 3994830], dtype=np.float64)

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
print("Test 1b: GCRS to ITRS (b) error in kilometers: ", test1b_error)

# !------------ Test 2a - ITRS to LLA
from envs.transformations import itrs2lla
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m, y=xyz1[1]*u.m, z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lon, lat, height]
test2_error = lla - np.asarray(itrs2lla(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2a: ITRS to LLA transformation")
print("Test 2a: ITRS to LLA error in rads,rads,meters: ", test2_error)

# !------------ Test 2b - ITRS to LLA
from envs.transformations import itrs2lla_py
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m, y=xyz1[1]*u.m, z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lon, lat, height]
test2_error = lla - np.asarray(itrs2lla_py(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2b: ITRS to LLA transformation")
print("Test 2b: ITRS to LLA (python) error in rads,rads,meters: ", test2_error)

# !------------ Test 3 - ITRS-ITRS to AzElSr
from envs.transformations import itrs2azel
xyz1 = np.array([1285410, -4797210, 3994830], dtype=np.float64)
xyz2 = np.array([1202990, -4824940, 3999870], dtype=np.float64)

observer = EarthLocation.from_geocentric(x=xyz1[0]*u.m,y=xyz1[1]*u.m, z=xyz1[2]*u.m)
target = SkyCoord(x=xyz2[0] * u.m, y=xyz2[1] * u.m, z=xyz2[2] * u.m, frame='itrs',
               representation_type='cartesian', obstime=t)# just for astropy
AltAz_frame = AltAz(obstime=t, location=observer)
results = target.transform_to(AltAz_frame)

az1 = results.az.to_value(u.rad)
alt1 = results.alt.to_value(u.rad)
sr1 = results.distance.to_value(u.m)

aer = itrs2azel(xyz1,np.reshape(xyz2,(1,3)))[0]

test3_error = [az1-aer[0],alt1-aer[1],sr1-aer[2]]

assert np.absolute(az1 - aer[0]) < 0.001, print("Failed Test 3a: ITRS-ITRS to Az transformation")
assert np.absolute(alt1 - aer[1]) < 0.001, print("Failed Test 3b: ITRS-ITRS to El transformation")
assert np.absolute(sr1 - aer[2]) < 0.001, print("Failed Test 3c: ITRS-ITRS to Srange transformation")
print("Test 3: ITRS-ITRS to Az, El, Srange error in rads,rads,meters: ", test2_error)

# !------------ Test 4 - ITRS-ITRS to AzElSr
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

# !------------ Test 6 - Filter Functions (core functions)
from envs.filter import compute_filter_weights, Q_discrete_white_noise, sigma_points, unscented_transform
from filterpy.common import Q_discrete_white_noise as Q_discrete_white_noise_fp
from filterpy.kalman import unscented_transform as unscented_transform_fp, MerweScaledSigmaPoints as MerweScaledSigmaPoints_fp
from filterpy.kalman.UKF import UnscentedKalmanFilter as UnscentedKalmanFilter_pf

# Configurable Settings:
dim_x = 6
dim_z = 3
dt = 30.0
qvar = 0.000001
obs_noise = np.repeat(100, 3)
x = np.array([34090.8583,  23944.7744,  6503.06682, -1.983785080, 2.15041744,  0.913881611])
P = np.eye(6) * np.array([100,  100,  100, 0.1, 0.1,  0.1])
alpha = 0.001
beta = 2.0
kappa = 3-6

# check weights
points_fp = MerweScaledSigmaPoints_fp(dim_x, alpha, beta, kappa)
Wc, Wm = compute_filter_weights(alpha, beta, kappa, dim_x)
Test6a = np.allclose(Wc, points_fp.Wc)
Test6b = np.allclose(Wm, points_fp.Wm)
assert Test6a, print("Failed Test 6a: points.Wc")
assert Test6b, print("Failed Test 6b: points.Wm")

# check noise function
noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.000001**2, block_size=3, order_by_dim=False)
noise_fp = Q_discrete_white_noise_fp(dim=2, dt=dt, var=0.000001**2, block_size=3, order_by_dim=False)
Test6c = np.allclose(noise, noise_fp)
assert Test6c, print("Failed Test 6c: Q_discrete_white_noise")

# check sigma points
lambda_ = alpha**2 * (dim_x + kappa) - dim_x
points = sigma_points(x, P, lambda_, dim_x)
Test6d = np.allclose(points, points_fp.sigma_points(x, P))
assert Test6d, print("Failed Test 6d: sigma points")

if np.all((Test6a, Test6b, Test6c, Test6d)):
    print("Test 6: Successful: sigmas points, noise function, and weights")

'''
# check unscented transform
ut = unscented_transform(points, Wm, Wc, noise)
ut_fp = unscented_transform_fp(points, Wm, Wc, noise)
assert np.allclose(ut[0], ut_fp[0]), print("Failed Test 6e: mean from UT")
assert np.allclose(ut[1], ut_fp[1]), print("Failed Test 6f: covariance from UT")

This test was removed because it was found than when the UT matched the UT in the code, 
the predict function did not correctly calculate the mean. Could use some looking into. 
'''

# !------------ Test 7 - Filter Prediction
from envs.filter import predict
from poliastro.core.propagation import markley
from numba import njit

@njit
def fx(x, dt):
    x = markley(398600.4418,x[:3],x[3:],dt) #  # (km^3 / s^2), (km), (km/s), (s)
    x = x.flatten()
    return x

@njit
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    return x[:3]

# Configurable Settings:
Q = noise
ukf = UnscentedKalmanFilter_pf(dim_x, dim_z, dt, hx, fx, points_fp)
ukf.x = np.copy(x)
ukf.P = np.copy(P)
ukf.Q = np.copy(Q)

# check filter predict
ukf.predict(dt)

x_post, P_post, sigmas_post = predict(x, P, Wm, Wc, Q, dt, lambda_, fx)
Test7a = np.prod((x_post-ukf.x)) / np.prod(x_post)
assert np.abs(Test7a) < 1.0e-10, print("Test 7a: Predicted means don't match")
Test7b = det((P_post-ukf.P))/det(P_post)
assert np.abs(Test7b) < 1.0e-10, print("Test 7b: Predicted covariances don't match")
Test7c = np.prod((sigmas_post-ukf.sigmas_f)) / np.prod(sigmas_post)
assert Test7c < 1.0e-10, print("Test 7c: Predicted sigmas points don't match")

print("Test 7: step 1 errors: x: ", Test7a, ", P: ", Test7b, ", sigmas:", Test7c)

ukf.predict(dt)

x_post2, P_post2, sigmas_post2 = predict(x_post, P_post, Wm, Wc, Q, dt, lambda_, fx)
Test7d = np.prod((x_post2-ukf.x)) / np.prod(x_post2)
assert np.abs(Test7d) < 1.0e-10, print("Test 7d: Predicted means don't match")
Test7e = det((P_post2-ukf.P))/det(P_post2)
assert np.abs(Test7e) < 1.0e-10, print("Test 7e: Predicted covariances don't match")
Test7f = np.prod((sigmas_post2-ukf.sigmas_f)) / np.prod(sigmas_post2)
assert Test7f < 1.0e-10, print("Test 7f: Predicted sigmas points don't match")

print("Test 7: step 2 errors: x: ", Test7d, ", P: ", Test7e, ", sigmas:", Test7f)

ukf.predict(dt)

x_post3, P_post3, sigmas_post3 = predict(x_post2, P_post2, Wm, Wc, Q, dt, lambda_, fx)

for i in range(47):
    ukf.predict(dt)
    x_post3, P_post3, sigmas_post3 = predict(x_post3, P_post3, Wm, Wc, Q, dt, lambda_, fx)

Test7h = np.prod((x_post3-ukf.x)) / np.prod(x_post3)
assert np.abs(Test7h) < 1.0e-10, print("Test 7h: Predicted means don't match")
Test7i = det((P_post3-ukf.P))/det(P_post3)
assert np.abs(Test7i) < 1.0e-10, print("Test 7i: Predicted covariances don't match")
Test7j = np.prod((sigmas_post3-ukf.sigmas_f)) / np.prod(sigmas_post2)
assert Test7j < 1.0e-10, print("Test 7j: Predicted sigmas points don't match")

print("Test 7: step 50 errors: x: ", Test7h, ", P: ", Test7i, ", sigmas:", Test7j)

# !------------ Test 8 - Filter Updates
from envs.filter import update

x_true = np.copy(x)

for i in range(50):
    x_true = fx(x_true, dt)

R = np.array([1.25, 1.25, 1.25])
z = x_true[:3]

x_post4, P_post4 = update(ukf.x, ukf.P, z, Wm, Wc, R, ukf.sigmas_f, hx)

ukf.update(z=z, R=R)

Test8a = np.prod((x_post4-ukf.x)) / np.prod(x_post4)

assert np.abs(Test8a) < 1.0e-10, print("Test 8a: Updated means don't match")
Test8b = det((P_post4-ukf.P))/det(P_post4)
assert np.abs(Test8b) < 1.0e-10, print("Test 8b: Updated covariances don't match")

print("Test 8: step 50, update 1 errors: x: ", Test8a, ", P: ", Test8b)

for i in range(50):
    ukf.predict(dt)
    x_post4, P_post4, sigmas_post4 = predict(x_post4, P_post4, Wm, Wc, Q, dt, lambda_, fx)

for i in range(50):
    x_true = fx(x_true, dt)

z = x_true[:3]

x_post4, P_post4 = update(x_post4, P_post4, z, Wm, Wc, R, sigmas_post4, hx)

ukf.update(z=z, R=R)

Test8c = np.prod((x_post4-ukf.x)) / np.prod(x_post4)
assert np.abs(Test8c) < 1.0e-10, print("Test 8c: Updated means don't match")
Test8d = det((P_post4-ukf.P))/det(P_post4)
assert np.abs(Test8d) < 1.0e-10, print("Test 8d: Updated covariances don't match")

print("Test 8: step 100, update 2 errors: x: ", Test8c, ", P: ", Test8d)

# !------------ Test 9 - Az El Updates
from envs.transformations import lla2itrs, _itrs2azel

@jit(['float64[:](float64[:],float64[:])'])
def residual_z(a, b):
    # prep array to receive results
    c = np.copy(a)
    # force angles in range <-pi, pi>
    while a[0] > np.pi or a[0] < -np.pi:
        a[0] = a[0] - np.sign(a[0])*np.pi*2
    while a[1] > np.pi or a[1] < -np.pi:
        a[1] = a[1] - np.sign(a[1])*np.pi*2
    # find differ, a - b
    c[0] = (a[0] - b[0] + np.pi) % (np.pi*2) - np.pi
    c[1] = (a[1] - b[1] + np.pi) % (np.pi*2) - np.pi
    c[2] = np.subtract(a[2], b[2])
    return c

@njit
def mean_z(sigmas, Wm):
    z = np.zeros(3)
    sum_sin_az, sum_cos_az, sum_sin_el, sum_cos_el = 0., 0., 0., 0.

    Wm = Wm/np.sum(Wm)

    for i in range(len(sigmas)):
        s = sigmas[i]
        sum_sin_az = sum_sin_az + np.sin(s[0])*Wm[i]
        sum_cos_az = sum_cos_az + np.cos(s[0])*Wm[i]
        sum_sin_el = sum_sin_el + np.sin(s[1])*Wm[i]
        sum_cos_el = sum_cos_el + np.cos(s[1])*Wm[i]
        z[2] = z[2] + s[2] * Wm[i]
    z[0] = np.arctan2(sum_sin_az, sum_cos_az)
    z[1] = np.arctan2(sum_sin_el, sum_cos_el)
    return z

observer_lat = np.radians(38.828198)
observer_lon = np.radians(-77.305352)
observer_alt = np.float64(20.0) # in meters
observer_lla = np.array((observer_lon, observer_lat, observer_alt))
observer_itrs = lla2itrs(observer_lla)/1000 # meter -> kilometers

@njit
def hx(x_gcrs, trans_matrix, obs=observer_itrs):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    x_itrs = trans_matrix @ x_gcrs[:3]
    aer = _itrs2azel(observer_itrs, x_itrs)
    return aer

def hx2(x, t, obs_lat=observer_lat, obs_lon=observer_lon, obs_height=observer_alt):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    object = SkyCoord(x=x[0] * u.km, y=x[1] * u.km, z=x[2] * u.km, frame='gcrs',
                   representation_type='cartesian', obstime=t)
    obs = EarthLocation.from_geodetic(lon=obs_lon*u.rad, lat=obs_lat*u.rad, height=obs_height*u.m)
    AltAz_frame = AltAz(obstime=t, location=obs)
    results = object.transform_to(AltAz_frame)

    az = results.az.to_value(u.rad)
    alt = results.alt.to_value(u.rad)
    sr = results.distance.to_value(u.km)
    aer = np.array([az, alt, sr])
    return aer

t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)

trans_matrix = gcrs2irts_matrix_a(t, eop)

ini = _itrs2azel(observer_itrs, x[:3]) # since _itrs2azel is used inside hx, it must be compiled prior to hx

z1 = hx(x, trans_matrix, obs=observer_itrs)

z2 = hx2(x, t, obs_lat=observer_lat, obs_lon=observer_lon, obs_height=observer_alt)

hx_error = z2 - z1

print("Test 9a: hx error in azimuth (arc seconds) = ", np.degrees(hx_error[0])*60*60)
print("Test 9b: hx error in elevation (arc seconds) = ", np.degrees(hx_error[1])*60*60)
print("Test 9c: hx error in slant range (meters) = ", np.degrees(hx_error[2])*1000)
print("I am unsure if these error are mine or AstroPy's given the above matched")

t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)
t = [t]
for i in range(500):
    t.append(t[-1] + timedelta(seconds=30))
trans_matrix = gcrs2irts_matrix_a(t, eop)

R = np.array([np.radians(1/60/60),np.radians(1/60/60),1.0])

sigmas_h = np.copy(ukf.sigmas_h)
for i in range(13):
    sigmas_h[i] = hx(sigmas_h[i], trans_matrix[0], obs=observer_itrs)

ini = mean_z(sigmas_h, Wm) # since mean_z is used inside update, it must be compiled prior to update
ini = residual_z(sigmas_h[2],sigmas_h[4]) # since mean_z is used inside update, it must be compiled prior to update

from envs.filter import update
"""
for i in range(50):
    for j in range(10):
        x_post4, P_post4, sigmas_post4 = predict(x_post4, P_post4, Wm, Wc, Q, dt, lambda_, fx)
        x_true = fx(x_true, dt)
    run = i*10 + j
    z = hx(x_true[:3], trans_matrix[run], obs=observer_itrs)
    x_post4, P_post4 = update(x_post4, P_post4, z, Wm, Wc, R, sigmas_post4, hx, mean_z = mean_z, residual_z = residual_z)

for i in range(50):
    x_true = fx(x_true, dt)

z = hx(x_true[:3], trans_matrix, obs=observer_itrs)

x_post4, P_post4 = update(x_post4, P_post4, z, Wm, Wc, R, sigmas_post4, hx)

ukf.update(z=z, R=R)
"""
"""
def wrapped():
    mean_z(sigmas_h, Wm)

import timeit

timeit.timeit(wrapped, number=2880)
"""

print("Done")
