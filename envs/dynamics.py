import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from numba import njit, jit
from poliastro.core.propagation import markley

from envs.transformations import _itrs2azel


#@jit(['float64[::](float64[::],float64)'],nopython=True)
@njit
def fx_xyz_markley(x, dt):
    x = markley(398600.4418,x[:3],x[3:],dt) #  # (km^3 / s^2), (km), (km/s), (s)
    x = x.flatten()
    return x


#@jit(['float64[::](float64[::])'],nopython=True)
@njit
def hx_xyz(x):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    return x[:3]


@njit
def hx_aer_erfa(x_gcrs, *hx_args):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    trans_matrix = hx_args[0][0]
    observer_itrs = hx_args[0][1]
    x_itrs = trans_matrix @ x_gcrs[:3]
    aer = _itrs2azel(observer_itrs, x_itrs)
    return aer


def hx_aer_kwargs(x_gcrs, **kwargs):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    trans_matrix = kwargs["trans_matrix"]
    observer_itrs = kwargs["observer_itrs"]
    x_itrs = trans_matrix @ x_gcrs[:3]
    aer = _itrs2azel(observer_itrs, x_itrs)
    return aer


def hx_aer_astropy(x, *hx_args):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    t = hx_args[0][0]
    obs_lat = hx_args[0][1]
    obs_lon = hx_args[0][2]
    obs_height = hx_args[0][3]
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


@jit(['float64[::1](float64[::1],float64[::1])'])
def residual_z_aer(a, b):
    # prep array to receive results
    c = np.copy(a)
    # force angles in range <-pi, pi>
    while a[1] > np.pi or a[1] < -np.pi:
        a[0] = a[0] - np.sign(a[0])*np.pi*2
    while a[1] > np.pi or a[1] < -np.pi:
        a[1] = a[1] - np.sign(a[1])*np.pi*2
    # find differ, a - b
    c[0] = (a[0] - b[0] + np.pi) % (np.pi*2) - np.pi
    c[1] = (a[1] - b[1] + np.pi) % (np.pi*2) - np.pi
    c[2] = np.subtract(a[2], b[2])
    while c[0] > np.pi or c[0] < -np.pi:
        c[0] = c[0] - np.sign(c[0]) * np.pi * 2
    while c[1] > np.pi or c[1] < -np.pi:
        c[1] = c[1] - np.sign(c[1]) * np.pi * 2
    return c


@njit
def residual_xyz(a, b):
    c = np.subtract(a,b)
    return c


@njit
def mean_z_aer(sigmas, Wm):
    z = np.zeros(3)
    sum_sin_az, sum_cos_az, sum_sin_el, sum_cos_el = 0., 0., 0., 0.

    Wm = Wm/np.sum(Wm)

    for i in range(len(sigmas)):
        s = sigmas[i]
        sum_sin_az = sum_sin_az + np.sin(s[0])*Wm[i]
        sum_cos_az = sum_cos_az + np.cos(s[0])*Wm[i]
        sum_sin_el = sum_sin_el + np.sin(s[1]+np.pi)*Wm[i]
        sum_cos_el = sum_cos_el + np.cos(s[1]+np.pi)*Wm[i]
        z[2] = z[2] + s[2] * Wm[i]
    z[0] = np.arctan2(sum_sin_az, sum_cos_az)
    z[1] = np.arctan2(sum_sin_el, sum_cos_el)-np.pi
    while z[0] > 2*np.pi or z[0] < 0:
        z[0] = z[0] - np.sign(z[0])*np.pi*2
    while z[1] > np.pi or z[1] < -np.pi:
        z[1] = z[1] - np.sign(z[1])*np.pi*2
    return z
