import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from numba import njit, jit
from poliastro.core.propagation import markley, vallado, pimienta, gooding, danby, farnocchia,  mikkola, func_twobody
from poliastro.core.elements import coe2rv
from envs.transformations import _itrs2azel
from poliastro.bodies import Earth
import functools
from scipy.integrate import DOP853, solve_ivp, RK45

k = Earth.k.to_value(u.m**3/u.s**2)
RE = Earth.R_mean.to_value(u.m)


@njit
def ad_none(t0, u_, k_):
    return 0, 0, 0


#@jit(['float64[::](float64[::],float64)'],nopython=True)
@njit
def fx_xyz_markley(x, dt, k=k):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = markley(k, r0, v0, tof) # (m^3 / s^2), (m), (m/s), (s)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


@njit
def fx_xyz_vallado(x, dt, k=k, numiter=350):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    # Compute Lagrange coefficients
    f, g, fdot, gdot = vallado(k, r0, v0, tof, numiter)

    assert np.abs(f * gdot - fdot * g - 1) < 1e-5  # Fixed tolerance

    # Return position and velocity vectors
    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0
    x_post = np.zeros(6)
    x_post[:3] = r
    x_post[3:] = v
    return x_post


@njit
def fx_xyz_pimienta(x, dt, k=k):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = pimienta(k, r0, v0, tof)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


@njit
def fx_xyz_gooding(x, dt, k=k, numiter=150, rtol=1e-8):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = gooding(k, r0, v0, tof, numiter=numiter, rtol=rtol)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


@njit
def fx_xyz_danby(x, dt, k=k):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = danby(k, r0, v0, tof)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


@njit
def fx_xyz_farnocchia(x, dt, k=k):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = farnocchia(k, r0, v0, tof)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


@njit
def fx_xyz_mikkola(x, dt, k=k):
    r0 = x[:3]
    v0 = x[3:]
    tof = dt
    rv = mikkola(k, r0, v0, tof)
    x_post = np.zeros(6)
    x_post[:3] = rv[0]
    x_post[3:] = rv[1]
    return x_post


def fx_xyz_cowell(x, dt, k=k, rtol=1e-11, *, events=None, ad=ad_none, **ad_kwargs):
    u0 = x
    tof = dt

    f_with_ad = functools.partial(func_twobody, k=k, ad=ad, ad_kwargs=ad_kwargs)

    result = solve_ivp(
        f_with_ad,
        (0, tof),
        u0,
        rtol=rtol,
        atol=1e-12,
        method=DOP853,
        dense_output=True,
        events=events,
    )
    if not result.success:
        raise RuntimeError("Integration failed")

    t_end = (
        min(result.t_events[0]) if result.t_events and len(result.t_events[0]) else None
    )

    return result.sol(tof)


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


def init_state_vec(sma=((RE + 400000), 42164000), ecc=(0.001, 0.3), inc=(0, 180), raan=(0, 360), argp=(0, 360),
                   nu=(0, 360), random_state=np.random.RandomState()):
    a = random_state.uniform(sma[0], sma[1]) # (m) – Semi-major axis.
    ecc = random_state.uniform(ecc[0], ecc[1]) # (Unit-less) – Eccentricity.
    inc = np.radians(random_state.uniform(inc[0], inc[1])) # (rad) – Inclination
    raan = np.radians(random_state.uniform(raan[0], raan[1])) # (rad) – Right ascension of the ascending node.
    argp = np.radians(random_state.uniform(argp[0], argp[1])) # (rad) – Argument of the pericenter.
    nu = np.radians(random_state.uniform(nu[0], nu[1])) # (rad) – True anomaly.
    p = a*(1-ecc**2) # (km) - Semi-latus rectum or parameter
    return np.concatenate(coe2rv(k, p, ecc, inc, raan, argp, nu))
