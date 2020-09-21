from collections.abc import Iterable
import numpy as np
from numpy import sin, cos, arctan2 as atan2, arctan as atan, tan, arcsin as asin, arccos as acos, sum, pi, sqrt, radians, float64, array, power, hypot
import pandas as pd
from astropy._erfa import DAYSEC, DAS2R, DMAS2R, DPI, eform
from astropy import _erfa as erfa
from numba import njit
import os

path = os.getcwd()
a, f = eform(1)  # WGS84 ellipsoid parameters - semi-major axis, flattening parameter
e = sqrt(f*(2-f))
b = (1-f)*a
arcsec2rad = pi/648000 # converts arc seconds to radians
deg2rad = pi/180 # converts degrees to radians
tau = 2*pi


def get_eops():
    """
   This function downloads the Earth Orientation Parameters (EOPs) from the IAU sources and returns them as a pandas
        dataframe; https://datacenter.iers.org/eop.php
    """
    url = 'ftp://hpiers.obspm.fr/iers/eop/eopc04/eopc04_IAU2000.62-now'
    ds = np.DataSource(path)
    file = ds.open(url)
    array = np.genfromtxt(file, skip_header=14)
    headers = ['Year', 'Month', 'Day', 'MJD', 'x', 'y', 'UT1-UTC', 'LOD', 'dX',
               'dY', 'x Err', 'y Err', 'UT1-UTC Err', 'LOD Err', 'dX Err', 'dY Err']
    eop = pd.DataFrame(data=array, index=array[:, 3], columns=headers)
    return eop


def utc2cel06a_parameters(t, eop, iau55=False):
    """
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    :param eop:
        eop is a dataframe containing the Earth Orientation Parameters as per IAU definitions
    :param t:
        t is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
            calculated for
    :return:
        jd is the julian date (always xxx.5 because it is based on a noon day break) in days
        ttb is the leap second offset in fractions of a day
        utb is the UT1 offset in fractions of a day
        xp and yp are the coordinates (in radians) of the Celestial Intermediate Pole with respect to the International
            Terrestrial Reference System (see IERS Conventions 2003), measured along the meridians to 0 and 90 deg west
            respectively (as extrapolated from between the two published points before and after).
        dx06 and dy06 are the CIP offsets wrt IAU 2006/2000A (mas->radians) as extrapolated from between the two
            published points before and after
    """
    year, month, day, hour, minute, second = t.year, t.month, t.day, t.hour, t.minute, t.second
    # TT (MJD). */
    djmjd0, date = erfa.cal2jd(iy=year, im=month, id=day)
    jd = djmjd0 + date
    day_frac = (60.0 * (60 * hour + minute) + second) / DAYSEC
    dat_s = erfa.dat(year, month, day, day_frac)
    ttb = dat_s / DAYSEC + 32.184 / DAYSEC

    # Polar motion (arcsec->radians)
    xp_l = eop["x"][date]
    yp_l = eop["y"][date]
    xp_h = eop["x"][date + 1]
    yp_h = eop["y"][date + 1]
    xp = (xp_l * (1 - day_frac) + xp_h * day_frac) * DAS2R
    yp = (yp_l * (1 - day_frac) + yp_h * day_frac) * DAS2R

    # UT1-UTC (s). */
    dut_l = eop["UT1-UTC"][date]
    dut_h = eop["UT1-UTC"][date + 1]
    dut1 = (dut_l * (1 - day_frac) + dut_h * day_frac)

    # CIP offsets wrt IAU 2006/2000A (mas->radians). */
    dx_l = eop["dX"][date]
    dx_h = eop["dX"][date + 1]
    dy_l = eop["dY"][date]
    dy_h = eop["dY"][date + 1]
    dx06 = (dx_l * (1 - day_frac) + dx_h * day_frac) * DAS2R
    dy06 = (dy_l * (1 - day_frac) + dy_h * day_frac) * DAS2R

    if iau55:
        # CIP offsets wrt IAU 2006/2000A (mas->radians). */
        dx06 = float64(0.1750 * DMAS2R, dtype="f64")
        dy06 = float64(-0.2259 * DMAS2R, dtype="f64")
        # UT1-UTC (s). */
        dut1 = float64(-0.072073685, dtype="f64")
        # Polar motion (arcsec->radians)
        xp = float64(0.0349282 * DAS2R, dtype="f64")
        yp = float64(0.4833163 * DAS2R, dtype="f64")

    # UT1. */
    utb = day_frac + dut1 / DAYSEC

    return jd, ttb, utb, xp, dx06, yp, dy06

def eraRZ(psi,array):
    s = np.sin(psi)
    c = np.cos(psi)
    a00 = c * array[0][0] + s * array[1][0]
    a01 = c * array[0][1] + s * array[1][1]
    a02 = c * array[0][2] + s * array[1][2]
    a10 = - s * array[0][0] + c * array[1][0]
    a11 = - s * array[0][1] + c * array[1][1]
    a12 = - s * array[0][2] + c * array[1][2]

    array[0][0] = a00
    array[0][1] = a01
    array[0][2] = a02
    array[1][0] = a10
    array[1][1] = a11
    array[1][2] = a12

    return array

def gcrs2irts_matrix_a(t, eop):
    """
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    :param eop:
        is a dataframe containing the Earth Orientation Parameters as per IAU definitions
    :param t:
        is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
        calculated for
    :return:
        matrix is a [3,3] numpy array or list of arrays used for transforming GCRS to ITRS or vice versa at the
        specified times; ITRS = matrix @ GCRS
    """
    if not (isinstance(t, Iterable)):
        t = [t]
    matrix = []
    for tt in t:
        jd, ttb, utb, xp, dx06, yp, dy06 = utc2cel06a_parameters(tt, eop)

        # celestial to terrestrial transformation matrix
        c2t06a_mat = erfa.c2t06a(tta=jd, ttb=ttb, uta=jd, utb=utb, xp=xp, yp=yp)

        matrix.append(c2t06a_mat)
    if len(matrix) == 1:
        matrix = matrix[0]
    return matrix

def gcrs2irts_matrix_b(t, eop):
    """
    Ref: http://www.iausofa.org/sofa_pn_c.pdf
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    :param eop:
        eop is a dataframe containing the Earth Orientation Parameters as per IAU definitions
    :param t:
        t is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
        calculated for
    :return:
        matrix is a [3,3] numpy array or list of arrays used for transforming GCRS to ITRS or vice versa at the
        specified times; ITRS = matrix @ GCRS
    """
    if not (isinstance(t, Iterable)):
        t = [t]
    matrix = []
    for ti in t:
        year = ti.year
        month = ti.month
        day = ti.day
        hour = ti.hour
        minute = ti.minute
        second = ti.second

        # TT (MJD). */
        djmjd0, date = erfa.cal2jd(iy=year, im=month, id=day)
        # jd = djmjd0 + date
        day_frac = (60.0 * (60.0 * hour + minute) + second) / DAYSEC
        utc = date + day_frac
        Dat = erfa.dat(year, month, day, day_frac)
        tai = utc + Dat / DAYSEC
        tt = tai + 32.184 / DAYSEC

        # UT1. */
        dut1 = eop["UT1-UTC"][date] * (1 - day_frac) + eop["UT1-UTC"][date + 1] * day_frac
        tut = day_frac + dut1 / DAYSEC
        # ut1 = date + tut

        # CIP and CIO, IAU 2006/2000A. */
        x, y, s = erfa.xys06a(djmjd0, tt)

        # X, Y offsets
        dx06 = (eop["dX"][date] * (1 - day_frac) + eop["dX"][date + 1] * day_frac) * DAS2R
        dy06 = (eop["dY"][date] * (1 - day_frac) + eop["dY"][date + 1] * day_frac) * DAS2R

        # Add CIP corrections. */
        x = x + dx06
        y = y + dy06

        # GCRS to CIRS matrix. */
        rc2i = erfa.c2ixys(x, y, s)

        # Earth rotation angle. */
        era = erfa.era00(djmjd0 + date, tut)

        # Form celestial-terrestrial matrix (no polar motion yet). */
        rc2ti = erfa.cr(rc2i)
        # rc2ti = eraRZ(era, rc2ti)
        rc2ti = erfa.rz(era, rc2ti)

        # Polar motion matrix (TIRS->ITRS, IERS 2003). */
        xp = (eop["x"][date] * (1 - day_frac) + eop["x"][date + 1] * day_frac) * DAS2R
        yp = (eop["y"][date] * (1 - day_frac) + eop["y"][date + 1] * day_frac) * DAS2R
        rpom = erfa.pom00(xp, yp, erfa.sp00(djmjd0, tt))

        # Form celestial-terrestrial matrix (including polar motion). */
        rc2it = erfa.rxr(rpom, rc2ti)
        matrix.append(rc2it)
    if len(matrix) == 1:
        matrix = matrix[0]
    return matrix

@njit
def lla2ecef(obs_lla, a=a, f=f, e=e):
    """
    :param lla: observations in lat, lon, height (deg, deg, m)
    :param a:
    :param f:
    :param e:
    :return: ECEF cartisian coordinates
    """
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y

    lat = obs_lla[0] # phi
    lon = obs_lla[1] # lambda
    alt = obs_lla[2] # h
    N = a/np.sqrt(1-e**2*sin(lat)**2) # (eq 2.48)
    x = (N + alt)*cos(lat)*cos(lon) # (eq 2.135)
    y = (N + alt)*cos(lat)*sin(lon) # (eq 2.135)
    z = (N*(1 - e**2) + alt)*sin(lat) # (eq 2.135)
    ecef = array([x, y, z])
    return ecef


@njit
def ecef2lla(ecef, a=a, b=b, f=f, e=e):
    """
    convert ECEF(meters) Cartesian coordinates to geodetic coordinates based on the ellipsoidal coordinates
    :param ecef:
        [x,y,z] : array of floats in ECEF coordinate (meters)

    :return:
        [lat, lon, alt] : array of floats; geodetic latitude (radians), geodetic longitude (radians), altitude (meters)
    based on:
    You, Rey-Jer. (2000). Transformation of Cartesian to Geodetic Coordinates without Iterations.
    Journal of Surveying Engineering. doi: 10.1061/(ASCE)0733-9453
    """
    x, y, z = ecef
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    E = sqrt(a ** 2 - b ** 2)

    # eqn. 4a
    u = sqrt(0.5 * (r ** 2 - E ** 2) + 0.5 * sqrt((r ** 2 - E ** 2) ** 2 + 4 * E ** 2 * z ** 2))
    Q = hypot(x, y)
    huE = hypot(u, E)
    # eqn. 4b
    if not(Q == 0 or u == 0):
        Beta = atan(huE / u * z / Q)
    else:
        if z >= 0:
            Beta = pi / 2
        else:
            Beta = -pi / 2
    # eqn. 13
    eps = ((b * u - a * huE + E ** 2) * sin(Beta)) / (a * huE * 1 / cos(Beta) - E ** 2 * cos(Beta))
    Beta += eps
    # %% final output
    lat = atan(a / b * tan(Beta))
    lon = atan2(y, x)
    # eqn. 7
    alt = hypot(z - b * sin(Beta), Q - a * cos(Beta))
    # inside ellipsoid?
    inside = x ** 2 / a ** 2 + y ** 2 / a ** 2 + z ** 2 / b ** 2 < 1
    if inside:
        alt = -alt

    return np.array([lat, lon, alt])


@njit
def aer2uvw(aer):
    """
    :param aer: array[azimuth, elevation, slant]
    :return: uvw observer centered instead earth centered --> u - north, v - east, w - up(azimuth)
    """
    # Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    # 2.2.2 Local Terrestrial Coordinates defined u, v, w
    az, el, r = aer
    u = r*cos(el)*cos(az)       # (eq 2.148)
    v = r*cos(el)*sin(az)       # (eq 2.148)
    w = r*sin(el)               # (eq 2.148)
    uvw = array([u, v, w])
    return uvw


@njit
def uvw2aer(uvw):
    """
    :param uvw:
    :return: array[azimuth, elevation, slant]
    """
    # Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    # 2.2.2 Local Terrestrial Coordinates defined u, v, w
    u, v, w = uvw
    r = sqrt(sum(uvw**2))       # (eq 2.156)
    az = atan2(v, u)            # (eq 2.154)
    if az < 0:
        az = az + tau
    el = asin(w/r)              # (eq 2.155)
    aer = array([az, el, r])
    return aer


def rrm2ddm(aer):
    """
    :param aer: array[azimuth, elevation, slant]
    :return: array[azimuth, elevation, slant] in degrees
    """
    aer[0] = np.degrees(aer[0])
    aer[1] = np.degrees(aer[1])
    return aer


@njit
def ecef2aer(obs_lla, ecef_sat, ecef_obs):
    """
    :Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
        https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    :param obs_lla: observations in lat, lon, height (deg, deg, m)
    :param ecef_sat:
    :return: array[azimuth, elevation, slant]
    """

    lat, lon = obs_lla[0], obs_lla[1]  # phi, lambda

    trans_uvw_ecef = array([[-sin(lat)*cos(lon),    -sin(lon),  cos(lat)*cos(lon)],
                            [-sin(lat)*sin(lon),    cos(lon),   cos(lat)*sin(lon)],
                            [cos(lat),              0,          sin(lat)]])         # (eq 2.153)
    delta_ecef = ecef_sat - ecef_obs        # (eq 2.149)
    R_enz = trans_uvw_ecef.T @ delta_ecef   # (eq 2.153)
    r = sqrt(sum(delta_ecef**2))            # (eq 2.156)
    az = atan2(R_enz[1], (R_enz[0]))        # (eq 2.154)
    if az < 0:
        az = az + 2*pi
    el = asin(R_enz[2]/r)                   # (eq 2.155)
    aer = array([az, el, r])
    return aer
