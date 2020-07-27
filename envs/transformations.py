from collections.abc import Iterable

import numpy as np
from numpy import sin, cos, arctan2 as atan2, arctan as atan, tan, arcsin as asin, arccos as acos, sum, pi, sqrt, radians, float64, array, power, hypot
import pandas as pd
from astropy._erfa import DAYSEC, DAS2R, DMAS2R, DPI, eform
from astropy import _erfa as erfa
from numba import njit

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
    ds = np.DataSource('C:/Users/dpawa/PycharmProjects/ssa-gym')
    file = ds.open(url)
    # datasource = np.DataSource(url)
    # file = datasource.open(url)
    array = np.genfromtxt(file, skip_header=14)
    headers = ['Year', 'Month', 'Day', 'MJD', 'x', 'y', 'UT1-UTC', 'LOD', 'dX',
               'dY', 'x Err', 'y Err', 'UT1-UTC Err', 'LOD Err', 'dX Err', 'dY Err']
    eop = pd.DataFrame(data=array, index=array[:, 3], columns=headers)
    return eop

def gcrs2irts_matrix_a(t, eop):
    """
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    Inputs:
        eop is a dataframe containing the Earth Orientation Parameters as per IAU definitions
        t is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
            calculated for
    Outputs:
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


def utc2cel06a_parameters(t, eop, iau55=False):
    """
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    Inputs:
        eop is a dataframe containing the Earth Orientation Parameters as per IAU definitions
        t is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
            calculated for
    Outputs:
        jd is the julian date (always xxx.5 because it is based on a noon day break) in days
        ttb is the leap second offset in fractions of a day
        utb is the UT1 offset in fractions of a day
        xp and yp are the coordinates (in radians) of the Celestial Intermediate Pole with respect to the International
            Terrestrial Reference System (see IERS Conventions 2003), measured along the meridians to 0 and 90 deg west
            respectively (as extrapolated from between the two published points before and after).
        dx06 and dy06 are the CIP offsets wrt IAU 2006/2000A (mas->radians) as extrapolated from between the two
            published points before and after
    """
    year = t.year
    month = t.month
    day = t.day
    hour = t.hour
    minute = t.minute
    second = t.second

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


def gcrs2irts_matrix_b(t, eop):
    """
    Purpose:
        This function calculates the cartesian transformation matrix for transforming GCRS to ITRS or vice versa
    Inputs:
        eop is a dataframe containing the Earth Orientation Parameters as per IAU definitions
        t is a datetime object or a list of datetime objects with the UTC times for the transformation matrix to be
            calculated for
    Outputs:
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
def itrs2azel(observer, targets):
    """
    Purpose:
        Calculate the observed locations of a set of ITRS target coordinates with respect to an observer coordinate
    Input:
        'observer' is assumed to be a numpy array of dimension [3], where 3 is the ITRS x,z,y cartesian coordinates in
            meters of the observing site at which the az el measurements are being generated
        'targets' is assumed to be a numpy array of dimension [n, 3], where 3 is the ITRS x,z,y cartesian coordinates in
            meters of a sets of n coordinates which are the distant points for the observer point
    Output:
        'aer' is a numpy array of dimension [n, 3], where 3 is the azimuth (radians), elevation (radians), slant range
            (meters) of the n target points from the perspective of the observer point
    Source:
        https://gis.stackexchange.com/questions/58923/calculating-view-angle
    """
    x = observer[0]
    y = observer[1]
    z = observer[2]
    dx = targets[:, 0] - x
    dy = targets[:, 1] - y
    dz = targets[:, 2] - z
    cos_azimuth = (-z * x * dx - z * y * dy + (x ** 2 + y ** 2) * dz) / sqrt(
        (x ** 2 + y ** 2) * (x ** 2 + y ** 2 + z ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    sin_azimuth = (-y * dx + x * dy) / sqrt((x ** 2 + y ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    az = atan2(sin_azimuth, cos_azimuth)
    cos_elevation = (x * dx + y * dy + z * dz) / sqrt((x ** 2 + y ** 2 + z ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    el = pi / 2 - acos(cos_elevation)
    sr = sqrt(sum(power((observer - targets), 2).T, axis=0))  # slant range
    az = az + (az < 0) * pi * 2
    aer = np.column_stack((az, el, sr))
    return aer


@njit
def _itrs2azel(observer, target):
    """
    Purpose:
        Calculate the observed location(s) of a set of ITRS target coordinate(s) with respect to an observer coordinate
    Input:
        observer is assumed to be a numpy array of dimension [3], where 3 is the ITRS x,z,y cartesian coordinates in
            meters of the observing site at which the az el measurements are being generated
        targets is assumed to be a numpy array of dimension [3] or [n,3], where 3 is the ITRS x,z,y cartesian
            coordinates in meters of n different sets of coordinates which are the distant point for the observer point
    Output:
        aer is a numpy array of dimension [3] or [n,3], where 3 is the azimuth (radians), elevation (radians), slant
            range (meters) of the target points from the perspective of the observer point
    """
    x = observer[0]
    y = observer[1]
    z = observer[2]
    dx = target[0] - x
    dy = target[1] - y
    dz = target[2] - z
    cos_azimuth = (-z * x * dx - z * y * dy + (x ** 2 + y ** 2) * dz) / sqrt(
        (x ** 2 + y ** 2) * (x ** 2 + y ** 2 + z ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    sin_azimuth = (-y * dx + x * dy) / sqrt((x ** 2 + y ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    az = atan2(sin_azimuth, cos_azimuth)
    cos_elevation = (x * dx + y * dy + z * dz) / sqrt((x ** 2 + y ** 2 + z ** 2) * (dx ** 2 + dy ** 2 + dz ** 2))
    el = pi / 2 - acos(cos_elevation)
    sr = sqrt(sum(power((observer - target), 2).T, axis=0))  # slant range
    az = az + (az < 0) * pi * 2
    aer = array([az, el, sr])
    return aer


@njit
def itrs2lla_py(xyz):
    '''
    # The below code was modified from its original ERFA c source files to run natively in python
    # Refer to the ERFA documentation below
    #  - - - - - - - - - -
    #   e r a G c 2 g d e
    #  - - - - - - - - - -
    #
    #  Transform geocentric coordinates to geodetic for a reference
    #  ellipsoid of specified form.
    #
    #  Given:
    #     a       double     equatorial radius (Notes 2,4)
    #     f       double     flattening (Note 3)
    #     xyz     double[3]  geocentric vector (Note 4)
    #
    #  Returned:
    #     elong   double     longitude (radians, east +ve)
    #     phi     double     latitude (geodetic, radians)
    #     height  double     height above ellipsoid (geodetic, Note 4)
    #
    #  Returned (function value):
    #             int        status:  0 = OK
    #                                -1 = illegal f
    #                                -2 = illegal a
    #
    #  Notes:
    #
    #  1) This function is based on the GCONV2H Fortran subroutine by
    #     Toshio Fukushima (see reference).
    #
    #  2) The equatorial radius, a, can be in any units, but meters is
    #     the conventional choice.
    #
    #  3) The flattening, f, is (for the Earth) a value around 0.00335,
    #     i.e. around 1/298.
    #
    #  4) The equatorial radius, a, and the geocentric vector, xyz,
    #     must be given in the same units, and determine the units of
    #     the returned height, height.
    #
    #  5) If an error occurs (status < 0), elong, phi and height are
    #     unchanged.
    #
    #  6) The inverse transformation is performed in the function
    #     eraGd2gce.
    #
    #  7) The transformation for a standard ellipsoid (such as ERFA_WGS84) can
    #     more conveniently be performed by calling eraGc2gd, which uses a
    #     numerical code to identify the required A and F values.
    #
    #  Reference:
    #
    #     Fukushima, T., "Transformation from Cartesian to geodetic
    #     coordinates accelerated by Halley's method", J.Geodesy (2006)
    #     79: 689-693
    #
    #  Copyright (C) 2013-2019, NumFOCUS Foundation.
    #  Derived, with permission, from the SOFA library.  See notes at end of file.
    # Functions of ellipsoid parameters (with further validation of f).
    '''
    aeps2 = a * a * 1e-32
    e2 = (2.0 - f) * f
    e4t = e2 * e2 * 1.5
    ec2 = 1.0 - e2
    if ec2 <= 0.0:
        print("ec2 in eraGc2gde = ", ec2)
    ec = sqrt(ec2)
    b = a * ec

    # Cartesian components. */
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # Distance from polar axis squared. */
    p2 = x * x + y * y

    # Longitude. */
    if p2 > 0.0:
        elong = atan2(y, x)
    else:
        elong = 0.0

    # Unsigned z-coordinate. */
    absz = np.abs(z)

    # Proceed unless polar case. */
    if p2 > aeps2:
        # Distance from polar axis. */
        p = sqrt(p2)

        # Normalization. */
        s0 = absz / a
        pn = p / a
        zc = ec * s0

        # Prepare Newton correction factors. */
        c0 = ec * pn
        c02 = c0 * c0
        c03 = c02 * c0
        s02 = s0 * s0
        s03 = s02 * s0
        a02 = c02 + s02
        a0 = sqrt(a02)
        a03 = a02 * a0
        d0 = zc * a03 + e2 * s03
        f0 = pn * a03 - e2 * c03

        # Prepare Halley correction factor. */
        b0 = e4t * s02 * c02 * pn * (a0 - ec)
        s1 = d0 * f0 - b0 * s0
        cc = ec * (f0 * f0 - b0 * c0)

        # Evaluate latitude and height. */
        phi = atan2(s1, cc)  # flag for review - arctan changed to arctan2
        s12 = s1 * s1
        cc2 = cc * cc
        height = (p * cc + absz * s1 - a * sqrt(ec2 * s12 + cc2)) / sqrt(s12 + cc2)
    else:
        # Exception: pole. */
        phi = DPI / 2.0
        height = absz - b

    # Restore sign of latitude. */
    if z < 0:
        phi = -phi

    '''
    # ----------------------------------------------------------------------
    #
    #
    #   Copyright (C) 2013-2019, NumFOCUS Foundation.
    #   All rights reserved.
    #
    #   This library is derived, with permission, from the International
    #   Astronomical Union's "Standards of Fundamental Astronomy" library,
    #   available from http://www.iausofa.org.
    #
    #   The ERFA version is intended to retain identical functionality to
    #   the SOFA library, but made distinct through different function and
    #   file names, as set out in the SOFA license conditions.  The SOFA
    #   original has a role as a reference standard for the IAU and IERS,
    #   and consequently redistribution is permitted only in its unaltered
    #   state.  The ERFA version is not subject to this restriction and
    #   therefore can be included in distributions which do not support the
    #   concept of "read only" software.
    #
    #   Although the intent is to replicate the SOFA API (other than
    #   replacement of prefix names) and results (with the exception of
    #   bugs;  any that are discovered will be fixed), SOFA is not
    #   responsible for any errors found in this version of the library.
    #
    #   If you wish to acknowledge the SOFA heritage, please acknowledge
    #   that you are using a library derived from SOFA, rather than SOFA
    #   itself.
    #
    #
    #   TERMS AND CONDITIONS
    #
    #   Redistribution and use in source and binary forms, with or without
    #   modification, are permitted provided that the following conditions
    #   are met:
    #
    #   1 Redistributions of source code must retain the above copyright
    #     notice, this list of conditions and the following disclaimer.
    #
    #   2 Redistributions in binary form must reproduce the above copyright
    #     notice, this list of conditions and the following disclaimer in
    #     the documentation and/or other materials provided with the
    #     distribution.
    #
    #   3 Neither the name of the Standards Of Fundamental Astronomy Board,
    #     the International Astronomical Union nor the names of its
    #     contributors may be used to endorse or promote products derived
    #     from this software without specific prior written permission.
    #
    #   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    #   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    #   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    #   FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
    #   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    #   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    #   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    #   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    #   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    #   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    #   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    #   POSSIBILITY OF SUCH DAMAGE.
    '''

    return elong, phi, height


def itrs2lla(xyz):
    """
    uses ERFA's geocentric to geodetic function and the WGS84 ellipsoid parameters
    Input:
        xyz is assumed to be a numpy array of dimension [3] or [n,3], where 3 is the x,z,y cartesian coordinates in meters
            of n different sets of coordinates
    Output:
        lla is a numpy array of dimension [3] or [n,3], where 3 is the lon (radians),lat (radians), height (meters) of
            the geodetic coordinates of n different sets of coordinates
    """
    lla = array(erfa.gc2gd(1, xyz), dtype=float64)
    lla = lla.T
    return lla


def lla2itrs(lla):
    """
    uses ERFA's geodetic to geocentric function and the WGS84 ellipsoid parameters
    Input:
        lla is assumed to be a numpy array of dimension [3] or [n,3], where 3 is the lon (radians),lat (radians),
            height (meters) of the geodetic coordinates of n different sets of coordinates
    Output:
        xyz is assumed to be a numpy array of dimension [3] or [n,3], where 3 is the x,z,y cartesian coordinates in
            meters of n different sets of coordinates
    """
    lla = np.atleast_2d(lla)
    xyz = array(erfa.gd2gc(n=1, elong=lla[:, 1], phi=lla[:, 0], height=lla[:, 2]), dtype=float64)
    if xyz.size == 3:
        xyz = xyz[0]
    return xyz

@njit
def lla2ecef(lla, a=a, f=f, e=e):
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y

    lat = lla[0] # phi
    lon = lla[1] # lambda
    alt = lla[2] # h
    N = a/np.sqrt(1-e**2*sin(lat)**2) # (eq 2.48)
    x = (N + alt)*cos(lat)*cos(lon) # (eq 2.135)
    y = (N + alt)*cos(lat)*sin(lon) # (eq 2.135)
    z = (N*(1 - e**2) + alt)*sin(lat) # (eq 2.135)
    ecef = array([x, y, z])
    return ecef


@njit
def ecef2lla(ecef, a=a, b=b, f=f, e=e):
    """
    convert ECEF (meters) to geodetic coordinates
    Parameters
    ----------
    [x,y,z] : array of floats in ECEF coordinate (meters)
    Returns
    -------
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
    # Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    az, el, r = aer
    u = r*cos(el)*cos(az)       # (eq 2.148)
    v = r*cos(el)*sin(az)       # (eq 2.148)
    w = r*sin(el)               # (eq 2.148)
    enu = array([u, v, w])
    return enu


@njit
def uvw2aer(uvw):
    # Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    u, v, w = uvw
    r = sqrt(sum(uvw**2))       # (eq 2.156)
    az = atan2(v, u)            # (eq 2.154)
    if az < 0:
        az = az + tau
    el = asin(w/r)              # (eq 2.155)
    aer = array([az, el, r])
    return aer


def rrm2ddm(aer):
    aer[0] = np.degrees(aer[0])
    aer[1] = np.degrees(aer[1])
    return aer


@njit
def ecef2aer(lla_obs, ecef_sat, ecef_obs):
    # Ref: Geometric Reference Systems in Geodesy by Christopher Jekeli, Ohio State University, August 2016
    # https://kb.osu.edu/bitstream/handle/1811/77986/Geom_Ref_Sys_Geodesy_2016.pdf?sequence=1&isAllowed=y
    lat = lla_obs[0] # phi
    lon = lla_obs[1] # lambda
    alt = lla_obs[2] # h
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
