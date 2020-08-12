import numpy as np
from numpy import cross
from numpy.core.umath import cos, sin, sqrt
from numpy.linalg import norm
from numba import njit

'''
This file contains all of the functions necessary to perform replicate the default functionality of PoliAstroy's
implementation of Farnocchia's mean motion based analytical orbit propagation algorithm. The implementing was writen by
by Juan Luis Cano Rodríguez. I have modify the structuring of several function or sub-functions to function in a gym 
environment running inside an RLLib worker. Credit for the development goes to Juan and the PoliAstro team.
'''


@njit
def rv_pqw(k, p, ecc, nu):
    r"""Returns r and v vectors in perifocal frame.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    nu: float
        True anomaly (rad).

    Returns
    -------

    r: ndarray
        Position. Dimension 3 vector
    v: ndarray
        Velocity. Dimension 3 vector

    Notes
    -----
    These formulas can be checked at Curtis 3rd. Edition, page 110. Also the
    example proposed is 2.11 of Curtis 3rd Edition book.

    .. math::

        \vec{r} = \frac{h^2}{\mu}\frac{1}{1 + e\cos(\theta)}\begin{bmatrix}
        \cos(\theta)\\
        \sin(\theta)\\
        0
        \end{bmatrix} \\\\\\

        \vec{v} = \frac{h^2}{\mu}\begin{bmatrix}
        -\sin(\theta)\\
        e+\cos(\theta)\\
        0
        \end{bmatrix}

    Examples
    --------
    # >>> from poliastro.constants import GM_earth
    # >>> k = GM_earth.value  # Earth gravitational parameter
    # >>> ecc = 0.3  # Eccentricity
    # >>> h = 60000e6  # Angular momentum of the orbit (m**2 / s)
    # >>> nu = np.deg2rad(120)  # True Anomaly (rad)
    # >>> p = h**2 / k  # Parameter of the orbit
    # >>> r, v = rv_pqw(k, p, ecc, nu)
    # >>> # Printing the results
    r = [-5312706.25105345  9201877.15251336    0] [m]
    v = [-5753.30180931 -1328.66813933  0] [m]/[s]

    """
    pqw = np.array([[cos(nu), sin(nu), 0], [-sin(nu), ecc + cos(nu), 0]]) * np.array(
        [[p / (1 + ecc * cos(nu))], [sqrt(k / p)]]
    )
    return pqw


@njit
def rotation_matrix(angle, axis):
    c = cos(angle)
    s = sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid axis: must be one of 'x', 'y' or 'z'")


@njit
def coe_rotation_matrix(inc, raan, argp):
    """Create a rotation matrix for coe transformation
    """
    r = rotation_matrix(raan, 2)
    r = r @ rotation_matrix(inc, 0)
    r = r @ rotation_matrix(argp, 2)
    return r


@njit
def coe2rv(k, p, ecc, inc, raan, argp, nu):
    r"""Converts from classical orbital to state vectors.

    Classical orbital elements are converted into position and velocity
    vectors by `rv_pqw` algorithm. A rotation matrix is applied to position
    and velocity vectors to get them expressed in terms of an IJK basis.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    inc : float
        Inclination (rad).
    raan : float
        Longitude of ascending node (rad).
    argp : float
        Argument of perigee (rad).
    nu : float
        True anomaly (rad).

    Returns
    -------
    r_ijk: np.array
        Position vector in basis ijk.
    v_ijk: np.array
        Velocity vector in basis ijk.

    Notes
    -----

    .. math::
        \begin{align}
            \vec{r}_{IJK} &= [ROT3(-\Omega)][ROT1(-i)][ROT3(-\omega)]\vec{r}_{PQW}
                               = \left [ \frac{IJK}{PQW} \right ]\vec{r}_{PQW}\\
            \vec{v}_{IJK} &= [ROT3(-\Omega)][ROT1(-i)][ROT3(-\omega)]\vec{v}_{PQW}
                               = \left [ \frac{IJK}{PQW} \right ]\vec{v}_{PQW}\\
        \end{align}

    Previous rotations (3-1-3) can be expressed in terms of a single rotation matrix:

    .. math::
        \left [ \frac{IJK}{PQW} \right ]

    .. math::
        \begin{bmatrix}
        \cos(\Omega)\cos(\omega) - \sin(\Omega)\sin(\omega)\cos(i) & -\cos(\Omega)\sin(\omega) - \sin(\Omega)\cos(\omega)\cos(i) & \sin(\Omega)\sin(i)\\
        \sin(\Omega)\cos(\omega) + \cos(\Omega)\sin(\omega)\cos(i) & -\sin(\Omega)\sin(\omega) + \cos(\Omega)\cos(\omega)\cos(i) & -\cos(\Omega)\sin(i)\\
        \sin(\omega)\sin(i) & \cos(\omega)\sin(i) & \cos(i)
        \end{bmatrix}

    """
    pqw = rv_pqw(k, p, ecc, nu)
    rm = coe_rotation_matrix(inc, raan, argp)

    ijk = pqw @ rm.T

    return ijk


@njit
def rv2coe(k, r, v):
    tol = 1e-8
    r"""Converts from vectors to classical orbital elements.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2)
    r : array
        Position vector (km)
    v : array
        Velocity vector (km / s)
    tol : float, optional
        Tolerance for eccentricity and inclination checks, default to 1e-8

    Returns
    -------
    p : float
        Semi-latus rectum of parameter (km)
    ecc: float
        Eccentricity
    inc: float
        Inclination (rad)
    raan: float
        Right ascension of the ascending nod (rad)
    argp: float
        Argument of Perigee (rad)
    nu: float
        True Anomaly (rad)

    Notes
    -----
    This example is a real exercise from Orbital Mechanics for Engineering
    students by Howard D.Curtis. This exercise is 4.3 of 3rd. Edition, page 200.

    1. First the angular momentum is computed:

    .. math::
        \vec{h} = \vec{r} \times \vec{v}

    2. With it the eccentricity can be solved:

    .. math::
        \begin{align}
        \vec{e} &= \frac{1}{\mu}\left [ \left ( v^{2} - \frac{\mu}{r}\right ) \vec{r}  - (\vec{r} \cdot \vec{v})\vec{v} \right ] \\
        e &= \sqrt{\vec{e}\cdot\vec{e}} \\
        \end{align}

    3. The node vector line is solved:

    .. math::
        \begin{align}
        \vec{N} &= \vec{k} \times \vec{h} \\
        N &= \sqrt{\vec{N}\cdot\vec{N}}
        \end{align}

    4. The rigth ascension node is computed:

    .. math::
        \Omega = \left\{ \begin{array}{lcc}
         cos^{-1}{\left ( \frac{N_{x}}{N} \right )} &   if  & N_{y} \geq  0 \\
         \\ 360^{o} -cos^{-1}{\left ( \frac{N_{x}}{N} \right )} &  if & N_{y} < 0 \\
         \end{array}
        \right.

    5. The argument of perigee:

    .. math::
        \omega  = \left\{ \begin{array}{lcc}
         cos^{-1}{\left ( \frac{\vec{N}\vec{e}}{Ne} \right )} &   if  & e_{z} \geq  0 \\
         \\ 360^{o} -cos^{-1}{\left ( \frac{\vec{N}\vec{e}}{Ne} \right )} &  if & e_{z} < 0 \\
         \end{array}
        \right.

    6. And finally the true anomaly:

    .. math::
        \nu  = \left\{ \begin{array}{lcc}
         cos^{-1}{\left ( \frac{\vec{e}\vec{r}}{er} \right )} &   if  & v_{r} \geq  0 \\
         \\ 360^{o} -cos^{-1}{\left ( \frac{\vec{e}\vec{r}}{er} \right )} &  if & v_{r} < 0 \\
         \end{array}
        \right.

    Examples
    --------
    >>> from poliastro.constants import GM_earth
    >>> from astropy import units as u
    >>> k = GM_earth.to(u.km ** 3 / u.s ** 2).value  # Earth gravitational parameter
    >>> r = np.array([-6045., -3490., 2500.])
    >>> v = np.array([-3.457, 6.618, 2.533])
    >>> p, ecc, inc, raan, argp, nu = rv2coe(k, r, v)
    >>> print("p:", p, "[km]")
    p: 8530.47436396927 [km]
    >>> print("ecc:", ecc)
    ecc: 0.17121118195416898
    >>> print("inc:", np.rad2deg(inc), "[deg]")
    inc: 153.2492285182475 [deg]
    >>> print("raan:", np.rad2deg(raan), "[deg]")
    raan: 255.27928533439618 [deg]
    >>> print("argp:", np.rad2deg(argp), "[deg]")
    argp: 20.068139973005362 [deg]
    >>> print("nu:", np.rad2deg(nu), "[deg]")
    nu: 28.445804984192122 [deg]

    """

    h = cross(r, v)
    n = cross([0, 0, 1], h)
    e = ((v.dot(v) - k / (norm(r))) * r - r.dot(v) * v) / k
    ecc = norm(e)
    p = h.dot(h) / k
    inc = np.arccos(h[2] / norm(h))

    circular = ecc < tol
    equatorial = abs(inc) < tol

    if equatorial and not circular:
        raan = 0
        argp = np.arctan2(e[1], e[0]) % (2 * np.pi)  # Longitude of periapsis
        nu = np.arctan2(h.dot(cross(e, r)) / norm(h), r.dot(e))
    elif not equatorial and circular:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = 0
        # Argument of latitude
        nu = np.arctan2(r.dot(cross(h, n)) / norm(h), r.dot(n))
    elif equatorial and circular:
        raan = 0
        argp = 0
        nu = np.arctan2(r[1], r[0]) % (2 * np.pi)  # True longitude
    else:
        a = p / (1 - (ecc ** 2))
        ka = k * a
        if a > 0:
            e_se = r.dot(v) / sqrt(ka)
            e_ce = norm(r) * v.dot(v) / k - 1
            nu = E_to_nu(np.arctan2(e_se, e_ce), ecc)
        else:
            e_sh = r.dot(v) / sqrt(-ka)
            e_ch = norm(r) * (norm(v) ** 2) / k - 1
            nu = F_to_nu(np.log((e_ch + e_sh) / (e_ch - e_sh)) / 2, ecc)

        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        px = r.dot(n)
        py = r.dot(cross(h, n)) / norm(h)
        argp = (np.arctan2(py, px) - nu) % (2 * np.pi)

    nu = (nu + np.pi) % (2 * np.pi) - np.pi

    return p, ecc, inc, raan, argp, nu


@njit
def _kepler_equation(E, M, ecc):
    return E_to_M(E, ecc) - M


@njit
def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


@njit
def _kepler_equation_hyper(F, M, ecc):
    return F_to_M(F, ecc) - M


@njit
def _kepler_equation_prime_hyper(F, M, ecc):
    return ecc * np.cosh(F) - 1


@njit
def newton(regime, x0, args=(), tol=1.48e-08, maxiter=50):
    p0 = 1.0 * x0
    for i in range(maxiter):
        if regime == "hyperbolic":
            fval = _kepler_equation_hyper(p0, *args)
            fder = _kepler_equation_prime_hyper(p0, *args)
        else:
            fval = _kepler_equation(p0, *args)
            fder = _kepler_equation_prime(p0, *args)

        newton_step = fval / fder
        p = p0 - newton_step
        if abs(p - p0) < tol:
            return p
        p0 = p

    return np.nan


@njit
def D_to_nu(D):
    r"""True anomaly from parabolic anomaly.

    Parameters
    ----------
    D : float
        Eccentric anomaly.

    Returns
    -------
    nu : float
        True anomaly.

    Notes
    -----
    From [1]_:

    .. math::

        \nu = 2 \arctan{D}

    """

    return 2.0 * np.arctan(D)


@njit
def nu_to_D(nu):
    r"""Parabolic anomaly from true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly in radians.

    Returns
    -------
    D : float
        Parabolic anomaly.

    Warnings
    --------
    The parabolic anomaly will be continuous in (-∞, ∞)
    only if the true anomaly is in (-π, π].
    No validation or wrapping is performed.

    Notes
    -----
    The treatment of the parabolic case is heterogeneous in the literature,
    and that includes the use of an equivalent quantity to the eccentric anomaly:
    [1]_ calls it "parabolic eccentric anomaly" D,
    [2]_ also uses the letter D but calls it just "parabolic anomaly",
    [3]_ uses the letter B citing indirectly [4]_
    (which however calls it "parabolic time argument"),
    and [5]_ does not bother to define it.

    We use this definition:

    .. math::

        B = \tan{\frac{\nu}{2}}

    References
    ----------
    .. [1] Farnocchia, Davide, Davide Bracali Cioci, and Andrea Milani.
       "Robust resolution of Kepler’s equation in all eccentricity regimes."
    .. [2] Bate, Muller, White.
    .. [3] Vallado, David. "Fundamentals of Astrodynamics and Applications",
       2013.
    .. [4] IAU VIth General Assembly, 1938.
    .. [5] Battin, Richard H. "An introduction to the Mathematics and Methods
       of Astrodynamics, Revised Edition", 1999.

    """
    # TODO: Rename to B
    return np.tan(nu / 2.0)


@njit
def nu_to_E(nu, ecc):
    r"""Eccentric anomaly from true anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly, between -π and π radians.

    Warnings
    --------
    The eccentric anomaly will be between -π and π radians,
    no matter the value of the true anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        E = 2 \atan \left( \sqrt{\frac{1 - e}{1 + e}} \tan{\frac{\nu}{2}}
        \in (-\pi, \pi]

    """
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(nu / 2))
    return E


@njit
def nu_to_F(nu, ecc):
    r"""Hyperbolic anomaly from true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    F : float
        Hyperbolic anomaly.

    Warnings
    --------
    The hyperbolic anomaly will be continuous in (-∞, ∞)
    only if the true anomaly is in (-π, π],
    which should happen anyway
    because the true anomaly is limited for hyperbolic orbits.
    No validation or wrapping is performed.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        F = 2 \operatorname{arctanh} \sqrt{\frac{e-1}{e+1}} \tan{\frac{\nu}{2}}

    """
    F = 2 * np.arctanh(np.sqrt((ecc - 1) / (ecc + 1)) * np.tan(nu / 2))
    return F


@njit
def E_to_nu(E, ecc):
    r"""True anomaly from eccentric anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    nu : float
        True anomaly, between -π and π radians.

    Warnings
    --------
    The true anomaly will be between -π and π radians,
    no matter the value of the eccentric anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        \nu = 2 \atan \left( \sqrt{\frac{1 + e}{1 - e}} \tan{\frac{E}{2}} \right)
        \in (-\pi, \pi]

    """
    nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return nu


@njit
def F_to_nu(F, ecc):
    r"""True anomaly from hyperbolic anomaly.

    Parameters
    ----------
    F : float
        Hyperbolic anomaly.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    nu : float
        True anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        \nu = 2 \atan \left( \sqrt{\frac{e + 1}{e - 1}} \tanh{\frac{F}{2}}
        \in (-\pi, \pi]

    """
    nu = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(F / 2))
    return nu


@njit
def M_to_E(M, ecc):
    """Eccentric anomaly from mean anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly.

    Notes
    -----
    This uses a Newton iteration on the Kepler equation.

    """
    assert -np.pi <= M <= np.pi
    if ecc < 0.8:
        E0 = M
    else:
        E0 = np.pi * np.sign(M)
    E = newton("elliptic", E0, args=(M, ecc), tol=1.48e-08, maxiter=50)
    return E


@njit
def M_to_F(M, ecc):
    """Hyperbolic anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    F : float
        Hyperbolic anomaly.

    Notes
    -----
    This uses a Newton iteration on the hyperbolic Kepler equation.

    """
    F0 = np.arcsinh(M / ecc)
    F = newton("hyperbolic", F0, args=(M, ecc), tol=1.48e-08, maxiter=100)
    return F


@njit
def M_to_D(M):
    """Parabolic anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.

    Returns
    -------
    D : float
        Parabolic anomaly.

    Notes
    -----
    This uses the analytical solution of Barker's equation from [5]_.

    """
    B = 3.0 * M / 2.0
    A = (B + (1.0 + B ** 2) ** 0.5) ** (2.0 / 3.0)
    D = 2 * A * B / (1 + A + A ** 2)
    return D


@njit
def E_to_M(E, ecc):
    r"""Mean anomaly from eccentric anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly.

    Warnings
    --------
    The mean anomaly will be outside of (-π, π]
    if the eccentric anomaly is.
    No validation or wrapping is performed.

    Notes
    -----
    The implementation uses the plain original Kepler equation:

    .. math::
        M = E - e \sin{E}

    """
    M = E - ecc * np.sin(E)
    return M


@njit
def F_to_M(F, ecc):
    r"""Mean anomaly from eccentric anomaly.

    Parameters
    ----------
    F : float
        Hyperbolic anomaly.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    M : float
        Mean anomaly.

    Notes
    -----
    As noted in [5]_, by manipulating
    the parametric equations of the hyperbola
    we can derive a quantity that is equivalent
    to the eccentric anomaly in the elliptic case:

    .. math::

        M = e \sinh{F} - F

    """
    M = ecc * np.sinh(F) - F
    return M


@njit
def D_to_M(D):
    r"""Mean anomaly from parabolic anomaly.

    Parameters
    ----------
    D : float
        Parabolic anomaly.

    Returns
    -------
    M : float
        Mean anomaly.

    Notes
    -----
    We use this definition:

    .. math::

        M = B + \frac{B^3}{3}

    Notice that M < ν until ν ~ 100 degrees,
    then it reaches π when ν ~ 120 degrees,
    and grows without bounds after that.
    Therefore, it can hardly be called an "anomaly"
    since it is by no means an angle.

    """
    M = D + D ** 3 / 3
    return M


@njit
def _kepler_equation_near_parabolic(D, M, ecc):
    return D_to_M_near_parabolic(D, ecc) - M


@njit
def _kepler_equation_prime_near_parabolic(D, M, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D ** 2)
    assert abs(x) < 1
    S = dS_x_alt(ecc, x)
    return np.sqrt(2.0 / (1.0 + ecc)) + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (D ** 2) * S


@njit
def S_x(ecc, x):
    atol = 1e-12
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * x ** k
        k += 1
        if abs(S - S_old) < atol:
            return S


@njit
def dS_x_alt(ecc, x):
    atol = 1e-12
    # Notice that this is not exactly
    # the partial derivative of S with respect to D,
    # but the result of arranging the terms
    # in section 4.2 of Farnocchia et al. 2013
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * x ** k
        k += 1
        if abs(S - S_old) < atol:
            return S


@njit
def D_to_M_near_parabolic(D, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D ** 2)
    assert abs(x) < 1
    S = S_x(ecc, x)
    return (
        np.sqrt(2.0 / (1.0 + ecc)) * D + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (D ** 3) * S
    )


@njit
def M_to_D_near_parabolic(M, ecc):
    tol = 1.48e-08
    maxiter = 50
    """Parabolic eccentric anomaly from mean anomaly, near parabolic case.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity (~1).

    Returns
    -------
    D : float
        Parabolic eccentric anomaly.

    """
    D0 = M_to_D(M)

    for _ in range(maxiter):
        fval = _kepler_equation_near_parabolic(D0, M, ecc)
        fder = _kepler_equation_prime_near_parabolic(D0, M, ecc)

        newton_step = fval / fder
        D = D0 - newton_step
        if abs(D - D0) < tol:
            return D

        D0 = D

    return np.nan


@njit
def delta_t_from_nu(nu, ecc, k, q):
    delta = 1e-2
    """Time elapsed since periapsis for given true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly.
    ecc : float
        Eccentricity.
    k : float
        Gravitational parameter.
    q : float
        Periapsis distance.
    delta : float
        Parameter that controls the size of the near parabolic region.

    Returns
    -------
    delta_t : float
        Time elapsed since periapsis.

    """
    assert -np.pi <= nu < np.pi
    if ecc < 1 - delta:
        # Strong elliptic
        E = nu_to_E(nu, ecc)  # (-pi, pi]
        M = E_to_M(E, ecc)  # (-pi, pi]
        n = np.sqrt(k * (1 - ecc) ** 3 / q ** 3)
    elif 1 - delta <= ecc < 1:
        E = nu_to_E(nu, ecc)  # (-pi, pi]
        if delta <= 1 - ecc * np.cos(E):
            # Strong elliptic
            M = E_to_M(E, ecc)  # (-pi, pi]
            n = np.sqrt(k * (1 - ecc) ** 3 / q ** 3)
        else:
            # Near parabolic
            D = nu_to_D(nu)  # (-∞, ∞)
            # If |nu| is far from pi this result is bounded
            # because the near parabolic region shrinks in its vicinity,
            # otherwise the eccentricity is very close to 1
            # and we are really far away
            M = D_to_M_near_parabolic(D, ecc)
            n = np.sqrt(k / (2 * q ** 3))
    elif ecc == 1:
        # Parabolic
        D = nu_to_D(nu)  # (-∞, ∞)
        M = D_to_M(D)  # (-∞, ∞)
        n = np.sqrt(k / (2 * q ** 3))
    elif 1 + ecc * np.cos(nu) < 0:
        # Unfeasible region
        return np.nan
    elif 1 < ecc <= 1 + delta:
        # NOTE: Do we need to wrap nu here?
        # For hyperbolic orbits, it should anyway be in
        # (-arccos(-1 / ecc), +arccos(-1 / ecc))
        F = nu_to_F(nu, ecc)  # (-∞, ∞)
        if delta <= ecc * np.cosh(F) - 1:
            # Strong hyperbolic
            M = F_to_M(F, ecc)  # (-∞, ∞)
            n = np.sqrt(k * (ecc - 1) ** 3 / q ** 3)
        else:
            # Near parabolic
            D = nu_to_D(nu)  # (-∞, ∞)
            M = D_to_M_near_parabolic(D, ecc)  # (-∞, ∞)
            n = np.sqrt(k / (2 * q ** 3))
    elif 1 + delta < ecc:
        # Strong hyperbolic
        F = nu_to_F(nu, ecc)  # (-∞, ∞)
        M = F_to_M(F, ecc)  # (-∞, ∞)
        n = np.sqrt(k * (ecc - 1) ** 3 / q ** 3)
    else:
        raise RuntimeError

    return M / n


@njit
def nu_from_delta_t(delta_t, ecc, k, q):
    delta = 1e-2
    """True anomaly for given elapsed time since periapsis.

    Parameters
    ----------
    delta_t : float
        Time elapsed since periapsis.
    ecc : float
        Eccentricity.
    k : float
        Gravitational parameter.
    q : float
        Periapsis distance.
    delta : float
        Parameter that controls the size of the near parabolic region.

    Returns
    -------
    nu : float
        True anomaly.

    """
    if ecc < 1 - delta:
        # Strong elliptic
        n = np.sqrt(k * (1 - ecc) ** 3 / q ** 3)
        M = n * delta_t
        # This might represent several revolutions,
        # so we wrap the true anomaly
        E = M_to_E((M + np.pi) % (2 * np.pi) - np.pi, ecc)
        nu = E_to_nu(E, ecc)
    elif 1 - delta <= ecc < 1:
        E_delta = np.arccos((1 - delta) / ecc)
        # We compute M assuming we are in the strong elliptic case
        # and verify later
        n = np.sqrt(k * (1 - ecc) ** 3 / q ** 3)
        M = n * delta_t
        # We check against abs(M) because E_delta could also be negative
        if E_to_M(E_delta, ecc) <= abs(M):
            # Strong elliptic, proceed
            # This might represent several revolutions,
            # so we wrap the true anomaly
            E = M_to_E((M + np.pi) % (2 * np.pi) - np.pi, ecc)
            nu = E_to_nu(E, ecc)
        else:
            # Near parabolic, recompute M
            n = np.sqrt(k / (2 * q ** 3))
            M = n * delta_t
            D = M_to_D_near_parabolic(M, ecc)
            nu = D_to_nu(D)
    elif ecc == 1:
        # Parabolic
        n = np.sqrt(k / (2 * q ** 3))
        M = n * delta_t
        D = M_to_D(M)
        nu = D_to_nu(D)
    elif 1 < ecc <= 1 + delta:
        F_delta = np.arccosh((1 + delta) / ecc)
        # We compute M assuming we are in the strong hyperbolic case
        # and verify later
        n = np.sqrt(k * (ecc - 1) ** 3 / q ** 3)
        M = n * delta_t
        # We check against abs(M) because F_delta could also be negative
        if F_to_M(F_delta, ecc) <= abs(M):
            # Strong hyperbolic, proceed
            F = M_to_F(M, ecc)
            nu = F_to_nu(F, ecc)
        else:
            # Near parabolic, recompute M
            n = np.sqrt(k / (2 * q ** 3))
            M = n * delta_t
            D = M_to_D_near_parabolic(M, ecc)
            nu = D_to_nu(D)
    # elif 1 + delta < ecc:
    else:
        # Strong hyperbolic
        n = np.sqrt(k * (ecc - 1) ** 3 / q ** 3)
        M = n * delta_t
        F = M_to_F(M, ecc)
        nu = F_to_nu(F, ecc)

    return nu


@njit
def farnocchia(k, r0, v0, tof):
    r"""Propagates orbit using mean motion.

    This algorithm depends on the geometric shape of the orbit.
    For the case of the strong elliptic or strong hyperbolic orbits:

    ..  math::

        M = M_{0} + \frac{\mu^{2}}{h^{3}}\left ( 1 -e^{2}\right )^{\frac{3}{2}}t

    .. versionadded:: 0.9.0

    Parameters
    ----------
    k : float
        Standar Gravitational parameter
    r0 : ~astropy.units.Quantity
        Initial position vector wrt attractor center.
    v0 : ~astropy.units.Quantity
        Initial velocity vector.
    tof : float
        Time of flight (s).

    Note
    ----
    This method takes initial :math:`\vec{r}, \vec{v}`, calculates classical orbit parameters,
    increases mean anomaly and performs inverse transformation to get final :math:`\vec{r}, \vec{v}`
    The logic is based on formulae (4), (6) and (7) from http://dx.doi.org/10.1007/s10569-013-9476-9

    """

    # get the initial true anomaly and orbit parameters that are constant over time
    p, ecc, inc, raan, argp, nu0 = rv2coe(k, r0, v0)
    q = p / (1 + ecc)

    delta_t0 = delta_t_from_nu(nu0, ecc, k, q)
    delta_t = delta_t0 + tof

    nu = nu_from_delta_t(delta_t, ecc, k, q)

    return coe2rv(k, p, ecc, inc, raan, argp, nu)


@njit
def fx_xyz_farnocchia(x, dt):
    """
    :param x:
    :param dt:
    :return:
    """
    k = 398600441800000.0
    x_post = np.reshape(farnocchia(k=k, r0=x[:3], v0=x[3:], tof=dt), 6)
    return x_post
