from numba import njit
import numpy as np
from scipy.linalg import block_diag, cholesky
from poliastro.core.propagation import markley

'''
@njit
def fx(x,dt):
    x = markley(398600.4418,x[:3],x[3:],dt) #  # (km^3 / s^2), (km), (km/s), (s)
    x = x.flatten()
    return x

@njit
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    return x[:3]
    
#@jit('f8[:](f8[:],f8[:])', nopython=True)
@njit
def residual_x(a,b):
    c = np.subtract(a,b)
    return c

#@jit('f8[:](f8[:],f8[:])', nopython=True)
@njit
def residual_z(a,b):
    c = np.subtract(a,b)
    return c

@njit
def mean_x(x):
    x_mean = np.mean(x)
    return x_mean

@njit
def mean_z(z):
    z_mean = np.mean(z)
    return z_mean
'''

@njit
def compute_filter_weights(alpha, beta, kappa, n):
    """ Computes the weights for the scaled unscented Kalman filter.
    """

    n = n
    lambda_ = alpha**2 * (n +kappa) - n

    c = .5 / (n + lambda_)
    Wc = np.full(2*n + 1, c)
    Wm = np.full(2*n + 1, c)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)
    return Wm, Wc

# no jit
def order_by_derivative(Q, dim, block_size):
    """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]
    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']
    This works for any covariance matrix or state transition function
    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder
    dim : int >= 1
       number of independent state variables. 3 for x, y, z
    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')
    """

    N = dim * block_size

    D = np.zeros((N, N))

    Q = np.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = np.eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix+block_size, iy:iy+block_size] = f

    return D

# no jit
def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):

    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.
    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.
    Parameters
    -----------
    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)
    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations
    var : float, default=1.0
        variance in the noise
    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.
    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)
        [x x' x'' y y' y'']
        whereas `False` interleaves the dimensions
        [x y z x' y' z' x'' y'' z'']
    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])
    References
    ----------
    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """

    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var

@njit
def unscented_transform(sigmas, Wm, Wc, noise_cov, mean_fn=np.dot, residual_fn=np.subtract):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.
    This works in conjunction with the UnscentedKalmanFilter class.
    Parameters
    ----------
    sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.
    Wm : ndarray [# sigmas per dimension]
        Weights for the mean.
    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.
    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.
    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.
        .. code-block:: Python
            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x
    residual_fn : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
        .. code-block:: Python
            def residual(a, b):
                y = a[0] - b[0]
                y = y % (2 * np.pi)
                if y > np.pi:
                    y -= 2*np.pi
                return y
    Returns
    -------
    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.
    P : ndarray
        covariance of the sigma points after passing throgh the transform.
    Examples
    --------
    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    kmax, n = sigmas.shape

    # x = np.dot(Wm, sigmas) # x = np.dot(Wm, sigmas)
    # x = np.average(sigmas,0,Wm)

    # normalize weights for mean
    Wm = Wm/np.sum(Wm)

    x = mean_fn(Wm, sigmas)

    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    '''if residual_fn is np.subtract:
        y = sigmas - np.array([mean_fn(Wm, sigmas)])
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)'''



    # compute residuals
    y = residual_fn(sigmas, x.T)
    # P = np.dot(y.T, np.dot(np.diag(Wc), y))

    # P = Wc @ y @ y.T

    if residual_fn is np.subtract or residual_fn is None:
        y = residual_fn(sigmas, x.T)
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        y = residual_fn(sigmas, x)
        for k in range(kmax):
            P = P + Wc[k] * np.outer(y[k], y[k])

    # P = np.dot(np.diag(Wc), y).T @ y

    P = P + noise_cov

    return x, P

@njit
def sigma_points(x, P, lambda_, n):
    """ Computes the sigma points for an unscented Kalman filter
    given the mean (x) and covariance(P) of the filter.
    Returns tuple of the sigma points and weights.
    Works with both scalar and array inputs:
    sigma_points (5, 9, 2) # mean 5, covariance 9
    sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
    Parameters
    ----------
    x : An np.array of the means of length n
        Can be a scalar if 1D.
        examples: 1, [1,2], np.array([1,2])
    P : np.array
       Covariance of the filter. If scalar, is treated as eye(n)*P.
    Returns
    -------
    sigmas : np.array, of size (n, 2n+1)
        Two dimensional array of sigma points. Each column contains all of
        the sigmas for one dimension in the problem space.
        Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
    """
    """
    print("##########################")
    print(x)
    print("*"*100)
    print(P)
    print("*"*100)
    print(lambda_)
    print("*"*100)
    print(n)
    print("$"*100)
    """

    #print("LAMBDA")
    #print(lambda_)


    #U = cholesky((lambda_ + n)*P)
    U = np.linalg.cholesky((lambda_ + n)*P).T
    #print("U")
    #print(U)
    sigmas = np.zeros((2*n+1, n))
    sigmas[0] = x
    for k in range(n):
        # pylint: disable=bad-whitespace
        sigmas[k+1]   = np.subtract(x, -U[k])
        sigmas[n+k+1] = np.subtract(x, U[k])

    #print("SIGMAS IN sigma_points FUNC")
    #print(sigmas)

    return sigmas

#@jit('f8[:,:](f8[:],f8[:],f8[:,:],f8[:,:],f8[:])', nopython=True)
@njit
def cross_variance(x, z, sigmas_f, sigmas_h, Wc, residual_x = np.subtract, residual_z = np.subtract):
    """
    Compute cross variance of the state `x` and measurement `z`.
    """

    Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
    N = sigmas_f.shape[0]
    for i in range(N):
        dx = residual_x(sigmas_f[i], x)
        dz = residual_z(sigmas_h[i], z)
        Pxz += Wc[i] * np.outer(dx, dz)
    return Pxz


#@jit('[f8[:],f8[:,:],f8[:,:]](f8[:],f8[:,:],f8[:,:],f8[:],f8[:],f8[:,:],f8,i8)', nopython=True)
@njit
def predict(x, P, Wm, Wc, Q, dt, lambda_, fx, mean_x = np.dot, residual_x = np.subtract):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '
        Important: this MUST be called before update() is called for the first
        time.
        Parameters
        ----------
        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.
        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """
        dim_x = len(x)

        sigmas = sigma_points(x, P, lambda_, dim_x)

        #print("SIGMAS IN PREDICT")
        #print(sigmas)

        n = len(sigmas)

        for i in range(n):
            sigmas[i] = fx(sigmas[i], dt)

        #and pass sigmas through the unscented transform to compute prior
        #x, P = unscented_transform(sigmas=sigmas_f, Wm=Wm, Wc=Wc, noise_cov=Q,
        #                           mean_fn=x_mean, residual_fn=residual_x)
        x, P = unscented_transform(sigmas=sigmas, Wm=Wm, Wc=Wc, noise_cov=Q,
                                   mean_fn=mean_x, residual_fn=residual_x)

        return x, P, sigmas


@njit
def update(x, P, z, Wm, Wc, R, sigmas_f, hx, residual_x = np.subtract, mean_z = np.dot, residual_z = np.subtract):
    """
    Update the UKF with the given measurements. On return,
    self.x and self.P contain the new mean and covariance of the filter.
    Parameters
    ----------
    z : numpy.array of shape (dim_z)
        measurement vector
    R : numpy.array((dim_z, dim_z)), optional
        Measurement noise. If provided, overrides self.R for
        this function call.
    UT : function(sigmas, Wm, Wc, noise_cov), optional
        Optional function to compute the unscented transform for the sigma
        points passed through hx. Typically the default function will
        work - you can use x_mean_fn and z_mean_fn to alter the behavior
        of the unscented transform.
    **hx_args : keyword argument
        arguments to be passed into h(x) after x -> h(x, **hx_args)
    """

    # pass prior sigmas through h(x) to get measurement sigmas
    # the shape of sigmas_h will vary if the shape of z varies, so
    # recreate each time
    dim_z = len(z)
    dim_x = len(x)
    dim_sigmas = dim_x * 2 +1

    sigmas_h = np.zeros((dim_sigmas, dim_z), np.float64)
    for i in range(dim_sigmas):
        sigmas_h[i] = hx(sigmas_f[i])

    # mean and covariance of prediction passed through unscented transform
    #zp, S = unscented_transform(sigmas_h, Wm, Wc, R, z_mean, residual_z) # S = system uncertainty
    zp, S = unscented_transform(sigmas=sigmas_h, Wm=Wm, Wc=Wc, noise_cov=R,
                                mean_fn=mean_z, residual_fn=residual_z) # S = system uncertainty

    # compute cross variance of the state and the measurements
    Pxz = cross_variance(x = x, z = zp, sigmas_f = sigmas_f, sigmas_h = sigmas_h, Wc = Wc, residual_x = residual_x,
                         residual_z = residual_z)

    SI = np.linalg.inv(S)
    K = np.dot(Pxz, SI)        # Kalman gain
    y = residual_z(z, zp)   # residual

    # update Gaussian state estimate (x, P)
    x = x + np.dot(K, y)
    P = P - np.dot(K, np.dot(S, K.T))

    return x, P

"""
    Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
        This is for convience, so everything is sized correctly on
        creation. If you are using multiple sensors the size of `z` can
        change based on the sensor. Just provide the appropriate hx function
    dt : float
        Time between steps in seconds.
    hx : function(x)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_z).
    fx : function(x,dt)
        function that returns the state x transformed by the
        state transistion function. dt is the time step in seconds.
    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. For example, MerweScaledSigmaPoints implements the alpha,
        beta, kappa parameterization of Van der Merwe, and
        JulierSigmaPoints implements Julier's original kappa
        parameterization. See either of those for the required
        signature of this class if you want to implement your own.
    sqrt_fn : callable(ndarray), default=None (implies scipy.linalg.cholesky)
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.
        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing as far as this class is concerned.
    x_mean_fn : callable  (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.
        .. code-block:: Python
            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x
    z_mean_fn : callable  (sigma_points, weights), optional
        Same as x_mean_fn, except it is called for sigma points which
        form the measurements after being passed through hx().
    residual_x : callable (x, y), optional
    residual_z : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars. One is for the state variable,
        the other is for the measurement state.
        .. code-block:: Python
            def residual(a, b):
                y = a[0] - b[0]
                if y > np.pi:
                    y -= 2*np.pi
                if y < -np.pi:
                    y = 2*np.pi
                return y
    Attributes
    ----------
    x : numpy.array(dim_x)
        state estimate vector
    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix
    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : ndarray
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        process noise matrix
    K : numpy.array
        Kalman gain
    y : numpy.array
        innovation residual
    log_likelihood : scalar
        Log likelihood of last measurement update.
    likelihood : float
        likelihood of last measurment. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the measurement. Read only.
    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead:
        .. code-block:: Python
            kf.inv = np.linalg.pinv

    For in depth explanations see my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    Also see the filterpy/kalman/tests subdirectory for test code that
    may be illuminating.
    References
    ----------
    .. [1] Julier, Simon J. "The scaled unscented transformation,"
        American Control Converence, 2002, pp 4555-4559, vol 6.
        Online copy:
        https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF
    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
        nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
        Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.
        Online Copy:
        https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
           the nonlinear transformation of means and covariances in filters
           and estimators," IEEE Transactions on Automatic Control, 45(3),
           pp. 477-482 (March 2000).
    .. [4] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
           Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
           Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.
           https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    .. [5] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
           Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.
    .. [6] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)
    """

"""
# Configurable Settings:
dim_x = 6
dim_z = 3
'''hx = hx
fx = fx'''
dt = 30.0
Qvar = 0.000001
obs_noise = np.repeat(100,3)
x = np.array([ 34090.8583,  23944.7744,  6503.06682, -1.983785080,2.15041744,  0.913881611])
P = np.eye(6) * np.array([ 100,  100,  100, 0.1, 0.1,  0.1])
alpha = 0.001
beta = 2.0
kappa = 3-6

# derived values
Wm, Wc = compute_filter_weights(alpha, beta, kappa, dim_x) # weights for the means and covariances.
R = np.eye(dim_z)*obs_noise
Q = Q_discrete_white_noise(dim=2, dt=dt, var=Qvar**2, block_size=3, order_by_dim=False)
lambda_ = alpha**2 * (dim_x + kappa) - dim_x
num_sigmas = 2*dim_x + 1
# sigma points transformed through f(x) and h(x)
# variables for efficiency so we don't recreate every update
sigmas_f = np.zeros((num_sigmas, dim_x)) # f(x)
sigmas_h = np.zeros((num_sigmas, dim_z)) # h(x)


Q_comp = pf_Q_discrete_white_noise(dim=2, dt=dt, var=Qvar**2, block_size=3, order_by_dim=False)
points_compare = pf_MerweScaledSigmaPoints(n=6, alpha=0.001, beta=2., kappa=3-6)
filter_comp = pf_UnscentedKalmanFilter(dim_x,dim_z,dt,hx,fx,points_compare)
filter_comp.R = R
filter_comp.x = x
filter_comp.P = P
filter_comp.Q = Q_comp

x_post, P_post, sigmas_f_post = predict(x, P, sigmas_f, Wm, Wc, Q, dt, dim_x)
filter_comp.predict()
np.array_equal(P_post,filter_comp.P)
np.array_equal(x_post,filter_comp.x)
#np.array_equal(sigmas_f_post,filter_comp.sigmas_f) # predict does not return sigmas_f
np.array_equal(R,filter_comp.R)
np.array_equal(Q,filter_comp.Q)
np.array_equal(Wm,filter_comp.Wm)
np.array_equal(Wc,filter_comp.Wc)
#np.array_equal(residual_x,filter_comp.residual_x)
#np.array_equal(residual_z,filter_comp.residual_z)
np.array_equal(Q,filter_comp.Q)


z = x[0:3]+np.diag(R)
x_post2, P_post2 = update(x_post, P_post, z, R, Wm, Wc, sigmas_f_post, sigmas_h)
filter_comp.update(z)

np.array_equal(P_post2,filter_comp.P)
np.array_equal(x_post2,filter_comp.x)

sim_noise = []
for i in range(10):
    sim_noise.append(np.random.normal(loc=0,scale=10, size=3))

import copy
state = copy.deepcopy(x)
for sn in sim_noise:
    state = fx(state,dt)
    z = state[:3] + sn
    filter_comp.predict()
    #filter_comp.update(z)
    x, P, sigmas_f = predict(x, P, sigmas_f, Wm, Wc, Q, dt, dim_x)
    #x, P = update(x, P, z, R, Wm, Wc, sigmas_f, sigmas_h)

np.array_equal(P,filter_comp.P)
np.array_equal(x,filter_comp.x)
np.subtract(np.linalg.det(P),np.linalg.det(filter_comp.P))/np.linalg.det(P)
np.mean(np.subtract(x,filter_comp.x)/x)
"""
