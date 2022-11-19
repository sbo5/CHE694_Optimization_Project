# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy.linalg import lu
from numpy.linalg import inv, cholesky
from numpy import dot, zeros, ones, eye, outer, transpose
from numpy.random import multivariate_normal
from numpy.linalg import solve, matrix_rank


class EnsembleKalmanFilter(object):
    """ This implements the ensemble Kalman filter (EnKF). The EnKF uses
    an ensemble of hundreds to thousands of state vectors that are randomly
    sampled around the estimate, and adds perturbations at each update and
    predict step. It is useful for extremely large systems such as found
    in hydrophysics. As such, this class is admittedly a toy as it is far
    too slow with large N.

    There are many versions of this sort of this filter. This formulation is
    due to Crassidis and Junkins [1]. It works with both linear and nonlinear
    systems.

    References
    ----------

    - [1] John L Crassidis and John L. Junkins. "Optimal Estimation of
      Dynamic Systems. CRC Press, second edition. 2012. pp, 257-9.
    """

    def __init__(self, x, u, P, Ny, N, hx):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------

        x : np.array(Ny)
            state mean

        u : np.array(Ny)
            input mean

        P : np.array((Nx, Nx))
            covariance of the state

        Ny : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), Ny would be 2.

        dt : float
            time step in seconds

        N : int
            number of sigma points (ensembles). Must be greater than 1.

        hx : function hx(x)
            Measurement function. May be linear or nonlinear - converts state
            x into a measurement. Return must be an np.array of the same
            dimensionality as the measurement vector.

        fx : function fx(x, dt)
            State transition function. May be linear or nonlinear. Projects
            state x into the next time period. Returns the projected state x.

        Examples
        --------

        .. code-block:: Python

            def hx(x):
               return np.array([x[0]])

            F = np.array([[1., 1.],
                          [0., 1.]])
            def fx(x, dt):
                return np.dot(F, x)

            x = np.array([0., 1.])
            P = np.eye(2) * 100.
            dt = 0.1
            f = EnKF(x=x, P=P, Ny=1, dt=dt, N=8,
                     hx=hx, fx=fx)

            std_noise = 3.
            f.R *= std_noise**2
            f.Q = Q_discrete_white_noise(2, dt, .01)

            while True:
                z = read_sensor()
                f.predict()
                f.update(np.asarray([z]))

        """

        assert Ny > 0
        self.Nx = len(x)  # number of states
        self.Ny = Ny  # number of measurements
        self.N = N  # number of sigma points
        self.u = u  # inputs
        self.Q = eye(self.Nx)  # process uncertainty
        self.R = eye(self.Ny)  # measurement uncertainty
        self.mean = np.array(zeros(self.Ny))  # not used in update step
        self.hx = hx
        # self.dt = dt
        # self.fx = fx
        # self.initialize(x, P)


    def initialize(self, x, P):
        """ Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__

        Parameters
        ----------

        x : np.array(Ny)
            state mean

        P : np.array((Nx, Nx))
            covariance of the state
        """
        assert x.ndim == 1
        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)
        self.x = x
        self.P = P
        return self.sigmas


    def update(self, ensemble, y, R=None):
        """
        Add a new measurement (y) to the kalman filter. If y is None, nothing
        is changed.

        Parameters
        ----------

        y : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """

        if y is None:
            return
        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.Ny) * R

        N = self.N
        ensemble_update_X = np.zeros((N, self.Nx))
        ensemble_Y = np.zeros((N, self.Ny))

        self.x = np.mean(ensemble, axis=0)

        for i in range(N):
            ensemble_Y[i] = self.hx(ensemble[i])

        y_mean = np.mean(ensemble_Y, axis=0)

# ------------------------------------------------
        P_xx = 0
        for sigma in ensemble:
            s = sigma - self.x
            P_xx += outer(s, s)
        P_xx = P_xx / (N-1)

        e_r = multivariate_normal(self.mean, R, N)
        R_test = np.matmul(e_r.transpose(), e_r)/(N-1)
# ------------------------------------------------

        P_yy = 0
        for sigma in ensemble_Y:
            s = sigma - y_mean
            P_yy += outer(s, s)
        P_yy = P_yy / (N-1) + R

        P_xy = 0
        for i in range(N):
            P_xy += outer(ensemble[i] - self.x, ensemble_Y[i] - y_mean)
        P_xy /= N-1

        if matrix_rank(P_yy) != self.Ny:
            print ("ALERT: P_yy IS SINGULAR")
            return ensemble

# This LU factorization only works when all states are measurements.
#        y = np.zeros(P_xy.shape)
#        Pe, L, U = lu(P_yy)
#        P_xzP = dot(Pe, P_xy)
#        y[0] = P_xzP[0]
#        for i in range(1, self.Nx):
#            alpha = L[i][0:i]
#            beta = P_xzP[i]
#            gamma = y[0:i]
#            temp = dot(alpha, gamma)
#            y[i] = beta - temp
#
#        K = np.zeros(P_xy.shape)
#        Ulast = U[self.Nx-1][-1:]
#        K[self.Nx-1] = y[self.Nx-1]/Ulast
#        for i in range(self.Nx-2, -1, -1):
#            alpha1 = U[i][i+1:]
#            beta1 = y[i]
#            gamma1 = K[i+1:]
#            temp1 = dot(alpha1, gamma1)
#            temp2 = beta1 - temp1
#            Udiag = U[i][i]
#            K[i] = temp2/Udiag
#        Kt = transpose(K)

        K = np.matmul(P_xy, inv(P_yy))

        for i in range(N):
            ensemble_update_X[i] = ensemble[i] + np.matmul(K, y - ensemble_Y[i])
            # ensemble_update_X[i] = ensemble[i] + np.matmul(K, y + e_r[i] - ensemble_Y[i])

#        self.x = np.mean(ensemble_update_X, axis=0)
#        self.P = self.P - dot3(K, P_yy, K.T)
        return ensemble_update_X


# ----------------------------------------------------------------------------------------------------------------------
#    def predict(self):
#        """ Predict next position. """
#
#        N = self.N
#        for i in range(N):
#            self.sigmas[i] = self.fx(self.sigmas[i], self.u)
#        e = multivariate_normal(self.mean, self.Q, N)
#        self.sigmas += e
#        self.x = np.mean(self.sigmas , axis=0)
#
#        P = 0
#        for s in self.sigmas:
#            sx = s - self.x
#            P += outer(sx, sx)
#
#        self.P = P / (N-1)
#        return self.x
