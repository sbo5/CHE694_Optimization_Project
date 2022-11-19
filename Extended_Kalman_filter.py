from __future__ import print_function, division # Grab some handy Python3 stuff.
import scipy.linalg
from casadi import *
import numpy as np
from Parameters import time_parameters, cmatrix_single


# ----------------------------------------------------------------------------------------------------------------------
# Continuous --> Discrete EKF
# ----------------------------------------------------------------------------------------------------------------------
def ekf_continuous_discrete(x_pre,x_cur,u_pre,y_cur,P_pre,Q,R,hx,f_jacxFun=None,f_jacuFun=None,h_jacxFun=None):
    """
    This ekf is discrete-time to continuous-time EKF. That is, it can be used to
    ODE models.

    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).


    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows

        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]

    Depending on your specific application, you will only be interested in
    some of these values.
    """
    DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
    # Nz, Nx, Nw, Ny, Nv, Nu, Np, dz, Dim, Nx_aug, Nw_aug, Nsigma = space_parameters()  # space related parameters

    Nx = x_pre.shape[0]  # size of x_pre or x_cur
    Nw = Nx

    # predict step
    A = np.array(f_jacxFun(x_pre, u_pre, np.zeros((Nw, 1))))
    B = np.array(f_jacuFun(x_pre, u_pre, np.zeros((Nw, 1))))

    [Ad, Bd] = c2d(A, B, DeltaT, Bp=None, f=None, asdict=False)

    # Get linearization of measurement.
    C = np.array(h_jacxFun(x_cur))

    # Prediction step for P
    P_kplus_k = mtimes(mtimes(Ad, P_pre), np.transpose(Ad)) + Q  # update matrix P (k+1|k)

    # update step
    Lk = mtimes(mtimes(P_kplus_k, np.transpose(C)), np.linalg.inv(
        mtimes(mtimes(C, P_kplus_k), np.transpose(C)) + R))  # L = Pc(:,:,i)*C'/(C*Pc(:,:,i)*C'+
    x_kplus_kplus = x_cur + mtimes(Lk, (y_cur - hx(x_cur)))  ## update current estimate
    # x_kplus_kplus = x_cur + mtimes(Lk, (y_cur - mtimes(C, x_cur)))  ## update current estimate
    P_kplus_kplus = mtimes((np.identity(Nx) - mtimes(Lk, C)),
                           P_kplus_k)  ## update current estimate of P(k|k) based on P(k|k-1)
    x_kplus_kplus = x_kplus_kplus.full().ravel()
    P_kplus_kplus = P_kplus_kplus.full()
    P_kplus_k = P_kplus_k.full()
    Lk = Lk.full()
    return [x_kplus_kplus, P_kplus_kplus, P_kplus_k]

# ----------------------------------------------------------------------------------------------------------------------
# Discrete --> Discrete EKF
# ----------------------------------------------------------------------------------------------------------------------
# def ekf_discrete_discrete(x_pre,x_cur,u_pre,y_cur,P_pre,Q,R,hx,f_jacxFun=None,f_jacuFun=None,h_jacxFun=None):
#     """
#     This ekf is discrete-time to discrete-time EKF. That is, it can be used to
#     discrete-time state-space model. For ODEs, please use "ekf_continuous_discrete"
#
#     Updates the prior distribution P^- using the Extended Kalman filter.
#
#     f and h should be casadi functions. f must be discrete-time. P, Q, and R
#     are the prior, state disturbance, and measurement noise covariances. Note
#     that f must be f(x,u,w) and h must be h(x).
#
#     If specified, f_jac and h_jac should be initialized jacobians. This saves
#     some time if you're going to be calling this many times in a row, althouth
#     it's really not noticable unless the models are very large.
#
#     The value of x that should be fed is xhat(k | k-1), and the value of P
#     should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
#     to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
#     P(k+1 | k). The return values are a list as follows
#
#         [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]
#
#     Depending on your specific application, you will only be interested in
#     some of these values.
#     """

# ----------------------------------------------------------------------------------------------------------------------
# Continuous --> Discrete EKF
# ----------------------------------------------------------------------------------------------------------------------
def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
    """
    Discretizes affine system (A, B, Bp, f) with timestep Delta.

    This includes disturbances and a potentially nonzero steady-state, although
    Bp and f can be omitted if they are not present.

    If asdict=True, return value will be a dictionary with entries A, B, Bp,
    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
    if Bp and f are provided, otherwise a 2-element list [A, B].
    """
    n_A = A.shape[0]
    I = np.eye(n_A)
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, I]),
                                             np.zeros((n_A, 2*n_A)))))
    Ad = D[:n_A,:n_A]
    Id = D[:n_A,n_A:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)   
            
    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd, fd]
    return retval
