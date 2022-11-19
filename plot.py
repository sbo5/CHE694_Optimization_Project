# -*- coding: utf-8 -*-
"""
Created on July 27 2019

@author: Song Bo
"""

# from __future__ import (print_function, division)  # Grab some handy Python3 stuff.
import sys
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy import linalg, integrate
import numpy as np
from matplotlib import pyplot as plt
import casadi as casadi

# sys.path.insert(0, "D:\Google Drive\Graduate\Research Project\Projects_Codes\MHE_paraEst\Chapter1\Results")

para = np.array([1.04/100/3600, 0.43, 0.078, 3.6, 1.56, 1.31/100/3600, 0.39, 0.100, 0.059*100, 1.48])

# data_dir = pjoin('studyOnPQR/1Px1e2Ppara1e3_Qx1e-1Qpara1e-20')
mat_fname_x = pjoin('x.mat')
x_1 = sio.loadmat(mat_fname_x)
x_exp_1 = x_1['xmea']
x_mhe = x_1['xmhe']  # no matter dis or central, x_mhe are the same
x_ol_1 = x_1['xol']
x_clean_1 = x_1['xclean']

# mat_fname_y = pjoin(data_dir, 'y.mat')
y_1 = sio.loadmat('Data/y.mat')
y_exp_1 = y_1['ymea']
y_mhe_1 = y_1['ymhe']

# mat_fname_SE = pjoin('SE.mat')
# SE_1 = sio.loadmat(mat_fname_SE)
# RMSE_mhe_y = SE_1['Jmhe_Y_SSE']
# RMSE_mhe_x = SE_1['Jmhe_X_SSE']
# RMSE_mhe_p = SE_1['Jmhe_P_SSE']

mat_fname_timePara = pjoin('time_parameters.mat')
time_para = sio.loadmat(mat_fname_timePara)
Tplot = time_para['Tplot'].ravel()
DeltaT = time_para['DeltaT'].item()
calNode = time_para['calNode'].item()
Nsim = time_para['Nsim'].item()
Tsim = time_para['Tsim'].item()
DeltaTmhe = time_para['DeltaTmhe'].item()

mat_fname_spacePara = pjoin('space_parameters.mat')
space_para = sio.loadmat(mat_fname_spacePara)
# Nest = space_para['Nest'].item()
NxPerEst = space_para['NxPerEst'].item()
# NyPerEst = space_para['NyPerEst'].item()
# Nsoil = space_para['Nsoil'].item()
Np = space_para['Np'].item()
Nx = space_para['Nx'].item()
Ny = space_para['Ny'].item()
NpTotal = space_para['NpTotal'].item()
# Nxx = space_para['Nxx'].item()
# Nxy = space_para['Nxy'].item()
# Nxz = space_para['Nxz'].item()
# NxPerSoil = space_para['NxPerSoil'].item()
# Np = Np*Nsoil
Nx_aug = space_para['Nx_aug'].item()

RMSE_exp = np.sqrt(np.sum((x_exp_1 - x_clean_1) ** 2, axis=1) / Nx)

# ------------------------------------------------------------------------------------------------------
x_mhe_timeS = np.zeros((Nsim+1, Nx_aug))

x_mhe_timeS[:,0:NxPerEst] = x_mhe[:,:NxPerEst,0]
x_mhe_timeS[:,NxPerEst:Nx] = x_mhe[:,:NxPerEst,1]
x_mhe_timeS[:,Nx:Nx+Np] = x_mhe[:,NxPerEst:,0]
x_mhe_timeS[:,Nx+Np:] = x_mhe[:,NxPerEst:,1]

RMSE_X = np.sqrt(np.sum((x_exp_1[:,0:Nx] - x_mhe_timeS[:,0:Nx]) ** 2, axis=1) / Nx)
RMSE_P = np.sqrt(np.sum((x_exp_1[:,Nx:] - x_mhe_timeS[:,Nx:]) ** 2, axis=1) / NpTotal)
# ----------------------------------------------------------------------------------------------------------------------
def thetaFun_np_5(psi,pars):
    Se = casadi.if_else(psi>=0., 1., (1+abs(psi*pars[3]*3.6+1.e-20)**(pars[4]*1.56)+1.e-20)**(-(1-1/((pars[4]*1.56)+1.e-20))))
    theta = (pars[2]*0.078+(pars[1]*0.43-pars[2]*0.078)*Se)
    theta = theta.full().ravel()
    return theta


def thetaFun_np_4(psi,pars):
    Se = casadi.if_else(psi>=0., 1., (1+abs(psi*pars[2]*3.6+1.e-20)**(pars[3]*1.56)+1.e-20)**(-(1-1/((pars[3]*1.56)+1.e-20))))
    theta = (1*0.078+(pars[1]*0.43-1*0.078)*Se)
    theta = theta.full().ravel()
    return theta


# theta_exp_1 = np.zeros((Nsim+1, Nx))
# theta_mhe = np.zeros((Nsim+1, Nx))
# # theta_ekf = np.zeros((Nsim+1, Nx))
# # theta_enkf = np.zeros((Nsim+1, Nx))
# for t in range(Nsim+1):
#     theta_exp_1[t,:] = thetaFun_np_5(x_exp_1[t, :Nx], x_exp_1[t, Nx:])
#     theta_mhe[t,:] = thetaFun_np_5(x_mhe[t, :Nx], x_mhe[t, Nx:])
#     # theta_ekf[t,:] = thetaFun_np_4(x_ekf[t, :Nx], x_ekf[t, Nx:])
#     # theta_enkf[t,:] = thetaFun_np_4(x_enkf[t, :Nx], x_enkf[t, Nx:])
# -----------------------------------------------------------------------------------------------------------------
end = 39+1
# f, axs = plt.subplots(5, 1, sharex=True, figsize=(6.0, 7.0))
# axs[0].plot(Tplot[:end] / DeltaT, theta_exp_1[:end, 0], 'r-')
# axs[0].plot(Tplot[:end] / DeltaT, theta_mhe[:end, 0], 'k--')
# # axs[0].plot(Tplot[:end] / DeltaT, theta_ekf[:end, 0], 'b-.')
# # axs[0].plot(Tplot[:end] / DeltaT, theta_enkf[:end, 0], 'g:', markersize=1)
# axs[0].set_xlim(0, end-1)
# axs[0].set_ylim(0.1, 0.4)
# axs[0].set_yticks([0.1, 0.2, 0.3, 0.4])
# axs[0].set_ylabel(r'$\theta_{1}$ (m)')
# # axs[0].grid()
#
# axs[1].plot(Tplot[:end] / DeltaT, theta_exp_1[:end, 5], 'r-')
# axs[1].plot(Tplot[:end] / DeltaT, theta_mhe[:end, 5], 'k--')
# # axs[1].plot(Tplot[:end] / DeltaT, theta_ekf[:end, 5], 'b-.')
# # axs[1].plot(Tplot[:end] / DeltaT, theta_enkf[:end, 5], 'g:')
# axs[1].set_xlim(0, end-1)
# axs[1].set_ylim(0.1, 0.4)
# axs[1].set_yticks([0.1, 0.2, 0.3, 0.4])
# axs[1].set_ylabel(r'$\theta_{6}$ (m)')
# # axs[1].grid()
#
# axs[2].plot(Tplot[:end] / DeltaT, theta_exp_1[:end, 11], 'r-')
# axs[2].plot(Tplot[:end] / DeltaT, theta_mhe[:end, 11], 'k--')
# # axs[2].plot(Tplot[:end] / DeltaT, theta_ekf[:end, 11], 'b-.')
# # axs[2].plot(Tplot[:end] / DeltaT, theta_enkf[:end, 11], 'g:')
# axs[2].set_xlim(0, end-1)
# axs[2].set_ylim(0.1, 0.4)
# axs[2].set_yticks([0.1, 0.2, 0.3, 0.4])
# axs[2].set_ylabel(r'$\theta_{12}$ (m)')
# # axs[2].grid()
#
# axs[3].plot(Tplot[:end] / DeltaT, theta_exp_1[:end, 17], 'r-')
# axs[3].plot(Tplot[:end] / DeltaT, theta_mhe[:end, 17], 'k--')
# # axs[3].plot(Tplot[:end] / DeltaT, theta_ekf[:end, 17], 'b-.')
# # axs[3].plot(Tplot[:end] / DeltaT, theta_enkf[:end, 17], 'g:')
# axs[3].set_xlim(0, end-1)
# axs[3].set_ylim(0.1, 0.4)
# axs[3].set_yticks([0.1, 0.2, 0.3, 0.4])
# axs[3].set_ylabel(r'$\theta_{18}$ (m)')
# # axs[3].grid()
#
# axs[4].plot(Tplot[:end] / DeltaT, theta_exp_1[:end, 29], 'r-')
# axs[4].plot(Tplot[:end] / DeltaT, theta_mhe[:end, 29], 'k--')
# # axs[4].plot(Tplot[:end] / DeltaT, theta_ekf[:end, 29], 'b-.')
# # axs[4].plot(Tplot[:end] / DeltaT, theta_enkf[:end, 29], 'g:')
# axs[4].set_xlim(0, end-1)
# axs[4].set_ylim(0.1, 0.4)
# axs[4].set_yticks([0.1, 0.2, 0.3, 0.4])
# axs[4].set_ylabel(r'$\theta_{30}$ (m)')
# # axs[4].grid()
#
# axs[4].set_xlabel('Time, t (day)')
# plt.xticks(Tplot[:end:calNode * 24 * (3600/DeltaTmhe)] / DeltaT, np.arange(Tsim / (DeltaTmhe * (3600/DeltaTmhe) * 24) + 1, step=1))
# f.legend(['EXP', 'MHE', 'EKF', 'EnKF'], loc=9, ncol=4, frameon=False)
# f.savefig('Y_Figure.pdf')

# -------------------------------------------------------------------------------
f, axs = plt.subplots(5, 1, sharex=True, figsize=(6.0, 7.0))
axs[0].plot(Tplot[:end] / DeltaT, x_exp_1[:end, 0], 'r-')
axs[0].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, 0], 'k--')
# axs[0].plot(Tplot[:end] / DeltaT, x_ekf[:end, 0], 'b-.')
# axs[0].plot(Tplot[:end] / DeltaT, x_enkf[:end, 0], 'g:', markersize=1)
axs[0].set_xlim(0, end-1)
axs[0].set_ylim(-0.8, -0.2)
axs[0].set_yticks([-0.75, -0.50, -0.25])
axs[0].set_ylabel(r'$h_{1}$ (m)')
# axs[0].grid()

axs[1].plot(Tplot[:end] / DeltaT, x_exp_1[:end, 5], 'r-')
axs[1].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, 5], 'k--')
# axs[1].plot(Tplot[:end] / DeltaT, x_ekf[:end, 5], 'b-.')
# axs[1].plot(Tplot[:end] / DeltaT, x_enkf[:end, 5], 'g:')
axs[1].set_xlim(0, end-1)
axs[1].set_ylim(-0.7, -0.3)
axs[1].set_yticks([-0.65, -0.50, -0.35])
axs[1].set_ylabel(r'$h_{6}$ (m)')
# axs[1].grid()

axs[2].plot(Tplot[:end] / DeltaT, x_exp_1[:end, 11], 'r-')
axs[2].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, 11], 'k--')
# axs[2].plot(Tplot[:end] / DeltaT, x_ekf[:end, 11], 'b-.')
# axs[2].plot(Tplot[:end] / DeltaT, x_enkf[:end, 11], 'g:')
axs[2].set_xlim(0, end-1)
axs[2].set_ylim(-0.7, -0.3)
axs[2].set_yticks([-0.65, -0.50, -0.35])
axs[2].set_ylabel(r'$h_{12}$ (m)')
# axs[2].grid()

axs[3].plot(Tplot[:end] / DeltaT, x_exp_1[:end, 17], 'r-')
axs[3].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, 17], 'k--')
# axs[3].plot(Tplot[:end] / DeltaT, x_ekf[:end, 17], 'b-.')
# axs[3].plot(Tplot[:end] / DeltaT, x_enkf[:end, 17], 'g:')
axs[3].set_xlim(0, end-1)
axs[3].set_ylim(-0.7, -0.3)
axs[3].set_yticks([-0.65, -0.50, -0.35])
axs[3].set_ylabel(r'$h_{18}$ (m)')
# axs[3].grid()

axs[4].plot(Tplot[:end] / DeltaT, x_exp_1[:end, 29], 'r-')
axs[4].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, 29], 'k--')
# axs[4].plot(Tplot[:end] / DeltaT, x_ekf[:end, 29], 'b-.')
# axs[4].plot(Tplot[:end] / DeltaT, x_enkf[:end, 29], 'g:')
axs[4].set_xlim(0, end-1)
axs[4].set_ylim(-0.7, -0.3)
axs[4].set_yticks([-0.65, -0.50, -0.35])
axs[4].set_ylabel(r'$h_{30}$ (m)')
# axs[4].grid()

axs[4].set_xlabel('Time, t (day)')
plt.xticks(Tplot[:end:calNode * 24 * (3600/DeltaTmhe)] / DeltaT, np.arange(Tsim / (DeltaTmhe * (3600/DeltaTmhe) * 24) + 1, step=1))
f.legend(['EXP', 'MHE'], loc=9, ncol=4, frameon=False)
# f.savefig('X_Figure_1.pdf')
f.show()

# -------------------------------------------------------------------------------
f, axs = plt.subplots(5, 1, sharex=True, figsize=(6.0, 7.0))
axs[0].plot(Tplot[:end] / DeltaT, x_exp_1[:end, Nx//2+0], 'r-')
axs[0].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, Nx//2+0], 'k--')
# axs[0].plot(Tplot[:end] / DeltaT, x_ekf[:end, 0], 'b-.')
# axs[0].plot(Tplot[:end] / DeltaT, x_enkf[:end, 0], 'g:', markersize=1)
axs[0].set_xlim(0, end-1)
axs[0].set_ylim(-0.8, -0.2)
axs[0].set_yticks([-0.75, -0.50, -0.25])
axs[0].set_ylabel(r'$h_{1}$ (m)')
# axs[0].grid()

axs[1].plot(Tplot[:end] / DeltaT, x_exp_1[:end, Nx//2+5], 'r-')
axs[1].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, Nx//2+5], 'k--')
# axs[1].plot(Tplot[:end] / DeltaT, x_ekf[:end, 5], 'b-.')
# axs[1].plot(Tplot[:end] / DeltaT, x_enkf[:end, 5], 'g:')
axs[1].set_xlim(0, end-1)
axs[1].set_ylim(-0.7, -0.3)
axs[1].set_yticks([-0.65, -0.50, -0.35])
axs[1].set_ylabel(r'$h_{6}$ (m)')
# axs[1].grid()

axs[2].plot(Tplot[:end] / DeltaT, x_exp_1[:end, Nx//2+11], 'r-')
axs[2].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, Nx//2+11], 'k--')
# axs[2].plot(Tplot[:end] / DeltaT, x_ekf[:end, 11], 'b-.')
# axs[2].plot(Tplot[:end] / DeltaT, x_enkf[:end, 11], 'g:')
axs[2].set_xlim(0, end-1)
axs[2].set_ylim(-0.7, -0.3)
axs[2].set_yticks([-0.65, -0.50, -0.35])
axs[2].set_ylabel(r'$h_{12}$ (m)')
# axs[2].grid()

axs[3].plot(Tplot[:end] / DeltaT, x_exp_1[:end, Nx//2+17], 'r-')
axs[3].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, Nx//2+17], 'k--')
# axs[3].plot(Tplot[:end] / DeltaT, x_ekf[:end, 17], 'b-.')
# axs[3].plot(Tplot[:end] / DeltaT, x_enkf[:end, 17], 'g:')
axs[3].set_xlim(0, end-1)
axs[3].set_ylim(-0.7, -0.3)
axs[3].set_yticks([-0.65, -0.50, -0.35])
axs[3].set_ylabel(r'$h_{18}$ (m)')
# axs[3].grid()

axs[4].plot(Tplot[:end] / DeltaT, x_exp_1[:end, Nx//2+29], 'r-')
axs[4].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, Nx//2+29], 'k--')
# axs[4].plot(Tplot[:end] / DeltaT, x_ekf[:end, 29], 'b-.')
# axs[4].plot(Tplot[:end] / DeltaT, x_enkf[:end, 29], 'g:')
axs[4].set_xlim(0, end-1)
axs[4].set_ylim(-0.7, -0.3)
axs[4].set_yticks([-0.65, -0.50, -0.35])
axs[4].set_ylabel(r'$h_{30}$ (m)')
# axs[4].grid()

axs[4].set_xlabel('Time, t (day)')
plt.xticks(Tplot[:end:calNode * 24 * (3600/DeltaTmhe)] / DeltaT, np.arange(Tsim / (DeltaTmhe * (3600/DeltaTmhe) * 24) + 1, step=1))
f.legend(['EXP', 'MHE'], loc=9, ncol=4, frameon=False)
# f.savefig('X_Figure_2.pdf')
f.show()

# -------------------------------------------------------------------------------
f, axs = plt.subplots(4, 1, sharex=True, figsize=(6.0, 7.0))
axs[0].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -10]*para[0], 'r-')
axs[0].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -10]*para[0], 'k--')
# axs[0].plot(Tplot[:end] / DeltaT, x_ekf[:end, 32]*para[0], 'b-.')
# axs[0].plot(Tplot[:end] / DeltaT, x_enkf[:end, 32]*para[0], 'g:')
axs[0].set_xlim(0, end-1)
axs[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
axs[0].set_ylim([0.75*para[0], 1.45*para[0]])
axs[0].set_yticks([0.8*para[0], 1.0*para[0], 1.2*para[0], 1.4*para[0]])
axs[0].set_ylabel(r'$K_{s}$ (m/s)')
# axs[0].grid()

axs[1].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -9]*para[1], 'r-')
axs[1].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -9]*para[1], 'k--')
# axs[1].plot(Tplot[:end] / DeltaT, x_ekf[:end, 33]*para[1], 'b-.')
# axs[1].plot(Tplot[:end] / DeltaT, x_enkf[:end, 33]*para[1], 'g:')
axs[1].set_xlim(0, end-1)
axs[1].set_ylim([0.35*para[1], 1.25*para[1]])
axs[1].set_yticks([0.4*para[1], 0.6*para[1], 0.8*para[1], 1.0*para[1], 1.2*para[1]])
axs[1].set_ylabel(r'$\theta_{s}$ $(m^{3}/m^{3})$')
# axs[1].grid()

axs[2].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -7]*para[3], 'r-')
axs[2].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -7]*para[3], 'k--')
# axs[2].plot(Tplot[:end] / DeltaT, x_ekf[:end, 34]*para[3], 'b-.')
# axs[2].plot(Tplot[:end] / DeltaT, x_enkf[:end, 34]*para[3], 'g:')
axs[2].set_xlim(0, end-1)
axs[2].set_ylim([0.75*para[3], 1.45*para[3]])
axs[2].set_yticks([0.8*para[3], 1.0*para[3], 1.2*para[3], 1.4*para[3]])
axs[2].set_ylabel(r'$\alpha$ (1/m)')
# axs[2].grid()

axs[3].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -6]*para[4], 'r-')
axs[3].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -6]*para[4], 'k--')
# axs[3].plot(Tplot[:end] / DeltaT, x_ekf[:end, 35]*para[4], 'b-.')
# axs[3].plot(Tplot[:end] / DeltaT, x_enkf[:end, 35]*para[4], 'g:')
axs[3].set_xlim(0, end-1)
axs[3].set_ylim([0.75*para[4], 1.65*para[4]])
axs[3].set_yticks([0.8*para[4], 1.0*para[4], 1.2*para[4], 1.4*para[4], 1.6*para[4]])
axs[3].set_ylabel(r'$n$')
# axs[3].grid()

axs[3].set_xlabel('Time, t (day)')
plt.xticks(Tplot[:end:calNode * 24 * (3600/DeltaTmhe)] / DeltaT, np.arange(Tsim / (DeltaTmhe * (3600/DeltaTmhe) * 24) + 1, step=1))
f.legend(['EXP', 'MHE'], loc=9, ncol=4, frameon=False)
# f.savefig('P_Figure_1.pdf')
f.show()

# -------------------------------------------------------------------------------
f, axs = plt.subplots(4, 1, sharex=True, figsize=(6.0, 7.0))
axs[0].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -5]*para[5], 'r-')
axs[0].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -5]*para[5], 'k--')
# axs[0].plot(Tplot[:end] / DeltaT, x_ekf[:end, 32]*para[0], 'b-.')
# axs[0].plot(Tplot[:end] / DeltaT, x_enkf[:end, 32]*para[0], 'g:')
axs[0].set_xlim(0, end-1)
axs[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
axs[0].set_ylim([0.75*para[5], 1.45*para[5]])
axs[0].set_yticks([0.8*para[5], 1.0*para[5], 1.2*para[5], 1.4*para[5]])
axs[0].set_ylabel(r'$K_{s}$ (m/s)')
# axs[0].grid()

axs[1].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -4]*para[6], 'r-')
axs[1].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -4]*para[6], 'k--')
# axs[1].plot(Tplot[:end] / DeltaT, x_ekf[:end, 33]*para[1], 'b-.')
# axs[1].plot(Tplot[:end] / DeltaT, x_enkf[:end, 33]*para[1], 'g:')
axs[1].set_xlim(0, end-1)
axs[1].set_ylim([0.35*para[6], 1.25*para[6]])
axs[1].set_yticks([0.4*para[6], 0.6*para[6], 0.8*para[6], 1.0*para[6], 1.2*para[6]])
axs[1].set_ylabel(r'$\theta_{s}$ $(m^{3}/m^{3})$')
# axs[1].grid()

axs[2].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -2]*para[8], 'r-')
axs[2].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -2]*para[8], 'k--')
# axs[2].plot(Tplot[:end] / DeltaT, x_ekf[:end, 34]*para[3], 'b-.')
# axs[2].plot(Tplot[:end] / DeltaT, x_enkf[:end, 34]*para[3], 'g:')
axs[2].set_xlim(0, end-1)
axs[2].set_ylim([0.75*para[8], 1.45*para[8]])
axs[2].set_yticks([0.8*para[8], 1.0*para[8], 1.2*para[8], 1.4*para[8]])
axs[2].set_ylabel(r'$\alpha$ (1/m)')
# axs[2].grid()

axs[3].plot(Tplot[:end] / DeltaT, x_exp_1[:end, -1]*para[9], 'r-')
axs[3].plot(Tplot[:end] / DeltaT, x_mhe_timeS[:end, -1]*para[9], 'k--')
# axs[3].plot(Tplot[:end] / DeltaT, x_ekf[:end, 35]*para[4], 'b-.')
# axs[3].plot(Tplot[:end] / DeltaT, x_enkf[:end, 35]*para[4], 'g:')
axs[3].set_xlim(0, end-1)
axs[3].set_ylim([0.75*para[9], 1.65*para[9]])
axs[3].set_yticks([0.8*para[9], 1.0*para[9], 1.2*para[9], 1.4*para[9], 1.6*para[9]])
axs[3].set_ylabel(r'$n$')
# axs[3].grid()

axs[3].set_xlabel('Time, t (day)')
plt.xticks(Tplot[:end:calNode * 24 * (3600/DeltaTmhe)] / DeltaT, np.arange(Tsim / (DeltaTmhe * (3600/DeltaTmhe) * 24) + 1, step=1))
f.legend(['EXP', 'MHE'], loc=9, ncol=4, frameon=False)
# f.savefig('P_Figure_2.pdf')
f.show()

# for i in range(Ny):  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:]/DeltaT, y[:, i], '-', label='y_exp')
#     # plt.plot(Tplot[:]/DeltaT, y_ekf_plot[:, i], '.', markersize=3, label='y_ekf')
#     # plt.plot(Tplot[:]/DeltaT, y_enkf_plot[:, i], '.', markersize=3, label='y_enkf')
#     plt.plot(Tplot[:]/DeltaT, y_mhe[:, i], '.', markersize=4, label='y_mhe')
#     plt.plot(Tplot[:]/DeltaT, y_ol[:, i], '--', label='y_ol')
#     plt.plot(Tplot[:]/DeltaT, y_clean[:, i], '-', label='y_clean')
#     plt.legend()
#     plt.xlabel('Time, t (hr)')
#     plt.ylabel('Soil Moisture, theta (m3/m3)')
#     plt.xticks(Tplot[:end:calNode * 24 * (3600 / DeltaTmhe)] / DeltaT,
#                np.arange(Tsim / (DeltaTmhe * (3600 / DeltaTmhe) * 24) + 1, step=1))
#     # plt.savefig('Y_Figure_%g.png' % (i+1))
#     plt.show()
#     plt.close()

# for i in range(Nx):  # choose how many states do yo wanna plot
#     plt.figure()
#     # plt.plot(Tplot[:]/DeltaT, yy[:, i], '-', label='y_exp')
#     plt.plot(Tplot[:end]/DeltaT, x_exp_1[:end, i], '-', label='x_exp')
#     # plt.plot(Tplot[:]/DeltaT, x_ekf[:, i], '.', markersize=3, label='x_ekf')
#     # plt.plot(Tplot[:]/DeltaT, x_enkf[:, i], '.', markersize=3, label='x_enkf')
#     plt.plot(Tplot[:end]/DeltaT, x_mhe[:end, i], '--', label='x_mhe')
#     plt.plot(Tplot[:end]/DeltaT, x_ol_1[:end, i], '--', label='x_ol')
#     # plt.plot(Tplot[:]/DeltaT, x_clean_1[:, i], '-', label='x_clean')
#     plt.legend()
#     plt.xlabel('Time, t (hr)')
#     plt.ylabel('Pressure, h (m)')
#     plt.xticks(Tplot[:end:calNode * 24 * (3600 / DeltaTmhe)] / DeltaT,
#                np.arange(Tsim / (DeltaTmhe * (3600 / DeltaTmhe) * 24) + 1, step=1))
#     plt.grid()
#     plt.savefig('X_Figure_%g.png' % (i + 1))
#     # plt.show()
#     # plt.close()
#
# for i in range(Nx,Nx+Np):  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:end]/DeltaT, x_exp_1[:end, i], '-', label='Para_exp')
#     # plt.plot(Tplot[:]/DeltaT, x_ekf[:, i], '.', markersize=3, label='Para_ekf')
#     # plt.plot(Tplot[:]/DeltaT, x_enkf[:, i], '.', markersize=3, label='Para_enkf')
#     plt.plot(Tplot[:end]/DeltaT, x_mhe[:end, i], '-',  label='Para_mhe')
#     plt.plot(Tplot[:end]/DeltaT, x_ol_1[:end, i], '--', label='Para_ol')
#     # plt.plot(Tplot[:end] / DeltaT, x_clean_1[:end, i], '-', label='Para_clean')
#     plt.legend()
#     plt.xlabel('Time, t (hr)')
#     plt.ylabel('Parameter')
#     plt.xticks(Tplot[:end:calNode * 24 * (3600 / DeltaTmhe)] / DeltaT,
#                np.arange(Tsim / (DeltaTmhe * (3600 / DeltaTmhe) * 24) + 1, step=1))
#     plt.savefig('P_Figure_%g.png' % (i + 1-Nx))
#     # plt.show()
#     # plt.close()

'''
# ----------------------------------------------------------------------------------------------------------------------
# y_mhe_1_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_1_timeS[:, :] = y_mhe_1[:, :]
# y_mhe_2_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_2_timeS[:, :] = y_mhe_2[:, :]
# y_mhe_3_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_3_timeS[:, :] = y_mhe_3[:, :]
# y_mhe_4_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_4_timeS[:, :] = y_mhe_4[:, :]
# y_mhe_5_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_5_timeS[:, :] = y_mhe_5[:, :]
# y_mhe_6_timeS = np.zeros((Nsim + 1, Ny))
# y_mhe_6_timeS[:, :] = y_mhe_6[:, :]
# y_mhe_1_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_1_timeS[:,0:NyPerEst] = y_mhe_1[:,:,0]
# y_mhe_1_timeS[:,NyPerEst:] = y_mhe_1[:,:,1]
# y_mhe_2_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_2_timeS[:,0:NyPerEst] = y_mhe_2[:,:,0]
# y_mhe_2_timeS[:,NyPerEst:] = y_mhe_2[:,:,1]
# y_mhe_3_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_3_timeS[:,0:NyPerEst] = y_mhe_3[:,:,0]
# y_mhe_3_timeS[:,NyPerEst:] = y_mhe_3[:,:,1]
# y_mhe_4_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_4_timeS[:,0:NyPerEst] = y_mhe_4[:,:,0]
# y_mhe_4_timeS[:,NyPerEst:] = y_mhe_4[:,:,1]
# y_mhe_5_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_5_timeS[:,0:NyPerEst] = y_mhe_5[:,:,0]
# y_mhe_5_timeS[:,NyPerEst:] = y_mhe_5[:,:,1]
# y_mhe_6_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_6_timeS[:,0:NyPerEst] = y_mhe_6[:,:,0]
# y_mhe_6_timeS[:,NyPerEst:] = y_mhe_6[:,:,1]
# y_mhe_7_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_7_timeS[:,0:NyPerEst] = y_mhe_7[:,:,0]
# y_mhe_7_timeS[:,NyPerEst:] = y_mhe_7[:,:,1]
# y_mhe_8_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_8_timeS[:,0:NyPerEst] = y_mhe_8[:,:,0]
# y_mhe_8_timeS[:,NyPerEst:] = y_mhe_8[:,:,1]
# y_mhe_9_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_9_timeS[:,0:NyPerEst] = y_mhe_9[:,:,0]
# y_mhe_9_timeS[:,NyPerEst:] = y_mhe_9[:,:,1]
# y_mhe_10_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_10_timeS[:,0:NyPerEst] = y_mhe_10[:,:,0]
# y_mhe_10_timeS[:,NyPerEst:] = y_mhe_10[:,:,1]

# RMSE_Y_1 = np.sqrt(np.sum((y_exp_1 - y_mhe_1_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_2 = np.sqrt(np.sum((y_exp_2 - y_mhe_2_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_3 = np.sqrt(np.sum((y_exp_3 - y_mhe_3_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_4 = np.sqrt(np.sum((y_exp_4 - y_mhe_4_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_5 = np.sqrt(np.sum((y_exp_5 - y_mhe_5_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_6 = np.sqrt(np.sum((y_exp_6 - y_mhe_6_timeS) ** 2, axis=1) / Ny)
# RMSE_Y_7 = np.sqrt(np.sum((y_exp_1-y_mhe_7_timeS)**2, axis=1)/Ny)
# RMSE_Y_8 = np.sqrt(np.sum((y_exp_1-y_mhe_8_timeS)**2, axis=1)/Ny)
# RMSE_Y_9 = np.sqrt(np.sum((y_exp_1-y_mhe_9_timeS)**2, axis=1)/Ny)
# RMSE_Y_10 = np.sqrt(np.sum((y_exp_1-y_mhe_10_timeS)**2, axis=1)/Ny)

# x_mhe_1_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_1_timeS[:, :] = x_mhe_1[:, :Nx]
# x_mhe_2_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_2_timeS[:, :] = x_mhe_2[:, :Nx]
# x_mhe_3_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_3_timeS[:, :] = x_mhe_3[:, :Nx]
# x_mhe_4_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_4_timeS[:, :] = x_mhe_4[:, :Nx]
# x_mhe_5_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_5_timeS[:, :] = x_mhe_5[:, :Nx]
# x_mhe_6_timeS = np.zeros((Nsim + 1, Nx))
# x_mhe_6_timeS[:, :] = x_mhe_6[:, :Nx]
# x_mhe_1_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_1_timeS[:,0:NxPerEst] = x_mhe_1[:,:NxPerEst,0]
# x_mhe_1_timeS[:,NxPerEst:] = x_mhe_1[:,:NxPerEst,1]
# x_mhe_2_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_2_timeS[:,0:NxPerEst] = x_mhe_2[:,:NxPerEst,0]
# x_mhe_2_timeS[:,NxPerEst:] = x_mhe_2[:,:NxPerEst,1]
# x_mhe_3_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_3_timeS[:,0:NxPerEst] = x_mhe_3[:,:NxPerEst,0]
# x_mhe_3_timeS[:,NxPerEst:] = x_mhe_3[:,:NxPerEst,1]
# x_mhe_4_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_4_timeS[:,0:NxPerEst] = x_mhe_4[:,:NxPerEst,0]
# x_mhe_4_timeS[:,NxPerEst:] = x_mhe_4[:,:NxPerEst,1]
# x_mhe_5_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_5_timeS[:,0:NxPerEst] = x_mhe_5[:,:NxPerEst,0]
# x_mhe_5_timeS[:,NxPerEst:] = x_mhe_5[:,:NxPerEst,1]
# x_mhe_6_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_6_timeS[:,0:NxPerEst] = x_mhe_6[:,:NxPerEst,0]
# x_mhe_6_timeS[:,NxPerEst:] = x_mhe_6[:,:NxPerEst,1]
# x_mhe_7_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_7_timeS[:,0:NxPerEst] = x_mhe_7[:,:NxPerEst,0]
# x_mhe_7_timeS[:,NxPerEst:] = x_mhe_7[:,:NxPerEst,1]
# x_mhe_8_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_8_timeS[:,0:NxPerEst] = x_mhe_8[:,:NxPerEst,0]
# x_mhe_8_timeS[:,NxPerEst:] = x_mhe_8[:,:NxPerEst,1]
# x_mhe_9_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_9_timeS[:,0:NxPerEst] = x_mhe_9[:,:NxPerEst,0]
# x_mhe_9_timeS[:,NxPerEst:] = x_mhe_9[:,:NxPerEst,1]
# x_mhe_10_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_10_timeS[:,0:NxPerEst] = x_mhe_10[:,:NxPerEst,0]
# x_mhe_10_timeS[:,NxPerEst:] = x_mhe_10[:,:NxPerEst,1]

# RMSE_X_1 = np.sqrt(np.sum((x_exp_1[:, :Nx] - x_mhe_1_timeS) ** 2, axis=1) / Nx)
# RMSE_X_2 = np.sqrt(np.sum((x_exp_2[:, :Nx] - x_mhe_2_timeS) ** 2, axis=1) / Nx)
# RMSE_X_3 = np.sqrt(np.sum((x_exp_3[:, :Nx] - x_mhe_3_timeS) ** 2, axis=1) / Nx)
# RMSE_X_4 = np.sqrt(np.sum((x_exp_4[:, :Nx] - x_mhe_4_timeS) ** 2, axis=1) / Nx)
# RMSE_X_5 = np.sqrt(np.sum((x_exp_5[:, :Nx] - x_mhe_5_timeS) ** 2, axis=1) / Nx)
# RMSE_X_6 = np.sqrt(np.sum((x_exp_6[:, :Nx] - x_mhe_6_timeS) ** 2, axis=1) / Nx)
# RMSE_X_7 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_7_timeS)**2, axis=1)/Nx)
# RMSE_X_8 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_8_timeS)**2, axis=1)/Nx)
# RMSE_X_9 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_9_timeS)**2, axis=1)/Nx)
# RMSE_X_10 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_10_timeS)**2, axis=1)/Nx)

# p_mhe_1_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_1_timeS[:, :] = x_mhe_1[:, Nx:]
# p_mhe_2_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_2_timeS[:, :] = x_mhe_2[:, Nx:]
# p_mhe_3_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_3_timeS[:, :] = x_mhe_3[:, Nx:]
# p_mhe_4_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_4_timeS[:, :] = x_mhe_4[:, Nx:]
# p_mhe_5_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_5_timeS[:, :] = x_mhe_5[:, Nx:]
# p_mhe_6_timeS = np.zeros((Nsim + 1, Np))
# p_mhe_6_timeS[:, :] = x_mhe_6[:, Nx:]
# p_mhe_1_timeS = np.zeros((Nsim+1, Np))
# p_mhe_1_timeS[:,0:Np] = x_mhe_1[:,NxPerEst:,0]
# p_mhe_1_timeS[:,Np:] = x_mhe_1[:,NxPerEst:,1]
# p_mhe_2_timeS = np.zeros((Nsim+1, Np))
# p_mhe_2_timeS[:,0:Np] = x_mhe_2[:,NxPerEst:,0]
# p_mhe_2_timeS[:,Np:] = x_mhe_2[:,NxPerEst:,1]
# p_mhe_3_timeS = np.zeros((Nsim+1, Np))
# p_mhe_3_timeS[:,0:Np] = x_mhe_3[:,NxPerEst:,0]
# p_mhe_3_timeS[:,Np:] = x_mhe_3[:,NxPerEst:,1]
# p_mhe_4_timeS = np.zeros((Nsim+1, Np))
# p_mhe_4_timeS[:,0:Np] = x_mhe_4[:,NxPerEst:,0]
# p_mhe_4_timeS[:,Np:] = x_mhe_4[:,NxPerEst:,1]
# p_mhe_5_timeS = np.zeros((Nsim+1, Np))
# p_mhe_5_timeS[:,0:Np] = x_mhe_5[:,NxPerEst:,0]
# p_mhe_5_timeS[:,Np:] = x_mhe_5[:,NxPerEst:,1]
# p_mhe_6_timeS = np.zeros((Nsim+1, Np))
# p_mhe_6_timeS[:,0:Np] = x_mhe_6[:,NxPerEst:,0]
# p_mhe_6_timeS[:,Np:] = x_mhe_6[:,NxPerEst:,1]
# p_mhe_7_timeS = np.zeros((Nsim+1, Np))
# p_mhe_7_timeS[:,0:Np] = x_mhe_7[:,NxPerEst:,0]
# p_mhe_7_timeS[:,Np:] = x_mhe_7[:,NxPerEst:,1]
# p_mhe_8_timeS = np.zeros((Nsim+1, Np))
# p_mhe_8_timeS[:,0:Np] = x_mhe_8[:,NxPerEst:,0]
# p_mhe_8_timeS[:,Np:] = x_mhe_8[:,NxPerEst:,1]
# p_mhe_9_timeS = np.zeros((Nsim+1, Np))
# p_mhe_9_timeS[:,0:Np] = x_mhe_9[:,NxPerEst:,0]
# p_mhe_9_timeS[:,Np:] = x_mhe_9[:,NxPerEst:,1]
# p_mhe_10_timeS = np.zeros((Nsim+1, Np))
# p_mhe_10_timeS[:,0:Np] = x_mhe_10[:,NxPerEst:,0]
# p_mhe_10_timeS[:,Np:] = x_mhe_10[:,NxPerEst:,1]

# RMSE_P_1 = np.sqrt(np.sum((x_exp_1[:, Nx:] - p_mhe_1_timeS) ** 2, axis=1) / Np)
# RMSE_P_2 = np.sqrt(np.sum((x_exp_2[:, Nx:] - p_mhe_2_timeS) ** 2, axis=1) / Np)
# RMSE_P_3 = np.sqrt(np.sum((x_exp_3[:, Nx:] - p_mhe_3_timeS) ** 2, axis=1) / Np)
# RMSE_P_4 = np.sqrt(np.sum((x_exp_4[:, Nx:] - p_mhe_4_timeS) ** 2, axis=1) / Np)
# RMSE_P_5 = np.sqrt(np.sum((x_exp_5[:, Nx:] - p_mhe_5_timeS) ** 2, axis=1) / Np)
# RMSE_P_6 = np.sqrt(np.sum((x_exp_6[:, Nx:] - p_mhe_6_timeS) ** 2, axis=1) / Np)
# RMSE_P_7 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_7_timeS)**2, axis=1)/Np)
# RMSE_P_8 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_8_timeS)**2, axis=1)/Np)
# RMSE_P_9 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_9_timeS)**2, axis=1)/Np)
# RMSE_P_10 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_10_timeS)**2, axis=1)/Np)

# startPlotPoint = 0
# interval = 1
# plt.figure()
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_1, label='Y_RMSE_dmhe_PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_2, label='Y_RMSE_dmhe_10*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_3, label='Y_RMSE_dmhe_100*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_4, label='Y_RMSE_dmhe_0.1*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_5, label='Y_RMSE_dmhe_0.01*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_Y_6, label='Y_RMSE_dmhe_0.00003*PQRnominal')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_7, label='Y_RMSE_dmhe_Ppara7e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_8, label='Y_RMSE_dmhe_Ppara8e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_9, label='Y_RMSE_dmhe_Ppara9e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_10, label='Y_RMSE_dmhe_Ppara1e3')
# plt.legend()
# plt.xlabel("Time, t (day)")
# plt.ylabel("RMSE of Measurements")
# plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
# plt.grid()
# # plt.savefig('RMSE_Y.png')
# plt.show()
# 
# plt.figure()
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_1, label='X_RMSE_dmhe_PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_2, label='X_RMSE_dmhe_10*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_3, label='X_RMSE_dmhe_100*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_4, label='X_RMSE_dmhe_0.1*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_5, label='X_RMSE_dmhe_0.01*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_X_6, label='X_RMSE_dmhe_0.00003*PQRnominal')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_7, label='X_RMSE_dmhe_Ppara7e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_8, label='X_RMSE_dmhe_Ppara8e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_9, label='X_RMSE_dmhe_Ppara9e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_10, label='X_RMSE_dmhe_Ppara1e3')
# plt.legend()
# plt.xlabel("Time, t (day)")
# plt.ylabel("RMSE of States")
# plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
# plt.grid()
# # plt.savefig('RMSE_X.png')
# plt.show()
# 
# plt.figure()
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_1, label='P_RMSE_dmhe_PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_2, label='P_RMSE_dmhe_10*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_3, label='P_RMSE_dmhe_100*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_4, label='P_RMSE_dmhe_0.1*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_5, label='P_RMSE_dmhe_0.01*PQRnominal')
# plt.plot(Tplot[startPlotPoint::interval] / DeltaT, RMSE_P_6, label='P_RMSE_dmhe_0.00003*PQRnominal')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_7, label='P_RMSE_dmhe_Ppara7e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_8, label='P_RMSE_dmhe_Ppara8e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_9, label='P_RMSE_dmhe_Ppara9e2')
# # plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_10, label='P_RMSE_dmhe_Ppara1e3')
# plt.legend()
# plt.xlabel("Time, t (day)")
# plt.ylabel("RMSE of Parameters")
# plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
# plt.grid()
# # plt.savefig('RMSE_P.png')
# plt.show()

# indexToPlot = np.arange(0,Ny,1)
# for i in indexToPlot:  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:]/DeltaT, y_exp_1[:, i], '-', label='y_exp')
#     # plt.plot(Tplot[:]/DeltaT, y_exp_2[:, i], '-', label='y_exp_2')
#     # plt.plot(Tplot[:]/DeltaT, y_exp_3[:, i], '-', label='y_exp_3')
#
#     plt.plot(Tplot[:]/DeltaT, y_mhe_1_timeS[:, i], '.', label='y_dmhe_PQRnominal')
#     # plt.plot(Tplot[:]/DeltaT, y_mhe_2_timeS[:, i], '.', label='y_dmhe_10*PQRnominal')
#     # plt.plot(Tplot[:]/DeltaT, y_mhe_3_timeS[:, i], '.', label='y_dmhe_100*PQRnominal')
#
#     # plt.plot(Tplot[:]/DeltaT, y_ol_1[:, i], '--', label='y_ol')
#     # plt.plot(Tplot[:]/DeltaT, y_clean_1[:, i], '-', label='y_clean')
#     plt.legend()
#     plt.xlabel('Time, t (day)')
#     plt.ylabel('Soil Moisture, theta (m3/m3)')
#     plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
#     plt.grid()
#     plt.show()
#     # plt.savefig('Y_Figure_%g.png' % (i+1))
# #
# indexToPlot = np.arange(0, Nx) + 0
# for i in indexToPlot:  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:] / DeltaT, x_exp_1[:, i], '-', label='x_true')
#     plt.plot(Tplot[:] / DeltaT, x_exp_2[:, i], '-', label='x_true_10*PQRnominal')
# 
#     plt.plot(Tplot[::calNode] / DeltaT, x_mhe_1_timeS[:, i], '.', markersize=4, label='x_dmhe_PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, x_mhe_2_timeS[:, i], '.', markersize=4, label='x_dmhe_10*PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, x_mhe_3_timeS[:, i], '.', markersize=4, label='x_dmhe_100*PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, x_mhe_4_timeS[:, i], '.', markersize=4, label='x_dmhe_0.1*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_5_timeS[:, i], '.', markersize=4, label='x_dmhe_0.01*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_6_timeS[:, i], '.', markersize=4, label='x_dmhe_0.00003*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_7_timeS[:, i], '.', markersize=4, label='x_dmhe_Ppara7e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_8_timeS[:, i], '.', markersize=4, label='x_dmhe_Ppara8e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_9_timeS[:, i], '.', markersize=4, label='x_dmhe_Ppara9e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_10_timeS[:, i], '.', markersize=4, label='x_dmhe_Ppara1e3')
# 
#     plt.plot(Tplot[:] / DeltaT, x_ol_1[:, i], '-', label='x_ol')
#     plt.plot(Tplot[:] / DeltaT, x_clean_1[:, i], '-', label='x_clean')
#     plt.legend()
#     plt.xlabel('Time, t (day)')
#     plt.ylabel('Pressure, h (m)')
#     plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
#     plt.grid()
#     # plt.show()
#     plt.savefig('X_Figure_%g.png' % (i + 1))
# # # #
# indexToPlot = np.arange(Nx, Nx + Np, 1)
# for i in indexToPlot:  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:] / DeltaT, x_exp_1[:, i], '-', label='p')
#     plt.plot(Tplot[::calNode] / DeltaT, p_mhe_1_timeS[:, i - Nx], '.', markersize=4, label='p_dmhe_PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, p_mhe_2_timeS[:, i - Nx], '.', markersize=4, label='p_dmhe_10*PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, p_mhe_3_timeS[:, i - Nx], '.', markersize=4, label='p_dmhe_100*PQRnominal')
#     plt.plot(Tplot[::calNode] / DeltaT, p_mhe_4_timeS[:, i - Nx], '.', markersize=4, label='p_dmhe_0.1*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_5_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_0.01*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_6_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_0.00003*PQRnominal')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_7_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_Ppara7e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_8_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_Ppara8e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_9_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_Ppara9e2')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_10_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_Ppara1e3')
#     plt.plot(Tplot[:] / DeltaT, x_ol_1[:, i], '--', label='p_ol')
#     # plt.plot(Tplot[:] / DeltaT, x_clean_1[:, i], '-', label='p_clean')
#     plt.legend()
#     plt.xlabel('Time, t (day)')
#     plt.ylabel('Parameters')
#     plt.xticks(Tplot[::calNode * 24] / DeltaT, np.arange(Tsim / (DeltaTmhe * 24) + 1, step=1))
#     plt.grid()
#     # plt.show()
#     plt.savefig('P_Figure_%g.png' % (i + 1 - Nx))
'''
