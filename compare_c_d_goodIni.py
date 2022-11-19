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

# sys.path.insert(0,"D:\Projects_Codes\MHE_Dist_MulLay_3D\V3_3D_vectorized\Results\finalized")

# data_dir = pjoin('L_SCL_yc_d_filter_Nsim4_44_midbounds_1lay2Y')
# mat_fname_x = pjoin(data_dir, 'x.mat')
x_1 = sio.loadmat('Data/x.mat')
x_exp_1 = x_1['xmea']
x_mhe_1 = x_1['xmhe']  # no matter dis or central, x_mhe are the same
x_ol_1 = x_1['xol']
x_clean_1 = x_1['xclean']

# mat_fname_y = pjoin(data_dir, 'y.mat')
y_1 = sio.loadmat('Data/y.mat')
y_exp_1 = y_1['ymea']
y_mhe_1 = y_1['ymhe']

# # mat_fname_noise = pjoin(data_dir, 'input_noise.mat')
# noise_1 = sio.loadmat('input_noise.mat')
# w_1 = noise_1['w']
# v_1= noise_1['v']
#
# # mat_fname_c = pjoin(data_dir, 'cMatrix.mat')
# cMatrix = sio.loadmat('cMatrix.mat')
# cMatrix_1 = cMatrix['c']
#
# # mat_fname_time = pjoin(data_dir, 'timeUsed')
# timeUsed_1 = sio.loadmat('timeUsed')
# mhe_timeUsed_list_1 = timeUsed_1['mhe_timeUsedSolve_list']
# fie_timeUsed_list_1 = timeUsed_1['fie_timeUsedSolve_list']

# middle sensor --------------------------------------------------------------------------------------------------------
# data_dir = pjoin('L_SCL_yc_d_filter_Nsim10_goodIni_33_midbounds')
# mat_fname_x = pjoin(data_dir, 'x.mat')
# x_2 = sio.loadmat(mat_fname_x)
# x_mhe_2 = x_2['xmhe']
#
# mat_fname_y = pjoin(data_dir, 'y.mat')
# y_2 = sio.loadmat(mat_fname_y)
# y_exp_2 = y_2['ymea']
# y_mhe_2 = y_2['ymhe']
#
# mat_fname_noise = pjoin(data_dir, 'input_noise.mat')
# noise_2 = sio.loadmat(mat_fname_noise)
# w_2 = noise_2['w']
# v_2= noise_2['v']
#
# mat_fname_c = pjoin(data_dir, 'cMatrix.mat')
# cMatrix = sio.loadmat(mat_fname_c)
# cMatrix_2 = cMatrix['c']
#
# mat_fname_time = pjoin(data_dir, 'timeUsed')
# timeUsed_2 = sio.loadmat(mat_fname_time)
# mhe_timeUsed_list_2 = timeUsed_2['mhe_timeUsedSolve_list']
# fie_timeUsed_list_2 = timeUsed_2['fie_timeUsedSolve_list']
# 
# # ----------------------------------------------------------------------------------------------------------
# data_dir = pjoin('L_SCL_yc_d_filter_Nsim10_goodIni_33_smallbounds')
# mat_fname_x = pjoin(data_dir, 'x.mat')
# x_3 = sio.loadmat(mat_fname_x)
# x_mhe_3 = x_3['xmhe']
# 
# mat_fname_y = pjoin(data_dir, 'y.mat')
# y_3 = sio.loadmat(mat_fname_y)
# y_exp_3 = y_3['ymea']
# y_mhe_3 = y_3['ymhe']
# 
# mat_fname_noise = pjoin(data_dir, 'input_noise.mat')
# noise_3 = sio.loadmat(mat_fname_noise)
# w_3 = noise_3['w']
# v_3 = noise_3['v']
# 
# mat_fname_c = pjoin(data_dir, 'cMatrix.mat')
# cMatrix = sio.loadmat(mat_fname_c)
# cMatrix_3 = cMatrix['c']
# 
# mat_fname_time = pjoin(data_dir, 'timeUsed')
# timeUsed_3 = sio.loadmat(mat_fname_time)
# mhe_timeUsed_list_3 = timeUsed_3['mhe_timeUsedSolve_list']
# fie_timeUsed_list_3 = timeUsed_3['fie_timeUsedSolve_list']

# mat_fname_timePara = pjoin(data_dir, 'time_parameters.mat')
time_para = sio.loadmat('Data/time_parameters.mat')
Tplot = time_para['Tplot'].ravel()
DeltaT = time_para['DeltaT'].item()
calNode = time_para['calNode'].item()
Nsim = time_para['Nsim'].item()
Tsim = time_para['Tsim'].item()
DeltaTmhe = time_para['DeltaTmhe'].item()

# mat_fname_spacePara = pjoin(data_dir, 'space_parameters.mat')
space_para = sio.loadmat('Data/space_parameters.mat')
Nest = space_para['Nest'].item()
NxPerEst = space_para['NxPerEst'].item()
NyPerEst = space_para['NyPerEst'].item()
Nsoil = space_para['Nsoil'].item()
Np = space_para['Np'].item()
Nx = space_para['Nx'].item()
Nxx = space_para['Nxx'].item()
Nxy = space_para['Nxy'].item()
Nxz = space_para['Nxz'].item()
Ny = space_para['Ny'].item()
NxPerSoil = space_para['NxPerSoil'].item()
NpTotal = Np*Nsoil

# # mat_fname_modelPara = pjoin(data_dir, 'model_parameters.mat')
# model_para = sio.loadmat('model_parameters.mat')
# ParsNp = model_para['ParsNp']

# --------------------------------------------------------------------------------------------------------
# data_dir = pjoin('L_SCL_yc_c_filter_Nsim20_goodIni')
# mat_fname_x = pjoin(data_dir, 'x.mat')
# x_1 = sio.loadmat(mat_fname_x)
# x_exp_1 = x_1['x_mea']
# x_mhe_1 = x_1['x_mhe']  # no matter dis or central, x_mhe are the same
# x_ol_1 = x_1['x_ol']
#
# mat_fname_y = pjoin(data_dir, 'y.mat')
# y_1 = sio.loadmat(mat_fname_y)
# y_exp_1 = y_1['y_mea']
# y_mhe_1 = y_1['y_mhe']
#
# mat_fname_noise = pjoin(data_dir, 'input_noise.mat')
# noise_1 = sio.loadmat(mat_fname_noise)
# w_1 = noise_1['w']
# v_1= noise_1['v']
#
# mat_fname_SE = pjoin(data_dir, 'SE.mat')
# error_1 = sio.loadmat(mat_fname_SE)
# Jekf_Y_SSE_1 = error_1['Jekf_Y_SSE'].ravel()
# Jekf_X_SSE_1 = error_1['Jekf_X_SSE'].ravel()
# Jekf_P_SSE_1 = error_1['Jekf_P_SSE'].ravel()
# Jenkf_Y_SSE_1 = error_1['Jenkf_Y_SSE'].ravel()
# Jenkf_X_SSE_1 = error_1['Jenkf_X_SSE'].ravel()
# Jenkf_P_SSE_1 = error_1['Jenkf_P_SSE'].ravel()
# Jmhe_Y_SSE_1 = error_1['Jmhe_Y_SSE_perEst'].ravel()
# Jmhe_X_SSE_1 = error_1['Jmhe_X_SSE_perEst'].ravel()
# Jmhe_P_SSE_1 = error_1['Jmhe_P_SSE_perEst'].ravel()
#
# mat_fname_c = pjoin(data_dir, 'cMatrix.mat')
# cMatrix = sio.loadmat(mat_fname_c)
# cMatrix_1 = cMatrix['c']
#
# mat_fname_time = pjoin(data_dir, 'timeUsed')
# timeUsed_1 = sio.loadmat(mat_fname_time)
# mhe_timeUsed_list_1 = timeUsed_1['mhe_timeUsedSolve_list']
# fie_timeUsed_list_1 = timeUsed_1['fie_timeUsedSolve_list']
# ----------------------------------------------------------------------------------------------------------------------
# y_mhe_1_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_1_timeS[:,:] = y_mhe_1[:,:,0]
y_mhe_1_timeS = np.zeros((Nsim+1, Ny))
y_mhe_1_timeS[:,0:NyPerEst] = y_mhe_1[:,:,0]
y_mhe_1_timeS[:,NyPerEst:] = y_mhe_1[:,:,1]
# y_mhe_2_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_2_timeS[:,0:NyPerEst] = y_mhe_2[:,:,0]
# y_mhe_2_timeS[:,NyPerEst:] = y_mhe_2[:,:,1]
# y_mhe_3_timeS = np.zeros((Nsim+1, Ny))
# y_mhe_3_timeS[:,0:NyPerEst] = y_mhe_3[:,:,0]
# y_mhe_3_timeS[:,NyPerEst:] = y_mhe_3[:,:,1]

RMSE_Y_1 = np.sqrt(np.sum((y_exp_1-y_mhe_1_timeS)**2, axis=1)/Ny)
# RMSE_Y_2 = np.sqrt(np.sum((y_exp_1-y_mhe_2_timeS)**2, axis=1)/Ny)
# RMSE_Y_3 = np.sqrt(np.sum((y_exp_1-y_mhe_3_timeS)**2, axis=1)/Ny)

# x_mhe_1_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_1_timeS[:,:] = x_mhe_1[:,:Nx,0]
x_mhe_1_timeS = np.zeros((Nsim+1, Nx))
x_mhe_1_timeS[:,0:NxPerEst] = x_mhe_1[:,:NxPerEst,0]
x_mhe_1_timeS[:,NxPerEst:] = x_mhe_1[:,:NxPerEst,1]
# x_mhe_2_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_2_timeS[:,0:NxPerEst] = x_mhe_2[:,:NxPerEst,0]
# x_mhe_2_timeS[:,NxPerEst:] = x_mhe_2[:,:NxPerEst,1]
# x_mhe_3_timeS = np.zeros((Nsim+1, Nx))
# x_mhe_3_timeS[:,0:NxPerEst] = x_mhe_3[:,:NxPerEst,0]
# x_mhe_3_timeS[:,NxPerEst:] = x_mhe_3[:,:NxPerEst,1]

RMSE_X_1 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_1_timeS)**2, axis=1)/Nx)
# RMSE_X_2 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_2_timeS)**2, axis=1)/Nx)
# RMSE_X_3 = np.sqrt(np.sum((x_exp_1[:,:Nx]-x_mhe_3_timeS)**2, axis=1)/Nx)

# p_mhe_1_timeS = np.zeros((Nsim+1, NpTotal))
# p_mhe_1_timeS[:,:] = x_mhe_1[:,Nx:,0]
p_mhe_1_timeS = np.zeros((Nsim+1, NpTotal))
p_mhe_1_timeS[:,0:Np] = x_mhe_1[:,NxPerEst:,0]
p_mhe_1_timeS[:,Np:] = x_mhe_1[:,NxPerEst:,1]
# p_mhe_2_timeS = np.zeros((Nsim+1, NpTotal))
# p_mhe_2_timeS[:,0:Np] = x_mhe_2[:,NxPerEst:,0]
# p_mhe_2_timeS[:,Np:] = x_mhe_2[:,NxPerEst:,1]
# p_mhe_3_timeS = np.zeros((Nsim+1, NpTotal))
# p_mhe_3_timeS[:,0:Np] = x_mhe_3[:,NxPerEst:,0]
# p_mhe_3_timeS[:,Np:] = x_mhe_3[:,NxPerEst:,1]

RMSE_P_1 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_1_timeS)**2, axis=1)/NpTotal)
# RMSE_P_2 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_2_timeS)**2, axis=1)/NpTotal)
# RMSE_P_3 = np.sqrt(np.sum((x_exp_1[:,Nx:]-p_mhe_3_timeS)**2, axis=1)/NpTotal)

startPlotPoint = 0
endPoint = 343
interval = 1
plt.figure()
plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_1, label='Y_RMSE_dmhe_bigBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_2, label='Y_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_3, label='Y_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of Measurements")
plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
plt.grid()
# plt.savefig('RMSE_Y.png')
plt.show()

plt.figure()
plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_1, label='X_RMSE_dmhe_bigBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_2, label='X_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_3, label='X_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of States")
plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
plt.grid()
# plt.savefig('RMSE_X.png')
plt.show()

plt.figure()
plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_1, label='P_RMSE_dmhe_bigBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_2, label='P_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_3, label='P_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of Parameters")
plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
plt.grid()
# plt.savefig('RMSE_P.png')
plt.show()

# indexToPlot = np.arange(0,Ny,1)
# for i in indexToPlot:  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:]/DeltaT, y_exp_1[:, i], '-', label='y_exp')
#     # plt.plot(Tplot[:]/DeltaT, y_exp_2[:, i], '-', label='y_exp_2')
#     # plt.plot(Tplot[:]/DeltaT, y_exp_3[:, i], '-', label='y_exp_3')
#
#     plt.plot(Tplot[:]/DeltaT, y_mhe_1_timeS[:, i], '.', label='y_dmhe_bigB')
#     # plt.plot(Tplot[:]/DeltaT, y_mhe_2_timeS[:, i], '.', label='y_dmhe_midB')
#     # plt.plot(Tplot[:]/DeltaT, y_mhe_3_timeS[:, i], '.', label='y_dmhe_smallB')
#
#     # plt.plot(Tplot[:]/DeltaT, y_ol_1[:, i], '--', label='y_ol')
#     # plt.plot(Tplot[:]/DeltaT, y_clean_1[:, i], '-', label='y_clean')
#     plt.legend()
#     plt.xlabel('Time, t (hr)')
#     plt.ylabel('Soil Moisture, theta (m3/m3)')
#     plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
#     plt.grid()
#     plt.show()
#     # plt.savefig('Y_Figure_%g.png' % (i+1))
# #
indexToPlot = np.arange(0,Nx,Nxx*Nxy)+3
for i in indexToPlot:  # choose how many states do yo wanna plot
    plt.figure()
    plt.plot(Tplot[:endPoint]/DeltaT, x_exp_1[:endPoint, i], '-', label='x_true')

    plt.plot(Tplot[:endPoint:calNode]/DeltaT, x_mhe_1_timeS[:endPoint, i], '.', markersize=4, label='x_dmhe_bigB')
    # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, i], '.', markersize=4, label='x_dmhe_midB')
    # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_3_timeS[:, i], '.', markersize=4, label='x_dmhe_smallB')

    plt.plot(Tplot[:endPoint]/DeltaT, x_ol_1[:endPoint, i], '-', label='x_ol')
    plt.plot(Tplot[:endPoint] / DeltaT, x_clean_1[:endPoint, i], '-', label='x_clean')
    plt.legend()
    plt.xlabel('Time, t (hr)')
    plt.ylabel('Pressure, h (m)')
    # plt.xticks(Tplot[:endPoint:calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
    plt.grid()
    # plt.show()
    plt.savefig('X_Figure_%g.png' % (i + 1))
# # #
# indexToPlot = np.arange(Nx,Nx+NpTotal,1)
# for i in indexToPlot:  # choose how many states do yo wanna plot
#     plt.figure()
#     plt.plot(Tplot[:]/DeltaT, x_exp_1[:, i], '-', label='p')
#
#     plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_bigB')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_midB')
#     # plt.plot(Tplot[::calNode]/DeltaT, p_mhe_3_timeS[:, i-Nx], '.', markersize=4, label='p_dmhe_smallB')
#
#     plt.plot(Tplot[:]/DeltaT, x_ol_1[:, i], '--', label='p_ol')
#     # plt.plot(Tplot[:] / DeltaT, x_clean_1[:, i], '-', label='p_clean')
#     plt.legend()
#     plt.xlabel('Time, t (hr)')
#     plt.ylabel('Parameters')
#     plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
#     plt.grid()
#     # plt.show()
#     plt.savefig('P_Figure_%g.png' % (i + 1-Nx))

'''
indexToPlot = np.arange(0,NyPerEst,1)
for j in range(Nest):
    for i in indexToPlot:  # choose how many states do yo wanna plot
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, y_exp_1[:, NyPerEst*j+i], '-', label='y_exp')
        # plt.plot(Tplot[:]/DeltaT, y_exp_2[:, NyPerEst*j+i], '-', label='y_exp_2')
        # plt.plot(Tplot[:]/DeltaT, y_exp_3[:, NyPerEst*j+i], '-', label='y_exp_3')

        plt.plot(Tplot[:]/DeltaT, y_mhe_1_timeS[:, NyPerEst*j+i], '.', label='y_dmhe_bigB')
        # plt.plot(Tplot[:]/DeltaT, y_mhe_2_timeS[:, NyPerEst*j+i], '.', label='y_dmhe_midB')
        # plt.plot(Tplot[:]/DeltaT, y_mhe_3_timeS[:, NyPerEst*j+i], '.', label='y_dmhe_smallB')

        # plt.plot(Tplot[::calNode]/DeltaT, y_mhe_1[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='y_dmhe_bigB')
        # plt.plot(Tplot[::calNode]/DeltaT, y_mhe_2[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='y_dmhe_midB')
        # plt.plot(Tplot[::calNode]/DeltaT, y_mhe_3[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='y_dmhe_smallB')
        # plt.plot(Tplot[:]/DeltaT, y_ol_1[:, NyPerEst*j+i], '--', label='y_ol')
        # plt.plot(Tplot[:]/DeltaT, y_clean_1[:, NyPerEst*j+i], '-', label='y_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Soil Moisture, theta (m3/m3)')
        plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
        plt.grid()
        plt.show()
        # plt.savefig('Y_Figure_%g.png' % (i+1))
#
indexToPlot = np.arange(0,NxPerEst,Nxx*Nxy) + 4
for j in range(Nest):
    for i in indexToPlot:  # choose how many states do yo wanna plot
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, x_exp_1[:, NxPerEst*j+i], '-', label='x_true')
        # plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='x_exp_d')

        plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1[:(Nsim)+1, i, j], '.', markersize=4, label='x_dmhe_bigB')
        # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2[:(Nsim)+1, i, j], '.', markersize=4, label='x_dmhe_midB')
        # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_3[:(Nsim)+1, i, j], '.', markersize=4, label='x_dmhe_smallB')
        plt.plot(Tplot[:]/DeltaT, x_ol_1[:, NxPerEst*j+i], '-', label='x_ol')
        # plt.plot(Tplot[:] / DeltaT, x_clean_1[:, NxPerEst*j+i], '-', label='x_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Pressure, h (m)')
        plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
        plt.grid()
        plt.show()
        # plt.savefig('X_Figure_%g.png' % (i + 1))
# # #
indexToPlot = np.arange(NxPerEst,NxPerEst+Np*max(1, Nsoil//Nest),1)
for j in range(Nest):
    for i in indexToPlot:  # choose how many states do yo wanna plot
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, x_exp_1[:, Nx-NxPerEst+i], '-', label='p')
        # plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
        plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1[:(Nsim)+1, i, j], '.', markersize=4, label='p_dmhe_bigB')
        # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2[:(Nsim)+1, i, j], '.', markersize=4, label='p_dmhe_midB')
        # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_3[:(Nsim)+1, i, j], '.', markersize=4, label='p_dmhe_smallB')
        # plt.plot(Tplot[:]/DeltaT, x_ol_1[:, Nx-NxPerEst+i], '--', label='p_ol')
        # plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Parameters')
        plt.xticks(Tplot[::calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
        plt.grid()
        plt.show()
        # plt.savefig('P_Figure_%g.png' % (i + 1-Nx))
         
plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 3], '-', label='$h_{1}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='x_exp_d')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1_timeS[:, 3], '.-', label='$h_{1,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, 3], '--', label='$h_{1,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 3], '-', label='$h_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, NxPerEst*j+i], '-', label='x_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel('Pressure, h (m)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('X_Figure_%g.png' % (1))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 13], '-', label=r'$h_{3}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='x_exp_d')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1_timeS[:, 13], '.-', label=r'$h_{3,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, 13], '--', label=r'$h_{3,dme}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 13], '-', label='$h_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, NxPerEst*j+i], '-', label='x_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'Pressure, $h_{3}$ (m)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('X_Figure_%g.png' % (2))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 18], '-', label=r'$h_{3}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='x_exp_d')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1_timeS[:, 18], '.-', label=r'$h_{3,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, 18], '--', label=r'$h_{3,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 18], '-', label='$h_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, NxPerEst*j+i], '-', label='x_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel('Pressure, $h_{3}$ (m)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('X_Figure_%g.png' % (3))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 28], '-', label=r'$h_{13}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='x_exp_d')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_1_timeS[:, 28], '.-', label=r'$h_{13, cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, 28], '--', label=r'$h_{13, dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 28], '-', label='$h_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, NxPerEst*j+i], '-', label='x_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'Pressure, $h_{13}$ (m)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('X_Figure_%g.png' % (4))
# ------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 32], '-', label='$K_{s}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 0], '.-', label='$K_{s,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 0], '--', label='$K_{s,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 32], '-', label='$K_{s,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel('Ks, (m/s)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (1))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 33], '-', label=r'$\theta_{s}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 1], '.-', label=r'$\theta_{s,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 1], '--', label=r'$\theta_{s,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 33], '-', label=r'$\theta_{s,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\theta$s' r'($m^3/m^3$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (2))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 34], '-', label=r'$\theta_{r}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 2], '.-', label=r'$\theta_{r,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 2], '--', label=r'$\theta_{r,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 34], '-', label=r'$\theta_{r,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\theta$r' r'($m^3/m^3$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (3))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 35], '-', label=r'$\alpha$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 3], '.-', label=r'$\alpha_{cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 3], '--', label=r'$\alpha_{dmhe}$')
plt.plot(Tplot[::]/DeltaT, x_ol_1[:, 35], '-', label=r'$\alpha_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\alpha$' r'($m^{-1}$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (4))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 36], '-', label=r'$n$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 4], '.-', label=r'$n_{cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 4], '--', label=r'$n_{dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 36], '-', label=r'$n_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$n$')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (5))
# ------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 37], '-', label='$K_{s}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 5], '.-', label='$K_{s,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 5], '--', label='$K_{s,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 37], '-', label='$K_{s,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel('Ks, (m/s)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (6))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 38], '-', label=r'$\theta_{s}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 6], '.-', label=r'$\theta_{s,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 6], '--', label=r'$\theta$s,dmhe')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 38], '-', label=r'$\theta_{s,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\theta$s' r'($m^3/m^3$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (7))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 39], '-', label=r'$\theta_{r}$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 7], '.-', label=r'$\theta_{r,cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 7], '--', label=r'$\theta_{r,dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 39], '-', label=r'$\theta_{r,ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\theta$r' r'($m^3/m^3$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (8))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 40], '-', label=r'$\alpha$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 8], '.-', label=r'$\alpha_{cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 8], '--', label=r'$\alpha_{dmhe}$')
plt.plot(Tplot[::]/DeltaT, x_ol_1[:, 40], '-', label=r'$\alpha_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$\alpha$' r'($m^{-1}$)')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (9))


plt.figure()
plt.plot(Tplot[:]/DeltaT, x_exp_1[:, 41], '-', label=r'$n$')
# plt.plot(Tplot[:]/DeltaT, x_exp_2_dis[:(Nsim)+1, i, j], '-', label='p')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_1_timeS[:, 9], '.-', label=r'$n_{cmhe}$')
plt.plot(Tplot[::calNode]/DeltaT, p_mhe_2_timeS[:, 9], '--', label=r'$n_{dmhe}$')
plt.plot(Tplot[:]/DeltaT, x_ol_1[:, 41], '-', label=r'$n_{ol}$')
# plt.plot(Tplot[:] / DeltaT, x_clean_1[:, Nx-NxPerEst+i], '-', label='p_clean')
plt.legend()
plt.xlabel('Time, t (day)')
plt.ylabel(r'$n$')
plt.xticks(Tplot[::calNode*144]/DeltaT, np.arange(Tsim/(DeltaTmhe*6*24)+1, step=1))
plt.grid()
# plt.show()
plt.savefig('P_Figure_%g.png' % (10))
'''