from __future__ import (print_function, division)  # Grab some handy Python3 stuff.
import numpy as np
from scipy import linalg, integrate, io
import matplotlib.pyplot as plt
from casadi import *
import time

from Model_3D_ODE_aug import getODE_mx_3D_process, FN_multiLayer_process
from Model_3D_ODE_aug import getODE_mx_3D_subsys_estimator, getOutputs_mx_aug_subsys, F_N_aug_mx_subsys
from Model_3D_ODE_aug import getOutputs_np_aug, getH0_np_multiLayer, KFun_np, F_N_np_V2#, F_N_np, F_N_np_V2
from Model_3D_ODE_aug import getOutputs_np_aug_subsys, F_N_aug_np_subsys
from Simulator_subsystem import simulator_dis

from Inputs import irr
from Parameters import model_parameters_multiLayer, initial_condition, time_parameters, space_parameters, cmatrix_single, bounds_para, variance
from MHE_multiShoot_casadi_DISTRIBUTED import mhe_discrete, filter_smoother_scheme, mhe_prepare, flaten, cal_cur_state
# from Ensemble_Kalman_filter import EnsembleKalmanFilter as EnKF
# from Extended_Kalman_filter import ekf_continuous_discrete
# from DOO import doo

# Basic parameter setting -------------------------------------------------------------------------------------
NumPara = 5
DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
spacePara_dict = space_parameters()  # space related parameters
Nx = spacePara_dict['Nx']
Nw = spacePara_dict['Nw']
Np = spacePara_dict['Np']
NpTotal = spacePara_dict['NpTotal']
Nx_aug = spacePara_dict['Nx_aug']
Nw_aug = spacePara_dict['Nw_aug']
Ny = spacePara_dict['Ny']
Nv = spacePara_dict['Nv']
Nsoil = spacePara_dict['Nsoil']
NxPerSoil = spacePara_dict['NxPerSoil']
Nest = spacePara_dict['Nest']
NxPerEst = spacePara_dict['NxPerEst']
NyPerEst = spacePara_dict['NyPerEst']
Nu = spacePara_dict['Nu']
Ninfo = spacePara_dict['Ninfo']
dz = spacePara_dict['dz']
Nxx = spacePara_dict['Nxx']
Nxy = spacePara_dict['Nxy']
Nxz = spacePara_dict['Nxz']

ParsDict, ParsNp, ParsScale = model_parameters_multiLayer(Nsoil)  # soil/model parameters: These are the true parameters
P0, Q, R, noise_w, noise_v, tol_desired = variance(Nx, Nx_aug, Nw_aug, Nv)
xlower, xupper, wlower, wupper, xlower_fie, xupper_fie = bounds_para(NxPerEst, NpTotal, Np, Nsoil, Nest)  # Nx, Nx_aug, Np are used to determine the dimension of bounds_para
CMatrix = cmatrix_single(Nx, Ny)

io.savemat('Data/space_parameters.mat', spacePara_dict)
io.savemat('Data/time_parameters.mat', dict(DeltaT=DeltaT, Tsim=Tsim, Nsim=Nsim, DeltaT_internal=DeltaT_internal, Nsim_internal=Nsim_internal, Tplot=Tplot, Nmhe=Nmhe, DeltaTmhe=DeltaTmhe, calNode=calNode))

# Make covariance matrices. --------------------------------------------------------------------------------------------
# In this study, P matrix is a tuning parameter and is not from EKF
# P_ekf = []
# P_ekf.append(P0)
# P_ekf_minus = []
# P_ekf_minus.append(P0)

seedpoint = 514
np.random.seed(seedpoint)  # Seed random number generator.  927 means seed will start from 927 all the time, which give random.randn the same results everytime.

# Initial state --------------------------------------------------------------------------------------------------------
# y0 = initial_condition(Nx)  # initial measurement
# x0 = getH0_np_multiLayer(y0, ParsNp, ParsScale, Nx, NxPerSoil, Nsoil, Np)  # Initial state, np is used since we only need the value
x0 = np.ones(Nx)*-0.51394922
for i in range(NpTotal//Np):
    x0 = np.append(x0, ParsScale)  # here we are not doing estimation. So we only add 1 set of pars at the end

# y0_est = 1*y0
ParsScale_est = ParsScale
# x0_est = getH0_np_multiLayer(y0_est, ParsNp, ParsScale, Nx, NxPerSoil, Nsoil, Np)
x0_est = x0[:Nx]
# Estimating all parameters --------------------
ratio = np.array([1.1, 0.9, 1.0, 0.9, 1.1])
ParsScale_est = np.multiply(ratio, ParsScale_est)
x0_est = 1.5*x0_est
for i in range(NpTotal//Np):
    x0_est = np.append(x0_est, ParsScale_est)

# Generate noises ------------------------------------------------------------------------------------------------------
v_all = noise_v*np.random.randn(Nsim//calNode+1,Nw)
v = np.matmul(v_all, cmatrix_single(Nx, Ny).T)
w_small = noise_w*np.random.randn(Nsim,Nw_aug)  # Process noise sequence
w_small[:,Nx_aug-Np*Nsoil:Nx_aug] = int(0)
w = np.zeros((Nsim,Nw_aug))  # Process noise sequence
for i in range(int(Tsim/DeltaTmhe)):  # Make w within DeltaTmhe be the same
    for j in range(int(DeltaTmhe/DeltaT)):
        w[j+i*int(DeltaTmhe/DeltaT),:] = w_small[i,:]
# ww_enkf = noise_w*np.random.randn(Nsim*Nsigma,Nw_aug)
# ww_enkf[:,Nx_aug-Np:Nx_aug] = int(0)

# == Generating matrices ===============================================================================================
# For centralized system in MX/SX ------------------------------------------------
x = np.zeros((Nsim+1,Nx_aug))
x[0, :] = x0
y = np.zeros((Nsim//calNode+1,Ny))
y[0, :] = getOutputs_np_aug(x0) + v[0, :]  # np is used, since we only need to know the value

x_ol = np.zeros((Nsim+1,Nx_aug))
x_ol[0,:] = x0_est
y_ol = np.zeros((Nsim//calNode+1,Ny))
y_ol[0, :] = getOutputs_np_aug(x0_est)# + v[0,:]  # np is used, since we only need to know the value

x_clean = np.zeros((Nsim+1,Nx_aug))
x_clean[0, :] = x0
y_clean = np.zeros((Nsim//calNode+1,Ny))
y_clean[0, :] = getOutputs_np_aug(x0)

x_int = np.zeros((Nsim+1,Nx_aug))
x_int[0, :] = x0
y_int = np.zeros((Nsim//calNode+1,Ny))
y_int[0, :] = getOutputs_np_aug(x0) + v[0, :]  # np is used, since we only need to know the value

x_ol_int = np.zeros((Nsim+1,Nx_aug))
x_ol_int[0,:] = x0_est
y_ol_int = np.zeros((Nsim//calNode+1,Ny))
y_ol_int[0, :] = getOutputs_np_aug(x0_est)# + v[0,:]  # np is used, since we only need to know the value

x_clean_int = np.zeros((Nsim+1,Nx_aug))
x_clean_int[0, :] = x0
y_clean_int = np.zeros((Nsim//calNode+1,Ny))
y_clean_int[0, :] = getOutputs_np_aug(x0)

# For centralized system in Numpy format -------------------------------------------------------------
x_np = np.zeros((Nsim+1,Nx_aug))
x_np[0, :] = x0
y_np = np.zeros((Nsim//calNode+1,Ny))
y_np[0, :] = getOutputs_np_aug(x0) + v[0, :]  # np is used, since we only need to know the value

x_ol_np = np.zeros((Nsim+1,Nx_aug))
x_ol_np[0,:] = x0_est
y_ol_np = np.zeros((Nsim//calNode+1,Ny))
y_ol_np[0, :] = getOutputs_np_aug(x0_est)# + v[0,:]  # np is used, since we only need to know the value

x_clean_np = np.zeros((Nsim+1,Nx_aug))
x_clean_np[0, :] = x0
y_clean_np = np.zeros((Nsim//calNode+1,Ny))
y_clean_np[0, :] = getOutputs_np_aug(x0)

# For distributed systems (subsystems)----------------------------------------------
x_clean_dis = np.zeros((Nsim+1, (NxPerEst + Np*max(1, Nsoil//Nest)), Nest))
x_dis = np.zeros((Nsim+1, (NxPerEst + Np*max(1, Nsoil//Nest)), Nest))
x_ol_dis = np.zeros((Nsim+1, (NxPerEst + Np*max(1, Nsoil//Nest)), Nest))
w_dis = np.zeros((Nsim, (NxPerEst + Np*max(1, Nsoil//Nest)), Nest))
v_dis = np.zeros((Nsim//calNode+1, NyPerEst, Nest))
y_clean_dis = np.zeros((Nsim//calNode+1, NyPerEst, Nest))
y_dis = np.zeros((Nsim//calNode+1, NyPerEst, Nest))
y_ol_dis = np.zeros((Nsim//calNode+1, NyPerEst, Nest))
if Nest ==1:
    x_clean_dis[0, :, 0] = x0
    x_dis[0, :, 0] = x0
    x_ol_dis[0, :, 0] = x0_est
    w_dis[:, :, 0] = w

    y_clean_dis[0, :, 0] = getOutputs_np_aug_subsys(x_clean_dis[0, :, 0], 0)
    y_dis[0, :, 0] = getOutputs_np_aug_subsys(x_dis[0, :, 0], 0) + v_dis[0, :, 0]
    y_ol_dis[0, :, 0] = getOutputs_np_aug_subsys(x_ol_dis[0, :, 0], 0)
else:  # more than 1 estimator
    if Nsoil == 1:
        for i in range(Nest):
            x_clean_dis[0, :, i] = np.append(x0[i*NxPerEst:(i+1)*NxPerEst], x0[Nx:Nx+Np])
            x_dis[0, :, i] = np.append(x0[i*NxPerEst:(i+1)*NxPerEst], x0[Nx:Nx+Np])
            x_ol_dis[0, :, i] = np.append(x0_est[i*NxPerEst:(i+1)*NxPerEst], x0_est[Nx:Nx+Np])
            w_dis[:,:,i] = np.concatenate((w[:,i*NxPerEst:(i+1)*NxPerEst], w[:,Nx:Nx+Np]), axis=1)
            v_dis[:,:,i] = v[:,i*NyPerEst:(i+1)*NyPerEst]

            y_clean_dis[0, :, i] = getOutputs_np_aug_subsys(x_clean_dis[0, :, i], i)
            y_dis[0, :, i] = getOutputs_np_aug_subsys(x_dis[0, :, i], i) + v_dis[0, :, i]
            y_ol_dis[0, :, i] = getOutputs_np_aug_subsys(x_ol_dis[0, :, i], i)
    else:  # assume number of estimators is the same as that of soil types
        for i in range(Nsoil):
            x_clean_dis[0, :, i] = np.append(x0[i*NxPerEst:(i+1)*NxPerEst], x0[Nx+Np*i:Nx+Np*(i+1)])
            x_dis[0, :, i] = np.append(x0[i * NxPerEst:(i + 1) * NxPerEst], x0[Nx+Np*i:Nx+Np*(i+1)])
            x_ol_dis[0, :, i] = np.append(x0_est[i * NxPerEst:(i + 1) * NxPerEst], x0_est[Nx+Np*i:Nx+Np*(i+1)])
            w_dis[:,:,i] = np.concatenate((w[:,i * NxPerEst:(i + 1) * NxPerEst], w[:,Nx+Np*i:Nx+Np*(i+1)]), axis=1)
            v_dis[:,:,i] = v[:,i*NyPerEst:(i+1)*NyPerEst]

            y_clean_dis[0, :, i] = getOutputs_np_aug_subsys(x_clean_dis[0, :, i], i)
            y_dis[0, :, i] = getOutputs_np_aug_subsys(x_dis[0, :, i], i) + v_dis[0, :, i]
            y_ol_dis[0, :, i] = getOutputs_np_aug_subsys(x_ol_dis[0, :, i], i)

# For estimator --------------------------------------------------
x_mhe = np.zeros((Nsim//calNode+1, (NxPerEst + Np*max(1, Nsoil//Nest)), Nest))  # mhe state estimation history
y_mhe = np.zeros((Nsim//calNode+1, NyPerEst, Nest))
if Nest ==1:
    x_mhe[0, :, 0] = x0_est
    y_mhe[0, :, 0] = getOutputs_np_aug_subsys(x_mhe[0, :, 0], 0)
else:
    if Nsoil == 1:
        for i in range(Nest):
            x_mhe[0, :, i] = np.append(x0_est[i*NxPerEst:(i+1)*NxPerEst], x0_est[Nx:Nx+Np])
            y_mhe[0, :, i] = getOutputs_np_aug_subsys(x_mhe[0, :, i], i)
    else:
        for i in range(Nsoil):
            x_mhe[0, :, i] = np.append(x0_est[i*NxPerEst:(i+1)*NxPerEst], x0_est[Nx+Np*i:Nx+Np*(i+1)])
            y_mhe[0, :, i] = getOutputs_np_aug_subsys(x_mhe[0, :, i], i)

X_allInWin_allEst = np.zeros((Nmhe*calNode+1, (NxPerEst+Np*max(1, Nsoil//Nest)), Nest))
W_allInWin_allEst = np.zeros((Nmhe*calNode, (NxPerEst+Np*max(1, Nsoil//Nest)), Nest))
Y_allInWin_allEst = np.zeros((Nmhe*calNode+1, (NyPerEst), Nest))

x_bar_allEst = np.zeros((NxPerEst+Np*max(1, Nsoil//Nest), Nest))
x_bar_allEst_alt = np.zeros((NxPerEst+Np*max(1, Nsoil//Nest), Nest))

# For iteration at each time instant -------------------------------
if Nest == 1:
    numIter = 1
else:
    numIter = 100
tol = 1e-8  # todo: can be modified later.
# initialize
x_sub1 = np.zeros((numIter + 1, NxPerEst+Np*max(1, Nsoil//Nest)))  # number of rows = iteration numbers + 1 (0 to numIter)
                                                                    # number of columns =  total # of elements (states and parameters) in each subsystem
x_sub2 = np.zeros((numIter + 1, NxPerEst+Np*max(1, Nsoil//Nest)))

# For error -------------------------------------------------------
RMSE_Y = np.zeros(Nsim//calNode+1)
RMSE_X = np.zeros(Nsim//calNode+1)
RMSE_P = np.zeros(Nsim//calNode+1)
if Nest == 1:
    y_mhe_timeS = np.zeros((Nsim+1, Ny))
    y_mhe_timeS[:,:] = y_mhe[:,:,0]
    x_mhe_timeS = np.zeros((Nsim+1, Nx))
    x_mhe_timeS[:,:] = x_mhe[:,:Nx,0]
    p_mhe_timeS = np.zeros((Nsim+1, NpTotal))
    p_mhe_timeS[:,:] = x_mhe[:,Nx:,0]

elif Nest == 2:
    y_mhe_timeS = np.zeros((Nsim+1, Ny))
    y_mhe_timeS[0,0:NyPerEst] = y_mhe[0,:,0]
    y_mhe_timeS[0,NyPerEst:] = y_mhe[0,:,1]

    x_mhe_timeS = np.zeros((Nsim+1, Nx))
    x_mhe_timeS[0,0:NxPerEst] = x_mhe[0,:NxPerEst,0]
    x_mhe_timeS[0,NxPerEst:] = x_mhe[0,:NxPerEst,1]

    p_mhe_timeS = np.zeros((Nsim+1, NpTotal))
    p_mhe_timeS[0,0:Np] = x_mhe[0,NxPerEst:,0]
    p_mhe_timeS[0,Np:] = x_mhe[0,NxPerEst:,1]

elif Nest == 4:
    y_mhe_timeS = np.zeros((Nsim+1, Ny))
    y_mhe_timeS[0,0:NyPerEst] = y_mhe[0,:,0]
    y_mhe_timeS[0,NyPerEst:NyPerEst*2] = y_mhe[0,:,1]
    y_mhe_timeS[0,NyPerEst*2:NyPerEst*3] = y_mhe[0,:,2]
    y_mhe_timeS[0,NyPerEst*3:NyPerEst*4] = y_mhe[0,:,3]

    x_mhe_timeS = np.zeros((Nsim+1, Nx))
    x_mhe_timeS[0,0:NxPerEst] = x_mhe[0,:NxPerEst,0]
    x_mhe_timeS[0,NxPerEst:NxPerEst*2] = x_mhe[0,:NxPerEst,1]
    x_mhe_timeS[0,NxPerEst*2:NxPerEst*3] = x_mhe[0,:NxPerEst,2]
    x_mhe_timeS[0,NxPerEst*3:NxPerEst*4] = x_mhe[0,:NxPerEst,3]

    p_mhe_timeS = np.zeros((Nsim+1, NpTotal))
    p_mhe_timeS[0,0:Np] = x_mhe[0,NxPerEst:,0]
    p_mhe_timeS[0,Np:Np*2] = x_mhe[0,NxPerEst:,1]
    p_mhe_timeS[0,Np*2:Np*3] = x_mhe[0,NxPerEst:,2]
    p_mhe_timeS[0,Np*3:Np*4] = x_mhe[0,NxPerEst:,3]

RMSE_Y[0] = np.sqrt(np.sum((y[0]-y_mhe_timeS[0])**2)/Ny)
RMSE_X[0] = np.sqrt(np.sum((x[0,:Nx]-x_mhe_timeS[0])**2)/Nx)
RMSE_P[0] = np.sqrt(np.sum((x[0,Nx:]-p_mhe_timeS[0])**2)/Np)

# For recording time -------------------------------------------------------
mhe_SolverTimeUsed_List = np.zeros((int(Nsim/calNode)-Nmhe, Nest))
mhe_SolverConstructed_List = np.zeros((int(Nsim/calNode)-Nmhe, Nest))
fie_SolverConstructed_List = np.zeros((Nmhe, Nest))
fie_SolverTimeUsed_List = np.zeros((Nmhe, Nest))

# For recording other information -----------------------------------------
filter_smoother = []
arr_cost = []
solverList = []

# For inputs - irrigation rate ---------------------------------------------
uu = irr(Nsim, DeltaT, Tsim)

# == Symbolic Models for Optimization Problems =========================================================================
# For centralized system --------------------------------------------------------------
x_symbol = MX.sym("x",Nx_aug)
u_symbol = MX.sym("u",Nu)
qleft_symbol = MX.sym("qleft", Ninfo)
qright_symbol = MX.sym("qright", Ninfo)
w_symbol = MX.sym("w",Nw_aug)
print('generating original symbolic 3D model')
timeS = -time.time()
F_symbol = getODE_mx_3D_process(x_symbol,u_symbol,qleft_symbol,qright_symbol,w_symbol)
# H_symbol = getOutputs_mx_aug(x_symbol)
timeE = time.time() + timeS
print('use', timeE/60, 'mins')
print('generating original symbolic 3D model - multi timestep')
timeS = -time.time()
F_N_symbol = FN_multiLayer_process(x_symbol,u_symbol,qleft_symbol,qright_symbol,w_symbol)
timeE = time.time() + timeS
print('use', timeE/60, 'mins')
F_N_casadi = Function('F_N_casadi', [x_symbol,u_symbol,qleft_symbol,qright_symbol,w_symbol], [F_N_symbol])

ode = {'x': x_symbol, 'p': vertcat(u_symbol,qleft_symbol,qright_symbol,w_symbol), 'ode': F_symbol}
opts = {'tf': DeltaT, 'regularity_check':True}  # seconds
I = integrator('I', 'cvodes', ode, opts)  # Build casadi integrator

# For distributed systems ----------------------------------------------------------------------------
x_symbol_est = MX.sym("x_est",NxPerEst+Np*max(1,Nsoil//Nest))
u_symbol_est = MX.sym("u_est",Nu)
xleft_symbol_est = MX.sym("xleft_est", Ninfo)
xright_symbol_est = MX.sym("xright_est", Ninfo)
w_symbol_est = MX.sym("w_est",NxPerEst+Np*max(1,Nsoil//Nest))
para_symbol_est = MX.sym("p_est",Np*max(1, Nsoil//Nest))
print('generating subsystem 1')
timeS = -time.time()
F_symbol_est = getODE_mx_3D_subsys_estimator(x_symbol_est, u_symbol_est, xleft_symbol_est, xright_symbol_est, w_symbol_est, para_symbol_est)
# H_symbol_est1 = getOutputs_mx_aug_subsys1(x_symbol_est, 0)
print('use', (time.time()+timeS)/60, 'mins')

print('generating subsystem - multistep')
timeS = -time.time()
F_N_symbol_est = F_N_aug_mx_subsys(x_symbol_est, u_symbol_est, xleft_symbol_est, xright_symbol_est, w_symbol_est, para_symbol_est)
F_N_casadi_est = Function('F_N_casadi_est', [x_symbol_est,u_symbol_est,xleft_symbol_est, xright_symbol_est,w_symbol_est,para_symbol_est], [F_N_symbol_est])
print('use', (time.time()+timeS)/60, 'mins')

ode_est = {'x': x_symbol_est, 'p': vertcat(u_symbol_est, xleft_symbol_est, xright_symbol_est, w_symbol_est, para_symbol_est), 'ode': F_symbol_est}
opts = {'tf': DeltaT, 'regularity_check':True}  # seconds
I_est = integrator('I_est', 'cvodes', ode_est, opts)  # Build casadi integrator

# Simulation begins here ===============================================================================================
solvetime = -time.time()
for i in range(1, Nsim+1):
    print('*****************************************************************************************')
    print('*****************************************************************************************')
    print('Previous time: ', i-1, ', current time: ', i)
    if seedpoint == []:
        print('Nmhe is:', Nmhe, ', dt_internal is:', DeltaT_internal, ', tol is:', tol_desired, ', seedpoint is empty')
    else:
        print('Nmhe is:', Nmhe, ', dt_internal is:', DeltaT_internal, ', tol is:', tol_desired, ', seedpoint is:', seedpoint)

# == Model simulation ==================================================================================================
    # Distributed simulator DT, when number of estimator is greater than 1 ---------------------------------------------
    if Nest != 1:
        print('subsystem simulator - clean')
        x_clean_dis[i,:,:] = simulator_dis(x_clean_dis[i-1,:,:], np.zeros((NxPerEst+Np*max(1,Nsoil//Nest),Nest)), Nest, NxPerEst, Nu, Ninfo, Nsoil, Np, dz, uu[i-1,:], ParsNp, F_N_casadi_est, I_est, F_N_aug_np_subsys, KFun_np)
        print('subsystem simulator - process')
        x_dis[i,:,:] = simulator_dis(x_dis[i-1,:,:], w_dis[i-1,:,:], Nest, NxPerEst, Nu, Ninfo, Nsoil, Np, dz, uu[i-1,:], ParsNp, F_N_casadi_est, I_est, F_N_aug_np_subsys, KFun_np)
        # x_dis[i,:,:] = simulator_dis(x_dis[i-1,:,:], np.zeros((NxPerEst+Np*max(1,Nsoil//Nest),Nest)), Nest, NxPerEst, Nu, Ninfo, Nsoil, Np, dz, uu[i-1,:], ParsNp, F_N_casadi_est, I_est, F_N_aug_np_subsys, KFun_np)
        print('subsystem simulator - ol')
        x_ol_dis[i,:,:] = simulator_dis(x_ol_dis[i-1,:,:], np.zeros((NxPerEst+Np*max(1,Nsoil//Nest),Nest)), Nest, NxPerEst, Nu, Ninfo, Nsoil, Np, dz, uu[i-1,:], ParsNp, F_N_casadi_est, I_est, F_N_aug_np_subsys, KFun_np)

        # Now get measurements from sensors
        for j in range(Nest):
            y_clean_dis[i//calNode, :, j] = getOutputs_np_aug_subsys(x_clean_dis[i, :, j], j)
            y_dis[i//calNode, :, j] = getOutputs_np_aug_subsys(x_dis[i, :, j], j) + v_dis[i//calNode, :, j]
            y_ol_dis[i//calNode, :, j] = getOutputs_np_aug_subsys(x_ol_dis[i, :, j], j)

    # Centralized simulator DT --------------------------------------------------------------------
    print('centralized simulator')
    qleft_clean = np.zeros(Ninfo)
    qleft_orig = np.zeros(Ninfo)
    qleft_ol = np.zeros(Ninfo)

    qright_clean = np.zeros(Ninfo)
    qright_orig = np.zeros(Ninfo)
    qright_ol = np.zeros(Ninfo)
    print('MX Simulator running')
    x_clean[i,:] = F_N_casadi(x_clean[i-1,:], uu[i-1,:]*np.ones(Nu), qleft_clean, qright_clean, np.zeros(Nw_aug)).full().ravel()
    x[i,:] = F_N_casadi(x[i-1,:],uu[i-1,:]*np.ones(Nu), qleft_orig, qright_orig, w[i-1,:]).full().ravel()
    x_ol[i,:] = F_N_casadi(x_ol[i-1,:], uu[i-1,:]*np.ones(Nu), qleft_ol, qright_ol, np.zeros(Nw_aug)).full().ravel()
    print('MX Simulator finished')

    # Get measurement from sensors
    y_clean[i//calNode,:] = getOutputs_np_aug(x_clean[i, :])    # current measurement, add measurement noise
    y[i//calNode,:] = getOutputs_np_aug(x[i, :]) + v[i//calNode, :]    # current measurement, add measurement noise
    y_ol[i//calNode,:] = getOutputs_np_aug(x_ol[i, :])# + v[i,:]

# == Moving Horizon Estimation =========================================================================================
    MHEFIEtotalTime = -time.time()
    x_in_allEst = np.zeros((min(i, Nmhe*calNode) + 1, NxPerEst + Np*max(1, Nsoil//Nest), Nest))  # for all estimators
    x_in_allEst_alt = np.zeros((min(i, Nmhe*calNode) + 1, NxPerEst + Np*max(1, Nsoil//Nest), Nest))

    # TODO: After here, we need to put them into iteration. Follow Haihan's code to modify
    for itr in range(numIter):  # Do Jacobi iteration
        print('Running iteration', itr, 'at time instant', i)

        # Todo: need to figure out what is the input and output for the following codes
        # Inputs: X_allInWin_allEST - For smoother scheme: Used to create the arrival state (x_bar) and initial guess (x_in)
        #           x_mhe - For filter: used to create the arrival state (x_bar) and initial guess (x_in)
        ''' Preparation '''
        # The following code is the most important, because it includes the information needed for exchanging
        x_bar_allEst, x_in_allEst = filter_smoother_scheme('filter', x_bar_allEst, x_in_allEst, x_mhe, x_ol,
                                                           X_allInWin_allEst, i, calNode, Nmhe, Nx, Np, Nsoil, Nest,
                                                           NxPerEst, NyPerEst, itr)  # Now we only have code for smoother
        for j in range(Nest):
            xbegin = j*NxPerEst
            xend = (j+1)*NxPerEst
            ybegin = j*NyPerEst
            yend = (j+1)*NyPerEst
            pbegin = j*Np
            pend = (j+1)*Np
            if i // calNode <= Nmhe:  # use FIE, else use mhe
                print('****************************************** FIE ******************************************')
                print('Current instant:', i, 'Current hour:', i / (3600 / DeltaT), 'Estimator No.', j + 1)
                filter_smoother += 'filter'
                arr_cost += 'yes'
                # The information exchange happens in the following line
                u_in1, qleft_in1, qright_in1, para_in1, x_in1, y_in1, P_in1, Q_in1, R_in1, x_bar = mhe_prepare(i, j, i, x_bar_allEst, y, uu, x_in_allEst, KFun_np, NxPerEst, Nu, Ninfo, Nest, Nsoil, Np, dz, ParsNp, calNode, Nmhe, Nx_aug, ybegin, yend, P0, Q, R)

                Pinv = linalg.inv(P_in1)

                fie_timeBuild = -time.time()
                print('Start constructing the solver')
                solver, lbdv, ubdv, lbg, ubg = mhe_discrete(Pinv, Q_in1, R_in1, i, tol_desired, I_est, F_N_casadi_est, 'fie',
                                                            getOutputs_mx_aug_subsys, Nu, Ninfo, Np, j,
                                                            calNode)  # full information estimation
                fie_timeBuild += time.time()
                # ---------------------------------------------------------------------------------------------------
                x_flat, y_flat, u_flat, qleft_flat, qright_flat, para_flat, _ = flaten(x_in1, y_in1, u_in1, qleft_in1, qright_in1, para_in1, P_in1, i, Nsoil, Nest, NxPerEst, Nu, Ninfo, Np, Nmhe, NyPerEst, calNode)
                x0 = np.concatenate(
                    (x_flat, np.zeros((NxPerEst + Np * max(1, Nsoil // Nest)) * i), y_flat, u_flat, qleft_flat, qright_flat, para_flat, x_bar))
                # ---------------------------------------------------------------------------------------------------
                fie_timeSolve = -time.time()
                sol = solver(x0=vertcat(x0), lbx=vertcat(lbdv, y_flat, u_flat, qleft_flat, qright_flat, para_flat, x_bar),
                             ubx=vertcat(ubdv, y_flat, u_flat, qleft_flat, qright_flat, para_flat, x_bar), lbg=lbg, ubg=ubg)
                x_opt = sol['x'].full().ravel()
                x_mhe_next, X_allInWin, W_allInWin, Y_allInWin = cal_cur_state(x_opt, u_in1, i, I_est, F_N_casadi_est,
                                                                               getOutputs_np_aug_subsys,
                                                                               NxPerEst + Np * max(1, Nsoil // Nest),
                                                                               NyPerEst,
                                                                               j)  # calculate current state estimate according to the optimization results
                fie_timeSolve += time.time()
                fie_SolverConstructed_List[i // calNode - 1, j] += fie_timeBuild  # Accumulate all iterations at 1 time instant
                fie_SolverTimeUsed_List[i // calNode - 1, j] += fie_timeSolve
                print('This FIE solver used', fie_timeBuild, 'secs to build')
                print('This FIE solver used', fie_timeSolve, 'secs to solve')
                print('This FIE solver used', fie_timeBuild + fie_timeSolve, 'secs')
            else:
                print('****************************************** MHE ******************************************')
                print('Current instant:', i, 'Current hour:', i / (3600 / DeltaT), 'Estimator No.', j + 1)
                filter_smoother += 'filter'
                arr_cost += 'yes'
                u_in2, qleft_in2, qright_in2, para_in2, x_in2, y_in2, P_in2, Q_in2, R_in2, x_bar = mhe_prepare(i, j, Nmhe*calNode, x_bar_allEst, y, uu, x_in_allEst, KFun_np, NxPerEst, Nu, Ninfo, Nest, Nsoil, Np, dz, ParsNp, calNode, Nmhe, Nx_aug, ybegin, yend, P0, Q, R)

                Pinv = linalg.inv(P_in2)

                if i // calNode == Nmhe + 1:
                    mhe_timeBuild = -time.time()
                    solver_mhe, lbdv, ubdv, lbg, ubg = mhe_discrete(Pinv, Q_in2, R_in2, Nmhe * calNode, tol_desired, I_est,
                                                                F_N_casadi_est, 'mhe', getOutputs_mx_aug_subsys, Nu, Ninfo, Np,
                                                                j, calNode)  # full information estimation
                    mhe_timeBuild += time.time()
                    print('Time used to build MHE is', mhe_timeBuild)
                    solverList += [solver_mhe]
                # ---------------------------------------------------------------------------------------------------
                x_flat, y_flat, u_flat, qleft_flat, qright_flat, para_flat, _ = flaten(x_in2, y_in2, u_in2, qleft_in2, qright_in2, para_in2, P_in2, Nmhe*calNode, Nsoil, Nest, NxPerEst, Nu, Ninfo, Np, Nmhe, NyPerEst, calNode)
                x0 = np.concatenate((x_flat, np.zeros((NxPerEst + Np * max(1, Nsoil // Nest)) * Nmhe * calNode), y_flat,
                                     u_flat, qleft_flat, qright_flat, para_flat, x_bar))
                # ---------------------------------------------------------------------------------------------------
                solver = solverList[j]
                mhe_timeSolve = -time.time()
                sol = solver(x0=vertcat(x0), lbx=vertcat(lbdv, y_flat, u_flat, qleft_flat, qright_flat, para_flat, x_bar),
                             ubx=vertcat(ubdv, y_flat, u_flat, qleft_flat, qright_flat, para_flat, x_bar), lbg=lbg, ubg=ubg)
                x_opt = sol['x'].full().ravel()
                x_mhe_next, X_allInWin, W_allInWin, Y_allInWin = cal_cur_state(x_opt, u_in2, Nmhe * calNode, I_est,
                                                                               F_N_casadi_est, getOutputs_np_aug_subsys,
                                                                               NxPerEst + Np * max(1, Nsoil // Nest),
                                                                               NyPerEst,
                                                                               j)  # calculate current state estimate according to the optimization results
                mhe_timeSolve += time.time()
                mhe_SolverConstructed_List[i // calNode - 1 - Nmhe, j] += mhe_timeBuild
                mhe_SolverTimeUsed_List[i // calNode - 1 - Nmhe, j] += mhe_timeSolve
                print('This MHE used', mhe_timeSolve, 'secs to solve')

            # Save the outputs into array for storing iteration results
            # Outputs for above MHE are: x_mhe_next, X_allInWin, W_allInWin, Y_allInWin
            x_mhe[i//calNode,:, j] = x_mhe_next
            y_mhe[i//calNode,:, j] = getOutputs_np_aug_subsys(x_mhe[i//calNode, :, j], j)

            X_allInWin_allEst[0:min(i, Nmhe*calNode)+1,:,j] = X_allInWin
            W_allInWin_allEst[0:min(i, Nmhe*calNode),:,j] = W_allInWin
            Y_allInWin_allEst[0:min(i, Nmhe*calNode)+1,:,j] = Y_allInWin

            # calculate the estimation performance index
            if Nest == 1:
                y_mhe_timeS[i//calNode, :] = y_mhe[i//calNode, :, 0]
                x_mhe_timeS[i//calNode, :] = x_mhe[i//calNode, :Nx, 0]
                p_mhe_timeS[i//calNode, :] = x_mhe[i//calNode, Nx:, 0]

            elif Nest == 2:
                y_mhe_timeS[i//calNode, ybegin:yend] = y_mhe[i//calNode, :, j]
                x_mhe_timeS[i//calNode, xbegin:xend] = x_mhe[i//calNode, :NxPerEst, j]
                p_mhe_timeS[i//calNode, pbegin:pend] = x_mhe[i//calNode, NxPerEst:, j]

            print('Para are: ', x_mhe_next[-Np*max(1, Nsoil//Nest):])
            MHEFIEtotalTime += time.time()

            # Save iterative results in each array
            if j == 0:
                x_sub1[itr] = x_mhe_next
            elif j == 1:
                x_sub2[itr] = x_mhe_next
            else:
                print('Current code only works for at most 2 subsystems.')
                quit()
        if itr > 0:  # Check tolerance after the second iteration is done
            if (abs(x_sub1[itr] - x_sub1[itr-1]) < tol).all() and (abs(x_sub2[itr] - x_sub2[itr-1]) < tol).all():
                break  # Break the for loop
        # TODO: Before here, we need to put then into iteration
    RMSE_Y[i] = np.sqrt(np.sum((y[i] - y_mhe_timeS[i]) ** 2) / Ny)
    RMSE_X[i] = np.sqrt(np.sum((x[i, :Nx] - x_mhe_timeS[i]) ** 2) / Nx)
    RMSE_P[i] = np.sqrt(np.sum((x[i, Nx:] - p_mhe_timeS[i]) ** 2) / NpTotal)
    print('RMSE_Y is', RMSE_Y[i-1:i+1])
    print('RMSE_X is', RMSE_X[i - 1:i + 1])
    print('RMSE_P is', RMSE_P[i - 1:i + 1])
    # io.savemat('Data/x.mat', dict(t=Tplot, dt=DeltaT, xmea=x, xmhe=x_mhe, xol=x_ol, xclean=x_clean))
    # io.savemat('Data/y.mat', dict(t=Tplot, dt=DeltaT, ymea=y, ymhe=y_mhe, yol=y_ol, yclean=y_clean))
    # io.savemat('Data/timeUsed.mat', dict(fie_timeUsedBuild_list=fie_SolverConstructed_List,
    #                                      fie_timeUsedSolve_list=fie_SolverTimeUsed_List,
    #                                      mhe_timeUsedBuild_list=mhe_SolverConstructed_List,
    #                                      mhe_timeUsedSolve_list=mhe_SolverTimeUsed_List))
# ======================================================================================================================
solvetime += time.time()
mhe_timeUsed_avg = np.average(mhe_SolverTimeUsed_List)

print ("Took" + "%5.3g s" % solvetime, 'to run from t0 to tf.', 'Average time that MHE used is', '%5.3g s.' % mhe_timeUsed_avg)

if seedpoint == []:
        print('Nmhe is: ', Nmhe, ', dt_internal is: ', DeltaT_internal, 'tol is: ', tol_desired, ', seedpoint is empty.', ' Initial state is NOT estimated.')
else:
        print('Nmhe is: ', Nmhe, ', dt_internal is: ', DeltaT_internal, 'tol is: ', tol_desired, ', seedpoint is: ',seedpoint, ' Initial state is NOT estimated.')

if Nest != 1:
    x_dis_timeS = np.zeros((Nsim+1, Nx_aug))
    x_clean_dis_timeS = np.zeros((Nsim+1, Nx_aug))
    x_ol_dis_timeS = np.zeros((Nsim+1, Nx_aug))

    x_dis_timeS[:,0:NxPerEst] = x_dis[:,:NxPerEst,0]
    x_dis_timeS[:,NxPerEst:Nx] = x_dis[:,:NxPerEst,1]
    x_dis_timeS[:,Nx:Nx+Np] = x_dis[:,NxPerEst:,0]
    x_dis_timeS[:,Nx+Np:] = x_dis[:,NxPerEst:,1]

    x_clean_dis_timeS[:,0:NxPerEst] = x_clean_dis[:,:NxPerEst,0]
    x_clean_dis_timeS[:,NxPerEst:Nx] = x_clean_dis[:,:NxPerEst,1]
    x_clean_dis_timeS[:,Nx:Nx+Np] = x_clean_dis[:,NxPerEst:,0]
    x_clean_dis_timeS[:,Nx+Np:] = x_clean_dis[:,NxPerEst:,1]

    x_ol_dis_timeS[:,0:NxPerEst] = x_ol_dis[:,:NxPerEst,0]
    x_ol_dis_timeS[:,NxPerEst:Nx] = x_ol_dis[:,:NxPerEst,1]
    x_ol_dis_timeS[:,Nx:Nx+Np] = x_ol_dis[:,NxPerEst:,0]
    x_ol_dis_timeS[:,Nx+Np:] = x_ol_dis[:,NxPerEst:,1]

# Make a plot of RMSE --------------------------------------------------------------------------------------------------
startPlotPoint = 0
endPoint = -1
interval = 1
plt.figure()
plt.plot(Tplot[startPlotPoint:endPoint:interval]/DeltaT, RMSE_Y[startPlotPoint:endPoint:interval], label='Y_RMSE')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_2, label='Y_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_Y_3, label='Y_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of Measurements")
plt.grid()
# plt.savefig('RMSE_Y.png')
plt.show()

plt.figure()
plt.plot(Tplot[startPlotPoint:endPoint:interval]/DeltaT, RMSE_X[startPlotPoint:endPoint:interval], label='X_RMSE')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_2, label='X_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_X_3, label='X_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of States")
plt.grid()
# plt.savefig('RMSE_X.png')
plt.show()

plt.figure()
plt.plot(Tplot[startPlotPoint:endPoint:interval]/DeltaT, RMSE_P[startPlotPoint:endPoint:interval], label='P_RMSE')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_2, label='P_RMSE_dmhe_midBounds')
# plt.plot(Tplot[startPlotPoint::interval]/DeltaT, RMSE_P_3, label='P_RMSE_dmhe_smallBounds')
plt.legend()
plt.xlabel("Time, t (hr)")
plt.ylabel("RMSE of Parameters")
plt.grid()
# plt.savefig('RMSE_P.png')
plt.show()

# All 32 measurements calculated from the 32 states --------------------------------------------------------------------
for j in range(Nest):
    for i in range(NyPerEst):
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, y[:, NyPerEst*j+i], '-', label='y_exp')
        #plt.plot(Tplot[:]/DeltaT, y_dis[:(Nsim//calNode)+1, i, j], '--', label='y_exp_subsys')
        # plt.plot(Tplot[:]/DeltaT, y_ekf_plot[:, i], '.', markersize=3, label='y_ekf')
        # plt.plot(Tplot[::calNode]/DeltaT, y_enkf_plot[::calNode, i], '.', markersize=3, label='y_enkf')
        plt.plot(Tplot[::calNode]/DeltaT, y_mhe[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='y_mhe')
        # plt.plot(Tplot[:]/DeltaT, y_ol[:, NyPerEst*j+i], '--', label='y_ol')
        # plt.plot(Tplot[:]/DeltaT, y_clean[:, NyPerEst*j+i], '-', label='y_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Soil Moisture, theta (m3/m3)')
        plt.xticks(Tplot[::calNode*2]/DeltaT, np.arange(Tsim/DeltaTmhe+1, step=2))
        plt.grid()
        plt.show()
        # plt.savefig('Y_Figure_%g.png' % (i+1))

# Plot states
for j in range(Nest):
    for i in range(0, NxPerEst):  # choose how many states do yo wanna plot
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, x[:, i], '-', label='x_exp')
        # plt.plot(Tplot[:]/DeltaT, x_ekf[:, i], '.', markersize=3, label='x_ekf')
        # plt.plot(Tplot[::calNode]/DeltaT, x_enkf[::calNode, i], '.', markersize=3, label='x_enkf')
        plt.plot(Tplot[::calNode]/DeltaT, x_mhe[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='x_mhe')
        # plt.plot(Tplot[:]/DeltaT, x_ol[:, Nx-NxPerEst+i], '--', label='p_ol')
        # plt.plot(Tplot[:] / DeltaT, x_clean[:, Nx-NxPerEst+i], '-', label='p_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Pressure, h (m)')
        # plt.xticks(Tplot[::calNode*2]/DeltaT, np.arange(Tsim/DeltaTmhe+1, step=2))
        plt.grid()
        plt.show()
        # plt.savefig('X_Figure_%g.png' % (j * Np + i + 1-Nx))

# Plot Parameters
for j in range(Nest):
    for i in range(NxPerEst, NxPerEst+Np*max(1, Nsoil//Nest)):  # choose how many states do yo wanna plot
        plt.figure()
        plt.plot(Tplot[:]/DeltaT, x[:, Nx-NxPerEst+i], '-', label='p_exp')
        # plt.plot(Tplot[:]/DeltaT, x_ekf[:, i], '.', markersize=3, label='x_ekf')
        # plt.plot(Tplot[::calNode]/DeltaT, x_enkf[::calNode, i], '.', markersize=3, label='x_enkf')
        plt.plot(Tplot[::calNode]/DeltaT, x_mhe[:(Nsim//calNode)+1, i, j], '.', markersize=4, label='p_mhe')
        # plt.plot(Tplot[:]/DeltaT, x_ol[:, Nx-NxPerEst+i], '--', label='p_ol')
        # plt.plot(Tplot[:] / DeltaT, x_clean[:, Nx-NxPerEst+i], '-', label='p_clean')
        plt.legend()
        plt.xlabel('Time, t (hr)')
        plt.ylabel('Parameters')
        # plt.xticks(Tplot[::calNode*2]/DeltaT, np.arange(Tsim/DeltaTmhe+1, step=2))
        plt.grid()
        plt.show()
        # plt.savefig('P_Figure_%g.png' % (j * Np + i + 1-Nx))

# Save data
io.savemat('x.mat', dict(t=Tplot, dt=DeltaT, xmea=x, xmhe=x_mhe, xol=x_ol, xclean=x_clean, xmea_dis=x_dis))
io.savemat('y.mat', dict(t=Tplot, dt=DeltaT, ymea=y, ymhe=y_mhe, yol=y_ol, yclean=y_clean, ymea_dis=y_dis))
io.savemat('input_noise.mat', dict(u=uu, w = w, v = v))
io.savemat('cMatrix.mat', dict(c = CMatrix))
io.savemat('SE.mat', dict(RMSE_Y=RMSE_Y, RMSE_X=RMSE_X, RMSE_P=RMSE_P))
io.savemat('timeUsed.mat', dict(fie_timeUsedBuild_list=fie_SolverConstructed_List, fie_timeUsedSolve_list=fie_SolverTimeUsed_List,
                                mhe_timeUsedBuild_list=mhe_SolverConstructed_List, mhe_timeUsedSolve_list=mhe_SolverTimeUsed_List))
# io.savemat('space_parameters.mat', dict(Nx=Nx, Nw=Nw, Ny=Ny, Nv=Nv, Nu=Nu, Np=Np, dz=dz, Nx_aug=Nx_aug, Nw_aug=Nx_aug, Nest=Nest, NxPerEst=NxPerEst, NyPerEst=NyPerEst))
io.savemat('space_parameters.mat', spacePara_dict)
io.savemat('time_parameters.mat', dict(DeltaT=DeltaT, Tsim=Tsim, Nsim=Nsim, DeltaT_internal=DeltaT_internal, Nsim_internal=Nsim_internal, Tplot=Tplot, Nmhe=Nmhe, DeltaTmhe=DeltaTmhe, calNode=calNode))
io.savemat('model_parameters.mat', dict(ParsDict=ParsDict, ParsNp=ParsNp, ParsScale=ParsScale))
io.savemat('weight_matrices.mat', dict(P0=P0, Q=Q, R=R, noise_w=noise_w, noise_v=noise_v))
io.savemat('mhe_bounds.mat', dict(xlower=xlower, xupper=xupper, wlower=wlower, wupper=wupper, xlower_fie=xlower_fie, xupper_fie=xupper_fie))
io.savemat('mhe_setup.mat', dict(filter_smooter=filter_smoother, arr_cost=arr_cost))

'''     
# Plots simulator results
indexToPlot = np.arange(0,Nxz,1)+Nxz*2
for i in indexToPlot:  # choose how many states do yo wanna plot
    plt.figure()
    plt.plot(Tplot[:endPoint]/DeltaT, x[:endPoint, i], '-', label='x_true')
    plt.plot(Tplot[:endPoint]/DeltaT, x_np_V2[:endPoint, i], '-', label='x_true_np')

    # plt.plot(Tplot[:endPoint:calNode]/DeltaT, x_mhe_1_timeS[:endPoint, i], '.', markersize=4, label='x_dmhe_bigB')
    # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_2_timeS[:, i], '.', markersize=4, label='x_dmhe_midB')
    # plt.plot(Tplot[::calNode]/DeltaT, x_mhe_3_timeS[:, i], '.', markersize=4, label='x_dmhe_smallB')

    plt.plot(Tplot[:endPoint]/DeltaT, x_ol[:endPoint, i], '-', label='x_ol')
    plt.plot(Tplot[:endPoint] / DeltaT, x_clean[:endPoint, i], '-', label='x_clean')
    plt.legend()
    plt.xlabel('Time, t (day)')
    plt.ylabel('Pressure, h (m)')
    # plt.xticks(Tplot[:endPoint:calNode * 144] / DeltaT, np.arange(Tsim / (DeltaTmhe * 6 * 24) + 1, step=1))
    plt.grid()
    plt.show()
    # plt.savefig('X_Figure_%g.png' % (i + 1))

# plt.figure()
# for i in range(w.shape[1]):
#     plt.plot(w[:, i])
# plt.show()
#
# plt.figure()
# for i in range(v.shape[1]):
#     plt.plot(v[:, i])
# plt.show()
'''