"""
Created on April 2 2019

@author: Song Bo

The states are indexed from the top (0) to the bottom (-1)
"""

from __future__ import (division)
from casadi import *
from Parameters import model_parameters_multiLayer, space_parameters, time_parameters, cmatrix_single


# ----------------------------------------------------------------------------------------------------------------------
# Create iterative MX model, used for optimization in MHE
# ----------------------------------------------------------------------------------------------------------------------
def FN_multiLayer_process(xk, uk, qleftk, qrightk, wk):  # used when dt is greater than 2 mins, because the FD model has numerical issue when dt > 2mins
    DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
    x_internal = xk
    for j in range(Nsim_internal):
        x_internal = x_internal + DeltaT_internal * getODE_mx_3D_process(x_internal, uk, qleftk, qrightk, wk)
    xk = x_internal
    return xk


# def F_N_np(xk, uk, qbotk, wk):  # used when dt is greater than 2 mins, because the FD model has numerical issue when dt > 2mins
#     DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
#     x_internal = xk
#     for j in range(Nsim_internal):
#         x_internal = x_internal + DeltaT_internal * getODE_np_3D_process(x_internal, uk, qbotk, wk)
#     xk = x_internal
#     return xk


def F_N_np_V2(xk, uk, qleftk, qrightk, wk):  # used when dt is greater than 2 mins, because the FD model has numerical issue when dt > 2mins
    DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
    x_internal = xk
    for j in range(Nsim_internal):
        x_internal = x_internal + DeltaT_internal * getODE_np_3D_process_V2(x_internal, uk, qleftk, qrightk, wk)
    xk = x_internal
    return xk


def F_N_aug_mx_subsys(xk, uk, xleftk, xrightk, wk, pk):  # used when dt is greater than 2 mins, because the FD model has numerical issue when dt > 2mins
    DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
    x_internal = xk
    for j in range(Nsim_internal):
        x_internal = x_internal + DeltaT_internal * getODE_mx_3D_subsys_estimator(x_internal, uk, xleftk, xrightk, wk, pk)
    xk = x_internal
    return xk


def F_N_aug_np_subsys(xk, uk, qleftk, qrightk, wk, pk):  # used when dt is greater than 2 mins, because the FD model has numerical issue when dt > 2mins
    DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters
    x_internal = xk
    for j in range(Nsim_internal):
        x_internal = x_internal + DeltaT_internal * getODE_np_3D_subsys(x_internal, uk, qleftk, qrightk, wk, pk)
    xk = x_internal
    return xk


# ----------------------------------------------------------------------------------------------------------------------
# Create np & Unscaled Richards DAE model
# ----------------------------------------------------------------------------------------------------------------------
# - K, C, theta, h - np & unscaled --------------------------------------------------------
def mean_KFun_np(psi1, psi2, pars, parsDict):
    K1 = KFun_np(psi1, pars, parsDict)
    K2 = KFun_np(psi2, pars, parsDict)
    K = (K1+K2)/2
    return K


def KFun_np(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+abs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    K = parsDict[0]*pars[0]*(Se+1.e-20)**0.5*(1-((1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)**2
    K = K.full().ravel()
    return K


def CFun_np(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+abs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    dSedh=(parsDict[3]*pars[3])*(1-1/((parsDict[4]*pars[4])+1.e-20))/(1-(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)*(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))*(1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))
    C = Se*0.00001+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*dSedh
    C = C.full().ravel()
    return C


def thetaFun_np_unscaled(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+abs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    theta = ((parsDict[2]*pars[2])+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*Se)
    theta = theta.full().ravel()
    return theta


def hFun_np_unscaled(theta, pars, parsDict):  # Assume all theta are <= theta_s
    h = (((((theta - (parsDict[2]*pars[2])) / ((parsDict[1]*pars[1]) - (parsDict[2]*pars[2]) + 1.e-20) + 1.e-20) ** (1. / (-(1-1/((parsDict[4]*pars[4])+1.e-20)) + 1.e-20))
              - 1) + 1.e-20) ** (1. / ((parsDict[4]*pars[4]) + 1.e-20))) / (-(parsDict[3]*pars[3]) + 1.e-20)
    return h


# - h0 - np & unscaled --------------------------------------------------------
# Calculated the initial state
# def getH0_np(thetaIni, p, numberOfNodes):
#     hMatrix = hFun_np_unscaled(thetaIni, p)
#
#     # hMatrix = np.zeros(numberOfNodes)
#     # hMatrix[0:9] = psiIni[0]  # 1st section has 8 states
#     # hMatrix[9:16] = psiIni[1]  # After, each section has 7 states
#     # hMatrix[16:23] = psiIni[2]
#     # hMatrix[23:numberOfNodes] = psiIni[3]
#
#     return hMatrix


def getH0_np_multiLayer(thetaIni, pList, pScale, Nx, NxPerSoil, Nsoil, Np):
    hMatrix = np.zeros(Nx)
    for i in range(Nsoil):
        start = i*NxPerSoil
        end = (i+1)*NxPerSoil
        p = pList[i*Np:(i+1)*Np]

        hMatrix[start:end] = hFun_np_unscaled(thetaIni[start:end], pScale, p)
    return hMatrix


# # - ODE - np & unscaled --------------------------------------------------------
def getODE_np_3D_process(xk, u, qbot, w):
    spacePara_dict = space_parameters()
    Nxx = spacePara_dict['Nxx']
    Nxy = spacePara_dict['Nxy']
    Nxz = spacePara_dict['Nxz']
    Np = spacePara_dict['Np']
    NpTotal = spacePara_dict['NpTotal']
    Nsoil = spacePara_dict['Nsoil']
    NxPerSoil = spacePara_dict['NxPerSoil']
    dx = spacePara_dict['dx']
    dy = spacePara_dict['dy']
    dz = spacePara_dict['dz']

    _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[:-NpTotal]
    Nx = x.shape[0]
    x_reshape = np.reshape(x, (Nxx, Nxy, Nxz), order='F')

    q_inX = np.zeros((Nxx+1, Nxy, Nxz))
    q_inY = np.zeros((Nxx, Nxy+1, Nxz))
    q_inZ = np.zeros((Nxx, Nxy, Nxz+1))

    # Lower boundary is updated within each time step
    q_inZ[:, :, -1] = qbot

    # Top boundary is updated within each time step
    q_inZ[:, :, 0] = u

    for index in range(Nsoil):
        start = int(index * Nxz / Nsoil)
        end = int((index + 1) * Nxz / Nsoil)
        ParsScaled = xk[Nx+Np*index: Nx+Np*(index+1)]

        # Left boundary is updated within each time step
        KLeft = KFun_np(x_reshape[0, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        KLeft = KLeft.reshape((Nxy, int(Nxz/Nsoil)), order='F')
        q_inX[0,:,start:end] = -KLeft*0

        # Right boundary is updated within each time step
        KRight = KFun_np(x_reshape[-1, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        KRight = KRight.reshape((Nxy, int(Nxz/Nsoil)), order='F')
        q_inX[-1,:,start:end] = -KRight*0

        # Front boundary is updated within each time step
        KFront = KFun_np(x_reshape[:, 0, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        KFront = KFront.reshape((Nxx, int(Nxz/Nsoil)), order='F')
        q_inY[:,0,start:end] = -KFront*0

        # Back boundary is updated within each time step
        KBack = KFun_np(x_reshape[:, -1, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        KBack = KBack.reshape((Nxx, int(Nxz/Nsoil)), order='F')
        q_inY[:,-1,start:end] = -KBack*0

    C = np.zeros(Nx)
    Knodes = np.zeros(Nx)
    for index in range(Nsoil):
        start = index * NxPerSoil
        end = (index + 1) * NxPerSoil
        ParsScaled = xk[Nx+Np*index: Nx+Np*(index+1)]

        C[start:end] = CFun_np(x[start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        Knodes[start:end] = KFun_np(x[start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
    C_reshape = np.reshape(C, (Nxx, Nxy, Nxz), order='F')
    Knodes_reshape = np.reshape(Knodes, (Nxx, Nxy, Nxz), order='F')

    Kmid_inX = np.zeros((Nxx-1, Nxy, Nxz))
    Kmid_inY = np.zeros((Nxx, Nxy-1, Nxz))
    Kmid_inZ = np.zeros((Nxx, Nxy, Nxz-1))

    i = np.arange(0, Nxx-1)
    Kmid_inX[i,:,:] = (Knodes_reshape[i,:,:] + Knodes_reshape[i+1,:,:])/2
    j = np.arange(0, Nxy-1)
    Kmid_inY[:,j,:] = (Knodes_reshape[:,j,:] + Knodes_reshape[:,j+1,:])/2
    k = np.arange(0, Nxz-1)
    Kmid_inZ[:,:,k] = (Knodes_reshape[:,:,k] + Knodes_reshape[:,:,k+1])/2

    ii = np.arange(1, Nxx)
    q_inX[ii,:,:] = -Kmid_inX*((x_reshape[i,:,:]-x_reshape[i+1,:,:])/dx)
    jj = np.arange(1, Nxy)
    q_inY[:,jj,:] = -Kmid_inY*((x_reshape[:,j,:]-x_reshape[:,j+1,:])/dy)
    kk = np.arange(1, Nxz)
    q_inZ[:,:,kk] = -Kmid_inZ*((x_reshape[:,:,k]-x_reshape[:,:,k+1])/dz+1.)

    iii = np.arange(0, Nxx)
    jjj = np.arange(0, Nxy)
    kkk = np.arange(0, Nxz)
    dhdt = ((-(q_inX[iii,:,:]-q_inX[iii+1,:,:])/dx)+(-(q_inY[:,jjj,:]-q_inY[:,jjj+1,:])/dy)+(-(q_inZ[:,:,kkk]-q_inZ[:,:,kkk+1])/dz))/C_reshape
    dhdt = np.reshape(dhdt, (Nx), order='F')

    dhdt = np.append(dhdt, np.zeros(NpTotal)) + w
    return dhdt


# *This version of numpy model is compatible with Casadi***************************************************************
def getODE_np_3D_process_V2(xk, u, qleft, qright, w):
    spacePara_dict = space_parameters()
    Nxx = spacePara_dict['Nxx']
    Nxy = spacePara_dict['Nxy']
    Nxz = spacePara_dict['Nxz']
    Np = spacePara_dict['Np']
    NpTotal = spacePara_dict['NpTotal']
    Nsoil = spacePara_dict['Nsoil']
    NxPerSoil = spacePara_dict['NxPerSoil']
    dx = spacePara_dict['dx']
    dy = spacePara_dict['dy']
    dz = spacePara_dict['dz']

    _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[:-NpTotal]
    Nx = x.shape[0]
    x_reshape = np.reshape(x, (Nxz * Nxy, Nxx), order='F')

    q_inZ = np.zeros(((Nxz + 1) * Nxy, Nxx))
    q_inY = np.zeros((Nxz * (Nxy + 1), Nxx))
    q_inX = np.zeros((Nxz * Nxy, Nxx + 1))

    C = np.zeros(Nx)
    Knodes = np.zeros(Nx)

    # Right boundary is updated within each time step
    q_inX[:, -1] = qright
    # Left boundary is updated within each time step
    q_inX[:, 0] = qleft

    for index in range(Nsoil):
        startLayer = int(index * Nxx / Nsoil)
        endLayer = int((index + 1) * Nxx / Nsoil)

        startX = index * NxPerSoil
        endX = (index + 1) * NxPerSoil
        ParsScaled = xk[Nx + Np * index: Nx + Np * (index + 1)]

        # Top boundary is updated within each time step
        # KLeft = KFun_np(x_reshape[0:, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KLeft = KLeft.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_inZ[0:Nxy, startLayer:endLayer] = u  # 1#-KLeft*0

        # Bottom boundary is updated within each time step
        # KRight = KFun_np(x_reshape[-1, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KRight = KRight.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_bot = -KFun_np(x[startX+Nxz-1:endX:Nxz], ParsScaled, ParsNp[index*Np:(index+1)*Np])  # -1#-KRight*0
        q_inZ[-Nxy:, startLayer:endLayer] = np.reshape(q_bot, (Nxy, int(Nxx/Nsoil)), order="F")

        # Front boundary is updated within each time step
        # KFront = KFun_np(x_reshape[:, 0, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KFront = KFront.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[0:Nxz, startLayer:endLayer] = 0  # 1#-KFront*0

        # Back boundary is updated within each time step
        # KBack = KFun_np(x_reshape[:, -1, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KBack = KBack.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[-Nxz:, startLayer:endLayer] = 0  # -1#-KBack*0

        C[startX:endX] = CFun_np(x[startX:endX], ParsScaled, ParsNp[index * Np:(index + 1) * Np])
        Knodes[startX:endX] = KFun_np(x[startX:endX], ParsScaled, ParsNp[index * Np:(index + 1) * Np])

    C_reshape = np.reshape(C, (Nxz * Nxy, Nxx), order='F')
    Knodes_reshape = np.reshape(Knodes, (Nxz * Nxy, Nxx), order='F')

    Kmid_inZ = np.zeros(((Nxz - 1) * Nxy, Nxx))
    Kmid_inY = np.zeros((Nxz * (Nxy - 1), Nxx))
    Kmid_inX = np.zeros((Nxz * Nxy, Nxx - 1))

    i = np.arange(0, Nxz * Nxy, step=Nxz)
    for layer in range(Nxz - 1):
        Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] = (Knodes_reshape[i + layer, :] + Knodes_reshape[i + 1 + layer, :]) / 2
        q_inZ[(layer + 1) * Nxy:(layer + 2) * Nxy, :] = -Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] * ((x_reshape[i + layer, :] - x_reshape[i + 1 + layer, :]) / dz + 1.)
    j = np.arange(0, Nxz)
    for layer in range(Nxy - 1):
        Kmid_inY[j + layer * (Nxz), :] = (Knodes_reshape[j + layer * (Nxz), :] + Knodes_reshape[j + Nxz + layer * (Nxz),:]) / 2
        q_inY[(layer + 1) * Nxz:(layer + 2) * Nxz, :] = -Kmid_inY[layer * Nxz:(layer + 1) * Nxz, :] * ((x_reshape[j + layer * (Nxz), :] - x_reshape[j + Nxz + layer * (Nxz), :]) / dy)
    k = np.arange(0, Nxx - 1)
    Kmid_inX[:, k] = (Knodes_reshape[:, k] + Knodes_reshape[:, k + 1]) / 2
    kk = np.arange(1, Nxx)
    q_inX[:, kk] = -Kmid_inX * ((x_reshape[:, k] - x_reshape[:, k + 1]) / dx)

    dhdtZ = np.zeros((Nxz * Nxy, Nxx))
    dhdtY = np.zeros((Nxz * Nxy, Nxx))
    iii = np.arange(0, Nxz) * Nxy
    jjj = np.arange(0, Nxy) * Nxz
    kkk = np.arange(0, Nxx)
    for layer in range(Nxy):
        dhdtZ[layer * Nxz:(layer + 1) * Nxz, :] = -(q_inZ[iii + layer, :] - q_inZ[iii + Nxy + layer, :]) / dz
    for layer in range(Nxz):
        dhdtY[jjj + layer, :] = -(q_inY[jjj + layer, :] - q_inY[jjj + layer + Nxz, :]) / dy

    dhdt = (dhdtZ + dhdtY + (-(q_inX[:, kkk] - q_inX[:, kkk + 1]) / dx)) / C_reshape
    dhdt = np.reshape(dhdt, (Nx, 1), order='F')

    dhdt = np.append(dhdt, np.zeros(NpTotal)) + w
    return dhdt
# **********************************************************************************************************************


# - Measurement - np & unscaled --------------------------------------------------------
def getOutputs_np_aug(x):
    spacePara_dict = space_parameters()
    Nx = spacePara_dict['Nx']
    Ny = spacePara_dict['Ny']
    CMatrix = cmatrix_single(Nx, Ny)
    # Pars_unscaled = x[Nx:Nx_aug]
    # y = thetaFun_np_unscaled(x[:Nx], Pars_unscaled)
    y = x[:Nx]
    y_pred = np.matmul(CMatrix, y)
    return y_pred


# # - Subsystem np model for estimator -----------------------------------------------------------------------------------
def getODE_np_3D_subsys(xk, u, qleft, qright, w, p):
    spacePara_dict = space_parameters()
    Nxx = spacePara_dict['Nxx']
    Nxy = spacePara_dict['Nxy']
    Nxz = spacePara_dict['Nxz']
    Np = spacePara_dict['Np']
    NpTotal = spacePara_dict['NpTotal']
    Nsoil = spacePara_dict['Nsoil']
    NxPerSoil = spacePara_dict['NxPerSoil']
    Nest = spacePara_dict['Nest']
    NxPerEst = spacePara_dict['NxPerEst']
    dx = spacePara_dict['dx']
    dy = spacePara_dict['dy']
    dz = spacePara_dict['dz']

    # _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[0:NxPerEst]
    Nx = x.shape[0]
    x_reshape = np.reshape(x, (Nxz * Nxy, int(Nxx / Nest)), order='F')

    q_inZ = np.zeros(((Nxz + 1) * Nxy, int(Nxx / Nest)))
    q_inY = np.zeros((Nxz * (Nxy + 1), int(Nxx / Nest)))
    q_inX = np.zeros((Nxz * Nxy, int(Nxx / Nest) + 1))

    C = np.zeros(Nx)
    Knodes = np.zeros(Nx)

    # Lower boundary is updated within each time step
    q_inX[:, -1] = qright
    # Top boundary is updated within each time step
    q_inX[:, 0] = qleft

    for index in range(max(1, Nsoil // Nest)):
        startLayer = int(index * Nxx / Nsoil)
        endLayer = int((index + 1) * Nxx / Nsoil)

        startX = index * min(NxPerEst, NxPerSoil)
        endX = (index + 1) * min(NxPerEst, NxPerSoil)
        ParsScaled = xk[Nx + Np * index: Nx + Np * (index + 1)]

        # Left boundary is updated within each time step
        # KLeft = KFun_np(x_reshape[0:, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KLeft = KLeft.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_inZ[0:Nxy, startLayer:endLayer] = u  # 1#-KLeft*0

        # Right boundary is updated within each time step
        # KRight = KFun_np(x_reshape[-1, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KRight = KRight.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_bot = -KFun_np(x[startX + Nxz - 1:endX:Nxz], ParsScaled, p[index * Np:(index + 1) * Np])  # -1#-KRight*0
        q_inZ[-Nxy:, startLayer:endLayer] = np.reshape(q_bot, (Nxy, int(Nxx / Nsoil)), order='F')

        # Front boundary is updated within each time step
        # KFront = KFun_np(x_reshape[:, 0, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KFront = KFront.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[0:Nxz, startLayer:endLayer] = 0  # 1#-KFront*0

        # Back boundary is updated within each time step
        # KBack = KFun_np(x_reshape[:, -1, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KBack = KBack.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[-Nxz:, startLayer:endLayer] = 0  # -1#-KBack*0

        C[startX:endX] = CFun_np(x[startX:endX], ParsScaled, p[index * Np:(index + 1) * Np])
        Knodes[startX:endX] = KFun_np(x[startX:endX], ParsScaled, p[index * Np:(index + 1) * Np])

    C_reshape = np.reshape(C, (Nxz * Nxy, int(Nxx / Nest)), order='F')
    Knodes_reshape = np.reshape(Knodes, (Nxz * Nxy, int(Nxx / Nest)), order='F')

    Kmid_inZ = np.zeros(((Nxz - 1) * Nxy, int(Nxx / Nest)))
    Kmid_inY = np.zeros((Nxz * (Nxy - 1), int(Nxx / Nest)))
    Kmid_inX = np.zeros((Nxz * Nxy, int(Nxx / Nest) - 1))

    i = np.arange(0, Nxz * Nxy, step=Nxz)
    for layer in range(Nxz - 1):
        Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] = (Knodes_reshape[i + layer, :] + Knodes_reshape[i + 1 + layer, :]) / 2
        q_inZ[(layer + 1) * Nxy:(layer + 2) * Nxy, :] = -Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] * (
                (x_reshape[i + layer, :] - x_reshape[i + 1 + layer, :]) / dz + 1.)
    j = np.arange(0, Nxz)
    for layer in range(Nxy - 1):
        Kmid_inY[j + layer * Nxz, :] = (Knodes_reshape[j + layer * Nxz, :] + Knodes_reshape[j + Nxz + layer * Nxz, :]) / 2
        q_inY[(layer + 1) * Nxz:(layer + 2) * Nxz, :] = -Kmid_inY[layer * Nxz:(layer + 1) * Nxz, :] * (
                (x_reshape[j + layer * Nxz, :] - x_reshape[j + Nxz + layer * Nxz, :]) / dy)
    k = np.arange(0, int(Nxx / Nest) - 1)
    Kmid_inX[:, k] = (Knodes_reshape[:, k] + Knodes_reshape[:, k + 1]) / 2
    kk = np.arange(1, int(Nxx / Nest))
    q_inX[:, kk] = -Kmid_inX * ((x_reshape[:, k] - x_reshape[:, k + 1]) / dx)

    dhdtZ = np.zeros((Nxz * Nxy, int(Nxx / Nest)))
    dhdtY = np.zeros((Nxz * Nxy, int(Nxx / Nest)))
    iii = np.arange(0, Nxz) * Nxy
    jjj = np.arange(0, Nxy) * Nxz
    kkk = np.arange(0, int(Nxx / Nest))
    for layer in range(Nxy):
        dhdtZ[layer * Nxz:(layer + 1) * Nxz, :] = -(q_inZ[iii + layer, :] - q_inZ[iii + Nxy + layer, :]) / dz
    for layer in range(Nxz):
        dhdtY[jjj + layer, :] = -(q_inY[jjj + layer, :] - q_inY[jjj + layer + Nxz, :]) / dy

    dhdt = (dhdtZ + dhdtY + (-(q_inX[:, kkk] - q_inX[:, kkk + 1]) / dx)) / C_reshape
    dhdt = np.reshape(dhdt, (Nx, 1), order='F')

    dhdt = np.append(dhdt, np.zeros(Np * max(1, Nsoil // Nest))) + w
    return dhdt


def getOutputs_np_aug_subsys(x, index):
    spacePara_dict = space_parameters()
    Nx = spacePara_dict['Nx']
    Ny = spacePara_dict['Ny']
    NxPerEst = spacePara_dict['NxPerEst']
    NyPerEst = spacePara_dict['NyPerEst']
    CMatrix = cmatrix_single(Nx, Ny)
    CMatrix = CMatrix[index*NyPerEst:(index+1)*NyPerEst, index*NxPerEst:(index+1)*NxPerEst]
    # Pars_unscaled = x[-Np:]
    # y = thetaFun_np_unscaled(x[:-Np], Pars_unscaled)
    y = x[:NxPerEst]
    y_pred = np.matmul(CMatrix, y)
    return y_pred


# ----------------------------------------------------------------------------------------------------------------------
# Create MX & Unscaled Richards ODE model
# ----------------------------------------------------------------------------------------------------------------------
# - K, C, theta, h - casadi MX & unscaled --------------------------------------------------------
def mean_KFun_mx(psi1, psi2, pars, parsDict):
    K1 = KFun_mx(psi1, pars, parsDict)
    K2 = KFun_mx(psi2, pars, parsDict)
    K = (K1+K2)/2
    return K

def KFun_mx(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+MX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    K = parsDict[0]*pars[0]*(Se+1.e-20)**0.5*(1-((1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)**2
    # K = K.full().ravel()
    return K


def CFun_mx(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+MX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    dSedh=(parsDict[3]*pars[3])*(1-1/((parsDict[4]*pars[4])+1.e-20))/(1-(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)*(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))*(1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))
    C = Se*0.00001+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*dSedh
    # C = C.full().ravel()
    return C


def thetaFun_mx_unscaled(psi,pars,parsDict):
    Se = if_else(psi>=0., 1., (1+MX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
    theta = ((parsDict[2]*pars[2])+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*Se)
    # theta = theta.full().ravel()
    return theta


def hFun_mx_unscaled(theta, pars,parsDict):  # Assume all theta are <= theta_s
    h = (((((theta - (parsDict[2]*pars[2])) / ((parsDict[1]*pars[1]) - (parsDict[2]*pars[2]) + 1.e-20) + 1.e-20) ** (1. / (-(1-1/((parsDict[4]*pars[4])+1.e-20)) + 1.e-20))
              - 1) + 1.e-20) ** (1. / ((parsDict[4]*pars[4]) + 1.e-20))) / (-(parsDict[3]*pars[3]) + 1.e-20)
    return h


# - h0 - casadi MX & unscaled --------------------------------------------------------
# - ODE - casadi MX & unscaled --------------------------------------------------------
def getODE_mx_3D_process(xk, u, qleft, qright, w):
    spacePara_dict = space_parameters()
    Nxx = spacePara_dict['Nxx']
    Nxy = spacePara_dict['Nxy']
    Nxz = spacePara_dict['Nxz']
    Np = spacePara_dict['Np']
    NpTotal = spacePara_dict['NpTotal']
    Nsoil = spacePara_dict['Nsoil']
    NxPerSoil = spacePara_dict['NxPerSoil']
    dx = spacePara_dict['dx']
    dy = spacePara_dict['dy']
    dz = spacePara_dict['dz']

    _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[:-NpTotal]
    Nx = x.shape[0]
    x_reshape = MX.reshape(x, (Nxz*Nxy, Nxx))

    q_inZ = MX.zeros(((Nxz+1)*Nxy, Nxx))
    q_inY = MX.zeros((Nxz*(Nxy+1), Nxx))
    q_inX = MX.zeros((Nxz*Nxy, Nxx+1))

    C = MX.zeros(Nx)
    Knodes = MX.zeros(Nx)


    # Right boundary is updated within each time step
    q_inX[:, -1] = qright
    # Left boundary is updated within each time step
    q_inX[:, 0] = qleft

    for index in range(Nsoil):
        startLayer = int(index * Nxx / Nsoil)
        endLayer = int((index + 1) * Nxx / Nsoil)

        startX = index * NxPerSoil
        endX = (index + 1) * NxPerSoil
        ParsScaled = xk[Nx+Np*index: Nx+Np*(index+1)]

        # Top boundary is updated within each time step
        # KLeft = KFun_mx(x_reshape[0:, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KLeft = KLeft.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_inZ[0:Nxy,startLayer:endLayer] = u#1#-KLeft*0

        # Bottom boundary is updated within each time step
        # KRight = KFun_mx(x_reshape[-1, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KRight = KRight.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_bot = -KFun_mx(x[startX+Nxz-1:endX:Nxz], ParsScaled, ParsNp[index*Np:(index+1)*Np])  # -1#-KRight*0
        q_inZ[-Nxy:, startLayer:endLayer] = MX.reshape(q_bot, (Nxy, int(Nxx/Nsoil)))

        # Front boundary is updated within each time step
        # KFront = KFun_mx(x_reshape[:, 0, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KFront = KFront.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[0:Nxz,startLayer:endLayer] = 0#1#-KFront*0

        # Back boundary is updated within each time step
        # KBack = KFun_mx(x_reshape[:, -1, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KBack = KBack.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[-Nxz:,startLayer:endLayer] = 0#-1#-KBack*0

        C[startX:endX] = CFun_mx(x[startX:endX], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        Knodes[startX:endX] = KFun_mx(x[startX:endX], ParsScaled, ParsNp[index*Np:(index+1)*Np])

    C_reshape = MX.reshape(C, (Nxz*Nxy, Nxx))
    Knodes_reshape = MX.reshape(Knodes, (Nxz*Nxy, Nxx))

    Kmid_inZ = MX.zeros(((Nxz-1)*Nxy, Nxx))
    Kmid_inY = MX.zeros((Nxz*(Nxy-1), Nxx))
    Kmid_inX = MX.zeros((Nxz*Nxy, Nxx-1))

    i = np.arange(0, Nxz * Nxy, step=Nxz)
    for layer in range(Nxz-1):
        Kmid_inZ[layer*Nxy:(layer+1)*Nxy,:] = (Knodes_reshape[i+layer,:] + Knodes_reshape[i+1+layer,:])/2
        q_inZ[(layer+1)*Nxy:(layer+2)*Nxy,:] = -Kmid_inZ[layer*Nxy:(layer+1)*Nxy,:]*((x_reshape[i+layer,:]-x_reshape[i+1+layer,:])/dz + 1.)
    j = np.arange(0, Nxz)
    for layer in range(Nxy-1):
        Kmid_inY[j+layer*(Nxz),:] = (Knodes_reshape[j+layer*(Nxz),:] + Knodes_reshape[j+Nxz+layer*(Nxz),:])/2
        q_inY[(layer+1)*Nxz:(layer+2)*Nxz,:] = -Kmid_inY[layer*Nxz:(layer+1)*Nxz,:]*((x_reshape[j+layer*(Nxz),:]-x_reshape[j+Nxz+layer*(Nxz),:])/dy)
    k = np.arange(0, Nxx-1)
    Kmid_inX[:,k] = (Knodes_reshape[:,k] + Knodes_reshape[:,k+1])/2
    kk = np.arange(1, Nxx)
    q_inX[:,kk] = -Kmid_inX*((x_reshape[:,k]-x_reshape[:,k+1])/dx)

    dhdtZ = MX.zeros((Nxz*Nxy, Nxx))
    dhdtY = MX.zeros((Nxz*Nxy, Nxx))
    iii = np.arange(0, Nxz)*Nxy
    jjj = np.arange(0, Nxy)*Nxz
    kkk = np.arange(0, Nxx)
    for layer in range(Nxy):
        dhdtZ[layer*Nxz:(layer+1)*Nxz,:] = -(q_inZ[iii+layer,:]-q_inZ[iii+Nxy+layer,:])/dz
    for layer in range(Nxz):
        dhdtY[jjj+layer,:] = -(q_inY[jjj+layer,:]-q_inY[jjj+layer+Nxz,:])/dy

    dhdt = (dhdtZ+dhdtY+(-(q_inX[:,kkk]-q_inX[:,kkk+1])/dx))/C_reshape
    dhdt = MX.reshape(dhdt, (Nx, 1))

    dhdt = vertcat(dhdt, MX.zeros(NpTotal)) + w
    return dhdt


def getODE_mx_3D_subsys_estimator(xk, u, qleft, qright, w, p):
    spacePara_dict = space_parameters()
    Nxx = spacePara_dict['Nxx']
    Nxy = spacePara_dict['Nxy']
    Nxz = spacePara_dict['Nxz']
    Np = spacePara_dict['Np']
    NpTotal = spacePara_dict['NpTotal']
    Nsoil = spacePara_dict['Nsoil']
    NxPerSoil = spacePara_dict['NxPerSoil']
    Nest = spacePara_dict['Nest']
    NxPerEst = spacePara_dict['NxPerEst']
    dx = spacePara_dict['dx']
    dy = spacePara_dict['dy']
    dz = spacePara_dict['dz']

    # _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[0:NxPerEst]
    Nx = x.shape[0]
    x_reshape = MX.reshape(x, (Nxz * Nxy, int(Nxx/Nest)))

    q_inZ = MX.zeros(((Nxz + 1) * Nxy, int(Nxx/Nest)))
    q_inY = MX.zeros((Nxz * (Nxy + 1), int(Nxx/Nest)))
    q_inX = MX.zeros((Nxz * Nxy, int(Nxx/Nest) + 1))

    C = MX.zeros(Nx)
    Knodes = MX.zeros(Nx)

    # Lower boundary is updated within each time step
    q_inX[:, -1] = qright
    # Top boundary is updated within each time step
    q_inX[:, 0] = qleft

    for index in range(max(1, Nsoil//Nest)):
        startLayer = int(index * Nxx/Nsoil)
        endLayer = int((index + 1) * Nxx/Nsoil)

        startX = index * min(NxPerEst, NxPerSoil)
        endX = (index + 1) * min(NxPerEst, NxPerSoil)
        ParsScaled = xk[Nx + Np * index: Nx + Np * (index + 1)]

        # Left boundary is updated within each time step
        # KLeft = KFun_mx(x_reshape[0:, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KLeft = KLeft.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_inZ[0:Nxy, startLayer:endLayer] = u  # 1#-KLeft*0

        # Right boundary is updated within each time step
        # KRight = KFun_mx(x_reshape[-1, :, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KRight = KRight.reshape((Nxy, int(Nxx/Nsoil)), order='F')
        q_bot = -KFun_mx(x[startX+Nxz-1:endX:Nxz], ParsScaled, p[index*Np:(index+1)*Np])  # -1#-KRight*0
        q_inZ[-Nxy:, startLayer:endLayer] = MX.reshape(q_bot, (Nxy, int(Nxx/Nsoil)))

        # Front boundary is updated within each time step
        # KFront = KFun_mx(x_reshape[:, 0, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KFront = KFront.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[0:Nxz, startLayer:endLayer] = 0  # 1#-KFront*0

        # Back boundary is updated within each time step
        # KBack = KFun_mx(x_reshape[:, -1, start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
        # KBack = KBack.reshape((Nxz, int(Nxx/Nsoil)), order='F')
        q_inY[-Nxz:, startLayer:endLayer] = 0  # -1#-KBack*0

        C[startX:endX] = CFun_mx(x[startX:endX], ParsScaled, p[index * Np:(index + 1) * Np])
        Knodes[startX:endX] = KFun_mx(x[startX:endX], ParsScaled, p[index * Np:(index + 1) * Np])

    C_reshape = MX.reshape(C, (Nxz * Nxy, int(Nxx/Nest)))
    Knodes_reshape = MX.reshape(Knodes, (Nxz * Nxy, int(Nxx/Nest)))

    Kmid_inZ = MX.zeros(((Nxz - 1) * Nxy, int(Nxx/Nest)))
    Kmid_inY = MX.zeros((Nxz * (Nxy - 1), int(Nxx/Nest)))
    Kmid_inX = MX.zeros((Nxz * Nxy, int(Nxx/Nest) - 1))

    i = np.arange(0, Nxz * Nxy, step=Nxz)
    for layer in range(Nxz - 1):
        Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] = (Knodes_reshape[i + layer, :] + Knodes_reshape[i + 1 + layer, :]) / 2
        q_inZ[(layer + 1) * Nxy:(layer + 2) * Nxy, :] = -Kmid_inZ[layer * Nxy:(layer + 1) * Nxy, :] * (
                    (x_reshape[i + layer, :] - x_reshape[i + 1 + layer, :]) / dz + 1.)
    j = np.arange(0, Nxz)
    for layer in range(Nxy - 1):
        Kmid_inY[j + layer * Nxz, :] = (Knodes_reshape[j + layer * Nxz, :] + Knodes_reshape[j + Nxz + layer * Nxz, :]) / 2
        q_inY[(layer + 1) * Nxz:(layer + 2) * Nxz, :] = -Kmid_inY[layer * Nxz:(layer + 1) * Nxz, :] * (
                    (x_reshape[j + layer * Nxz, :] - x_reshape[j + Nxz + layer * Nxz, :]) / dy)
    k = np.arange(0, int(Nxx/Nest) - 1)
    Kmid_inX[:, k] = (Knodes_reshape[:, k] + Knodes_reshape[:, k + 1]) / 2
    kk = np.arange(1, int(Nxx/Nest))
    q_inX[:, kk] = -Kmid_inX * ((x_reshape[:, k] - x_reshape[:, k + 1]) / dx)

    dhdtZ = MX.zeros((Nxz * Nxy, int(Nxx/Nest)))
    dhdtY = MX.zeros((Nxz * Nxy, int(Nxx/Nest)))
    iii = np.arange(0, Nxz) * Nxy
    jjj = np.arange(0, Nxy) * Nxz
    kkk = np.arange(0, int(Nxx/Nest))
    for layer in range(Nxy):
        dhdtZ[layer * Nxz:(layer + 1) * Nxz, :] = -(q_inZ[iii + layer, :] - q_inZ[iii + Nxy + layer, :]) / dz
    for layer in range(Nxz):
        dhdtY[jjj + layer, :] = -(q_inY[jjj + layer, :] - q_inY[jjj + layer + Nxz, :]) / dy

    dhdt = (dhdtZ + dhdtY + (-(q_inX[:, kkk] - q_inX[:, kkk + 1]) / dx)) / C_reshape
    dhdt = MX.reshape(dhdt, (Nx, 1))

    dhdt = vertcat(dhdt, MX.zeros(Np*max(1,Nsoil//Nest))) + w
    return dhdt


# - Measurement - casadi MX & unscaled --------------------------------------------------------
def getOutputs_mx_aug(x):
    spacePara_dict = space_parameters()
    Nx = spacePara_dict['Nx']
    Ny = spacePara_dict['Ny']
    NxPerEst = spacePara_dict['NxPerEst']
    NyPerEst = spacePara_dict['NyPerEst']
    CMatrix = cmatrix_single(Nx, Ny)
    # Pars_unscaled = x[Nx:Nx_aug]
    # y = thetaFun_mx_unscaled(x[:Nx], Pars_unscaled)
    y = x[:Nx]
    y_pred = casadi.mtimes(CMatrix, y)
    return y_pred


def getOutputs_mx_aug_subsys(x, index):
    spacePara_dict = space_parameters()
    Nx = spacePara_dict['Nx']
    Ny = spacePara_dict['Ny']
    NxPerEst = spacePara_dict['NxPerEst']
    NyPerEst = spacePara_dict['NyPerEst']
    CMatrix = cmatrix_single(Nx, Ny)
    CMatrix = CMatrix[index*NyPerEst:(index+1)*NyPerEst, index*NxPerEst:(index+1)*NxPerEst]
    # Pars_unscaled = x[-Np:]
    # y = thetaFun_mx_unscaled(x[:-Np], Pars_unscaled)
    y = x[:NxPerEst]
    y_pred = casadi.mtimes(CMatrix, y)
    return y_pred


'''
def getODE_multiLayer_process(xk, u, qbot, w):
    spacePara_dict = space_parameters()
    Nsoil = spacePara_dict['Nsoil']
    _, ParsNp, _ = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[:-NpTotal]  # Assume the process will always be centralized
    Nx = x.shape[0]

    q = MX.zeros(Nx + 1, 1)
    # Bottom boundary
    # KBot = KFun_mx(psiBot, ParsScaled)
    # q[-1] = - KBot * ((x[-1] - psiBot) / dz * 2 + 1.0)
    q[-1] = qbot
    # Top boundary
    q[0] = u

    C = MX.zeros(Nx, 1)
    for index in range(Nsoil):
        start = index * NxPerSoil
        end = (index + 1) * NxPerSoil

        ParsScaled = xk[Nx+Np*index: Nx+Np*(index+1)]

        C[start:end] = CFun_mx(x[start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])

    # C = CFun_mx(x, ParsScaled, ParsDict)

    Knodes = MX.zeros(Nx, 1)
    for index in range(Nsoil):
        start = index * NxPerSoil
        end = (index + 1) * NxPerSoil
        ParsScaled = xk[Nx+Np*index: Nx+Np*(index+1)]
        Knodes[start:end] = KFun_mx(x[start:end], ParsScaled, ParsNp[index*Np:(index+1)*Np])
    # Knodes = KFun_mx(x, ParsScaled, ParsDict)
    i = np.arange(0, Nx - 1)
    Kmid = (Knodes[i] + Knodes[i + 1]) / 2

    j = np.arange(1, Nx)
    q[j] = - Kmid * ((x[i] - x[i + 1]) / dz + 1.)

    i = np.arange(0, Nx)
    dhdt = (-(q[i] - q[i + 1]) / dz) / C
    dhdt = vertcat(dhdt, MX.zeros(NpTotal)) + w  # Assume the process is always centralized
    return dhdt


# - Subsystem MX model for estimator -----------------------------------------------------------------------------------
def getODE_multiLayer_subsys_estimator(xk, u, qbot, w, p):
    spacePara_dict = space_parameters()
    # _, ParsNp, _ = model_parameters_multiLayer(Np, Nsoil)  # soil/model parameters: This are the true parameters

    x = xk[0:NxPerEst]
    Nx = x.shape[0]

    q = MX.zeros(NxPerEst + 1, 1)
    # Bottom boundary
    q[-1] = qbot
    # Top boundary
    q[0] = u

    C = MX.zeros(Nx, 1)
    for index in range(max(1, Nsoil//Nest)):
        start = index * min(NxPerEst, NxPerSoil)
        end = (index + 1) * min(NxPerEst, NxPerSoil)
        ParsScaled = xk[NxPerEst + Np * index: NxPerEst + Np * (index + 1)]
        C[start:end] = CFun_mx(x[start:end], ParsScaled, p[index*Np:(index+1)*Np])

    # C = CFun_mx(x, Pars_unscaled, pdict)

    Knodes = MX.zeros(Nx, 1)
    for index in range(max(1, Nsoil//Nest)):
        start = index * min(NxPerEst, NxPerSoil)
        end = (index + 1) * min(NxPerEst, NxPerSoil)
        ParsScaled = xk[NxPerEst+Np*index: NxPerEst+Np*(index+1)]
        Knodes[start:end] = KFun_mx(x[start:end], ParsScaled, p[index*Np:(index+1)*Np])
    # Knodes = KFun_mx(x, Pars_unscaled, pdict)
    i = np.arange(0, NxPerEst - 1)
    Kmid = (Knodes[i] + Knodes[i + 1]) / 2

    j = np.arange(1, NxPerEst)
    q[j] = - Kmid * ((x[i] - x[i + 1]) / dz + 1.)

    i = np.arange(0, NxPerEst)
    dhdt = (-(q[i] - q[i + 1]) / dz) / C
    dhdt = vertcat(dhdt, MX.zeros(Np*max(1,Nsoil//Nest))) + w
    return dhdt
'''


'''
# ----------------------------------------------------------------------------------------------------------------------
# Create casadi SX & Unscaled Richards ODE model
# ----------------------------------------------------------------------------------------------------------------------
# - K, C, theta, h - casadi & unscaled --------------------------------------------------------
# def KFun_sx_unscaled_aug(psi,pars):
#     Se = if_else(psi>=0., 1., (1+SX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
#     K = parsDict[0]*pars[0]*(Se+1.e-20)**0.5*(1-((1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)**2
#     # K = K.full().ravel()
#     return K
#
#
# def CFun_sx_unscaled(psi,pars):
#     Se = if_else(psi>=0., 1., (1+SX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
#     dSedh=(parsDict[3]*pars[3])*(1-1/((parsDict[4]*pars[4])+1.e-20))/(1-(1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20)*(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))*(1-(Se+1.e-20)**(1/((1-1/((parsDict[4]*pars[4])+1.e-20))+1.e-20))+1.e-20)**(1-1/((parsDict[4]*pars[4])+1.e-20))
#     C = Se*0.00001+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*dSedh
#     # C = C.full().ravel()
#     return C
#
#
# def thetaFun_sx_unscaled(psi,pars):
#     Se = if_else(psi>=0., 1., (1+SX.fabs(psi*(parsDict[3]*pars[3])+1.e-20)**(parsDict[4]*pars[4])+1.e-20)**(-(1-1/((parsDict[4]*pars[4])+1.e-20))))
#     theta = ((parsDict[2]*pars[2])+((parsDict[1]*pars[1])-(parsDict[2]*pars[2]))*Se)
#     # theta = theta.full().ravel()
#     return theta
#
#
# def hFun_sx_unscaled(theta, pars):  # Assume all theta are <= theta_s
#     h = (((((theta - (parsDict[2]*pars[2])) / ((parsDict[1]*pars[1]) - (parsDict[2]*pars[2]) + 1.e-20) + 1.e-20) ** (1. / (-(1-1/((parsDict[4]*pars[4])+1.e-20)) + 1.e-20))
#               - 1) + 1.e-20) ** (1. / ((parsDict[4]*pars[4]) + 1.e-20))) / (-(parsDict[3]*pars[3]) + 1.e-20)
#     return h
#
#
# # - h0 - casadi & unscaled --------------------------------------------------------
# # Calculated the initial state
# def getH0_sx_unscaled(thetaIni, p, numberOfNodes):
#     psiIni = hFun_sx_unscaled(thetaIni, p)
#
#     hMatrix = SX.zeros(numberOfNodes)
#     hMatrix[0:9] = psiIni[0]  # 1st section has 8 states
#     hMatrix[9:16] = psiIni[1]  # After, each section has 7 states
#     hMatrix[16:23] = psiIni[2]
#     hMatrix[23:numberOfNodes] = psiIni[3]
#
#     return hMatrix, psiIni
#
#
# # - ODE - casadi & unscaled --------------------------------------------------------
# def getODE_sx_unscaled_aug(x,u,w):
#     Nz, Nx, Nw, Ny, Nv, Nu, Np, dz, Dim, Nx_aug, Nw_aug, Nsigma = space_parameters()
#     Pars_unscaled = x[Nx:Nx_aug]
#     x = x[0:Nx]
#     psiBot = x[-1]
#
#     q = SX.zeros(Nx + 1, 1)
#     # Bottom boundary
#     KBot = KFun_sx_unscaled_aug(psiBot, Pars_unscaled)
#     q[-1] = - KBot * ((x[-1] - psiBot) / dz * 2 + 1.0)
#     # Top boundary
#     q[0] = u
#
#     C = CFun_sx_unscaled(SX(x), Pars_unscaled)
#
#     i = np.arange(0, Nx - 1)
#     Knodes = KFun_sx_unscaled_aug(SX(x), Pars_unscaled)
#     Kmid = (Knodes[i] + Knodes[i + 1]) / 2
#
#     j = np.arange(1, Nx)
#     q[j] = - Kmid * ((x[i] - x[i + 1]) / dz + 1.)
#
#     i = np.arange(0, Nx)
#     dhdt = (-(q[i] - q[i + 1]) / dz) / C
#     dhdt = vertcat(dhdt, SX.zeros(Np)) + SX(w)
#     return dhdt
#
#
# # - Measurement - casadi & unscaled --------------------------------------------------------
# def getOutputs_sx_unscaled_aug(x):
#     Nz, Nx, Nw, Ny, Nv, Nu, Np, dz, Dim, Nx_aug, Nw_aug, Nsigma = space_parameters()
#     CMatrix = cmatrix_single(Nx, Ny, Dim)
#     # Pars_unscaled = x[Nx:Nx_aug]
#     # y = thetaFun_sx_unscaled(SX(x[:Nx]), Pars_unscaled)
#     y = x[:Nx]
#     y_pred = casadi.mtimes(CMatrix, y)
#     return y_pred
'''


if __name__ == '__main__':
    Pars, Pars_unscaled, _ = model_parameters(5)

    h0 = -0.514
    h1 = -0.45
    h2 = -0.65
    y0 = thetaFun_np_unscaled(h0, Pars_unscaled)
    y1 = thetaFun_np_unscaled(h1, Pars_unscaled)
    y2 = thetaFun_np_unscaled(h2, Pars_unscaled)

    print(y0, y1, y2)
