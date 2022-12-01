from __future__ import (print_function, division)  # Grab some handy Python3 stuff.
import numpy as np


def Loam():
    pars = {}
    pars['thetaR'] = 0.078
    pars['thetaS'] = 0.43
    pars['alpha'] = 0.036*100
    pars['n'] = 1.56
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.04/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def SandyClayLoam():
    pars = {}
    pars['thetaR'] = 0.100
    pars['thetaS'] = 0.39
    pars['alpha'] = 0.059*100
    pars['n'] = 1.48
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 1.31/100/3600
    pars['neta'] = 0.5
    pars['Ss'] = 0.00001
    pars['mini'] = 1.e-20
    return pars


def model_parameters_multiLayer(Nsoil, Np=5):
    parsDict1 = Loam()
    parsDict2 = SandyClayLoam()
    # parsDict3 = SandyClayLoam()
    # parsDict4 = SandyClayLoam()

    if Np == 5:
        parsNP1 = np.array([parsDict1['Ks'], parsDict1['thetaS'], parsDict1['thetaR'], parsDict1['alpha'], parsDict1['n']])  # Ks, thetaS, thetaR, alpha, n
        parsNP2 = np.array([parsDict2['Ks'], parsDict2['thetaS'], parsDict2['thetaR'], parsDict2['alpha'], parsDict2['n']])  # Ks, thetaS, thetaR, alpha, n
        # parsNP3 = np.array([parsDict3['Ks'], parsDict3['thetaS'], parsDict3['thetaR'], parsDict3['alpha'], parsDict3['n']])  # Ks, thetaS, thetaR, alpha, n
        # parsNP4 = np.array([parsDict4['Ks'], parsDict4['thetaS'], parsDict4['thetaR'], parsDict4['alpha'], parsDict4['n']])  # Ks, thetaS, thetaR, alpha, n

    elif Np == 4:
        parsNP1 = np.array([parsDict1['Ks'], parsDict1['thetaS'], parsDict1['alpha'], parsDict1['n']])  # Ks, thetaS, thetaR, alpha, n
        parsNP2 = np.array([parsDict2['Ks'], parsDict2['thetaS'], parsDict2['alpha'], parsDict2['n']])  # Ks, thetaS, thetaR, alpha, n
        # parsNP3 = np.array([parsDict3['Ks'], parsDict3['thetaS'], parsDict3['alpha'], parsDict3['n']])  # Ks, thetaS, thetaR, alpha, n

    elif Np == 3:
        parsNP1 = np.array([parsDict1['Ks'], parsDict1['thetaS'], parsDict1['n']])  # Ks, thetaS, thetaR, alpha, n
        parsNP2 = np.array([parsDict2['Ks'], parsDict2['thetaS'], parsDict2['n']])  # Ks, thetaS, thetaR, alpha, n
        # parsNP3 = np.array([parsDict3['Ks'], parsDict3['thetaS'], parsDict3['n']])  # Ks, thetaS, thetaR, alpha, n
    else:
        print('In def model_parameters_multiLayer, the Np is wrong')
        quit()
    pars_sc = np.ones(Np)
    if Nsoil == 1:
        return [parsDict1], parsNP1, pars_sc
    elif Nsoil == 2:
        return [parsDict1, parsDict2], np.append(parsNP1, parsNP2), pars_sc
    # elif Nsoil == 4:
    #     return [parsDict1, parsDict2, parsDict3, parsDict4], np.append(np.append(parsNP1, parsNP2), np.append(parsNP3, parsNP4)), pars_sc
    # elif Nsoil == 3:
        # return [parsDict1, parsDict2, parsDict3], np.append(parsNP1, parsNP2, parsNP3), pars_sc


def initial_condition(Nx):
    y0 = np.array([0.30, 0.30, 0.30, 0.30])  # Assume these 4 Ys are measured from the sensor at t0
    y0 = np.ones(Nx)*0.30
    # y0 = np.array([0.20, 0.20, 0.20, 0.20])
    # y0 = np.array([0.15, 0.15, 0.15, 0.15])
    return y0


def time_parameters():
    DeltaT = 1*60  # Time step [sec]
    Tsim = 5*24*60*(DeltaT)  # Simulation time: 5 days
    Nsim = int(Tsim/DeltaT)  # Simulation nodes
    DeltaT_internal = 1*60  # sec  # smaller dt for FD model - for integration
    Nsim_internal = int(DeltaT/DeltaT_internal)
    Tplot = np.arange(Nsim + 1) * DeltaT  # used for plot
    Nmhe = 8 # size of moving window
    DeltaTmhe = DeltaT  # how often MHE/EnKF updates/estimates
    calNode = DeltaTmhe // DeltaT
    return DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode


def space_parameters(Np=5):
    ratio = 1
    spacePara_dict = {}
    spacePara_dict['Nz'] = 2  # Number of algebraic states
    spacePara_dict['Nxz'] = int(16*ratio) #32
    spacePara_dict['Nxx'] = 6
    spacePara_dict['Nxy'] = 1
    spacePara_dict['Nx'] = int(spacePara_dict['Nxx']*spacePara_dict['Nxy']*spacePara_dict['Nxz'])  # Number of differential states

    spacePara_dict['Nsoil'] = 2
    spacePara_dict['NxPerSoil'] = int(spacePara_dict['Nx']/spacePara_dict['Nsoil'])

    spacePara_dict['Nw'] = spacePara_dict['Nx']  # Number of process/model noise
    spacePara_dict['Np'] = Np
    spacePara_dict['NpTotal'] = spacePara_dict['Np']*spacePara_dict['Nsoil']  # Number of parameters to estimate (4 parameters are going to be estimated)
    spacePara_dict['Nx_aug'] = spacePara_dict['Nx']+spacePara_dict['NpTotal']  # Number of augmented states
    spacePara_dict['Nw_aug'] = spacePara_dict['Nx_aug']
    spacePara_dict['Ny'] = 4*2#*spacePara_dict['Nxx']*spacePara_dict['Nxy']#spacePara_dict['Nx']  # Number of measurements#
    spacePara_dict['Nv'] = spacePara_dict['Ny']  # Number of meassurements noise
    spacePara_dict['Nu'] = 1# spacePara_dict['Nxx']*spacePara_dict['Nxy']  # Number of inputs
    spacePara_dict['Ninfo'] = spacePara_dict['Nxz']*spacePara_dict['Nxy']

    spacePara_dict['H_soil'] = 0.335*ratio  # meter # Depth of soil
    spacePara_dict['dz'] = spacePara_dict['H_soil'] / spacePara_dict['Nxz']
    spacePara_dict['Lx_soil'] = 0.18
    spacePara_dict['Ly_soil'] = 0.03
    spacePara_dict['dx'] = spacePara_dict['Lx_soil'] / spacePara_dict['Nxx']
    spacePara_dict['dy'] = spacePara_dict['Ly_soil'] / spacePara_dict['Nxy']

    spacePara_dict['Dim'] = 3  # dimension of the model

    spacePara_dict['Nsigma'] = 100  # Number of sigma points used for EnKF

    spacePara_dict['NestX'] = 2  # Number of estimators in X direction
    spacePara_dict['NestY'] = 1
    spacePara_dict['NestZ'] = 1
    spacePara_dict['Nest'] = spacePara_dict['NestX']*spacePara_dict['NestY']*spacePara_dict['NestZ']
    spacePara_dict['NxPerEst'] = int(spacePara_dict['Nx']/spacePara_dict['Nest'])
    spacePara_dict['NyPerEst'] = int(spacePara_dict['Ny']/spacePara_dict['Nest'])
    return spacePara_dict
    # return Nz, Nx, Nw, Ny, Nv, Nu, Np, NpTotal, dz, Dim, Nx_aug, Nw_aug, Nsigma, Nsoil, NxPerSoil, Nxx, Nxy, Nxz, dx, dy

# def positionOfNodes(Nxx, Nxy, Nxz, start=0):
#     coordinate = []
#     for i in range(Nxx):
#         for j in range(Nxy):
#             for k in range(Nxz):
#                 coordinate.append([i+start,j,k])
#     return coordinate

def variance(Nx, Nx_aug, Nw_aug, Nv):
    tol_desired = 1.e-8  # the default value of ipopt is e-8

    sigma_P_para = 1.e3  # Standard deviation for prior
    sigma_P_x = 1.e2

    sigma_Q_para = 1.e-20  # Standard deviation for the process noise
    sigma_Q_x = 1.e-1

    noise_w = 3.e-6  # 3.e-6 # 0.0036
    noise_v = 8.e-3  # 0.01

    sigma_R = sigma_Q_x * (noise_v / (noise_w))
    P0 = np.diag((sigma_P_para*np.ones((Nx_aug,)))**2)    # Covariance for prior.
    for i in range(0,Nx):
        P0[i,i]=sigma_P_x**2
    Q = np.diag((sigma_Q_para * np.ones((Nw_aug,))) ** 2)
    for i in range(0, Nx):
        Q[i, i] = sigma_Q_x ** 2
    R = np.diag((sigma_R * np.ones((Nv,))) ** 2)
    #
    # P0[-1,-1]=1.e2**2
    # P0[-6,-6]=1.e2**2
    return P0, Q, R, noise_w, noise_v, tol_desired

def bounds_para(Nx_aug, NpTotal, Np, Nsoil, Nest):  # bounds for decision variables (x and w) used in MHE
    # if Np == 5:
    #     from Model_1D_ODE_aug import hFun_np_unscaled  # either from augmented model or original model. Does not matter.
    # elif Np == 4:
    #     from Model_1D_ODE_aug_FourPara import hFun_np_unscaled  # either from augmented model or original model. Does not matter.
    # elif Np == 3:
    #     from ThreePara_Model_1D_ODE_aug import hFun_np_unscaled  # either from augmented model or original model. Does not matter.
    _, _, parsScale = model_parameters_multiLayer(Nsoil)  # soil/model parameters: This are the true parameters
    #
    # ymax = 0.43  # reasonable max and min of soil moistures (given Y0 and u)
    # ymin = 0.078
    # y0 = initial_condition()[0]
    # xmax = hFun_np_unscaled(ymax, Pars_np)
    # xmin = hFun_np_unscaled(ymin, Pars_np)
    # x0 = hFun_np_unscaled(y0, Pars_np)

    lowRatio = 0.8  # ratio used for deviating parameters
    upRatio = 1.2

    # bounds for states x
    # augmented model - state and parameter estimation
    if Np == 5:
        plower = lowRatio*np.array([parsScale[0]/lowRatio, parsScale[1]/lowRatio, parsScale[2]/lowRatio, parsScale[3]/lowRatio, parsScale[4]/lowRatio])
        pupper = upRatio*np.array([parsScale[0]/upRatio, parsScale[1]/upRatio, parsScale[2]/upRatio, parsScale[3]/upRatio, parsScale[4]/upRatio])
    elif Np == 4:
        plower = lowRatio*np.array([parsScale[0], parsScale[1], parsScale[2], parsScale[3]])
        pupper = upRatio*np.array([parsScale[0], parsScale[1], parsScale[2], parsScale[3]])
    elif Np == 3:
        plower = lowRatio * np.array([parsScale[0], parsScale[1], parsScale[2]])
        pupper = upRatio * np.array([parsScale[0], parsScale[1], parsScale[2]])

    xlower = np.append(-1 * np.ones(Nx_aug), plower)
    xupper = np.append(-0.0001 * np.ones(Nx_aug), pupper)
    xlower_fie = np.append(-1 * np.ones(Nx_aug), plower)
    xupper_fie = np.append(-0.0001 * np.ones(Nx_aug), pupper)
    wlower = np.append(-np.inf* np.ones(Nx_aug), 0*np.ones(Np))
    wupper = np.append(np.inf * np.ones(Nx_aug), 0*np.ones(Np))
    for i in range(Nsoil//Nest-1):
        xlower = np.append(xlower, plower)
        xupper = np.append(xupper, pupper)
        xlower_fie = np.append(xlower_fie, plower)
        xupper_fie = np.append(xupper_fie, pupper)
        wlower = np.append(wlower, 0*np.ones(Np))
        wupper = np.append(wupper, 0*np.ones(Np))

    # xlower = np.append(-1*np.inf * np.ones(Nx-Np), plower)
    # xupper = np.append(0 * np.ones(Nx-Np), pupper)
    # xlower = np.append(xmin * np.ones(Nx-Np), plower)
    # xupper = np.append(xmax * np.ones(Nx-Np), pupper)

    # bounds for process noise w
    # wlower = -np.inf* np.ones(Nx_aug)
    # wupper = np.inf * np.ones(Nx_aug)
    # wlower = -1.e-13 * np.ones(Nx)
    # wupper = 1.e-13*np.ones(Nx)
    return xlower, xupper, wlower, wupper, xlower_fie, xupper_fie


def cmatrix_single(Nx, Ny):
    # CMatrix = np.zeros((Ny, Nx))
    # for i in range(0, Ny):
    #     CMatrix[i,i] = 1
    # CMatrix[0,4] = 1.
    # CMatrix[1,12] = 1.
    # CMatrix[2,20] = 1.
    # CMatrix[3,28] = 1.
    #
    # CMatrix[0,40] = 1.
    # CMatrix[1,120] = 1.
    # CMatrix[2,200] = 1.
    # CMatrix[3,280] = 1.
    # --sensors at bottom----------------------------------------------------------------
    # CMatrix = np.zeros((Ny, Nx))
    # for i in range(32-Ny-1, 32-1):
    #     CMatrix[i-(32-Ny-1),i] = 1
    # --sensors at top----------------------------------------------------------------
    # CMatrix = np.zeros((Ny, Nx))
    # for i in range(0+1, Ny+1):
    #     CMatrix[i-1,i] = 1
    # -------------------------------------------------------------------------
    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    # CMatrix[0,64] = 1.
    # CMatrix[1,192] = 1.
    # CMatrix[2,320] = 1.
    # CMatrix[3,448] = 1.
    # -------------------------------------------------------------------------
    # spacePara_dict = space_parameters()
    # Nxx = spacePara_dict['Nxx']
    # Nxy = spacePara_dict['Nxy']
    #
    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    #
    # position = [2,6,10,14,18,22,26,30]
    # len = position.__len__()
    #
    # if Ny == len:
    #     i = 0
    #     for j in range(int(Ny)):
    #         for k in range(int(Ny//len)):
    #             CMatrix[i, position[j] * Nxx * Nxy + k] = 1
    #             i += 1
    # elif Ny//len == Nxx*Nxy:
    #     i = 0
    #     for j in range(int(Ny/Nxx/Nxy)):
    #         for k in range(int(Ny//len)):
    #             CMatrix[i, position[j]*Nxx*Nxy+k] = 1
    #             i += 1

    # center of 4*4---------------------------------------------------
    '''
    location determines, in 1 horizontal layer, which nodes have sensors burried below
    postion shows, in 1 location, how many sensors are burried and their depths.
    '''
    # spacePara_dict = space_parameters()
    # Nxz = spacePara_dict['Nxz']
    # Nxy = spacePara_dict['Nxy']
    #
    # CMatrix = np.zeros((Ny, Nx))
    #
    # location = [5, 10]
    # position = [2,6,10,14,18,22,26,30]  # the depth of sensors in that location
    # len = position.__len__()
    #
    # k = 0
    # for loc in location:
    #     for pos in position:
    #         CMatrix[k, loc*Nxz+pos] = 1
    #         k+=1

    '''
    location determines, in 1 horizontal layer, which nodes have sensors burried below
    postion shows, in 1 location, how many sensors are burried and their depths.
    '''
    spacePara_dict = space_parameters()
    Nxz = spacePara_dict['Nxz']
    Nxy = spacePara_dict['Nxy']

    CMatrix = np.zeros((Ny, Nx))

    location = [1, 4]
    position = [2,6,10,14]  # the depth of sensors in that location
    len = position.__len__()

    k = 0
    for loc in location:
        for pos in position:
            CMatrix[k, loc*Nxz+pos] = 1
            k+=1

    # # center of 3*3---------------------------------------------------
    # spacePara_dict = space_parameters()
    # Nxx = spacePara_dict['Nxx']
    # Nxy = spacePara_dict['Nxy']
    #
    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    #
    # position = [2,6,10,14,18,22,26,30]
    # len = position.__len__()
    #
    # i = 0
    # for j in range(int(Ny)):
    #     CMatrix[i, position[j] * Nxx * Nxy + 4] = 1
    #     i += 1

    # center of 2*2---------------------------------------------------
    # spacePara_dict = space_parameters()
    # Nxx = spacePara_dict['Nxx']
    # Nxy = spacePara_dict['Nxy']
    #
    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    #
    # position = [2,6,10,14,18,22,26,30]
    # len = position.__len__()
    #
    # i = 0
    # for j in range(int(Ny)):
    #     CMatrix[i, position[j] * Nxx * Nxy + 0] = 1
    #     i += 1

    # spacePara_dict = space_parameters()
    # Nxz = spacePara_dict['Nxz']
    # position = [1,4]
    # layer = [2,6,10,14,18,22,26,30]
    # len = layer.__len__()
    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    # index = 0
    # for i in range(Ny//len):
    #     for j in range(len):
    #         CMatrix[index, position[i]*Nxz + layer[j]] = 1.
    #         index += 1

    # CMatrix = np.zeros((Ny, Nx))  # 4 measurements
    # CMatrix[0,2] = 1.
    # CMatrix[1,6] = 1.
    # CMatrix[2,10] = 1.
    # CMatrix[3,14] = 1.
    # CMatrix[4,18] = 1.
    # CMatrix[5,22] = 1.
    # CMatrix[6,26] = 1.
    # CMatrix[7,30] = 1.

    # # 100*1 -------------------------------------------------------
    # '''
    # location determines, in 1 horizontal layer, which nodes have sensors burried below
    # postion shows, in 1 location, how many sensors are burried and their depths.
    # '''
    # spacePara_dict = space_parameters()
    # Nxz = spacePara_dict['Nxz']
    #
    # CMatrix = np.zeros((Ny, Nx))
    #
    # location = [10, 40, 60, 90]
    # position = [2,6,10,14,18,22,26,30]  # the depth of sensors in that location
    # len = position.__len__()
    #
    # k = 0
    # for loc in location:
    #     for pos in position:
    #         CMatrix[k, loc*Nxz+pos] = 1
    #         k+=1
    return CMatrix


'''
def steady_state(Nx):
    # x_ss = np.ones((Nx))*-0.113271

    # x_ss = np.loadtxt('x_ss')
    # x_ss = x_ss[0:32]

    # x_ss = np.arange(-0.113211, -0.113210, 0.000001/31)
    # x_ss = np.arange(-0.113, -0.112, 0.001/31)
    x_ss = np.arange(-0.11, -0.22, -0.11/32)
    # x_ss = np.arange(-0.01, -0.52, -0.51/32)
    return x_ss


def scale_parameters(Nx, Nw_aug, Ny, Np):  # all max and min are true values of each x, p, w, y
    Pars_dict, Pars_unscaled, Pars_scaled = model_parameters(Np)  # soil/model parameters: This are the true parameters
    pmax = 1.3*np.array([Pars_unscaled[0], Pars_unscaled[1], Pars_unscaled[2], Pars_unscaled[3]])
    pmin = 0.7*np.array([Pars_unscaled[0], Pars_unscaled[1], Pars_unscaled[2], Pars_unscaled[3]])
    xmax = np.append(-0.0001*np.ones(Nx), pmax)
    xmin = np.append(-1*np.ones(Nx), pmin)
    ymax = -0.0001*np.ones(Ny)
    ymin = -1*np.ones(Ny)
    # wmax = np.append(3*5e-6* np.ones(Nw_aug-Np), 1.e-20*np.ones(Np))
    # wmin = np.append(-3*5e-6 * np.ones(Nw_aug-Np), -1.e-20*np.ones(Np))
    wmax = 3*5e-6* np.ones(Nw_aug)
    wmin = -3*5e-6 * np.ones(Nw_aug)
    return xmax, xmin, ymax, ymin, wmax, wmin, pmax, pmin


def bounds_scaled(Nx, Nx_aug, Ny, Np):  # bounds for decision variables (x and w) used in MHE
    # bounds for states x
    if Nx == Nx_aug:  # augmented model - state and parameter estimation
        xlower = 0*np.ones(Nx_aug)
        xupper = 1*np.ones(Nx_aug)

        xlower_fie = 0*np.ones(Nx_aug)
        xupper_fie = 1*np.ones(Nx_aug)
    else:  # original model - state estimation only
        xlower = 0*np.ones(Nx)
        xupper = 1*np.ones(Nx)

        xlower_fie = 0*np.ones(Nx)
        xupper_fie = 1*np.ones(Nx)

    # bounds for process noise w
    wlower = 0*np.ones(Nx_aug)
    wupper = 1*np.ones(Nx_aug)

    return xlower, xupper, wlower, wupper, xlower_fie, xupper_fie


def bounds_state(Nx, Nx_aug, Np):
    # original model - state estimation only
    xlower = -1 * np.ones(Nx)
    xupper = -0.0001 * np.ones(Nx)
    # xlower = xmin * np.ones(Nx)
    # xupper = xmax * np.ones(Nx)

    xlower_fie = -1 * np.ones(Nx)
    xupper_fie = -0.0001 * np.ones(Nx)

    # bounds for process noise w
    wlower = -np.inf * np.ones(Nx)
    wupper = np.inf*np.ones(Nx)
    return xlower, xupper, wlower, wupper, xlower_fie, xupper_fie


def model_parameters(Np):
    pars = Loam()
    if Np == 5:
        pars_unsc = np.array([1., 1., 1., 1., 1.])  # Ks, thetaS, thetaR, alpha, n
    elif Np == 4:
        pars_unsc = np.array([1., pars['thetaS'],pars['alpha'],pars['n']])  # Ks, thetaR, alpha, n
    elif Np == 3:
        pars_unsc = np.array([1., pars['alpha'], pars['n']])  # Ks, thetaR, n
    pars_sc = np.ones(Np)
    return pars, pars_unsc, pars_sc
'''