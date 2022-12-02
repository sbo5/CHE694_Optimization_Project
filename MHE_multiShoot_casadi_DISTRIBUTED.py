from __future__ import print_function, division # Grab some handy Python3 stuff.
from scipy import linalg, integrate, io
from casadi import *
from Parameters import bounds_para, space_parameters
import time


# ---------------Discrete moving horizon estimation--------------------
def mhe_discrete(Pbar,Q,R,M,tol,I,F_N_casadi,switch,hx,Nu,Ninfo,NumPara,index,calNode):
    spacePara_dict = space_parameters()  # space related parameters
    NxPerEst = spacePara_dict['NxPerEst']
    NpTotal = spacePara_dict['NpTotal']
    Np = spacePara_dict['Np']
    Nsoil = spacePara_dict['Nsoil']
    Nest = spacePara_dict['Nest']
    Nx = Q.shape[0]  # Number of states
    Ny = R.shape[0]  # Number of measurements

    xlower, xupper, wlower, wupper, xlower_fie, xupper_fie = bounds_para(NxPerEst, NpTotal, Np, Nsoil, Nest)  # Nx, Nx_aug, Np are used to determine the dimension of bounds

    # define optimization variables
    dv=[]  # decision varibales
    lbdv = []  # lower limits of dv
    ubdv = []  # upper limits of dv
    wm = []  # process noises. This is also the decision variable. Later, we combine it with dv
    lbwm = []  # lower and upper limits of wm
    ubwm = []
    ym = []  # measurements
    um = []  # inputs
    # qbotm = []  # bottom boundary of each estimator
    qleftm = []
    qrightm = []
    pm = []
    xbarm = []  # xbar in arrival cost
    # Pbarm = []  # P in arrival cost
    g = []  # nonlinear constraints
    lbg = []  # lower limits
    ubg = []  # upper limits

    J = 0

    # Formulate the NLP
    Xk = MX.sym('X0',Nx)
    dv += [Xk]  # add X0 in decision variable list
    if switch == 'mhe':
        lbdv.extend(xlower)
        ubdv.extend(xupper)
    else:
        lbdv.extend(xlower_fie)
        ubdv.extend(xupper_fie)

    xbar = MX.sym('xbar', Nx)
    xbarm += [xbar]
    # Pbar = MX.sym('Pbar', Nx*Nx)
    # Pbarm += [Pbar]
    if switch == 'mhe':
        J_arr = mtimes(mtimes((Xk-xbar).T, reshape(Pbar,(Nx,Nx))), (Xk-xbar))               # arrival cost
        # J_arr = 0
    else:  # FIE
        # J_arr = 0
        J_arr = mtimes(mtimes((Xk-xbar).T, reshape(Pbar,(Nx,Nx))), (Xk-xbar))               # arrival cost
    J = J + J_arr

    pk = MX.sym('p', Np * max(1, Nsoil // Nest))
    pm += [pk]

    for k in range(M):
        if k%calNode == 0:
            yk = MX.sym('y' + str(k//calNode), Ny)  # measurement y from sensor
            ym += [yk]
            vk = hx(Xk, index) - yk

        Wk = MX.sym('W' + str(k), Nx)
        wm += [Wk]
        lbwm.extend(wlower)
        ubwm.extend(wupper)

        uk = MX.sym('u' + str(k), Nu)
        um += [uk]

        # qbotk = MX.sym('qbot' + str(k), Nu)
        # qbotm += [qbotk]

        qleftk = MX.sym('qleft' + str(k), Ninfo)
        qleftm += [qleftk]

        qrightk = MX.sym('qright' + str(k), Ninfo)
        qrightm += [qrightk]

        # u_cur = u[k,:]    # has included delay u information
        # -------------------------------------------------------------------------------------
        # Fk = I(x0=Xk, p=vertcat(uk, qbotk, Wk))
        # xk = Fk['xf']
        # -------------------------------------------------------------------------------------
        # xk = xk + DeltaT*getODE_mx_unscaled(xk, u_cur, MX.zeros(Nx_aug)) + DeltaT*wk
        # -------------------------------------------------------------------------------------
        # x_internal = xk
        # for j in range(Nsim_internal):
        #     x_internal = x_internal + DeltaT_internal*getODE_mx_unscaled(x_internal, u_cur, MX.zeros(Nw_aug)) + DeltaT_internal*wk
        # xk = x_internal
        # -------------------------------------------------------------------------------------
        # xk = F_N_casadi(Xk, uk, qbotk, Wk, pk)
        xk = F_N_casadi(Xk, uk, qleftk, qrightk, Wk, pk)

        Xk = MX.sym('X'+str(k+1), Nx)
        dv += [Xk]
        lbdv.extend(xlower)
        ubdv.extend(xupper)

        if k%calNode == 0:
            J = J + mtimes(mtimes((Wk).T,linalg.inv(Q)),(Wk)) + mtimes(mtimes(vk.T,linalg.inv(R)),vk)
        else:
            J = J + mtimes(mtimes((Wk).T,linalg.inv(Q)),(Wk))

        # Add inequality constraint
        g += [Xk-xk]  # Xk - f(Xk-1, Wk-1) = 0. Continuity condition between segments
        lbg.extend(np.zeros(Nx))
        ubg.extend(np.zeros(Nx))
        # lbg.extend(-1.e-10*np.ones(Nx))
        # ubg.extend(1.e-10*np.ones(Nx))

    yk = MX.sym('y' + str(M//calNode), Ny)
    ym += [yk]
    vk = hx(Xk, index) - yk
    J = J + mtimes(mtimes(vk.T, linalg.inv(R)), vk)

    dv.extend(wm)
    dv.extend(ym)
    dv.extend(um)
    # dv.extend(qbotm)
    dv.extend(qleftm)
    dv.extend(qrightm)
    dv.extend(pm)
    dv.extend(xbarm)
    # dv.extend(Pbarm)

    lbdv.extend(lbwm)
    ubdv.extend(ubwm)

    # Create an NLP solver
    # NLP solver options
    opts = {}
    # opts["expand"] = True
    # opts["ipopt.max_iter"] = 1000.0
    # opts["ipopt.tol"] = tol
    # opts["ipopt.acceptable_tol"] = tol*100
    # opts["ipopt.linear_solver"] = "ma57"
    opts["ipopt.print_level"] = 5
    opts["ipopt.print_info_string"] = "yes"
    # opts["regularity_check"] = True
    # opts["ipopt.check_derivatives_for_naninf"] = 'yes'
    # opts["ipopt.derivative_test"] = "first-order"
    # opts["ipopt.derivative_test_print_all"] = "no"
    # opts["verbose"] = True
    # opts["ipopt.hessian_approximation"] = 'limited-memory'
    ### ======== nlp_f,nlp_g,nlp_grad,nlp_grad_f,nlp_hess_l,nlp_jac_g ===========================
    # opts['monitor'] = 'nlp_g'
    nlp = {'x': vertcat(*dv), 'f': J, 'g':vertcat(*g)}
    # nlp = {'x': vertcat(*dv), 'f': J}
    print('Generating solver')
    timeS = -time.time()
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    print('Solver generated')
    print('use',(time.time()+timeS)/60,'mins')
    # V_allInWin = y - Y_allInWin
    # J_cal = np.sum(mtimes(V_allInWin**2, linalg.inv(R))) + np.sum(mtimes(W_allInWin**2, linalg.inv(Q))) #+ np.sum(mtimes(X_allInWin[0,].T, linalg.inv(P), X_allInWin[0,]))
    return solver, lbdv, ubdv, lbg, ubg


def filter_smoother_scheme(indicator, x_bar_allEst, x_in_allEst, x_mhe, xx_ol, X_allInWin_allEst, i, calNode, Nmhe, Nx, Np, Nsoil, Nest, NxPerEst, NyPerEst, itr):
    if indicator == 'smoother':
        for j in range(Nest):
            xbegin = j * NxPerEst
            xend = (j + 1) * NxPerEst
            ybegin = j * NyPerEst
            yend = (j + 1) * NyPerEst
            if i // calNode <= Nmhe:  # FIE
                if i // calNode == 1:
                    if Nsoil == 1:
                        x_bar_allEst[:, j] = x_mhe[0, :, j]
                        x_in_allEst[:, :, j] = x_mhe[0:i + 1, :, j]
                        # x_in_allEst[1, :, j] = np.append(xx_ol[1, xbegin:xend], xx_ol[1,Nx:Nx + Np])  # when use centralized simulator
                        # x_in_allEst[1, :, j] = xx_ol[1,:,j]  # when use distributed simulator
                        x_in_allEst[1, :, j] = x_in_allEst[0, :, j]
                    else:
                        x_bar_allEst[:, j] = x_mhe[0, :, j]
                        x_in_allEst[:, :, j] = x_mhe[0:i + 1, :, j]
                        # x_in_allEst[1, :, j] = np.append(xx_ol[1, xbegin:xend], xx_ol[1,Nx + Np * max(1, Nsoil // Nest) * j:Nx + Np * max(1,Nsoil // Nest) * (j + 1)])  # when use centralized simulator
                        # x_in_allEst[1, :, j] = xx_ol[1, :, j]  # when use distributed simulator
                        x_in_allEst[1, :, j] = x_in_allEst[0, :, j]
                        # x_bar_allEst[:, j] = np.append(xx_ol[0, xbegin:xend], xx_ol[0, Nx_aug-Np:Nx_aug])
                        # x_in_allEst[:, :, j] = np.append(xx_ol[0:i+1, xbegin:xend], xx_ol[0:i+1, Nx_aug-Np:Nx_aug], axis=1)
                        # x_bar_allEst_alt[:, :] = x_mhe[0]  # this one is the same as x_bar_allEst
                        # x_in_allEst_alt[:, :, :] = x_mhe[0:i+1]  # don't use this one, since x_mhe does not has results at the next time instant
                else:
                    # print('Smoother scheme is used')
                    x_bar_allEst[:, j] = X_allInWin_allEst[0, :, j]
                    x_in_allEst[:, :, j] = X_allInWin_allEst[0:i + 1, :, j]
                    x_in_allEst[-calNode:, :, j] = x_in_allEst[-calNode - 1, :,j]  # this critical when deltaTmhe > deltaT, since initial guess is not accurate anymore
            else:  # MHE
                x_bar_allEst[:, j] = X_allInWin_allEst[1 * calNode, :, j]
                x_in_allEst[:-calNode, :, j] = X_allInWin_allEst[1 * calNode:, :, j]
                x_in_allEst[-calNode:, :, j] = x_in_allEst[-calNode - 1, :, j]
    elif indicator == 'filter':
        for j in range(Nest):
            xbegin = j * NxPerEst
            xend = (j + 1) * NxPerEst
            ybegin = j * NyPerEst
            yend = (j + 1) * NyPerEst
            if i // calNode <= Nmhe:  # FIE
                # print('Smoother scheme is used')
                x_bar_allEst[:, j] = x_mhe[0, :, j]
                x_in_allEst[:, :, j] = x_mhe[0:i + 1, :, j]
                if itr == 0:
                    x_in_allEst[-calNode:, :, j] = x_in_allEst[-calNode - 1, :,j]  # this critical when deltaTmhe > deltaT, since initial guess is not accurate anymore
            else:  # MHE
                x_bar_allEst[:, j] = x_mhe[i - Nmhe*calNode, :, j]
                x_in_allEst[:-calNode, :, j] = x_mhe[i - Nmhe * calNode:i, :, j]
                if itr == 0:
                    x_in_allEst[-calNode:, :, j] = x_in_allEst[-calNode - 1, :, j]
    return x_bar_allEst, x_in_allEst


def mhe_prepare(i, j, dataSize, x_bar_allEst, yy, uu, x_in_allEst, KFun, NxPerEst, Nu, Ninfo, Nest, Nsoil, Np, dz, ParsNp, calNode, Nmhe, Nx_aug, ybegin, yend, P0, Q, R):
    # smoother -----------------------------------------------------------------------------------------------------
    if j == 0:  # first estimator
        uin = uu[i - dataSize:i, :]*np.ones(Nu)  # top boundary of 1st estimator is irrigation
        qleftin = 0*np.ones(Ninfo*dataSize)

        xlastk = x_in_allEst[0:dataSize, NxPerEst - Ninfo:NxPerEst, j]
        if Nest == 1:  # bottom boundary of 1st estimator. If its centralized, then free drainage
            xbotk = xlastk
        else:
            xbotk = x_in_allEst[0:dataSize, 0:Ninfo, j + 1]
        if Nest == 1:
            para = ParsNp  # Only 1 layer of soil, ParsNp will always contains 5 parameters
            Klast = KFun(xlastk, x_in_allEst[0, -Np:, j], para[-Np:])
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j],para[-Np:])
        elif Nest > 1 and Nsoil > 1:
            para = ParsNp[j * Np:(j + 1) * Np]  # First estimator, the 1st layer of soil
            Klast = KFun(xlastk, x_in_allEst[0, -Np:, j], para)
            # Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j],para)  # we assume the bottom boundary share the same soil properties of the subsystem
            # Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j], ParsNp[(j+1)*Np: (j+2)*Np])  # here the bottom boundary share the different soil properties of the subsys
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j+1], ParsNp[(j + 1) * Np: (j + 2) * Np])  # here the bottom boundary share the different soil properties of the subsy
        elif Nest > 1 and Nsoil == 1:  # This is the same as Nest == 1
            para = ParsNp  # Only 1 layer of soil, ParsNp will always contains 5 parameters
            Klast = KFun(xlastk, x_in_allEst[0, -Np:, j], para[-Np:])
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j],para[-Np:])
        else:
            print('Nsoil and Nest are not defined well')
        Kmid = (Klast + Kbot) / 2
        qrightin = - Kmid * ((xlastk.ravel() - xbotk.ravel()) / dz)

    elif j == Nest - 1:  # last estimator
        uin = uu[i - dataSize:i, :]*np.ones(Nu)  # top boundary of 1st estimator is irrigation

        xtopk = x_in_allEst[0:dataSize, NxPerEst - Ninfo:NxPerEst, j - 1]
        x1k = x_in_allEst[0:dataSize, 0:Ninfo, Nest - 1]
        if Nsoil == 1:
            para = ParsNp  # only 1 type of soil
            Ktop = KFun(xtopk, x_in_allEst[0, -Np:, j], para)
            K1 = KFun(x1k, x_in_allEst[0, -Np:, j], para)
        else:
            para = ParsNp[j * Np:(j + 1) * Np]  # the last estimator will have the last layer of soil
            # Ktop = KFun(xtopk, x_in_allEst[0, -Np:, j], para)  # we assume the top boundary share the same soil properties of the subsystem
            # Ktop = KFun(xtopk, x_in_allEst[0, -Np:,j], ParsNp[(j-1)*Np: j*Np])  # here the top boundary share the different soil properties of the subsys
            Ktop = KFun(xtopk, x_in_allEst[0, -Np:,j-1], ParsNp[(j-1)*Np: j*Np])  # here the top boundary share the different soil properties of the subsys
            K1 = KFun(x1k, x_in_allEst[0, -Np:, j], para)
        Kmid = (Ktop + K1) / 2
        qleftin = - Kmid * ((xtopk.ravel() - x1k.ravel()) / dz)

        xbotk = x_in_allEst[0:dataSize, NxPerEst - Ninfo:NxPerEst, Nest - 1]
        if Nsoil == 1:
            para = ParsNp
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j], para)
        else:
            para = ParsNp[j * Np:(j + 1) * Np]  # this para replace para in the previous if-else-, but they are the same
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j], para)
        qrightin = 0*np.ones(Ninfo*dataSize)
    else:
        uin = uu[i - dataSize:i, :]*np.ones(Nu)  # top boundary of 1st estimator is irrigation

        xtopk = x_in_allEst[0:dataSize, NxPerEst - Ninfo:NxPerEst, j - 1]
        x1k = x_in_allEst[0:dataSize, 0:Ninfo, j]
        if Nsoil == 1:  # this is always false
            para = ParsNp
            Ktop = KFun(xtopk, x_in_allEst[0, -Np:, j], para)
            K1 = KFun(x1k, x_in_allEst[0, -Np:, j], para)
        else:
            para = ParsNp[j * Np:(j + 1) * Np]
            Ktop = KFun(xtopk, x_in_allEst[0, -Np:, j-1], ParsNp[(j-1)*Np: j*Np])
            K1 = KFun(x1k, x_in_allEst[0, -Np:, j], para)
        Kmid = (Ktop + K1) / 2
        qleftin = - Kmid * ((xtopk.ravel() - x1k.ravel()) / dz)

        xlastk = x_in_allEst[0:dataSize, NxPerEst - Ninfo:NxPerEst, j]
        xbotk = x_in_allEst[0:dataSize, 0:Ninfo, j + 1]
        if Nsoil == 1:
            para = ParsNp
            Klast = KFun(xlastk, x_in_allEst[0, -Np:, j], para)
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j], para)
        else:
            para = ParsNp[j * Np:(j + 1) * Np]
            Klast = KFun(xlastk, x_in_allEst[0, -Np:, j], para)
            Kbot = KFun(xbotk, x_in_allEst[0, -Np:, j+1], ParsNp[(j + 1) * Np: (j + 2) * Np])
        Kmid = (Klast + Kbot) / 2
        qrightin = - Kmid * ((xlastk.ravel() - xbotk.ravel()) / dz)

    x_bar = x_bar_allEst[:, j]
    x_in = x_in_allEst[:, :, j]
    y_in = yy[max(0, i // calNode - Nmhe):i // calNode + 1, ybegin:yend]  # when use centralized simulator
    # y_in = yy[max(0, i // calNode - Nmhe):i // calNode + 1, :, j]  # when use ditributed simulator

    if Nest == 2:
        b = Nx_aug - Np * Nsoil - NxPerEst
        e = b + NxPerEst + Np
        P_in = P0[b:e,b:e]
        Q_in = Q[b:e,b:e]
    elif Nest == 1:
        P_in = P0
        Q_in = Q
    R_in = R[ybegin:yend, ybegin:yend]
    return uin, qleftin, qrightin, para, x_in, y_in, P_in, Q_in, R_in, x_bar


def flaten(x, y, u, qleft, qright, para, P, i, Nsoil, Nest, NxPerEst, Nu, Ninfo, Np, Nmhe, NyPerEst, calNode):
    x_flat = x.reshape(1, (i + 1) * (NxPerEst + Np * max(1, Nsoil // Nest))).ravel()
    y_flat = y.reshape(1, (min(Nmhe, i // calNode) + 1) * NyPerEst).ravel()

    u_flat = u.reshape(i * Nu, 1).ravel()
    # qbot_flat = qbot.reshape(i * Nu, 1).ravel()
    qleft_flat = qleft.reshape(i * Ninfo, 1).ravel()
    qright_flat = qright.reshape(i * Ninfo, 1).ravel()
    para_flat = para.reshape(Np * max(1, Nsoil // Nest), 1).ravel()

    Pinv = linalg.inv(P)
    P_flat = Pinv.reshape(1, (NxPerEst + Np * max(1, Nsoil // Nest)) * (NxPerEst + Np * max(1, Nsoil // Nest))).ravel()
    return x_flat, y_flat, u_flat, qleft_flat, qright_flat, para_flat, P_flat


def cal_cur_state(w_opt,u,M,I,F_N_casadi,hx,Nx,Ny,index):  # calculate current state
    # DeltaT, Tsim, Nsim, DeltaT_internal, Nsim_internal, Tplot, Nmhe, DeltaTmhe, calNode = time_parameters()  # time related parameters

    X0 = w_opt[:Nx]  # X0 is always the first one
    xk = X0
    W = np.zeros((M,Nx))
    X = np.zeros((M+1,Nx))
    X[0,:] = xk
    Y = np.zeros((M+1, Ny))
    Y[0,:] = hx(xk,index)
    for i in range(M):
        X[i+1,:] = w_opt[i*Nx+Nx:Nx+(i+1)*Nx]
        W[i,:] = w_opt[i*Nx+Nx*(M+1):(i+1)*Nx+Nx*(M+1)]
        # SimInterval = [Tplot[i], Tplot[i+1]]
        # u_cur = u[i,:]
        # ---------------------------------------------------------------------------------------------
        # x_next = integrate.odeint(getODE_np_unscaled, xk, SimInterval, args=(u_cur, W[i,:]))  # for ode
        # xk = x_next[-1]
        # ---------------------------------------------------------------------------------------------
        # Fk = I(x0=xk, p=vertcat(u_cur, W[i,:]))
        # xk = Fk['xf'].full().ravel()
        # ---------------------------------------------------------------------------------------------
        # x_next = xk + DeltaT * getODE_np_unscaled(xk, DeltaT, u_cur, np.zeros(Nx_aug)) + DeltaT*W[i,:]
        # xk = x_next
        # ---------------------------------------------------------------------------------------------
        # x_internal = xk
        # for j in range(Nsim_internal):
        #     x_internal = x_internal + DeltaT_internal * getODE_np_unscaled(x_internal, DeltaT_internal, u_cur, np.zeros(Nw_aug)) + DeltaT_internal*W[i,:]
        # xk = x_internal
        # ---------------------------------------------------------------------------------------------
        # xk = F_N_casadi(xk, u_cur, W[i,:]).full().ravel()  # This one is only used for comparing with X[i+1,:]
        # ---------------------------------------------------------------------------------------------
        Y[i+1,:] = hx(X[i+1,:], index)
    return X[M], X, W, Y
