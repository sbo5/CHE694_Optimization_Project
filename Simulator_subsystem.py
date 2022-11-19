import numpy as np
from casadi import *

def simulator_dis(xin, win, Nest, NxPerEst, Nu, Ninfo, Nsoil, Np, dz, uu, ParsNp, F_N_Fun, I_est_Fun, F_N_aug_np_subsys, KFun):
    xk = np.zeros((NxPerEst+Np*max(1, Nsoil//Nest), Nest))
    for j in range(Nest):
        if j == 0:  # first estimator
            uin = uu*np.ones(Nu)  # top boundary of 1st estimator is irrigation
            qleftin = 0.*np.ones(Ninfo)
            
            xlastk = xin[NxPerEst - Ninfo:NxPerEst, j]
            if Nest == 1:  # bottom boundary of 1st estimator. If its centralized, then free drainage
                xbotk = xlastk
            else:
                xbotk = xin[0:Ninfo, j + 1]

            if Nest == 1:
                para = ParsNp  # with only 1 estimator, there could be more than 1 layer of soil
                Klast = KFun(xlastk, xin[-Np:, j], para[-Np:])  # if there is only 1 estimator, the bottom boundary is always using the bottom layer of soil
                Kbot = KFun(xbotk, xin[-Np:, j], para[-Np:])
            elif Nest > 1 and Nsoil > 1:
                para = ParsNp[j * Np:(j + 1) * Np]  # First estimator, the 1st layer of soil
                Klast = KFun(xlastk, xin[-Np:, j], para)
                # Kbot = KFun(xbotk, xin[-Np:, j],para)  # we assume the bottom boundary share the same soil properties of the subsystem
                Kbot = KFun(xbotk, xin[-Np:, j], ParsNp[(j + 1) * Np: (j + 2) * Np])  # here the bottom boundary share the different soil properties of the subsys
            elif Nest > 1 and Nsoil == 1:  # This is the same as Nest == 1
                para = ParsNp  # Only 1 layer of soil, ParsNp will always contains 5 parameters
                Klast = KFun(xlastk, xin[-Np:, j], para[-Np:])
                Kbot = KFun(xbotk, xin[-Np:, j], para[-Np:])
            else:
                print('Nsoil and Nest are not defined well')
            Kmid = (Klast + Kbot) / 2
            qrightin = - Kmid * ((xlastk - xbotk) / dz)
        elif j == Nest - 1:  # last estimator
            uin = uu*np.ones(Nu)  # top boundary of 1st estimator is irrigation

            xtopk = xin[NxPerEst - Ninfo:NxPerEst, j - 1]
            x1k = xin[0:Ninfo, Nest - 1]
            if Nsoil == 1:
                para = ParsNp  # only 1 type of soil
                Ktop = KFun(xtopk, xin[-Np:, j], para)
                K1 = KFun(x1k, xin[-Np:, j], para)
            else:
                para = ParsNp[j * Np:(j + 1) * Np]  # the last estimator will have the last layer of soil
                # Ktop = KFun(xtopk, xin[-Np:, j],para)  # we assume the top boundary share the same soil properties of the subsystem
                Ktop = KFun(xtopk, xin[-Np:, j], ParsNp[(j - 1) * Np: j * Np])  # here the top boundary share the different soil properties of the subsys
                K1 = KFun(x1k, xin[-Np:, j], para)
            Kmid = (Ktop + K1) / 2
            qleftin = - Kmid * ((xtopk - x1k) / dz)

            xbotk = xin[NxPerEst - Ninfo:NxPerEst, Nest - 1]
            if Nsoil == 1:
                para = ParsNp
                Kbot = KFun(xbotk, xin[-Np:, j], para)
            else:
                para = ParsNp[j * Np:(j + 1) * Np]  # this para replace para in the previous if-else-, but they are the same
                Kbot = KFun(xbotk, xin[-Np:, j], para)
            qrightin = 0.*np.ones(Ninfo)
        else:
            uin = uu*np.ones(Nu)  # top boundary of 1st estimator is irrigation

            xtopk = xin[NxPerEst - Ninfo:NxPerEst, j - 1]
            x1k = xin[0:Ninfo, j]
            if Nsoil == 1:  # this is always false
                para = ParsNp
                Ktop = KFun(xtopk, xin[-Np:, j], para)
                K1 = KFun(x1k, xin[-Np:, j], para)
            else:
                para = ParsNp[j * Np:(j + 1) * Np]
                # Ktop = KFun(xtopk, xin[-Np:, j], para)
                Ktop = KFun(xtopk, xin[-Np:, j], ParsNp[(j - 1) * Np: j * Np])
                K1 = KFun(x1k, xin[-Np:, j], para)
            Kmid = (Ktop + K1) / 2
            qleftin = - Kmid * ((xtopk - x1k) / dz)

            xlastk = xin[NxPerEst - Ninfo:NxPerEst, j]
            xbotk = xin[0:Ninfo, j + 1]
            if Nsoil == 1:
                para = ParsNp
                Klast = KFun(xlastk, xin[-Np:, j], para)
                Kbot = KFun(xbotk, xin[-Np:, j], para)
            else:
                para = ParsNp[j * Np:(j + 1) * Np]
                Klast = KFun(xlastk, xin[-Np:, j], para)
                # Kbot = KFun(xbotk, xin[-Np:, j], para)
                Kbot = KFun(xbotk, xin[-Np:, j], ParsNp[(j + 1) * Np: (j + 2) * Np])
            Kmid = (Klast + Kbot) / 2
            qrightin = - Kmid * ((xlastk - xbotk) / dz)
        xk[:,j] = F_N_Fun(xin[:,j], uin, qleftin, qrightin, win[:,j], para).full().ravel()

        # Ik = I_est_Fun(x0=xin[:,j], p=vertcat(uin, qrightin, win[:,j], para))
        # xk[:,j] = Ik['xf'].full().ravel()

        # xk[:, j] = F_N_aug_np_subsys(xin[:, j], uin, qleftin, qrightin, win[:, j], para)
    return xk