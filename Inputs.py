import numpy as np


def irr(Nsim, DeltaT, Tsim):
    Nday = Tsim/86400
    # NperDay = 24*3600/DeltaT
    NperHr = 3600/DeltaT
    # Top boundaries
    irr = np.ones((Nsim,1))*0  # irr is in shape(Nsim,1), since it is time variant
    # -----------------------------------
    # irr[0:20] = - 5.399e-07  # 46.6 mm/day
    # irr[41:60] = - 5.399e-07
    # irr[81:100] = - 5.399e-07
    # irr[121:140] = - 5.399e-07
    # irr[161:180] = - 5.399e-07
    # irr[201:220] = - 5.399e-07
    # irr[241:260] = - 5.399e-07
    # irr[281:300] = - 5.399e-07
    # irr[321:340] = - 5.399e-07
    # irr[361:380] = - 5.399e-07
    # -----------------------------------
    # irr[:] = 0#-2.89e-07
    for i in range(Nday):
        for l in range(4*NperHr):  # 4 hr irrigation
            irr[i*NperHr*24 + 12*NperHr + l] = -2.89e-07  # everydat, start at 12 pm
    # -----------------------------------
    # irr[1:4] = - 5.399e-8
    # irr[10:13] = - 5.399e-8
    # irr[19:22] = - 5.399e-8
    # ------------------------------------
    # irr[:] = - 8.1e-8  # 7 mm/day
    return irr
