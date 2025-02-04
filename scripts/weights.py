import numpy as np
def sh_weight(x,a=39):
    if x==0:
        x=10e-9
    return 1.j*(1-np.cos(a*x))/np.sinh(2*np.pi*x)


def ch_weight(x):
    return 1./np.cosh(2*np.pi*x)