import numpy as np
import quimb.tensor as qtn


def kraus_mps(jump_op,site_number,length):

    mps_sites=[np.eye(2).reshape(4)]*length
    mps_sites[site_number]=jump_op.reshape(4)
    mps=qtn.MPS_product_state(mps_sites)
    return mps


def id_mps(length):
    mps=qtn.MPS_product_state([np.eye(2).reshape(4)]*length)
    return mps


