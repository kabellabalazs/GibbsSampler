import numpy as np
import quimb.tensor as qtn
from move_to_gpu import move2gpu


def kraus_mps(jump_op,site_number,length,gpu=False):

    mps_sites=[np.eye(2).reshape(4)]*length
    mps_sites[site_number]=jump_op.reshape(4)
    mps=qtn.MPS_product_state(mps_sites)
    if gpu:
        mps.apply_to_arrays(move2gpu)
    return mps


def id_mps(length,gpu=False):
    mps=qtn.MPS_product_state([np.eye(2).reshape(4)]*length)
    if gpu:
        mps.apply_to_arrays(move2gpu)
    return mps


