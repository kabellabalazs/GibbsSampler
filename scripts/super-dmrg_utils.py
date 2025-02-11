import pickle
from mpo_utils import mirror_mpo
import os
#Loading the data ( half of the system)

def load_disc_parts(folder):
    folder='DATA/new_data02_L6/'
    
    with open(folder+'sca_xs.pkl', 'rb') as file:
        sca_xs = pickle.load(file)
        
    with open(folder+'sca_zs.pkl', 'rb') as file:
        sca_zs = pickle.load(file)
    with open(folder+'scd_xs.pkl', 'rb') as file:
        scd_xs = pickle.load(file)
    with open(folder+'scd_zs.pkl', 'rb') as file:
        scd_zs = pickle.load(file)


    #Mirror the data to get the full system
    sca_xs_mirrored = [mirror_mpo(sca_x) for sca_x in reversed(sca_xs)]
    sca_zs_mirrored = [mirror_mpo(sca_z) for sca_z in reversed(sca_zs)]
    scd_xs_mirrored = [mirror_mpo(scd_x) for scd_x in reversed(scd_xs)]
    scd_zs_mirrored = [mirror_mpo(scd_z) for scd_z in reversed(scd_zs)]

    scas=sca_xs+sca_xs_mirrored+sca_zs+sca_zs_mirrored
    scds=scd_xs+scd_xs_mirrored+scd_zs+scd_zs_mirrored
    return scds,scas

