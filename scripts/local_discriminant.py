from weights import ch_weight, sh_weight
#from superoperator_construct import get_left_action, get_right_action
import numpy as np
from options import options
from mpo_utils import mps_to_mpo, mirror_mpo
from transformations import s_transform
from mps_generators import kraus_mps, id_mps
import gc
import pickle



def disc_parts(jump_op,site_num,length,delta_t,num_steps,ham_plus,ham_minus,opts=options,gpu=False):
    '''
    This function calcullates Sc[D] and Sc[A] for a given jump operator and site number.
    Sc[D]=(Id/2+Sc[S[A]@A[-Sc[A@S[A]]-Sc[A]@S[A]])/2
    Parameters:
    jump_op: np.ndarray
        The jump operator to be used in the transformation.
    site_num: int
        The site number where the jump operator acts.
    length: int
        The length of the chain.
    delta_t: float
        The time step for the time evolution.
    num_steps: int
        The number of time steps to be used in the time evolution.
    ham_plus: qtn.localHam1D
        The Hamiltonian for the forward time evolution.
    ham_minus: qtn.localHam1D
        The Hamiltonian for the backward time evolution.
    opts: dict  
        The options for the tensor network compression.
    gpu: bool
        If True the tensors are moved to the GPU.
    Returns:
    qtn.qtn.tensor_1d.MatrixProductOperator
        The Sc[D] operator.
    qtn.qtn.tensor_1d.MatrixProductOperator
        The Sc[A] operator.
    '''
    jump_mps=kraus_mps(jump_op,site_num,length)

    sa=s_transform(jump_mps,sh_weight,delta_t,num_steps,ham_plus,ham_minus,**opts)
    print(f'sa {site_num} done')
    left_gate=np.kron(jump_op,np.eye(2))
    right_gate=np.kron(np.eye(2),jump_op)
    saa=sa.gate(right_gate,site_num,contract=True)
    asa=sa.gate(left_gate,site_num,contract=True)
    del sa
    scsaa=s_transform(saa,ch_weight,delta_t,num_steps,ham_plus,ham_minus,**opts)
    del saa
    gc.collect()
    scd_p1=(id_mps(length)*0.5).add_MPS(scsaa)
    del scsaa
    gc.collect()
    scd_p1.compress(**opts['comp_opts'])
    scd_p1=mps_to_mpo(scd_p1)
    print(f'scd_p1 {site_num} done')

    scasa=mps_to_mpo(s_transform(asa,ch_weight,delta_t,num_steps,ham_plus,ham_minus,**opts))
    print(f'scasa {site_num} done')
    sca=mps_to_mpo(s_transform(jump_mps,ch_weight,delta_t,num_steps,ham_plus,ham_minus))
    scasca=sca.apply(sca, compress=True, **opts['comp_opts'])
    print(f'scasca {site_num} done')
    scd_p2=(scasa.add_MPO(scasca))
    del scasa
    del scasca
    gc.collect()
    scd_p2.compress(**opts['comp_opts'])
    scd=(scd_p1.add_MPO(scd_p2*-1))*0.5
    scd.compress(**opts['comp_opts'])

    return scd,sca


def load_disc_parts(folder):
    
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