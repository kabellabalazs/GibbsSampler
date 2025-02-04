from weights import ch_weight, sh_weight
#from superoperator_construct import get_left_action, get_right_action
import quimb.tensor as qtn
import numpy as np
from utils import binary_sum
from options import options
from mpo_utils import mps_to_mpo
from transformations import s_transform
from mps_generators import kraus_mps, id_mps
from move_to_gpu import move2gpu



from weights import ch_weight, sh_weight
import gc
#from superoperator_construct import get_left_action, get_right_action
import quimb.tensor as qtn
import numpy as np
from utils import binary_sum
from options import options
from mpo_utils import mps_to_mpo
from transformations import s_transform
from mps_generators import kraus_mps, id_mps
from move_to_gpu import move2gpu



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


"""
def local_discriminant(scd,sca):
    left_scd=get_left_action(scd*-1)
    right_scd=get_right_action(scd*-1)
    sca_sca=get_right_action(sca).apply(get_left_action(sca))
    local_disc=(sca_sca.add_MPO(left_scd)).add_MPO(right_scd)
    local_disc.compress()
    return local_disc
"""

"""
def local_discriminant(scd,sca):
    left_scd=get_left_action(scd*-1)
    right_scd=get_right_action(scd*-1)
    sca_sca=get_right_action(sca).apply(get_left_action(sca))
    local_disc=(sca_sca.add_MPO(left_scd)).add_MPO(right_scd)
    local_disc.compress()
    return local_disc
"""