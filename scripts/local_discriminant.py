from weights import ch_weight, sh_weight
#from superoperator_construct import get_left_action, get_right_action
import numpy as np
from options import options
from mpo_utils import mps_to_mpo, mirror_mpo
from transformations import s_transform
from mps_generators import kraus_mps, id_mps
import gc
import pickle
import os


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
def generate_disc_parts(ham_tp,ham_tm,delta_t,num_steps,jump_ops,folder,options=options):
    length=ham_tp.L
    scas={key:[] for key in jump_ops}
    scds={key:[] for key in jump_ops}
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(int(length/2)):
        for item in jump_ops.items():
            print(f'Site: {item[0]}_{i}')
            #generate parts
            scd_a,sca_a=disc_parts(item[1],i,length,delta_t,num_steps,ham_tp,ham_tm,opts=options,gpu=False)
            scas[item[0]].append(sca_a)
            scds[item[0]].append(scd_a)
            #save
            with open(folder+f'scas.pkl','wb') as f:
                pickle.dump(scas,f)
            with open(folder+f'scds.pkl','wb') as f:
                pickle.dump(scds,f)
    #Mirror the data to get the full system

    for item in scas.items():
        key=item[0]
        lst=item[1]
        scas[key]=lst+ [mirror_mpo(sca) for sca in reversed(lst)]
    for item in scds.items():
        key=item[0]
        lst=item[1]
        scds[key]=lst+ [mirror_mpo(scd) for scd in reversed(lst)]
    with open(folder+f'scas.pkl','wb') as f:
        pickle.dump(scas,f)
    with open(folder+f'scds.pkl','wb') as f:
        pickle.dump(scds,f)
    return scds,scas

def load_disc_parts(folder):
    
    with open(folder+'scas.pkl', 'rb') as file:
        scas = pickle.load(file)
    with open(folder+'scds.pkl', 'rb') as file:
        scds = pickle.load(file)
    
    return scds,scas