import quimb.tensor as qtn
import numpy as np
from utils import binary_sum
from options import options
from utils import progress_bar
import torch
def s_transform(kraus_mps,weight_func,delta_t,num_steps,ham_tp,ham_tm,use_torch=False,split_opts=options['split_opts'],tebd_opts=options['tebd_opts'],comp_opts=options['comp_opts']):
    '''
    Implements the S-transform of a given MPS.
    S[.]=Integrate_{-inf}^{inf} dt weight_func(t) U(t)[.]U(-t)
    where U(t) is the time evolution operator of the Hamiltonian H ==> exp(-iHt)
    Parameters:
    kraus_mps: qtn.tensor_1d.MatrixProductState
        The MPS to be transformed.
    weight_func: function
        The weight function to be used in the transformation.
    delta_t: float
        The time step for the time evolution.
    num_steps: int
        The number of time steps to be used in the time evolution.
    ham_tp: qtn.localHam1D
        The Hamiltonian for the forward time evolution.
    ham_tm: qtn.localHam1D
        The Hamiltonian for the backward time evolution.
    split_opts: dict
        The options for the tensor network splitting.
    tebd_opts: dict 
        The options for the TEBD time evolution.
    comp_opts: dict
        The options for the tensor network compression.
    Returns:
    qtn.qtn.tensor_1d.MatrixProductState
        The transformed MPS.
    '''
    if use_torch:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure kraus_mps, ham_tp, and ham_tm are using PyTorch tensors on the GPU
        kraus_mps.apply_to_arrays(lambda x: torch.tensor(x,dtype=torch.complex128, device=device))
        ham_tp.apply_to_arrays(lambda x: torch.tensor(x,dtype=torch.complex128, device=device))
        ham_tm.apply_to_arrays(lambda x: torch.tensor(x,dtype=torch.complex128, device=device))
     #create the TEBD object for the forward and backward time evolution
    tebd_plus=qtn.TEBD(kraus_mps,ham_tp,split_opts=split_opts)
    tebd_minus=qtn.TEBD(kraus_mps,ham_tm,split_opts=split_opts)
    #initialize the time steps
    times=np.linspace(delta_t,num_steps*delta_t,num_steps)
    #initialize the binary tree addition
    rule=binary_sum(num_steps+1)
    sum_buffer=[]
    i=0
    for t in times:
        #progress_bar(i,num_steps)
        tebd_plus.update_to(t,**tebd_opts)
        tebd_minus.update_to(t,**tebd_opts)
        summed=(tebd_plus.pt*weight_func(t)*delta_t).add_MPS(tebd_minus.pt*weight_func(-t)*delta_t)
        summed.compress(**comp_opts)
        sum_buffer.append(summed)
        for _ in range(rule[i]):
            part1=sum_buffer.pop()
            part2=sum_buffer.pop()
            summed=part1.add_MPS(part2)
            summed.compress(**comp_opts)
            sum_buffer.append(summed)
        i+=1
    sum_buffer.append(kraus_mps*(weight_func(0)*delta_t))
    while len(sum_buffer) > 1:
        part1=sum_buffer.pop()
        part2=sum_buffer.pop()
        summed=part1.add_MPS(part2)
        summed.compress(**comp_opts)
        sum_buffer.append(summed)
    transformed=sum_buffer[0]
    return transformed
