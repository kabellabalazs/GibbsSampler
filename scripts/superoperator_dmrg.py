import quimb.tensor as qtn
import quimb as qu
import numpy as np
import pickle
import scipy.sparse.linalg as spla
from utils import progress_bar
from mpo_utils import mps_to_mpo

def get_rho_conj(rho,upper_ind_id='b{}',lower_ind_id='a{}',tags='rho*'):
    """
    Generate the hermitian conjugate of a given matrix product operator (MPO), 
    with the correct upper and lower inds suitable for the super operator Tensor Network .
    Parameters:
    -----------
    rho : qtn.tensor_1d.MatrixProductOperator
        The input matrix product operator to be conjugated.
    upper_ind_id : str, optional
        The format string for the upper indices of the MPO (default is 'b{}').
    lower_ind_id : str, optional
        The format string for the lower indices of the MPO (default is 'a{}').
    tags : str or list of str, optional
        The tags to be assigned to the conjugated MPO (default is 'rho*').
    Returns:
    --------
    qtn.tensor_1d.MatrixProductOperator
        The conjugated matrix product operator.
    """
    #get the tensors in the correct order
    site_tag_len=len(rho.site_tag_id)-2
    ind_order=[int(t[site_tag_len:]) for t in rho.tags if t in rho.site_tags]
    tens=[rho.arrays[j].conj() for j in [ind_order.index(i) for i in range(rho.L)]]
    
    #create the conjugated MPO
    rho_conj=qtn.tensor_1d.MatrixProductOperator(tens,
                                                upper_ind_id=upper_ind_id,
                                                lower_ind_id=lower_ind_id,
                                                tags=tags,shape='lrud') #conjugate the operator
 
    return rho_conj

def mirror_mpo(mpo):
    """
    Mirror a Matrix Product Operator (MPO) by reversing the order of its arrays and 
    transposing the internal arrays appropriately.
    Parameters:
    mpo (qtn.tensor_1d.MatrixProductOperator): The input MPO to be mirrored.
    Returns:
    qtn.tensor_1d.MatrixProductOperator: The mirrored MPO with arrays in reversed order 
    and internal arrays transposed.
    """

    mpo.permute_arrays(shape='lrud')
    mirrored_arrays = [mpo.arrays[-1]]
    for arr in list(reversed(mpo.arrays))[1:-1]:
        mirrored_arrays.append(arr.transpose(1,0,2,3))
    mirrored_arrays.append(mpo.arrays[0])
    return qtn.tensor_1d.MatrixProductOperator(mirrored_arrays)



def project_out(rho,rho_conj,rho0,i,lix,rix):
    """
    Locally projects out the reduced ground state density matrix 
    by contracting the gound state and the itareted density matrix.
    
    

    Parameters:
    -----------
    rho : qtn.MatrixProductOperator
        The first density matrix operator.
    rho_conj : qtn.MatrixProductOperator
        The conjugate of the first density matrix operator.
    rho0 : qtn.MatrixProductOperator
        The initial density matrix operator.
    i : int
        The site index to project out.
    lix : list of str
        The left indices for the linear operator.
    rix : list of str
        The right indices for the linear operator.
    Returns:
    --------
    qtn.tensor_core.TNLinearOperator
        The projected tensor network linear operator.
    """
    
    site_tag_len=len(rho0.site_tag_id)-2
    ind_order=[int(t[site_tag_len:]) for t in rho0.tags if t in rho0.site_tags]
    
    tens=[rho0.arrays[j] for j in [ind_order.index(i) for i in range(rho0.L)]]
    rho0=qtn.MatrixProductOperator(tens,
                                    upper_ind_id='b{}', lower_ind_id='a{}',
                                    tags='rho0',shape='lrud')
    
    rho0_conj=get_rho_conj(rho0,upper_ind_id='c{}',lower_ind_id='d{}',tags='rho0*')
    
    full_network=qtn.TensorNetwork1D((rho,rho0,rho_conj,rho0_conj))
    full_network._L=rho.L
    full_network._site_tag_id='I{}'
    menv=qtn.tensor_dmrg.MovingEnvironment(full_network,begin='left',bsz=2)
    menv.move_to(i)
    curnet = menv()
    linop_tens=[curnet['rho0*',f'I{i}'],
                curnet['rho0*',f'I{i+1}'],
                curnet['rho0',f'I{i}'],
                curnet['rho0',f'I{i+1}'],
                curnet['_LEFT'],
                curnet['_RIGHT']]
    return qtn.tensor_core.TNLinearOperator(linop_tens,left_inds=lix,right_inds=rix,optimize='auto-hq')