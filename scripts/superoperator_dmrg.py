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


def gibbs_mpo(local_ham,tebd_tol,split_opts):
    rho0=qtn.MPS_product_state([np.eye(2)]*local_ham.L)
    tebd=qtn.TEBD(rho0,local_ham,imag=True,progbar=False,split_opts=split_opts)
    tebd.update_to(.5,tol=tebd_tol)
    rho=mps_to_mpo(tebd.pt)
    rho=rho*np.sqrt(1/(rho.apply(rho)).trace())
    return rho

def super_dmrg(rho0,left_mpos,right_mpos,rho_g,num_iters,tolarances,which_eval='LA'):

    which_eval='LA'
    
    rho=rho0
    rho_conj=get_rho_conj(rho) 

    new_left_mpos=[]
    new_right_mpos=[]
    for i in range(len(left_mpos)):
        left_mpo=qtn.MatrixProductOperator(left_mpos[i].arrays, upper_ind_id='b{}', lower_ind_id='c{}', tags='lmpo',shape='lrud')
        right_mpo=qtn.MatrixProductOperator(right_mpos[i].arrays, upper_ind_id='d{}', lower_ind_id='a{}', tags='rmpo',shape='lrud')
        new_left_mpos.append(left_mpo)
        new_right_mpos.append(right_mpo)
    right_mpos=new_right_mpos
    left_mpos=new_left_mpos

    menvs=[]
    for k in range(len(left_mpos)):
        full_network=qtn.TensorNetwork1D((rho,rho_conj,left_mpos[k],right_mpos[k]))
        full_network._L=rho.L
        full_network._site_tag_id='I{}'
        menvs.append(qtn.tensor_dmrg.MovingEnvironment(full_network,begin='right',bsz=2))

    if isinstance(tolarances,float):
        tolarances=[tolarances]*num_iters
    n=0
    while n<num_iters:
        tol=tolarances[n]
        split_opts = {'method': 'svd','cutoff_mode':'abs', 'max_bond': 100, 'cutoff': tol}
        print('\nright_sweep: ',n)
        
        for i in range(rho.L-2):
            progress_bar(i,rho.L-1)
            v0 = (rho[f'I{i}'] & rho[f'I{i+1}']) # get the double site to optimize
            v0c=v0.contract()
            # get the indicies of the linear operator
            rix = v0c.inds
            lix = (rho_conj[f'I{i}'] & rho_conj[f'I{i+1}']).outer_inds() 
            v_shape=v0c.shape
            #get all the linear operators from the left and tight mpos
            linop=-1*project_out(rho,rho_conj,rho_g,i,lix,rix)
            for _,menv in enumerate(menvs):
                menv.move_to(i)
                curnet = menv()
                linop_tens=[curnet['lmpo',f'I{i}'],
                            curnet['lmpo',f'I{i+1}'],
                            curnet['rmpo',f'I{i}'],
                            curnet['rmpo',f'I{i+1}'],
                            curnet['_LEFT'],
                            curnet['_RIGHT']]
                linop+=qtn.tensor_core.TNLinearOperator(linop_tens,left_inds=lix,right_inds=rix,optimize='auto-hq')
        
            #find the eigenvector of the full linear operator
            _,evecs=spla.eigsh(linop,k=1,which=which_eval,v0=v0c.data.reshape(-1),tol=tol)
            #split the linear operator into two sites
            res_vec=evecs.T[0]
            combined_sites=qtn.Tensor(res_vec.reshape(v_shape),inds=rix)
            
            if i==rho.L-2:
                split_inds=rix[:int(len(rix)/2+0.5)]
            else:
                split_inds=rix[:int(len(rix)/2)]
            new_sites=combined_sites.split(left_inds=split_inds,
                                        absorb='right',
                                        rtags=['rho',f'I{i+1}'],
                                        ltags=['rho',f'I{i}'],
                                        bond_ind=v0.inner_inds()[0],
                                        **split_opts)
            new_rho1=new_sites.tensors[0]
            new_rho2=new_sites.tensors[1]
            #replace the sites in the density mpo rho
            rho[f'I{i}']=new_rho1.transpose(*rho[f'I{i}'].inds)
            rho[f'I{i+1}']=new_rho2.transpose(*rho[f'I{i+1}'].inds)
            rho_conj=get_rho_conj(rho)
            #update the menvs
            menvs=[]
            for k in range(len(left_mpos)):
                full_network=qtn.TensorNetwork1D((rho,rho_conj,left_mpos[k],right_mpos[k]))
                full_network._L=rho.L
                full_network._site_tag_id='I{}'
                menvs.append(qtn.tensor_dmrg.MovingEnvironment(full_network,begin='left',bsz=2))
        print('\nleft_sweep: ',n) 
        for i in reversed(list(range(1,rho.L-1))):
            progress_bar(i,rho.L-1)
            v0 = (rho[f'I{i}'] & rho[f'I{i+1}']) # get the double site to optimize
            v0c=v0.contract()
            # get the indicies of the linear operator
            rix = v0c.inds
            lix = (rho_conj[f'I{i}'] & rho_conj[f'I{i+1}']).outer_inds() 
            v_shape=v0c.shape
            linop=-1*project_out(rho,rho_conj,rho_g,i,lix,rix)
            for _,menv in enumerate(menvs):
                menv.move_to(i)
                curnet = menv()
                linop_tens=[curnet['lmpo',f'I{i}'],
                            curnet['lmpo',f'I{i+1}'],
                            curnet['rmpo',f'I{i}'],
                            curnet['rmpo',f'I{i+1}'],
                            curnet['_LEFT'],
                            curnet['_RIGHT']]
                linop+=qtn.tensor_core.TNLinearOperator(linop_tens,left_inds=lix,right_inds=rix,optimize='auto-hq')
            
            _,evecs=spla.eigsh(linop,k=1,which=which_eval,v0=v0c.data.reshape(-1),tol=tol)

            res_vec=evecs.T[0]
            combined_sites=qtn.Tensor(res_vec.reshape(v_shape),inds=rix)
            if i==rho.L-2:
                split_inds=rix[:int(len(rix)/2+0.5)]
            else:
                split_inds=rix[:int(len(rix)/2)]
            new_sites=combined_sites.split(left_inds=split_inds,absorb='left',rtags=[f'I{i+1}','rho'],ltags=[f'I{i}','rho'],bond_ind=v0.inner_inds()[0],**split_opts)
            new_rho1=new_sites.tensors[0]
            new_rho2=new_sites.tensors[1]
            rho[f'I{i}']=new_rho1.transpose(*rho[f'I{i}'].inds)
            rho[f'I{i+1}']=new_rho2.transpose(*rho[f'I{i+1}'].inds)
            rho_conj=get_rho_conj(rho)
            menvs=[]
            for k in range(len(left_mpos)):
                full_network=qtn.TensorNetwork1D((rho,rho_conj,left_mpos[k],right_mpos[k]))
                full_network._L=rho.L
                full_network._site_tag_id='I{}'
                menvs.append(qtn.tensor_dmrg.MovingEnvironment(full_network,begin='right',bsz=2))
        eval=sum([menv().contract(all) for menv in menvs])
        print('\n',eval)
        n+=1
    return eval,rho
