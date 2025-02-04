import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator

def create_hamiltonian(local_hamiltonians):
    l=len(local_hamiltonians)+1
    ham=np.zeros((2**l,2**l),dtype=complex)
    for i in range(l-1):
        ham+=np.kron(np.eye(2**i),np.kron(local_hamiltonians[(i,i+1)],np.eye(2**(l-2-i))))
    return ham

def transform(weight_func,op,local_hamiltonians):
    hamiltonian=create_hamiltonian(local_hamiltonians)
    evals,evecs = np.linalg.eigh(hamiltonian)
    nus=np.subtract.outer(evals,evals)
    return evecs@((evecs.T@op@evecs)*weight_func(nus))@evecs.T

def t_weight(nu):
    return .5*np.tanh(nu/4)

def c_weight(nu):
    return .5/np.cosh(nu/4)

def jump_op(kraus,site_num,length):
   return np.kron(np.eye(2**site_num),np.kron(kraus,np.eye(2**(length-1-site_num))))

def sc_transform(kraus,site_num,local_hamiltonians):
    return transform(c_weight,jump_op(kraus,site_num,len(local_hamiltonians)+1),local_hamiltonians)

def s_transform(kraus,site_num,local_hamiltonians):
    return transform(t_weight,jump_op(kraus,site_num,len(local_hamiltonians)+1),local_hamiltonians)

def disc_part(jump_op, site_num, local_hamiltonians):
    length=len(local_hamiltonians)+1
    a=np.kron(np.eye(2**site_num),np.kron(jump_op,np.eye(2**(length-1-site_num))))
    sca=transform(c_weight,a,local_hamiltonians)
    sa=transform(t_weight,a,local_hamiltonians)
    sc_saa=transform(c_weight,sa@a,local_hamiltonians)
    sc_asa=transform(c_weight,a@sa,local_hamiltonians)
    part1=np.eye(2**length)/2+sc_saa
    part2=(sca@sca)+sc_asa
    scd=(part1-part2)/2
    return scd,sca

def local_operation(scd,sca,vec):
    mat=vec.reshape(int(np.sqrt(len(vec))),int(np.sqrt(len(vec))))
    result=sca@mat@sca-scd@mat-mat@scd
    return result.reshape(len(vec))

def full_discriminant(kraus_ops,local_hamiltonians):
    length=len(local_hamiltonians)+1
    lin_ops=[]
    for i in range(length):
        for jump_op in kraus_ops:
            scd,sca=disc_part(jump_op,i,local_hamiltonians)
            lin_op=LinearOperator((2**(2*length),2**(2*length)),matvec=lambda x: local_operation(scd,sca,x))
            lin_ops.append(lin_op)
    
    return lin_ops
