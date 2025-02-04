import numpy as np
import quimb.tensor as qtn

def mps_to_mpo(mps):
    tensor_order=[int(tag.replace('I','')) for tag in list(mps.tags)]  
    real_order=[tensor_order.index(i) for i in range(len(mps.arrays))]      
    mps.permute_arrays(shape='lrp')
    arrays=[mps.arrays[i] for i in real_order]
    mpo_arrays=[arr.reshape(arr.shape[:-1]+(int(np.sqrt(arr.shape[-1])),int(np.sqrt(arr.shape[-1])))) for arr in arrays]
    return qtn.tensor_1d.MatrixProductOperator(mpo_arrays)

def mpo_to_mps(mpo):
    tensor_order=[int(tag.replace('I','')) for tag in list(mpo.tags)]  
    real_order=[tensor_order.index(i) for i in range(len(mpo.arrays))]      
    mpo.permute_arrays(shape='lrud')
    arrays=[mpo.arrays[i] for i in real_order]
    mps_arrays=[arr.reshape(arr.shape[:-2]+(np.prod(arr.shape[-2:]),)) for arr in arrays]
    return qtn.tensor_1d.MatrixProductState(mps_arrays)

def mirror_mpo(mpo):
    mpo.permute_arrays(shape='lrud')
    mirrored_arrays = [mpo.arrays[-1]]
    for arr in list(reversed(mpo.arrays))[1:-1]:
        mirrored_arrays.append(arr.transpose(1,0,2,3))
    mirrored_arrays.append(mpo.arrays[0])
    return qtn.tensor_1d.MatrixProductOperator(mirrored_arrays)


def embed_to_identity(mpo,first_site,full_length):
    partial_length=len(mpo.arrays)
    mpo_arrays=[]
    mpo.permute_arrays(shape='lrud')
    if first_site==0:
        mpo_arrays+=mpo.arrays[:-1]
        mpo_arrays+=[mpo.arrays[-1].reshape(mpo.arrays[-1].shape[0],1,*mpo.arrays[-1].shape[1:])]
        mpo_arrays+=[np.eye(4).reshape(1,1,4,4)]*(full_length-partial_length-1)
        mpo_arrays+=[np.eye(4).reshape(1,4,4)]

    elif first_site==full_length-partial_length:
        mpo_arrays+=[np.eye(4).reshape(1,4,4)]
        mpo_arrays+=[np.eye(4).reshape(1,1,4,4)]*(first_site-1)
        mpo_arrays+=[mpo.arrays[0].reshape(1,*mpo.arrays[0].shape)]
        mpo_arrays+=mpo.arrays[1:]
    else:
        mpo_arrays+=[np.eye(4).reshape(1,4,4)]
        mpo_arrays+=[np.eye(4).reshape(1,1,4,4)]*(first_site-1)
        mpo_arrays+=[mpo.arrays[0].reshape(1,*mpo.arrays[0].shape)]
        mpo_arrays+=mpo.arrays[1:-1]
        mpo_arrays+=[mpo.arrays[-1].reshape(mpo.arrays[-1].shape[0],1,*mpo.arrays[-1].shape[1:])]
        mpo_arrays+=[np.eye(4).reshape(1,1,4,4)]*(full_length-partial_length-first_site-1)
        mpo_arrays+=[np.eye(4).reshape(1,4,4)]
    return qtn.tensor_1d.MatrixProductOperator(mpo_arrays)
