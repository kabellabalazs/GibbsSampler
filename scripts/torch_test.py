import quimb.tensor as qtn
import numpy as np
from weights import ch_weight,sh_weight
from transformations import s_transform
from mps_generators import kraus_mps
import time
use_torch=True


L=10 #use even numbers
J=1
h=1.5
tol=1e-8
delta_t=.08
final_t=3.2
a=33 #0.05 33 #0.1 #32
num_steps=int(final_t/delta_t)

options={}
options['split_opts']={'method':'svd','cutoff':tol,'cutoff_mode':'rel','max_bond':100}
options['tebd_opts']={'tol':tol,'order':4,'progbar':False}
options['comp_opts']={'method':'svd','cutoff':tol,'cutoff_mode':'rel'}



I=np.eye(2)
X=np.array([[0,1],[1,0]])
Z=np.array([[1,0],[0,-1]])

h2=-J*(np.kron(np.kron(Z,I),np.kron(Z,I))-np.kron(np.kron(I,Z),np.kron(I,Z)))
h1=h*(np.kron(X,I)-np.kron(I,X))

ham_tp=qtn.LocalHam1D(L,H2=h2,H1=h1,cyclic=False)
ham_tm=qtn.LocalHam1D(L,H2=-h2,H1=-h1,cyclic=False)
kraus=kraus_mps(Z,int(L/2),L)
weight_func=lambda x: sh_weight(x,a)
print('Starting S-transform')
start_time = time.time()
sa = s_transform(kraus, weight_func, delta_t, num_steps, ham_tp, ham_tm, use_torch=use_torch, **options)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
