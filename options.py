

prec=1e-9
max_bond=200
options={}
options['split_opts']={'method':'svd','cutoff':prec,'cutoff_mode':'rel','max_bond':max_bond}
options['tebd_opts']={'tol':prec,'order':4,'progbar':False}
options['comp_opts']={'method':'svd','cutoff':prec,'cutoff_mode':'rel'}

