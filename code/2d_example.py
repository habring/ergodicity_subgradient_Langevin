import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os
from pathlib import Path
import pandas as pd


## Choose your experimental setting
method = 'subgrad' #'subgrad' 'prox' 'myula'
G_name = 'lp_l1' # l1, lp_l1
K_name = 'grad' # identity, grad
F_name = 'l2sq'


# parameters to set
d = 2
gt = np.array([0,1])
sigma = 1
u0 = gt+1
niter = int(5001)
niter = int(20001)
# n_parallel_chains = int(1e4)
n_parallel_chains = int(1)
metropolis_check = False
save_every = 100
check = 0
data_par = 1/(sigma**2)
reg_par = 5
tau_list = [1e-5,1e-4, 1e-3, 1e-2,1e-1]
save_all_iterates = True

# results folder
folder = Path('results/2d_example/'+ G_name)
if method=='prox':
    folder = folder / 'prox_grad'
elif method=='subgrad':
    folder = folder / 'subgrad'
elif method=='myula':
    folder = folder / 'myula'


# define the potential function based on user choices
if K_name == 'identity':
    K = identity([d])
    folder = folder / 'no_K' / f'n_chains_{n_parallel_chains}'
elif K_name=='grad':
    K = gradient_1d([*gt.shape,n_parallel_chains])
    folder = folder / 'non_trivial_K' / f'n_chains_{n_parallel_chains}'

data_extended = np.concatenate(n_parallel_chains*[gt[...,np.newaxis]],axis = -1)
F = nfun(F_name, npar=data_par, mshift=data_extended, dims = tuple(range(len(gt.shape))))
G = nfun(G_name, npar=reg_par, dims = tuple(range(len(gt.shape))))


folder = folder / ('data_par_'+str(data_par)+'reg_par_'+str(reg_par))
folder.mkdir(parents=True, exist_ok=True)
if metropolis_check:
    folder = folder / 'metropolis'
    folder.mkdir(parents=True, exist_ok=True)

computation_times = pd.DataFrame(columns = ['tau','ld','time_per_1000_iter'])

# start simulation
for tau in tau_list:

    print('START NEW RUN')
    print(f'tau: {tau}')

    if method=='prox':
        algo = subgradient_Langevin(u0=u0,data=gt,niter=niter,n_parallel_chains=n_parallel_chains,
                            metropolis_check=metropolis_check,folder=str(folder)+'/',check=check,save_every=save_every,
                            F=F,G = G, tau=tau,K=K)
        res = algo.prox_grad(save_all_iterates=save_all_iterates)
        row = {'tau':tau,'ld':-1,'time_per_1000_iter':np.mean(res['times'])}
        computation_times = computation_times._append(row,ignore_index=True)

    elif method=='subgrad':
        algo = subgradient_Langevin(u0=u0,data=gt,niter=niter,n_parallel_chains=n_parallel_chains,
                            metropolis_check=metropolis_check,folder=str(folder)+'/',check=check,save_every=save_every,
                            F=F, G=G, tau=tau,K=K)
        res = algo.subgrad(save_all_iterates=save_all_iterates)
        row = {'tau':tau,'ld':-1,'time_per_1000_iter':np.mean(res['times'])}
        computation_times = computation_times._append(row,ignore_index=True)

    elif method=='myula':
        ld = tau/(1-data_par*tau) # condition from Pereyra paper
        algo = MYULA(u0=u0,data=gt,niter=niter,n_parallel_chains=n_parallel_chains,
                            metropolis_check=metropolis_check,folder=str(folder)+'/',check=check,save_every=save_every,
                            F=F, G=G, tau=tau,K=K,ld=ld)
        res = algo.sample()
        row = {'tau':tau,'ld':ld,'time_per_1000_iter':np.mean(res['times'])}
        computation_times = computation_times._append(row,ignore_index=True)
    else:
        raise NotImplementedError('Method not implemented')

    try:
        np.save(f'{str(folder)}/tau_{tau}',res['times'])
    except:
        print('No sampling done')

print(method)
print(f'K: {K_name}')
print(computation_times)

