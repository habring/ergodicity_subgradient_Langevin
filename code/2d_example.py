import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os
from pathlib import Path



## Parameters to set
prox = True
G_name = 'lp_l1' # l1, lp_l1
K_name = 'identity' # identity, grad



d = 2
gt = np.array([0,1])
sigma = 1
u0 = gt+1
niter = int(20001)
n_parallel_chains = int(1e5)
metropolis_check = False
save_every = 100
check = 0
burnin = 1




data_par = 1/(sigma**2)
reg_par = 5
L_F = data_par
L_G = reg_par*np.sqrt(d)
K_nrm = 1


folder = Path('results/2d_example/'+ G_name)

if prox:
    folder = folder / 'prox_grad'
else:
    folder = folder / 'subgrad'

if K_name == 'identity':
    K = identity([d])
elif K_name=='grad':
    K = []
    folder = folder / 'non_trivial_K'

folder.mkdir(parents=True, exist_ok=True)

for tau in [1e-1,1e-2, 1e-3, 1e-4,1e-5]:

    print('START NEW RUN')
    print(f'tau: {tau}')

    if prox:
        algo = subgradient_Langevin(u0=u0,data=gt,niter=niter,n_parallel_chains=n_parallel_chains,
                            metropolis_check=metropolis_check,folder=str(folder)+'/',check=check,save_every=save_every,
                            reg_par=reg_par, data_par=data_par, G_name = G_name, tau=tau,K=K)
        res = algo.prox_grad()

    else:
        algo = subgradient_Langevin(u0=u0,data=gt,niter=niter,n_parallel_chains=n_parallel_chains,
                            metropolis_check=metropolis_check,folder=str(folder)+'/',check=check,save_every=save_every,
                            reg_par=reg_par, data_par=data_par, G_name = G_name, tau=tau,K=K)
        res = algo.subgrad()

    




