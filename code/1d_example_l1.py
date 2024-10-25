import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os



d = 1

gt = np.array([1])
sigma = 1
u0 = gt

niter = int(50000)
n_parallel_chains = int(1e6)
metropolis_check = False
save_every = 1000
check = 0
burnin = 1


folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/1d_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder+'l1/'
if not os.path.exists(folder):
	os.makedirs(folder)


data_par = 1/(sigma**2)
reg_par = 1
L_F = data_par
L_G = reg_par*np.sqrt(d)
K_nrm = 1

prox = False


if prox:
    folder = folder + 'prox_subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)
else:
    folder = folder + 'subgrad/'
    if not os.path.exists(folder):
        os.makedirs(folder)

for tau in [1e-2, 1e-3, 1e-4]:

    if prox:

        algo = subgradient_Langevin(u0=u0,F_name = 'l1',niter=niter,n_parallel_chains=n_parallel_chains,
                                metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par, tau=tau,K=identity([d]))
        res = algo.prox_subgrad(average_distribution=True,burnin=burnin)
    else:
        algo = subgradient_Langevin(u0=u0,F_name = 'l1',niter=niter,n_parallel_chains=n_parallel_chains,
                                metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par, tau=tau,K=identity([d]))
        res = algo.subgrad(burnin=burnin)