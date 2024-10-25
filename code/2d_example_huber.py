import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from util import *
from algorithms import *



niter = int(50000)
n_parallel_chains = int(1e4)
metropolis_check = False
save_every = 500
check = 0
burnin=0
method = 'prox'
method = 'grad'
method = 'subgrad'



tau_list = [1e-5,1e-4,1e-3,1e-2]


d = 2
gt = np.array([-1,1])
np.random.seed(0)

sigma = 1
u0 = gt

data_par = 1/(sigma**2)
reg_par = 5
L_F = data_par
L_G = reg_par*np.sqrt(d)
K_nrm = 2


folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/2d_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder+'huber/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder + method + '/'
if not os.path.exists(folder):
    os.makedirs(folder)


if method == 'prox':

    

    for tau in tau_list:
        algo = subgradient_Langevin(u0=u0,F_name = 'huber',niter=niter,n_parallel_chains=n_parallel_chains,
                                metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par, tau=tau)
        res = algo.prox_subgrad(average_distribution=True,burnin=burnin)


elif method == 'grad':


    for tau in tau_list:
        algo = subgradient_Langevin(u0=u0,F_name = 'huber',niter=niter,n_parallel_chains=n_parallel_chains,
                                metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
                                reg_par=reg_par, data_par=data_par, tau=tau)
        res = algo.grad_subgrad(average_distribution=True,burnin=burnin)



elif method == 'subgrad':

    
    for tau in tau_list:
        algo = subgradient_Langevin(u0=u0,niter=niter,n_parallel_chains=n_parallel_chains,metropolis_check=metropolis_check,
                                    folder=folder,check=check,save_every=save_every,
                                    F_name='huber', 
                                    reg_par=reg_par, data_par=data_par,tau=tau)

        res = algo.subgrad(burnin=burnin)


else:
    print('Method not valid')



