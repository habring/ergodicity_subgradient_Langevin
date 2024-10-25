import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os



d=300
b = 1
gt = np.zeros(300)
for i in range(100,200):
	gt[i] = (i-100)/10.

np.random.seed(0)

u0 = gt+np.random.laplace(scale = b, size = gt.shape)



niter = int(1e6+1)
n_parallel_chains = 1000
metropolis_check = True
save_every = 500
check = 0
burnin = 5000
eps = .1



folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/1d_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)




folder = folder+'prox_subgrad/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder+'l1/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder+'metro/'
if not os.path.exists(folder):
	os.makedirs(folder)

for reg_par in [1, .1, 50,10,100,1]:

	data_par = 1/b
	
	L_F = np.sqrt(d)/b
	L_G = reg_par*np.sqrt(d)
	K_nrm = 2

	tau = (-L_F*np.sqrt(2*d)+np.sqrt(L_F**2*2*d+2*eps*L_G**2*K_nrm**2))/(L_G**2*K_nrm**2)
	print(tau)

	algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
							F_name='l1', reg_par=reg_par, data_par=data_par,tau=tau)


	res = algo.prox_subgrad(burnin=burnin, average_distribution=True)
