import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os



d = 300
gt = np.zeros(d)
for i in range(100,200):
	gt[i] = (i-100)/10.

np.random.seed(0)

sigma = 1
u0 = gt+sigma*np.random.normal(size = gt.shape)



niter = int(1e6+1)
n_parallel_chains = 1000
metropolis_check = True
save_every = 500
check = 0
eps = .1

prox = True

folder = 'results/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = 'results/1d_ex/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder + 'grad_subgrad/'
if not os.path.exists(folder):
	os.makedirs(folder)

folder = folder + 'metro/'
if not os.path.exists(folder):
	os.makedirs(folder)

for reg_par in [1,0.1,10,50,100]:


	data_par = 1/(sigma**2)


	m = 1/(sigma**2)
	L_F = 1/(sigma**2)
	L_G = reg_par*np.sqrt(d)
	K_nrm = 2



	tau = (eps*m)/(2*L_F*d + L_G**2*K_nrm**2)
	print(tau)
	algo = subgradient_Langevin(u0=u0,niter =niter,n_parallel_chains=n_parallel_chains,
							metropolis_check=metropolis_check,folder=folder,check=check,save_every=save_every,
							reg_par=reg_par, data_par=data_par, tau=tau)

	res = algo.grad_subgrad()
