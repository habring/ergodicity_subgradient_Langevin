import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os
from pathlib import Path
import pandas as pd
np.random.seed(0)


method = 'subgrad' # 'subgrad' 'myula'
F_name = '2d_l2blur' # 'l2sq' '2d_l2blur'
G_name = 'l1'
metropolis_check = True
save_every = 500
check = 0
n_parallel_chains = int(1)

burnin = int(5e5)
niter = burnin + int(5e5)

tau_list = [1e-6,1e-5,1e-4]
ld = 1e-4

if metropolis_check:
    burnin = int(1e6)
    niter = burnin + int(1e6)

if F_name == 'l2sq':
    sigma = 0.05
    data_par = 1/(sigma**2)
    reg_par = 30
    folder = Path('results/imaging_examples/denoising/') / method

elif F_name == '2d_l2blur':
    sigma = 0.01
    data_par_strongly_convex_list = [1e-3, 1e-1]
    data_par = 1/(sigma**2)

    ### blur kernel
    sz = 5
    sig = 1
    mu = 0
    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)
    gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)

    reg_par = 20
    folder = Path('results/imaging_examples/deconvolution/') / method

if metropolis_check:
    folder = folder / 'metro'

if metropolis_check:
    tau_list = [1e-6]

computation_times = pd.DataFrame(columns = ['tau','ld','time_per_1000_iter'])
print(method)
for tau in tau_list:
    for name in ['peppers_bw','barbara']:
        for data_par_strongly_convex in data_par_strongly_convex_list:

            folder_im = folder / name / ('data_par_'+str(data_par)+'reg_par_'+str(reg_par)+'delta_'+str(data_par_strongly_convex))
            folder_im.mkdir(parents=True, exist_ok=True)

            # initialize potential
            
            if F_name=='l2sq':
                u0 = np.load('images/'+name+'_noisy.npy')
                data_ext = np.concatenate(n_parallel_chains*[u0[...,np.newaxis]],axis = -1)
                F = nfun(F_name, npar=data_par, mshift=data_ext, dims = tuple(range(len(u0.shape))))

            elif F_name=='2d_l2blur':
                u0 = np.load('images/'+name+'.npy')
                u0 = scipy.signal.convolve2d(u0,gauss_kernel,mode='same',boundary='wrap')
                u0 = u0+sigma*np.random.normal(size=u0.shape)
                data_ext = np.concatenate(n_parallel_chains*[u0[...,np.newaxis]],axis = -1)
                F = nfun(F_name, npar=data_par, eps = data_par_strongly_convex, blur_kernel=gauss_kernel, mshift=data_ext, dims = tuple(range(len(u0.shape))))

            K = gradient([*u0.shape,n_parallel_chains])
            vdims = () # anisotropic TV
            dims = (0,1,2)
            G = nfun(G_name, npar=reg_par, dims = dims,vdims=vdims)


            if method=='subgrad':
                algo = subgradient_Langevin(u0=u0,data=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                        metropolis_check=metropolis_check,folder=str(folder_im)+'/',check=check,save_every=save_every,
                                        F=F, G=G, K=K, tau=tau)

                res = algo.subgrad(burnin=burnin)
                row = {'tau':tau,'ld':-1,'time_per_1000_iter':np.mean(res['times'])}
                computation_times = computation_times._append(row,ignore_index=True)
            elif method=='myula':
                algo = MYULA(u0=u0,data=u0,niter=niter,n_parallel_chains=n_parallel_chains,
                                    metropolis_check=False,folder=str(folder_im)+'/',check=check,save_every=save_every,
                                    F=F, G=G, tau=tau,K=K,ld=ld)

                res = algo.sample(burnin=burnin)
                row = {'tau':tau,'ld':ld,'time_per_1000_iter':np.mean(res['times'])}
                computation_times = computation_times._append(row,ignore_index=True)

            else:
                raise NotImplementedError('Method not implemented')

            try:
                np.save(f'{str(folder_im)}/tau_{tau}',res['times'])
            except:
                print('No sampling done')