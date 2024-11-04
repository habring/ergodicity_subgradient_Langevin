import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image
import os
from pathlib import Path
np.random.seed(0)


F_name = '2d_l2blur' # 'l2sq' '2d_l2blur'
G_name = 'l1'


if F_name == 'l2sq':
    sigma = 0.05
    data_par = 1/(sigma**2)
    niter = int(1e6+100000+1)
    n_parallel_chains = int(1)
    n_parallel_chains = int(100)
    metropolis_check = False
    save_every = 500
    check = 500
    burnin = 1e6

    if metropolis_check:
        tau_list = [1e-6]
        niter = int(3e6+1e6+1)
        burnin = 3e6
    else:
        tau_list = [1e-6,1e-5,1e-4]

    reg_par_list = [10,15,20,30,50]
    reg_par_list = [30]
    folder = Path('results/imaging_examples/denoising/')

elif F_name == '2d_l2blur':
    sigma = 0.01
    data_par_strongly_convex = 0.001
    data_par = 1/(sigma**2)
    niter = int(1e6+500000+1)
    n_parallel_chains = int(1)
    n_parallel_chains = int(100)
    metropolis_check = False
    save_every = 5000
    check = 500
    burnin = 1e6

    ### blur kernel
    sz = 5
    sig = 1
    mu = 0
    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)
    gauss_kernel = ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)

    if metropolis_check:
        tau_list = [1e-6]
        niter = int(3e6+1e6+1)
        burnin = 3e6
    else:
        tau_list = [1e-6,1e-5]

    reg_par_list = [20,30,10,15,50]
    reg_par_list = [20]
    folder = Path('results/imaging_examples/deconvolution/')


for tau in tau_list:
    for reg_par in reg_par_list:

        for name in ['peppers_bw','barbara']:

            folder_im = folder / name / ('data_par_'+str(data_par)+'reg_par_'+str(reg_par))
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
                F = nfun(F_name, npar=data_par, blur_kernel=gauss_kernel, mshift=data_ext, dims = tuple(range(len(u0.shape))))
                eps_reg = nfun('l2sq', npar=data_par_strongly_convex, dims = tuple(range(len(u0.shape))))
                F = nfun(F=F, G=eps_reg)

            K = gradient([*u0.shape,n_parallel_chains])
            vdims = () # anisotropic TV
            dims = (0,1,2)
            G = nfun(G_name, npar=reg_par, dims = dims,vdims=vdims)

            algo = subgradient_Langevin(u0=u0,data=u0,niter =niter,n_parallel_chains=n_parallel_chains,
                                    metropolis_check=metropolis_check,folder=str(folder_im),check=check,save_every=save_every,
                                    F=F, G=G, K=K, tau=tau)

            res = algo.subgrad(burnin=burnin)
