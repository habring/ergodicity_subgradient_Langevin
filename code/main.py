import numpy as np
import matplotlib.pyplot as plt
from util import *
from algorithms import *
from PIL import Image




# gt = imread('images/lena.png')
# u0 = gt+0.1*(gt.max()-gt.min())*np.random.normal(size = gt.shape)
# #u0 = gt+0.05*(gt.max()-gt.min())*np.random.normal(size = gt.shape)
# u0 = u0[100:150,100:150]



folder = './'
gt = np.zeros(500)
for i in range(100,225):
	gt[i] = i-100
#gt[:] = .5

u0 = gt+0.1*(gt.max()-gt.min())*np.random.normal(size = gt.shape)
u0 = gt+10*np.random.normal(size = gt.shape)



niter = 10000
n_parallel_chains = 5000

algo = subgradient_Langevin(u0=u0,niter =niter)


for tau in [1]:#[1e-3, 1e-5,5e-5,1e-4,5e-4,1e-2]:
	for reg_par in [1]:#[5000,100,1,10,50]:
		fac = .01*.5
		reg_par *= fac
		data_par = fac

		#res = metropolis_hastings1d(u0=u0,niter =niter)
		res = algo.subgrad()

		Sn = res['Sn']
		x0 = res['x0']

		f, axarr = plt.subplots(2,2)
		axarr[0,0].plot(Sn)
		axarr[0,1].plot(Sn)
		axarr[1,0].plot(x0)
		plt.show()
		plt.savefig(folder+'test.png')

		#np.save(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.npy',xk)
		#plt.savefig(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.png')













# niter = 50000
# n_parallel_chains = 10000
# for tau in [1e-3, 1e-5,5e-5,1e-4,5e-4,1e-2]:
# 	for ld in [500,100,1,10,50]:
# 		res = grad_subgrad(u0=u0,niter =niter, n_parallel_chains = n_parallel_chains, tau=tau)

# 		xk = res['xk']
# 		x0 = res['x0']

# 		f, axarr = plt.subplots(2,2)
# 		axarr[0,0].plot(np.mean(xk,axis=-1))
# 		axarr[0,0].set_title('mean')
# 		axarr[0,1].plot(np.var(xk,axis=-1))
# 		axarr[0,1].set_title('var')
# 		axarr[1,0].plot(np.mean(x0,axis=-1))
# 		axarr[1,0].set_title('data')

# 		np.save(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.npy',xk)
# 		plt.savefig(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.png')


















# gt = imread('images/lena.png')
# gt = imread('images/lena.png')
# folder = '2d_ex/'
# u0 = gt[100:150,100:150]
# u0 = u0+0.1*(u0.max()-u0.min())*np.random.normal(size = u0.shape)


# niter = 50000
# n_parallel_chains = 5000
# for tau in [1e-2, 1e-5,5e-5,1e-4,5e-4,1e-2]:
# 	for ld in [500,100,1,10,50]:
# 		res = grad_subgrad(u0=u0,niter =niter, n_parallel_chains = n_parallel_chains, tau=tau,ld = ld)

# 		xk = res['xk']
# 		x0 = res['x0']

# 		f, axarr = plt.subplots(2,2)
# 		axarr[0,0].imshow(np.mean(xk,axis=-1),cmap='gray')
# 		axarr[0,0].set_title('mean')
# 		sns.heatmap(np.log(np.var(xk,axis=-1)), ax = axarr[0,1])
# 		axarr[0,1].set_title('var')
# 		axarr[1,0].imshow(np.mean(x0,axis=-1),cmap='gray')
# 		axarr[1,0].set_title('data')

# 		np.save(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.npy',xk)
# 		plt.savefig(folder+'tau='+str(tau)+'_regpar='+str(ld)+'_niter='+str(niter)+'_n_chains='+str(n_parallel_chains)+'.png')

