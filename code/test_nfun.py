import numpy as np
import util
import matpy as mp










x_dim = 100
y_dim = 100
channels = 100

grad_mp = mp.gradient([x_dim,y_dim])
grad_util = util.gradient([x_dim,y_dim,channels])


x = np.random.rand(x_dim,y_dim,channels)
y = np.random.rand(x_dim,y_dim)
y = np.concatenate(channels*[y[...,np.newaxis]],axis = -1)

# test gradient operator

grad_mp_val = np.zeros([x_dim,y_dim,2,channels])
grad_mp_adj_val = np.zeros([x_dim,y_dim,channels])


# compute with mp separately
for i in range(100):
	grad_mp_val[:,:,:,i] = grad_mp.fwd(x[:,:,i])
	grad_mp_adj_val[:,:,i] = grad_mp.adj(grad_mp_val[:,:,:,i])


# compute with util in parallel
grad_util_val = grad_util.fwd(x)
grad_util_adj_val = grad_util.adj(grad_mp_val)

print(np.sum(np.abs(grad_mp_val-grad_util_val)))
print(np.sum(np.abs(grad_util_adj_val-grad_mp_adj_val)))



# test 1 norm

for ntype in ['l1','l2sq']:

	util_l1 = util.nfun(ntype,npar=np.pi,mshift=y,dims = (0,1))
	mp_l1 = mp.nfun(ntype,npar=np.pi,mshift=y[...,0])

	# value:
	norm_mp_val = np.zeros(channels)

	# compute with mp separately
	for i in range(100):
		norm_mp_val[i] = mp_l1.val(x[:,:,i])


	# compute with util in parallel
	norm_util_val = util_l1.val(x)

	print(np.sum(np.abs(norm_mp_val-norm_util_val)))

	# prox
	norm_mp_prox = np.zeros(x.shape)

	# compute with mp separately
	for i in range(100):
		norm_mp_prox[:,:,i] = mp_l1.prox(x[:,:,i],ppar = np.pi)


	# compute with util in parallel
	norm_util_prox = util_l1.prox(x,ppar = np.pi)

	print(np.sum(np.abs(norm_mp_prox-norm_util_prox)))

	if ntype =='l2sq':
		# grad
		norm_mp_grad = np.zeros(x.shape)

		# compute with mp separately
		for i in range(100):
			norm_mp_grad[:,:,i] = mp_l1.grad(x[:,:,i])


		# compute with util in parallel
		norm_util_grad = util_l1.subgrad(x)

		print(np.sum(np.abs(norm_mp_grad-norm_util_grad)))


ntype='l1'
x = grad_util.fwd(x)
y = np.random.rand(*x[:,:,:,0].shape)
y = np.concatenate(channels*[y[...,np.newaxis]],axis = -1)
util_l1 = util.nfun(ntype,npar=np.pi,mshift=y,vdims=(2),dims = (0,1))
mp_l1 = mp.nfun(ntype,npar=np.pi,vdims=(2),mshift=y[...,0])

# value:
norm_mp_val = np.zeros(channels)

# compute with mp separately
for i in range(100):
	norm_mp_val[i] = mp_l1.val(x[:,:,:,i])


# compute with util in parallel
norm_util_val = util_l1.val(x)

print(np.sum(np.abs(norm_mp_val-norm_util_val)))

# prox
norm_mp_prox = np.zeros(x.shape)

# compute with mp separately
for i in range(100):
	norm_mp_prox[:,:,:,i] = mp_l1.prox(x[:,:,:,i],ppar = np.pi)


# compute with util in parallel
norm_util_prox = util_l1.prox(x,ppar = np.pi)

print(np.sum(np.abs(norm_mp_prox-norm_util_prox)))