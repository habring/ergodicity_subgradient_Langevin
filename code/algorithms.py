import numpy as np
from util import *
from tqdm import tqdm
import seaborn as sns
import time
from matpy import tv_denoise

# proximal-subgradient method for sampling from potential F(x)+G(Kx). A proximal step wrt. F and a subgradient step wrt. GoK is used.


class subgradient_Langevin(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        ##Set data
        data_in.u0 = 0 #Direct image input
        data_in.data = 0
        par.niter = 100000
        par.n_parallel_chains = int(500)
        par.tau = 5e-3
        par.check = 500
        par.save_every = 0
        par.metropolis_check = False
        par.folder = './'

        par.F = []
        par.G = []
        par.blur_kernel = np.ones([5,5])/25
        par.K = []
        par_parse(par_in,[par,data_in])

        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)
        data = np.concatenate(par.n_parallel_chains*[data_in.data[...,np.newaxis]],axis = -1)

        # if len(data_in.u0.shape)==1:
        #     if par.K==[]:
        #         par.K = gradient_1d(x_0.shape)

        #     par.F = nfun(par.F_name, npar=par.data_par, mshift=data, dims = tuple(range(len(data_in.u0.shape))))
        #     par.G = nfun(par.G_name, npar=par.reg_par, dims = tuple(range(len(data_in.u0.shape))))

        # elif len(data_in.u0.shape)==2:
        #     if par.K==[]:
        #         par.K = gradient(x_0.shape)

        #     par.F = nfun(par.F_name, npar=par.data_par, blur_kernel=par.blur_kernel, mshift=data, dims = tuple(range(len(data_in.u0.shape))))
        #     vdims = ()#2
        #     dims = (0,1,2)
        #     par.G = nfun(par.G_name, npar=par.reg_par, dims = dims,vdims=vdims)

        self.par = par
        self.data_in = data_in

    # old and proposal are the old iterate and the proposed new one. p_old_given_proposal denotes the log 
    # of the transition probability from the proposal to the old iterate wrt. the proposal transition kernel
    # and p_proposal_given_old the opposite
    def metropolis_check(self, old, proposal, p_old_given_proposal, p_proposal_given_old):
        acceptance_crit = ( - (self.par.F.val(proposal) + self.par.G.val(self.par.K.fwd(proposal))) + 
                            (self.par.F.val(old) + self.par.G.val(self.par.K.fwd(old)))).flatten()
        acceptance_crit = acceptance_crit + p_old_given_proposal - p_proposal_given_old
        # cap acceptance crit at 1 to avoid overflow in exp
        acceptance_crit = np.minimum(1,acceptance_crit)
        acceptance_crit = np.exp(acceptance_crit)
        acceptance = np.random.uniform(low = 0., high = 1.,size = self.par.n_parallel_chains)
        acceptance = (acceptance_crit>acceptance)*1.0
        acc = np.copy(acceptance)
        acceptance = acceptance[np.newaxis,...]
        if len(self.data_in.u0.shape)==2:
            acceptance = acceptance[np.newaxis,...]
        return proposal*acceptance + old*(1 - acceptance),acc


    def prox_grad(self,burnin=0,save_all_iterates=False):
        
        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        if save_all_iterates:
            xx = np.zeros([self.par.niter,*x_0.shape])

        measure_times_of_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_of_this_many_iterates)


        x_k = np.copy(x_0)
        if self.par.save_every:
            np.save(self.par.folder+'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)
        
        t = 0

        for k in range(self.par.niter):

            if k%measure_times_of_this_many_iterates==0 and k>0:
                times[k//measure_times_of_this_many_iterates-1] = t
                t = 0
        
            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        axarr[0].imshow(np.mean(running_mmse,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.mean(x_0,axis=-1),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        axarr[2].imshow(np.mean(running_var_uncentered-running_mmse**2,axis=-1),cmap = 'hot')
                        axarr[2].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(x_k_intermediate,axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[0,1].plot(np.var(x_k_intermediate,axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
            
            
            dt = time.time()

            subgrad_F = self.par.F.subgrad(x_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau * subgrad_F
            gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
            x_k = self.par.G.prox(x_k_intermediate+gauss, ppar = self.par.tau)

            dt = time.time()-dt
            t = t + dt

            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k_intermediate/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k_intermediate**2/(k-burnin+1)


            if self.par.save_every:
                if k%self.par.save_every==0:
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',x_k_intermediate)
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'_mmse.npy',running_mmse)
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'_variance.npy',running_var_uncentered - running_mmse**2)
                    if save_all_iterates:
                        np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                        'all_iterates.npy',xx)

            if save_all_iterates:
                xx[k,...] = x_k[:]


        if save_all_iterates:
            res = {'xk':x_k,'all_iterates':xx,'x0':x_0,'times':times}
        else:
            res = {'xk':x_k,'x0':x_0,'times':times}
        return res

    def subgrad(self,burnin=0,save_all_iterates=False):

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        if save_all_iterates:
            xx = np.zeros([self.par.niter,*x_0.shape])

        if self.par.save_every:
            np.save(self.par.folder+'tau_'+str(self.par.tau)+'_x0.npy',x_0)

        measure_times_of_this_many_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_of_this_many_iterates)

        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)

        t = 0

        for k in range(self.par.niter):

            if k%measure_times_of_this_many_iterates==0 and k>0:
                times[k//measure_times_of_this_many_iterates-1] = t
                t = 0

            if self.par.check and k >= burnin:
                if k%self.par.check==0 and k>0:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].imshow(np.mean(running_mmse,axis=-1),cmap = 'gray')
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations. ')
                        axarr[0,1].imshow(np.mean(x_0,axis=-1),cmap = 'gray')
                        axarr[0,1].set_title('Data')
                        sns.heatmap(np.log(np.squeeze(running_var_uncentered-running_mmse**2)), ax = axarr[1,0])
                        axarr[1,0].set_title('Marginal posterior variances')
                        plt.show()

                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations. '+str(stopping_time_index))
                        axarr[0,1].plot(np.var(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()


            dt = time.time()

            Kx_k = self.par.K.fwd(x_k)
            subgrad_G = self.par.G.subgrad(Kx_k)
            x_k_intermediate = x_k - self.par.tau*(self.par.F.subgrad(x_k) + self.par.K.adj(subgrad_G))
            gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
            x_k_proposed = x_k_intermediate + gauss

            if self.par.metropolis_check:
                Kx_k_proposed = self.par.K.fwd(x_k_proposed)
                subgrad_G = self.par.G.subgrad(Kx_k_proposed)
                x_k_proposed_reverse = x_k_proposed - self.par.tau*(self.par.F.subgrad(x_k_proposed) + self.par.K.adj(subgrad_G))

                p_old_given_proposal = -1/(4*self.par.tau)*np.sum((x_k-x_k_proposed_reverse)**2,axis=tuple(range(len(x_k.shape)-1)))
                p_proposal_given_old = -1/(4*self.par.tau)*np.sum(gauss**2,axis=tuple(range(len(x_k.shape)-1)))

                x_k,_ = self.metropolis_check(x_k, x_k_proposed, p_old_given_proposal, p_proposal_given_old)
            else:
                x_k = x_k_proposed


            dt = time.time() - dt
            t = t + dt
            
            if save_all_iterates:
                xx[k,...] = x_k[:]


            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k**2/(k-burnin+1)

            if self.par.save_every:
                if k%self.par.save_every==0:
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',x_k)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'_mmse.npy',running_mmse)
                    np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'_variance.npy',running_var_uncentered.squeeze() - running_mmse.squeeze()**2)
                    if save_all_iterates:
                        np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                        'all_iterates.npy',xx)


            if k==0:
                np.save(self.par.folder+'tau_'+str(self.par.tau)+'_'
                                    'iter_0.npy',x_0)

        if save_all_iterates:
            res = {'xk':x_k,'all_iterates':xx,'x0':x_0,'times':times}
        else:
            res = {'xk':x_k,'x0':x_0,'times':times}

        return res


class MYULA(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        data_in.u0 = 0 #Direct image input
        data_in.data = 0
        par.niter = 100000
        par.n_parallel_chains = int(500)
        par.tau = 5e-3
        par.check = 500
        par.save_every = 0
        par.metropolis_check = False
        par.folder = './'
        par.ld = 1

        par.F = []
        par.G = []
        par.blur_kernel = np.ones([5,5])/25
        par.K = []
        par_parse(par_in,[par,data_in])


        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)

        # if len(data_in.u0.shape)==1:
        #     if par.K==[]:
        #         par.K = gradient_1d(x_0.shape)

        #     par.F = nfun(par.F_name, npar=par.data_par, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
        #     par.G = nfun(par.G_name, npar=par.reg_par, dims = tuple(range(len(data_in.u0.shape))))

        # elif len(data_in.u0.shape)==2:
        #     if par.K==[]:
        #         par.K = gradient(x_0.shape)

        #     par.n_parallel_chains=1

        #     par.F = nfun(par.F_name, npar=par.data_par, blur_kernel=par.blur_kernel, mshift=x_0, dims = tuple(range(len(data_in.u0.shape))))
        #     vdims = ()#2
        #     dims = (0,1,2)
        #     par.G = nfun(par.G_name, npar=par.reg_par, dims = dims,vdims=vdims)
        
        self.par = par
        self.data_in = data_in

# only for images
    def sample(self,burnin = 0, prox_iter=-1):

        measure_times_of_this_may_iterates = 1000
        times = np.zeros(self.par.niter//measure_times_of_this_may_iterates)

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        
        ld_string = '%s' % float('%.1g' % self.par.ld)

        if self.par.save_every:
            np.save(self.par.folder+'ld_'+ld_string+'_'
                                        'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        running_mmse = np.zeros(x_k.shape)
        running_var_uncentered = np.zeros(x_k.shape)


        t = 0
        for k in range(self.par.niter):

            if k%measure_times_of_this_may_iterates==0 and k>0:
                times[k//measure_times_of_this_may_iterates-1] = t
                t = 0

            if self.par.check:
                if k%self.par.check==0 and k>burnin:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].imshow(np.mean(running_mmse,axis=-1),cmap = 'gray')
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations. ')
                        axarr[0,1].imshow(np.mean(x_0,axis=-1),cmap = 'gray')
                        axarr[0,1].set_title('Data')
                        sns.heatmap(np.log(np.squeeze(running_var_uncentered-running_mmse**2)), ax = axarr[1,0])
                        axarr[1,0].set_title('Marginal posterior variances')
                        plt.show()

            dt = time.time()

            grad_F = self.par.F.subgrad(x_k)
            if self.par.K.name=='identity':
                prox_G_K = self.par.G.prox(x_k,ppar=self.par.ld)
            elif 'gradient' in self.par.K.name:
                prox_G_K = tv_denoise(u0=x_k,ld = 1/(self.par.G.npar*self.par.ld), niter = prox_iter).u
                
            else:
                raise NotImplementedError('Operator K not implemented')

            x_k_intermediate = x_k*(1-self.par.tau/self.par.ld) - self.par.tau*grad_F + prox_G_K*(self.par.tau/self.par.ld) 
            x_k = x_k_intermediate + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)


            dt = time.time()-dt
            t += dt

            if  k>= burnin:
                running_mmse = running_mmse*(k-burnin)/(k-burnin+1) + x_k/(k-burnin+1)
                running_var_uncentered = running_var_uncentered*(k-burnin)/(k-burnin+1) + x_k**2/(k-burnin+1)


            if self.par.save_every:


                if k%self.par.save_every==0:
                    np.save(self.par.folder+'ld_'+ld_string+'_'
                                        'tau_'+str(self.par.tau)+'_computation_times.npy',times)

                if k%self.par.save_every==0 and k>=burnin:
                    np.save(self.par.folder+'ld_'+ld_string+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',np.squeeze(x_k))


                    np.save(self.par.folder+'ld_'+ld_string+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'_mmse.npy',np.mean(running_mmse,axis=-1))
                    np.save(self.par.folder+'ld_'+ld_string+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'_variance.npy',running_var_uncentered.squeeze()- running_mmse.squeeze()**2)


        res = {'xk':x_k,'x0':x_0,'times':times}
        return res
