import numpy as np
from util_torch import *
from tqdm import tqdm
import seaborn as sns
import time
import torch

# proximal-subgradient method for sampling from potential F(x)+G(Kx). A proximal step wrt. F and a subgradient step wrt. GoK is used.
device = 'cuda'

class subgradient_Langevin(object):
    def __init__(self,**par_in):

        par = parameter({})
        data_in = data_input({})

        ##Set data
        data_in.u0 = 0 #Direct image input

        par.niter = 100000
        par.n_parallel_chains = int(500)
        fac = .01*.5
        par.reg_par = fac*200
        par.data_par = fac
        par.noise = 0
        par.tau = 5e-3
        par.check = 500
        par.save_every = 0
        par.metropolis_check = False
        par.folder = './'

        #Data type: {'l1','l2sq','inpaint','I0'}
        #par.F_name='l2sq'
        par.F_name='l2sq'
        par.G_name='l1'
        par.K = []
        par_parse(par_in,[par,data_in])


        x_0 = np.concatenate(par.n_parallel_chains*[data_in.u0[...,np.newaxis]],axis = -1)

        if len(data_in.u0.shape)==1:
            if par.K==[]:
                par.K = gradient_1d(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, mshift=torch.tensor(x_0,device=device), dims = tuple(range(len(data_in.u0.shape))))
            par.G = nfun(par.G_name, npar=par.reg_par, mshift=torch.tensor(0.0,device=device), dims = tuple(range(len(data_in.u0.shape))))

        elif len(data_in.u0.shape)==2:
            if par.K==[]:
                par.K = gradient(x_0.shape)

            par.F = nfun(par.F_name, npar=par.data_par, mshift=torch.tensor(x_0,device=device), dims = tuple(range(len(data_in.u0.shape))))
            vdims = 2
            par.G = nfun(par.G_name, npar=par.reg_par, mshift=torch.tensor(0.0,device=device), dims = tuple(range(len(data_in.u0.shape))),vdims=vdims)


        self.par = par
        self.data_in = data_in


    def metropolis_check(self, old, proposal, p_old_proposal, p_proposal_old):
        acceptance_crit = torch.exp( - (self.par.F.val(proposal) + self.par.G.val(self.par.K.fwd(proposal))) + (self.par.F.val(old) + self.par.G.val(self.par.K.fwd(old)))).flatten()
        acceptance_crit = acceptance_crit*p_proposal_old/p_old_proposal
        acceptance_rate = torch.rand(size = tuple([self.par.n_parallel_chains]), device = old.get_device())

        acceptance = (acceptance_crit>acceptance_rate)*1.0
        acceptance = acceptance[np.newaxis,...]
        if len(self.data_in.u0.shape)==2:
            acceptance = acceptance[np.newaxis,...]

        return proposal*acceptance + old*(1 - acceptance)

    def grad_subgrad(self):

        
        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)
        for k in range(self.par.niter):

            if self.par.check:
                if k%self.par.check==0 and k>0:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(3)
                        axarr[0].imshow(np.mean(x_k_intermediate,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[1].imshow(np.mean(x_0,axis=-1),cmap = 'gray')
                        axarr[1].set_title('Initial')
                        sns.heatmap(np.log(np.var(x_k_intermediate,axis=-1)), ax = axarr[2])
                        axarr[2].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(x_k_intermediate,axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[0,1].plot(np.var(x_k_intermediate,axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
            

            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx via prox
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau * self.par.K.adj(subgrad_G)
            
            if self.par.metropolis_check:

                gauss = np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)
                x_k_prop = x_k_intermediate - self.par.tau*self.par.F.subgrad(x_k_intermediate) + gauss

                q_x_y = np.exp( -(gauss*gauss).sum( axis=tuple(range(len(gauss.shape)-1)) )/(4*self.par.tau) )

                Kx_k_prop = self.par.K.fwd(x_k_prop)
                # compute subgradient of G at point Kx via prox
                subgrad_G = self.par.G.subgrad(Kx_k_prop)
                # subgrad step
                x_k_prop_rev = x_k_prop - self.par.tau * self.par.K.adj(subgrad_G)
                x_k_prop_rev = x_k_prop_rev - self.par.tau*self.par.F.subgrad(x_k_prop_rev)

                q_y_x = np.exp( -((x_k_prop_rev-x_k)**2).sum( axis=tuple(range(len(x_k.shape)-1)) ) /(4*self.par.tau))

                x_k = self.metropolis_check(x_k,x_k_prop,q_x_y,q_y_x)
                x_k_intermediate = x_k

            else:
                x_k = x_k_intermediate - self.par.tau*self.par.F.subgrad(x_k_intermediate) + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)

            if self.par.save_every:
                if k%self.par.save_every==0:
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy',x_k_intermediate)

        return {'x_k_interm':x_k_intermediate,'x0':x_0}


    def prox_subgrad(self,average_distribution=False,burnin=0):
        device = 'cuda'
        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)
        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)

        x_0 = torch.tensor(x_0, device = device)
        x_k = x_0.clone().detach()
        
        stopping_times = torch.randint(low = burnin+1, high = self.par.niter, size = tuple([self.par.n_parallel_chains]), device = device)
        stopping_times = torch.sort(stopping_times, dim = -1)[0]

        sample = torch.zeros([*x_0.shape,stopping_times.shape[-1]],device = device)
        stopping_time_index = 0
        stopping_condition = False

        normal_tensor = torch.zeros(x_0.shape,device = device)

        

        for k in range(self.par.niter):
            if self.par.check:
                x_k_intermediate_plot = x_k_intermediate.detach().cpu().numpy()

                if k%self.par.check==0 and k>0:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(2)
                        axarr[0].imshow(np.mean(x_k_intermediate_plot,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations')
                        sns.heatmap(np.log(np.var(x_k_intermediate_plot,axis=-1)), ax = axarr[1])
                        axarr[1].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(x_k_intermediate_plot,axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations')
                        axarr[0,1].plot(np.var(x_k_intermediate_plot,axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
            

            if average_distribution:
                picks = 1.0*(stopping_times==k)
                picks = torch.unsqueeze(picks,dim=0)
                if len(self.data_in.u0.shape)==2:
                    picks = torch.unsqueeze(picks,dim=0)
                sample = sample + picks*torch.unsqueeze(x_k,dim=-1)


            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx via prox
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k_intermediate = x_k - self.par.tau * self.par.K.adj(subgrad_G)

            if self.par.metropolis_check:
                gauss = np.sqrt(2*self.par.tau)*normal_tensor.normal_()
                x_k_prop = self.par.F.prox(x_k_intermediate, ppar = self.par.tau) + gauss

                q_x_y = torch.exp( -torch.sum((gauss*gauss), dim =tuple(range(len(gauss.shape)-1)) )/(4*self.par.tau) )

                Kx_k_prop = self.par.K.fwd(x_k_prop)
                # compute subgradient of G at point Kx via prox
                subgrad_G = self.par.G.subgrad(Kx_k_prop)
                # subgrad step
                x_k_prop_rev = x_k_prop - self.par.tau * self.par.K.adj(subgrad_G)
                x_k_prop_rev = self.par.F.prox(x_k_prop_rev, ppar = self.par.tau)

                q_y_x = torch.exp( -torch.sum( ((x_k_prop_rev-x_k)**2), dim=tuple(range(len(gauss.shape)-1)) ) /(4*self.par.tau))

                x_k = self.metropolis_check(x_k,x_k_prop,q_x_y,q_y_x)
                x_k_intermediate = x_k

            else:
                x_k = self.par.F.prox(x_k_intermediate, ppar = self.par.tau) + np.sqrt(2*self.par.tau)*normal_tensor.normal_()


            if self.par.save_every:

                if k%self.par.save_every==0:
                    torch.save(x_k_intermediate,self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'.npy')

                    if average_distribution:
                        torch.save(sample[...,:stopping_time_index],self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                        'data_par_'+str(self.par.data_par)+'_'
                                        'tau_'+str(self.par.tau)+'_'
                                        'iter_'+str(k)+'average_distribution.npy')


        return {'xk':x_k_intermediate,'average_distribution_sample':sample,'x0':x_0}

    def subgrad(self,burnin=0):

        assert (not self.par.metropolis_check), "Metropolis correction not implemented for this algorithm"

        x_0 = np.concatenate(self.par.n_parallel_chains*[self.data_in.u0[...,np.newaxis]],axis = -1)
        x_k = np.copy(x_0)

        if self.par.save_every:
            np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_x0.npy',x_0)


        stopping_times = np.random.randint(burnin+1, high = self.par.niter,size = self.par.n_parallel_chains)
        stopping_times = np.sort(stopping_times)

        sample = np.zeros(x_0.shape)
        stopping_time_index = 0
        stopping_condition = False

        for k in range(self.par.niter):

            if self.par.check:
                if k%self.par.check==0 and k>0:
                    if len(self.data_in.u0.shape)==2:
                        f, axarr = plt.subplots(2)
                        axarr[0].imshow(np.mean(x_k,axis=-1),cmap = 'gray')
                        axarr[0].set_title('MMSE after '+str(k)+' iterations. '+str(x_k.shape[-1]))
                        sns.heatmap(np.log(np.var(x_k,axis=-1)), ax = axarr[1])
                        axarr[1].set_title('Marginal posterior variances')
                        plt.show()
                    else:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].plot(np.mean(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[0,0].set_title('MMSE after '+str(k)+' iterations. '+str(stopping_time_index))
                        axarr[0,1].plot(np.var(sample[...,:stopping_time_index-1],axis=-1))
                        axarr[1,0].plot(np.mean(x_0,axis=-1))
                        plt.show()
                    

            if (stopping_time_index < self.par.n_parallel_chains):
                if k==stopping_times[stopping_time_index]:
                    same_index = True
                    while same_index:
                        stopping_time_index += 1
                        sample[...,stopping_time_index-1] = x_k[...,stopping_time_index-1]
                        same_index = k == stopping_times[stopping_time_index] if stopping_time_index<self.par.n_parallel_chains else False



            Kx_k = self.par.K.fwd(x_k)
            # compute subgradient of G at point Kx via prox
            subgrad_G = self.par.G.subgrad(Kx_k)
            # subgrad step
            x_k = x_k - self.par.tau*(self.par.F.subgrad(x_k) + self.par.K.adj(subgrad_G)) + np.sqrt(2*self.par.tau)*np.random.normal(size = x_k.shape)


            if self.par.save_every:
                if k%self.par.save_every==0:                    
                    np.save(self.par.folder+'reg_par_'+str(self.par.reg_par)+'_'
                                    'data_par_'+str(self.par.data_par)+'_'
                                    'tau_'+str(self.par.tau)+'_'
                                    'iter_'+str(k)+'average_distribution.npy',sample[...,:stopping_time_index])


        return {'average_distribution_sample':sample,'x0':x_0}


