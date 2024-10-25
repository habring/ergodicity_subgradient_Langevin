import numpy as np
import numba
import time
import scipy.sparse

# Directions
LEFT=0
RIGHT=1
UP=2
DOWN=3

# Algorithms for computing messages
FULL=0
TV=1
NARROWBAND=2

@numba.njit
def fast_dist_trans(h, w):
    K = h.shape[-1]
    for k in range(1,K):
        h[:,k] = np.minimum(h[:,k], h[:,k-1]+w)
    for k in range(K-2,-1,-1):            
        h[:,k] = np.minimum(h[:,k], h[:,k+1]+w)
    return h

@numba.njit
def compute_min(h,f,w):
    K = h.shape[-1]
    m = h[:,0] + f[:,0]*w
    for l in range(1,K):
        m = np.minimum(m, h[:,l] + f[:,l]*w)
    return m

@numba.njit
def compute_softmin(h,f,w):
    K = h.shape[-1]

    # determine minimum
    m = compute_min(h,f,w)

    # compute numerically stable softmin
    sum_exp = np.exp(m - h[:,0] - f[:,0]*w)
    for l in range(1,K):
         sum_exp += np.exp(m - h[:,l] - f[:,l]*w)
    softmin = m - np.log(sum_exp)
    return softmin

@numba.njit
def truncated_cost(h,f,w,NB,T):
    K = h.shape[-1]

    mt = h[:,0]
    for k in range(1,K):
        mt = np.minimum(mt, h[:,k])
    mt += T*w
    
    m = np.zeros_like(h)
    for k in range(K):
        sl = np.maximum(0, k-NB)
        el = np.minimum(K, k+NB)
        m[:,k] = np.minimum(mt, compute_min(h[:,sl:el], f[:,k,sl:el], w))
    return m
    
@numba.njit
def full_cost(h,f,w, softmin=False):
    # pairwise costs
    K = h.shape[-1]
    m = np.zeros_like(h)
    for k in range(K):
        if softmin:
            m[:,k] = compute_softmin(h, f[:,k,:], w)
        else:
            m[:,k] = compute_min(h, f[:,k,:], w)
    return m

def dp_chain(g, f, w, algorithm, NB=None, T=None, softmin=False, normalize=True):
    '''
        g: unary costs with shape M x N x K (nodes times labels)
        f: pairwise costs with shape M x N x K x K (edges times labels squared)
        w: pairwise weights independent of the label
        NB: |l-k| <= NB
    '''
    M, N, K = g.shape
    m = np.zeros_like(g)
    
    # loop over nodes in chain
    for i in range(N - 1):
        
        # preciding unary costs + messages
        hi = g[:, i, :] + m[:, i, :]
        # pairwise terms
        fi = f[:,i,:,:]
        # weights
        wi = w[:,i]
        
        if algorithm==FULL:
            # compute messages for full costs
            mip = full_cost(hi, fi, wi, softmin)
            
        elif algorithm==NARROWBAND:
            # compute messages in case of truncated cost
            mip = truncated_cost(hi, fi, wi, NB, T)
            
        elif algorithm==TV:
            # Fast distance transform (K running time)
            mip = fast_dist_trans(hi, wi)
            
        if normalize==True: # normalize messages over labels
            mip -= mip.min(axis=-1, keepdims=True)
        
        m[:, i + 1, :] = mip
    return m

def init_data(g, f=None, w=None, NB=None):
    
    M,N,K = g.shape

    T = None
    
    if f is None:
        print("Pairwise cost: TV")
        f = np.broadcast_to(f, (M,N,K,K))
        algorithm=TV
    elif np.prod(f.shape)==K*K and NB is None:
        print("Pairwise cost: FULL")       
        f = np.broadcast_to(f, (M,N) + f.shape)
        algorithm=FULL
    elif np.prod(f.shape)==K*K and not NB is None:
        T = f[0, NB+1]
        print("Pairwise cost: TRUNCATED with ","T =", T)  
        f = np.broadcast_to(f, (M,N) + f.shape)
        algorithm=NARROWBAND

    if w is None:
        w = np.broadcast_to(1.0, (2,M,N))
    elif np.isscalar(w):
        w = np.broadcast_to(w, (2,M,N))
        
    if not np.prod(w.shape)==2*M*N:
        print("Error: w.shape is ", w.shape, " but should be 2 x ", M, " x ",N)
    return algorithm, f, w, T

# Compute costs
def compute_energy(g,f,l):
    H,W,C = g.shape
    uc = np.take_along_axis(g, l[:,:,None], axis=2).squeeze()
    if len(f.shape)==2:
        f = np.broadcast_to(f, (H,W) + f.shape)
    
    lhp = np.hstack((l[:,1:], l[:,-2:-1]))
    lvp = np.vstack((l[1:,:], l[-2:-1,:]))
    
    pw = np.take_along_axis(f, l[:,:,None,None], axis=3).squeeze()
    pwh = np.take_along_axis(pw, lhp[:,:,None,], axis=2).squeeze()
    pwv = np.take_along_axis(pw, lvp[:,:,None,], axis=2).squeeze()
    
    return uc.sum()+ pwh.sum()+pwv.sum()
    

# Semi-global matching
def sgm(g, f=None, w=None, NB=None, maxit=1, verbose=0):

    M,N,K = g.shape
    
    # init data
    algorithm, f, w, T = init_data(g, f, w, NB)

    # init messages
    m = np.zeros((4,)+g.shape)

    # messages L -> R
    m[RIGHT] = dp_chain(g, f, w[0], algorithm, NB, T)

    # messages R -> L
    m[LEFT] = np.flip(dp_chain(np.flip(g, axis=1),
                               np.flip(f, axis=1),
                               np.flip(w[0], axis=1),
                               algorithm, NB, T), axis=1)                  
    # messages U -> D
    m[DOWN] = dp_chain(g.transpose(1,0,2),
                       f.transpose(1,0,2,3),
                       w[1].transpose(1,0),
                       algorithm, NB, T).transpose(1,0,2)

    # messages D -> U
    m[UP] = np.flip(dp_chain(np.flip(g.transpose(1,0,2), axis=1),
                             np.flip(f.transpose(1,0,2,3), axis=1),
                             np.flip(w[1].transpose(1,0), axis=1),
                             algorithm, NB, T), axis=1).transpose(1,0,2)
                    
    # Compute min-marginals
    b = g + m.sum(axis=0)
     
    # compute minimum labeling
    l = b.argmin(axis=-1)
    
    # compute one-hot labeling
    x = (np.arange(K) == l[...,None]).astype(int)
    
    return x, l, b

# sweep belief propagation
def sbp(g, f=None, w=None, NB=None, maxit=1, softmin=False, verbose=0):

    M,N,K = g.shape
    
    # init data
    algorithm, f, w, T = init_data(g, f, w, NB)
    
    # init messages
    m = np.zeros((4,)+g.shape)
    t0 = time.time()
    for it in range(maxit):

        # Augment unaries with messages from the other trees
        gm = g + m[DOWN] + m[UP]

        # messages L -> R
        m[RIGHT] = dp_chain(gm, f, w[0], algorithm, NB, T, softmin=softmin, normalize=True)

        # messages R -> L
        m[LEFT] = np.flip(dp_chain(np.flip(gm, axis=1),
                                   np.flip(f, axis=1),
                                   np.flip(w[0], axis=1),
                                   algorithm, NB, T, softmin=softmin, normalize=True), axis=1)              

        # Augment the unaries with messages from the other trees
        gm = g + m[RIGHT] + m[LEFT]
          
        # messages U -> D
        m[DOWN] = dp_chain(gm.transpose(1,0,2),
                           f.transpose(1,0,2,3),
                           w[1].transpose(1,0),
                           algorithm, NB, T, softmin=softmin, normalize=True).transpose(1,0,2)

        # messages D -> U
        m[UP] = np.flip(dp_chain(np.flip(gm.transpose(1,0,2), axis=1),
                                 np.flip(f.transpose(1,0,2,3), axis=1),
                                 np.flip(w[1].transpose(1,0), axis=1),
                                 algorithm, NB, T, softmin=softmin, normalize=True), axis=1).transpose(1,0,2)
        
        # compute cost of optimal labeling
        b = g + m.sum(axis=0)
        
        # compute minimum labeling
        l = b.argmin(axis=-1)
        
        if verbose > 0:
            E = compute_energy(g,f,l)
            print("iter = ", it,
                  ", time = ", "{:3.5f}".format(time.time()-t0),
                  ", E = ", "{:3.5f}".format(E),end="\n")

    # compute one-hot labeling
    x = (np.arange(K) == l[...,None]).astype(int)
    
    return x, l, b

# Call of min oracle
def solve_hv_chains(g, f=None, w=None, direction="h", NB=None, verbose=0):

    # init data
    algorithm, f, w, T = init_data(g, f, w, NB)
    M,N,K = g.shape

    m = g.copy()
    if direction=="h":
        # messages L -> R
        m += dp_chain(g, f, w[0], algorithm, NB, T)

        # messages R -> L
        m += np.flip(dp_chain(np.flip(g, axis=1),
                               np.flip(f, axis=1),
                               np.flip(w[0], axis=1),
                               algorithm, NB, T), axis=1)
        
        # compute cost of optimal labeling 
        c = m.min(axis=-1)
        c = np.sum(c[:,0])
    
    elif direction=="v":
        # messages U -> D
        m += dp_chain(g.transpose(1,0,2),
                       f.transpose(1,0,2,3),
                       w[1].transpose(1,0),
                       algorithm, NB, T).transpose(1,0,2)

        # messages D -> U
        m += np.flip(dp_chain(np.flip(g.transpose(1,0,2), axis=1),
                               np.flip(f.transpose(1,0,2,3), axis=1),
                               np.flip(w[1].transpose(1,0), axis=1),
                               algorithm, NB, T), axis=1).transpose(1,0,2)
        
        # compute cost of optimal labeling 
        c = m.min(axis=-1)
        c = np.sum(c[0,:])
    
    # compute minimum labeling
    l = m.argmin(axis=-1)
    
    # compute one-hot labeling
    x = (np.arange(K) == l[...,None]).astype(int)
    
    return x, c