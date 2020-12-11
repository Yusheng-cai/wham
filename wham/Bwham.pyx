import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp

def Bwham(Ni,Ml,Wil,tol=1e-10,maxiter=1e5,print_every=-1):
    """
    Ni: Number of data counts in simulation i (S,)
    
    Ml: Number of data in from simulation i=1,...,S in bin l (M,)
    
    Wil: 0.5*k*beta*((n-nstar)**2) (S,M) 

    returns:
        1. gi = ln(fi)
        2. -log(pl) = Free energy distribution in each bin 
        3. pl = probability as each bin l
    """
    S = Ni.shape[0] 
    M = Ml.shape[0]
    gi = np.ones((S,1))*1e-15 # gi with shape (S,1)

    Ni = Ni[:,np.newaxis] # Ni with shape (S,1)
    Ml = Ml[:,np.newaxis] # Ml with shape (M,1)
    iter_ = 1    

    print_flag = False if print_every==-1 else True

    while True:
        # calculate log_pl of shape (M,)
        log_pl = -logsumexp(gi-Wil,b=Ni,axis=0) + np.log(Ml[:,0])
        
        # make log_pl into shape (S,M)
        log_pl = np.repeat(log_pl[np.newaxis,:],S,axis=0)

        # store gi from previous iteration 
        gi_prev = gi

        # update current gi 
        gi = -logsumexp(log_pl - Wil,axis=1)
        gi = gi[:,np.newaxis] #make gi into shape (S,1) again

        # normalize gi by the gi at -1
        gi = gi - gi[-1]

        # find error
        error = np.linalg.norm(gi-gi_prev,2)

        if print_flag:
            if iter_ % print_every == 0:
                print("Error at iteration {} is {}".format(iter_,error))

        if error < tol:
            break

        iter_ += 1
        if iter_ > maxiter:
            print("Maximum iterations reached, now exiting")
            break

    return gi,-log_pl[0],np.exp(log_pl[0])

def Bwham_NLL_eq(x,Ni,Ml,Wil):
    """
    x: shape (S,)

    Ni: Number of data counts in simulation i (S,)

    Ml: Number of data in from simulation i=1,...,S in bin l (M,)

    Wil: 0.5*k*beta*(n-nstar)**2 (S,M)
    """
    S = Wil.shape[0]
    M = Wil.shape[1]
    x = x - x[-1]

    first_term = -(Ni*x).sum()

    log_pl = nup.log(Ml) - nup.log(nup.sum(Ni[:,np.newaxis]*nup.exp(x[:,np.newaxis]-Wil),axis=0)) 

    second_term = (Ml * log_pl).sum(axis=0)

    return first_term - second_term

def Bwham_NLL(gi0,Ni,Ml,Wil):
    """
    gi0: the initial guess of the gi's where gi=ln(fi)

    Ni: Number of data counts in simulation i (S,)

    Ml: Number of data from simulation i=1,...,S in bin l(M,)

    Wil: 0.5*k*beta*(n-nstar)**2 (S,M)

    returns:
        gi: where gi = ln(fi) (S,)

        Fl: where Fl = -log(pl) (M,)

        pl: shape (M,)
    """
    result = minimize(value_and_grad(Bwham_NLL_eq),gi0,args=(Ni,Ml,Wil),jac=True,method='L-BFGS-B') 
    
    gi = result.x
    gi = gi - gi[-1]
    if result.success == True:
        print("Optimization has converged")
    else:
        print("Optimization has not converged")

    log_pl = np.log(Ml) - logsumexp(gi[:,np.newaxis]-Wil,b=Ni[:,np.newaxis],axis=0) 
    F = -log_pl

    return gi, -log_pl,np.exp(log_pl)


