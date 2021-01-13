import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp

def Bwham(fi,Ni,Ml,Wil,tol=1e-10,maxiter=1e5,print_every=-1):
    """
    Implementation of binned wham

    fi: initial guess for -log(Zi/Z0), passed in as a numpy array (S,)

    Ni: Number of data counts in simulation i (S,)
    
    Ml: Number of data in from simulation i=1,...,S in bin l (M,)
    
    Wil: 0.5*k*beta*((n-nstar)**2) (S,M) 

    returns:
        1. fi = -log(Zi/Z0)
        2. -log(pl) = Free energy distribution in each bin 
        3. pl = probability as each bin l
    """
    S = Ni.shape[0] 
    M = Ml.shape[0]

    iter_ = 1    

    print_flag = False if print_every==-1 else True
    fi_prev = fi

    while True:
        # calculate log_pl of shape (M,)
        log_pl = -logsumexp(np.repeat(fi[:,np.newaxis],M,axis=1)-Wil,b=np.repeat(Ni[:,np.newaxis],M,axis=1),axis=0) + np.log(Ml)
        
        # update current gi 
        fi = -logsumexp(np.repeat(log_pl[np.newaxis,:],S,axis=0) - Wil,axis=1)

        # normalize fi by the fi at -1
        fi = fi - fi[-1]

        # find error
        error = np.linalg.norm(fi-fi_prev,2)

        if print_flag:
            if iter_ % print_every == 0:
                print("Error at iteration {} is {}".format(iter_,error))

        if error < tol:
            break

        iter_ += 1
        if iter_ > maxiter:
            print("Maximum iterations reached, now exiting")
            break

        fi_prev = fi

    return fi,-log_pl,np.exp(log_pl)

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
    
    log_pl = nup.log(Ml) - \
            autograd_logsumexp(nup.repeat(x[:,nup.newaxis],M,axis=1)-Wil,b=nup.repeat(Ni[:,nup.newaxis],M,axis=1),axis=0)

    second_term = (Ml * log_pl).sum(axis=0)


    return first_term - second_term

def Bwham_NLL(fi0,Ni,Ml,Wil,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000):
    """
    fi0: the initial guess of the gi's where gi=ln(fi)

    Ni: Number of data counts in simulation i (S,)

    Ml: Number of data from simulation i=1,...,S in bin l(M,)

    Wil: 0.5*k*beta*(n-nstar)**2 (S,M)

    returns:
        fi: where fi = -ln(Zi/Z0) (S,)

        Fl: where Fl = -log(pl) (M,)

        pl: shape (M,)
    """
    result = minimize(value_and_grad(Bwham_NLL_eq),fi0,args=(Ni,Ml,Wil),jac=True,method='L-BFGS-B',options={'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':10}) 
    
    gi = result.x
    gi = gi - gi[-1]

    if result.success == True:
        print("Optimization has converged")
    else:
        print("Optimization has not converged")

    log_pl = np.log(Ml) - logsumexp(gi[:,np.newaxis]-Wil,b=Ni[:,np.newaxis],axis=0) 
    F = -log_pl
    pl = np.exp(log_pl)

    return gi, F, pl
