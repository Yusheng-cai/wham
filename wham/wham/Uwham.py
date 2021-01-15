import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp

def Uwham(fi,uji,Ni,maxiter=1e5,tol=1e-8,print_every=-1):
    """
    calculates unbinned wham 
    
    fi: initial guess of fi (-ln(Zi/Z0))
    S: number of simulation
    Ntot: the total amount of observations

    uji: a numpy array of shape (S,Ntot) that corresponds to 0.5*beta*k*(N-Nstar)**2
    Ni: a numpy array of shape (S,)

    returns:
        wji: the weights of all the observations in the simulation (Ntot,)
        fi: -ln(Zi/Z0) (S,)
    """
    S = uji.shape[0] # number of simulations
    Ntot = uji.shape[1] # total number of observations 
    iter_ = 1
    fi_prev = fi

    print_flag = False if print_every == -1 else True

    while True:
        lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-uji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 
        fi = - logsumexp(-uji + np.repeat(lnwji[np.newaxis,:],S,axis=0),axis=1)

        fi = fi - fi[-1] #subtract the unbiased fi
        error = np.max(np.abs(fi-fi_prev))
        if print_flag == True:
            if iter_ % print_every == 0:
                print("Error is {} at iteration {}".format(error,iter_))

        iter_ += 1

        if iter_ > maxiter:
            print("UWham did not converge within ", maxiter," iterations")
            break

        if error < tol:
            break

        fi_prev = fi

    lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-uji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 
    return np.exp(lnwji),fi

def Uwham_NLL_eq(fi,uji,Ni):
    """
    fi: initial guess of the log of the partition coefficients Zk normalized by Z0 e.g. f1 = -ln(Z1/Z0) (shape (S,)) 
    uji: beta*Wji energy matrix (shape(S,Ntot))
    Ni: the count of observations in each simulation (shape(S,))
    
    returns:
        Negative Log Likelihood of Uwham
    """
    S = uji.shape[0]
    Ntot = uji.shape[1]
    fi = fi - fi[-1]

    first_term = 1/Ntot*nup.sum(autograd_logsumexp(nup.repeat(fi[:,nup.newaxis],Ntot,axis=1)-uji,\
                                                b=np.repeat(Ni[:,np.newaxis]/Ntot,Ntot,axis=1),axis=0))

    second_term = -nup.sum(Ni*fi)/nup.sum(Ni)

    return first_term + second_term

def Uwham_NLL(fi0,uji,Ni,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,disp=None,iprint=-1,verbose=False):
    """
    fi0: initial guess for Uwham
    uji: a matrix that holds all the energy for all the observations (beta*Wji) (shape(S,Ntot))
    Ni: the count of observations in each simulation (shape(S,))
    ftol: The iteration stops when (f^k-f^{k+1})/max(|f^{k}|,|f^{k+1}|,1) <= ftol
    gtol: The iteration stop when max{|proj g_i | i=1,...,n}<=gtol

    returns:
        the optimal wji for each observation
    """
    S = uji.shape[0]
    Ntot = uji.shape[1]

    if verbose:
        result = minimize(value_and_grad(Uwham_NLL_eq),fi0,args=(uji,Ni),\
                jac=True,method='L-BFGS-B',options={'disp':disp,'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':iprint},callback=callback)
    else:
        result = minimize(value_and_grad(Uwham_NLL_eq),fi0,args=(uji,Ni),\
                jac=True,method='L-BFGS-B',options={'disp':disp,'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':iprint})

    if result.success == False:
        print("Optimization has not converged")
    else:
        print("Optimization has converged")

    fi = result.x # shape (S,)
    fi = fi - fi[-1]

    lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-uji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) #shape (Ntot,)

    return np.exp(lnwji),fi

def callback(fi):
    print(fi)
