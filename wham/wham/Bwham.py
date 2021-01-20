import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp


class Bwham:
    def __init__(self,xji,Ni,Ntwiddle,k,min_,max_,bins=101,beta=0.4036,unbiased=True):
        """
        xji: all the observation in dataset (Ntot,)
        Ni: The number of observations in each simulation (S,)
        Ntwiddle: The Ntwiddle for all the biased simulations (S-1,)
        k: The parameter for the harmonic potential where U=0.5*k(x-xstar)**2
        min_: the minimum of the bins
        max_: the maximum of the bins
        bins: number of bins
        beta: beta=1/kbT where the default value is at T=298K
        """

        self.xji = xji
        self.Ni = Ni
        self.Ntwiddle = Ntwiddle
        self.min_ = min_
        self.max_ = max_
        self.bins = bins
        self.beta = beta
        self.k = k
        self.Ml,self.Wil,self.fi0,self.bins = self.initialize(unbiased=unbiased)
        
    def initialize(self,unbiased=True):
        """
        xji: All the observations from the umbrella sampling simulations (Ntot,)

        Ntwiddle: The Ntwiddle in the biased simulations --> exclude the unbiased

        min_: minimum of the bins

        max_: maximum of the bins

        k: the harmonic constant for the bias potential (in kJ/mol)

        beta: 1/KbT (in kJ/mol), default value is for T=298K

        bins: number of bins

        unbiased: a boolean value signifying whether or not to include the unbiased simulation, 
                  if True, a list of zero's will be included for Wil
        
        returns:
            Ml: Number of observations in bin l (M,)
            Wil: beta*k*0.5*(n-nstar)**2 (S,M)
            fi0: The initial guesses for the Bwham (array of 0's)
            bins: the bins for the binned free energy 
        """
        xji = self.xji
        Ntwiddle = self.Ntwiddle
        bins = self.bins
        min_ = self.min_
        max_ = self.max_
        beta = self.beta
        k = self.k

        if unbiased == True:
            S = len(Ntwiddle)+1
        else:
            S = len(Ntwiddle)

        M = bins-1

        fi0 = np.zeros((S,))
        
        # Find Ml
        bins_ = np.linspace(min_,max_,bins)
        digitized = np.digitize(xji,bins_,right=False)
        Ml = np.array([(digitized == i).sum() for i in range(1,bins)]) # shape bins-1 (M,)
        
        # Find Wil
        Wil = np.zeros((S,M))
        for i in range(M):
            mini = bins_[i]
            maxi = bins_[i+1]
            middle = (mini+maxi)/2
            for j in range(len(Ntwiddle)):
                N = Ntwiddle[j]
                Wil[j,i] = 0.5*beta*k*(N-middle)**2
        
        return Ml, Wil,fi0,bins_[:-1]

    def self_consistent(self,tol=1e-10,maxiter=1e5,print_every=-1):
        """
        Implementation of self consistent solver of binned wham

        fi: initial guess for -log(Zi/Z0), passed in as a numpy array (S,)

        Ni: Number of data counts in simulation i (S,)
        
        Ml: Number of data in from simulation i=1,...,S in bin l (M,)
        
        Wil: 0.5*k*beta*((n-nstar)**2) (S,M) 

        returns:
            1. fi = -log(Zi/Z0)
            2. -log(pl) = Free energy distribution in each bin 
            3. pl = probability as each bin l
        """
        Ni,Ml,fi,Wil = self.Ni, self.Ml, self.fi0,self.Wil

        S = Ni.shape[0] 
        M = Ml.shape[0]

        iter_ = 1    

        print_flag = False if print_every==-1 else True
        fi_prev = fi
        converged = False

        while not converged:
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
                converged = True

            iter_ += 1
            if iter_ > maxiter:
                print("Maximum iterations reached, now exiting")
                break

            fi_prev = fi

        if converged == True:
            F = -log_pl
            F = F - F.min()
            pl = np.exp(log_pl)
            return fi,F,pl
        else:
            return None

    def Maximum_likelihood(self,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,iprint=-1):
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
        fi0,Ni,Wil,Ml = self.fi0,self.Ni,self.Wil,self.Ml

        result = minimize(value_and_grad(Bwham_NLL_eq),fi0,args=(Ni,Ml,Wil),jac=True,method='L-BFGS-B',options={'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':iprint}) 
        
        fi = result.x
        fi = fi - fi[-1]

        if result.success == True:
            print("Optimization has converged")
            log_pl = np.log(Ml) - logsumexp(fi[:,np.newaxis]-Wil,b=Ni[:,np.newaxis],axis=0) 
            F = -log_pl
            F = F - F.min()
            pl = np.exp(log_pl)

            return fi, F, pl
        else:
            print("Optimization has not converged")
            return None


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


