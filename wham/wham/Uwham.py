import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp

class Uwham:
    def __init__(self,xji,k,Ntwiddle,Ni,beta=0.4036,unbiased=True):
        """
        xji: all the observations in the shape of (Ntot,) including all the biased and unbiased simulations

        k:   the parameter in harmonic potential

        Ntwiddle: the Ntwiddle of all the biased simulations in the shape of (S-1,) where S is the number of 
        simulation including the unbiased simulation

        Ni: The number of observations in each simulation

        beta:beta=1/kbT (default value is at T=298K)
        """
        self.xji = xji
        self.k = k
        self.beta = beta
        self.Ntwiddle = Ntwiddle
        self.uji,self.fi0 = self.initialize(unbiased=unbiased)
        self.Ni = Ni

        self.wji = None
        self.fi = None
        
    def initialize(self,unbiased=True):
        """
        unbiased: a boolean, if True, the unbiased energy will be added ( a list of 0's)
        
        returns:
            uji: beta*k*0.5*(n-nstar)**2 (S,Ntot)
            fi0: initial guesses for Uwham (S,)
        """ 
        Ntwiddle = self.Ntwiddle
        beta = self.beta
        k = self.k
        xji = self.xji

        # The number of simulations
        if unbiased == True:
            S = len(Ntwiddle) + 1
        else:
            S = len(Ntwiddle)
        
        fi0 = np.zeros((S,))
        Ntot = len(xji)

        uji = np.zeros((S,Ntot))
        for i in range(len(Ntwiddle)):
            uji[i] = 0.5*beta*k*(xji-Ntwiddle[i])**2

        return uji,fi0

    def self_consistent(self,maxiter=1e5,tol=1e-8,print_every=-1):
        """
        performs self-consistent iterations of unbinned Wham 

        input: 
            fi0: initial guess of fi (-ln(Zi/Z0))
            Ntot: the total amount of observations

            uji: a numpy array of shape (S,Ntot) that corresponds to 0.5*beta*k*(N-Nstar)**2
            Ni: a numpy array of shape (S,)

        returns:
            if converged:
                wji: the weights of all the observations in the simulation (Ntot,)
                fi: -ln(Zi/Z0) (S,)
            else: 
                returns None
        """
        # define variables
        uji = self.uji
        fi = self.fi0
        Ni = self.Ni

        S = uji.shape[0] # number of simulations
        Ntot = uji.shape[1] # total number of observations 
        iter_ = 1
        fi_prev = fi

        print_flag = False if print_every == -1 else True
        converged = False

        while not converged:
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
                converged = True

            fi_prev = fi
        
        if converged == True:
            lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-uji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 

            self.wji = np.exp(lnwji)
            self.fi = fi
            
            return np.exp(lnwji),fi
        else:
            return None
 
    def Maximum_likelihood(self,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,disp=None,iprint=-1):
        """
        fi0: initial guess for Uwham
        uji: a matrix that holds all the energy for all the observations (beta*Wji) (shape(S,Ntot))
        Ni: the count of observations in each simulation (shape(S,))
        ftol: The iteration stops when (f^k-f^{k+1})/max(|f^{k}|,|f^{k+1}|,1) <= ftol
        gtol: The iteration stop when max{|proj g_i | i=1,...,n}<=gtol

        returns:
            if converged:
                the optimal wji for each observation
                fi = -ln(Zi/Z0)
            else:
                returns None
        """
        uji = self.uji
        fi0 = self.fi0
        Ni = self.Ni

        Ntot = uji.shape[1]

        result = minimize(value_and_grad(Uwham_NLL_eq),fi0,args=(uji,Ni),\
                    jac=True,method='L-BFGS-B',options={'disp':disp,'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':iprint})

        if result.success == True:
            print("Optimization has converged")

            fi = result.x # shape (S,)
            fi = fi - fi[-1]

            lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-uji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) #shape (Ntot,)
            wji = np.exp(lnwji)

            self.wji = wji
            self.fi = fi

            return wji,fi
        else:
            print("Optimization has not converged")

            return None

    def compute_betaF_profile(self,min_,max_,bins=100):
        """
        Function that calculates the Free energy for Uwham from the observations xji and
        weights wji

        xji: Total observations from umbrella simulations (Ntot,)
        wji: the weights associated with each of the observations (Ntot,)
        min_: the minimum of the binned vector (float/int)
        max_: the maximum of the binned vector (float/int)
        bins: Number of bins between min_ and max_ (int)

        returns:
            bins_vec: binned vector from min_ to max_ (bins-1,)
            p: the probability in each bin
            F: The free energy in the binned vectors from min_ to max_ for all the simulations performed(S,bins-1)
        """
        xji = self.xji
        if self.wji is None:
            raise RuntimeError("Please run Maximum_likelihood or self_consistent first to obtain weights wji")

        S = self.uji.shape[0]

        bins_vec = np.linspace(min_,max_,bins)
        pji = self.get_pji()
        sum_ = pji.sum(axis=1)

        # The weighted probability for each bin
        p = np.zeros((S,bins-1))
        F = np.zeros_like(p)

        # This will be a vector such as np.array([1,2,3,2,..]) indicating which bin each element falls into
        digitized = np.digitize(xji,bins_vec)
        indices = [np.argwhere(digitized == j) for j in range(1,bins)]

        for i in range(S):
            pi = np.array([np.sum(pji[i][idx])/sum_[i] for idx in indices]) 
            p[i] = pi 

            Fi = -np.log(pi)
            Fi = Fi - Fi.min()
            F[i] = Fi
        
        return (bins_vec[:-1],F,p)

    def get_pji(self):
        """
        Function that obtains all the weights for unbiased as well as biased simulations following the equation
            pji_k = np.exp(fi)*np.exp(-Uji_k)*wji
        where wji is the unbiased weights 

        input:
            fi: -Log(Zi/Z0) (S,)
            Uji: energy matrix=0.5*beta*k*(x-xji)**2 (S,Ntot)
            wji: weight matrix(Ntot,)

        returns:
            pji: shape (S,Ntot)
        """
        if self.fi is None and self.wji is None:
            raise RuntimeError("Please run either self_consistent or Maximum_likelihood first")
        
        uji = self.uji
        fi = self.fi
        wji = self.wji
        
        S = uji.shape[0]
        Ntot = uji.shape[1]

        pji = np.exp(np.repeat(fi[:,np.newaxis],Ntot,axis=1))*np.exp(-uji)*np.repeat(wji[np.newaxis,:],S,axis=0)

        return pji
    
def Uwham_NLL_eq(fi,uji,Ni):
    """
    fi: initial guess of the log of the partition coefficients Zk normalized by Z0 e.g. f1 = -ln(Z1/Z0) (shape (S,)) 
    uji: beta*Wji energy matrix (shape(S,Ntot))
    Ni: the count of observations in each simulation (shape(S,))
    
    returns:
        Negative Log Likelihood of Uwham
    """
    Ntot = uji.shape[1]
    fi = fi - fi[-1]

    first_term = 1/Ntot*nup.sum(autograd_logsumexp(nup.repeat(fi[:,nup.newaxis],Ntot,axis=1)-uji,\
                                                b=np.repeat(Ni[:,np.newaxis]/Ntot,Ntot,axis=1),axis=0))

    second_term = -nup.sum(Ni*fi)/nup.sum(Ni)

    return first_term + second_term


