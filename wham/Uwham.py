import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp

class Uwham:
    """
    A class that performs the unbinned calculations

    Args:
        xji(np.ndarray): all the observations inclduing biased and unbiased simulations (Ntot,)
        k(float): the parameter in harmonic potential where U(x)=0.5*k*(x-xstar)**2
        Ntwiddle(np.ndarray): The Ntwiddle of all the biased simulations (S-1,)
        Ni(np.ndarray): Number of observations in each simulations (S,)
        beta(float): 1/kbT, the default is at T=298K
        uji(np.ndarray): The energy matrix, is zero for unbiased simulation (S,Ntot)
        fi0(np.ndarray): The initial guess of fi (-ln(Zi/Z0)) for optimization (S,) 
    """
    def __init__(self,xji,k,Ntwiddle,Ni,beta=0.4036,unbiased=True):
        self.xji = xji
        self.k = k
        self.beta = beta
        self.Ntwiddle = Ntwiddle
        self.uji,self.fi0 = self.initialize(unbiased=unbiased)
        self.Ni = Ni

        self.wji = None
        self.fi = None
        
    def initialize(self):
        """initialize some parameters of the class 
        Returns: 
            1. uji= beta*k*0.5*(n-nstar)**2 (S,Ntot)
            2. fi0=initial guesses for Uwham (S,)
        """ 
        Ntwiddle = self.Ntwiddle
        beta = self.beta
        k = self.k
        xji = self.xji

        # The number of simulations
        S = len(Ntwiddle) + 1
        
        fi0 = np.zeros((S,))
        Ntot = len(xji)

        uji = np.zeros((S,Ntot))
        for i in range(len(Ntwiddle)):
            uji[i] = 0.5*beta*k*(xji-Ntwiddle[i])**2

        return uji,fi0

    def self_consistent(self,maxiter=1e5,tol=1e-8,print_every=-1):
        """performs self-consistent iterations of unbinned Wham 
        Args:
            maxiter(int): specifies the maximum number of self consistent iterations are allowed
            tol(float): specifies the tolerance of the iteration
            print_every(int): The frequency at which the programs outputs the result. If the number is less than zero, the program will not output result. 

        Returns
            1. wji=the weights of all the observations in the simulation (Ntot,)
            2. fi=-ln(Zi/Z0) (S,)
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
        Optimizes the negative likelihood equation using LBFGS algorithm

        Args:
            ftol(float): the tolerance as set forth by scipy.minimize (default 2.22e-09)

            gtol(float): the tolerance as set forth by scipy.minimize (default 1e-05)  

            maxiter(int): the maximum number of iterations as set forth by scipy.minimize. (default 15000)

            maxfun(int): the maximum number of function evaluations as set forth by scipy.minimize. (default 15000)

            iprint(int): the frequency at which the program outputs the result. Will not output if less than 0. (default -1)

        Return:
            wji(np.ndarray): the optimal weights for each observation if converged else None
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

    def compute_betaF_profile(self,min,max,bins=100):
        """
        Function that calculates the Free energy for Uwham from the observations xji and
        weights wji
        
        Args:
            min(float): the minimum of the binned vector (float/int)
            max(float): the maximum of the binned vector (float/int)
            bins(int): number of bins 

        Returns:
            1. bins_vec= binned vector from min to max (bins-1,)
            2. p=the probability in each bin
            3. F=The free energy in the binned vectors from min to max for all the simulations performed(S,bins-1)
        """
        xji = self.xji
        if self.wji is None:
            raise RuntimeError("Please run Maximum_likelihood or self_consistent first to obtain weights wji")

        S = self.uji.shape[0]

        bins_vec = np.linspace(min,max,bins)
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

        Return:
            pji(np.ndarray): pji matrix with shape (S,Ntot)
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
    Args:
        fi(np.ndarray): initial guess of the log of the partition coefficients Zk normalized by Z0 e.g. f1 = -ln(Z1/Z0) (shape (S,)) 
        uji(np.ndarray): beta*Wji energy matrix (shape(S,Ntot))
        Ni(np.ndarray): the count of observations in each simulation (shape(S,))
        
    Return:
        Negative Log Likelihood value of Uwham
    """
    Ntot = uji.shape[1]
    fi = fi - fi[-1]

    first_term = 1/Ntot*nup.sum(autograd_logsumexp(nup.repeat(fi[:,nup.newaxis],Ntot,axis=1)-uji,\
                                                b=np.repeat(Ni[:,np.newaxis]/Ntot,Ntot,axis=1),axis=0))

    second_term = -nup.sum(Ni*fi)/nup.sum(Ni)

    return first_term + second_term

