import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import alogsumexp

class Uwham:
    """
    A class that performs the binless Wham calculations or Mbar

    Args:
        xji(np.ndarray): all the observations inclduing biased and unbiased simulations (Ntot,)
        k(float): the parameter in harmonic potential where U(x)=0.5*k*(x-xstar)**2
        Ntwiddle(np.ndarray): The Ntwiddle of all the biased simulations (S-1,)
        Ni(np.ndarray): Number of observations in each simulations (S,)
        beta(float): 1/kbT, the default is at T=298K
        buji(np.ndarray): The energy matrix, is zero for unbiased simulation (S,Ntot)
        fi0(np.ndarray): The initial guess of fi (-ln(Zi/Z0)) for optimization (S,) 
    """

    def __init__(self,xji,k,Ntwiddle,Ni,beta=0.4036):
        self.xji = xji
        self.k = k
        self.beta = beta
        self.Ntwiddle = Ntwiddle
        self.Ni = Ni
        self.buji,self.fi0 = self.initialize()

        self.lnwji = None
        self.fi = None
        
    def initialize(self):
        """
        initialize some parameters of the class 

        Return: 
            1. buji= beta*k*0.5*(n-nstar)**2 (S,Ntot)
            2. fi0=initial guesses for Uwham (S,)
        """ 
        Ntwiddle = self.Ntwiddle
        beta = self.beta
        k = self.k
        xji = self.xji
        Ni = self.Ni
        S,Ntot = Ni.shape[0],xji.shape[0]
        
        # Find beta*uji by using vectorized operation in numpy
        buji = np.zeros((S,Ntot))
        temp = 0.5*beta*k*(np.expand_dims(xji,axis=0) - np.expand_dims(Ntwiddle,axis=1))**2
        buji[:-1,:] = temp
        
        
        # Initialize fi0 according to mbar paper fk0 = 1/Nk*sum_n ln(exp(-beta*Uk))
        fi0 = 1/Ni*(-buji.sum(axis=1))
        fi0 = fi0 - fi0[-1] 

        return buji,fi0

    def self_consistent(self,maxiter=1e5,tol=1e-7,print_every=-1):
        """
        performs self-consistent iteration optimization of binless Wham 

        Args:
            maxiter(int): specifies the maximum number of self consistent iterations are allowed (default 1e5)
            tol(float): specifies the tolerance of the iteration, max|fn+1-fn|/max|fn| (default 1e-7)
            print_every(int): The frequency at which the programs outputs the result. If the number is less than zero, the program will not output result.(default -1, so nothing will be printed) 

        Return:
            1. wji=the weights of all the observations in the simulation (Ntot,)
            2. fi=-ln(Zi/Z0) (S,)
        """
        # define variables
        buji = self.buji
        fi = self.fi0
        Ni = self.Ni

        S = buji.shape[0] # number of simulations
        Ntot = buji.shape[1] # total number of observations 
        iter_ = 1
        fi_prev = fi

        print_flag = False if print_every == -1 else True
        converged = False

        while not converged:
            lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 
            fi = - logsumexp(-buji + np.repeat(lnwji[np.newaxis,:],S,axis=0),axis=1)
            fi = fi - fi[-1] #subtract the unbiased fi 

            lnpjik = self.get_lnpji_k(lnwji,fi)
            g = self.gradient(lnpjik,Ni)
            gnorm = np.dot(g,g)
            
            error = np.max(np.abs(fi-fi_prev)[:-1])/np.max(np.abs(fi_prev[:-1]))

            if print_flag == True:
                if iter_ % print_every == 0:
                    print("Error is {} at iteration {}".format(error,iter_))
                    print("gradient norm is {} at iteration {}".format(gnorm,iter_)) 

            iter_ += 1

            if iter_ > maxiter:
                print("UWham did not converge within ", maxiter," iterations")
                break

            if error < tol:
                converged = True

            fi_prev = fi
            print(converged)
        
        if converged == True:
            lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 

            self.lnwji = lnwji
            self.fi = fi
            
            return lnwji,fi
        else:
            return None

 
    def Maximum_likelihood(self,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,disp=None,iprint=-1):
        """
        Optimizes the negative likelihood equation of binless Wham using LBFGS algorithm as implemented by scipy.minimize, the derivatives of the MLE is found by automatic differentiation using autograd 

        Args:
            ftol(float): the tolerance as set forth by scipy.minimize (default 2.22e-09)

            gtol(float): the tolerance as set forth by scipy.minimize (default 1e-05)  

            maxiter(int): the maximum number of iterations as set forth by scipy.minimize. (default 15000)

            maxfun(int): the maximum number of function evaluations as set forth by scipy.minimize. (default 15000)

            iprint(int): the frequency at which the program outputs the result. Will not output if less than 0. (default -1)

        Return:
            lnwji(np.ndarray): log of the optimal weights for each observation if converged else None
        """
        buji = self.buji
        fi0 = self.fi0
        Ni = self.Ni

        Ntot = buji.shape[1]

        result = minimize(value_and_grad(Uwham_NLL_eq),fi0,args=(buji,Ni),\
                    jac=True,method='L-BFGS-B',options={'disp':disp,'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxfun':maxfun,'iprint':iprint})

        if result.success == True:
            print("Optimization has converged")

            fi = result.x # shape (S,)
            fi = fi - fi[-1]

            lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) #shape (Ntot,)

            self.lnwji = lnwji
            self.fi = fi

            return lnwji,fi
        else:
            print("Optimization has not converged")

            return None

    def compute_betaF_profile(self,min,max,bins=100):
        """
        Function that calculates the Free energy for Uwham from the observations xji and
        log of the weights lnwji
        
        Args:
            min(float): the minimum of the binned vector (float/int)
            max(float): the maximum of the binned vector (float/int)
            bins(int): number of bins 

        Returns:
            1. bins_vec = binned vector from min to max (bins-1,)
            2. F = The free energy in the binned vectors from min to max for all the simulations performed(S,bins-1)
            3. logp = log of the probability in each bin
        """
        xji = self.xji
        if self.lnwji is None:
            raise RuntimeError("Please run Maximum_likelihood or self_consistent first to obtain weights lnwji")

        S = self.buji.shape[0]
        bins_vec = np.linspace(min,max,bins)

        # the bin size dl 
        dl = bins_vec[1] - bins_vec[0]
        
        # find lnpji from all simulations (S,Ntot)
        lnpji_k = self.get_lnpji_k(self.lnwji,self.fi)

        # log(sum(exp(lnpji)))
        lnsum_ = logsumexp(lnpji_k,axis=1)

        # The log probability for each bin (S,bins-1)
        logp = np.zeros((S,bins-1))

        # Create the free energy matrix with the same shape
        F = np.zeros_like(logp)

        # This will be a vector such as np.array([1,2,3,2,..]) indicating which bin each element falls into
        digitized = np.digitize(xji,bins_vec)
        indices = [np.argwhere(digitized == j) for j in range(1,bins)]

        for i in range(S):
            logpi = np.array([logsumexp(lnpji_k[i][idx]) for idx in indices]) - lnsum_[i] 
            logp[i] = logpi 

            Fi = np.log(dl)-logpi
            Fi = Fi - Fi.min()
            F[i] = Fi
        
        return (bins_vec[:-1],F,logp)

    def get_lnpji_k(self,lnwji,fi):
        """
        Function that obtains all the weights for unbiased as well as biased simulations following the equation lnpji_k = fi-buji_k+lnwji where wji is the unbiased weights 

        Args:
            lnwji(np.ndarray): log of the unbiased weights (Ntot,)
            fi(np.ndarray): -log(Zi/Z0) (S,)

        Return:
            lnpji_k(np.ndarray): log of the pji matrix with shape (S,Ntot)
        """
        buji = self.buji
        
        S = buji.shape[0]
        Ntot = buji.shape[1]

        lnpji_k = np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji+ np.repeat(lnwji[np.newaxis,:],S,axis=0)

        return lnpji_k

    def gradient(self,lnpjik,Ni):
        """
        Function that calculates the gradient of the Negative likelihood function. The function form is as follows
            gk(f) = Nk - Nk*sum_n Wnk(f)
        where Wnk=exp(fk)exp(-beta Uk(xn))wn
        
        Args:
            lnpjik(np.ndarray): The log of the probability matrix shape(S,Ntot)
            Ni(np.ndarray): The number of observations in each simulation (S,)

        return:
            The gradient of the negative likelihood function (S,)
        """
        Ntot = lnpjik.shape[1]

        lnpk = logsumexp(lnpjik,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=1)

        return Ni - np.exp(lnpk)
    
def Uwham_NLL_eq(fi,buji,Ni):
    """
    Args:
        fi(np.ndarray): initial guess of the log of the partition coefficients Zk normalized by Z0 e.g. f1 = -ln(Z1/Z0) (shape (S,)) 
        buji(np.ndarray): beta*Wji energy matrix (shape(S,Ntot))
        Ni(np.ndarray): the count of observations in each simulation (shape(S,))
        
    Return:
        Negative Log Likelihood value of Uwham
    """
    Ntot = buji.shape[1]
    fi = fi - fi[-1]
    first_term = 1/Ntot*nup.sum(alogsumexp(nup.repeat(fi[:,nup.newaxis],Ntot,axis=1)-buji,\
                                                b=np.repeat(Ni[:,np.newaxis]/Ntot,Ntot,axis=1),axis=0))

    second_term = -nup.sum(Ni*fi)/Ntot

    return first_term + second_term
