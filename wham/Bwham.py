import autograd.numpy as nup
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.special import logsumexp
from wham.lib.numeric import autograd_logsumexp

class Bwham:
    """
    Args:
        xji(np.ndarray): all the observation in dataset (Ntot,)
        Ntwiddle(np.ndarray): The Ntwiddle for all the biased simulations (S-1,)
        Ni(np.ndarray): The number of observations in each simulation (S,)
        k(float): The parameter for the harmonic potential where U=0.5*k(x-xstar)**2
        min_(float): the minimum of the bins
        max_(float): the maximum of the bins
        bins(int): number of bins
        beta(float): beta=1/kbT where the default value is at T=298K
        Ml(np.ndarray): Number of simulations in the lth bin from all simulations (M,)
        Wil(np.ndarray): Biased energy from different simulations, 0 for unbiased simulation
        fi0(np.ndarray): An array of zero's which can be used as the initial guess for the optimization
    """

    def __init__(self,xji,Ntwiddle,Ni,k,min_,max_,bins=101,beta=0.4036,unbiased=True):
        self.xji = xji
        self.Ni = Ni
        self.Ntwiddle = Ntwiddle
        self.min_ = min_
        self.max_ = max_
        self.bins = bins
        self.beta = beta
        self.k = k
        self.Ml,self.Wil,self.fi0,self.bins = self.initialize(unbiased=unbiased)

        self.fi = None
        self.pl = None
        
    def initialize(self,unbiased=True):
        """
        Function that initializes some variables

        returns:
            1. Ml= observations in the lth bin from all simulations (M,)

            2. Wil= The energy matrix where Wil = 0.5*beta*ki*(xl-xi)**2

            3. bins= the binned vector (M,)
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
        
        Args:
            tol(float): the tolerance for convergence  (default 1e-10)
            maxiter(int): maximum iterations possible for self consistent solver (default 1e5)
            print_every(int): how many iterations to print result. A number less than 0 indicates never. (default -1)


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

            self.fi = fi
            self.pl = pl

            return fi,F,pl
        else:
            return None

    def Maximum_likelihood(self,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,iprint=-1):
        """
        Args:
            ftol(float): tolerance parameter as set forth by scipy.minimize's 'L-BFGS-B' option (default 2.22e-09)
            gtol(float): tolerance parameter as set forth by scipy.minimize's 'L-BFGS-B' option (default 1e-05)
            maxiter(int): Maximum iteration as set forth by scipy.minimize (default 15000)
            maxfun(int): Maximum function evaluation as set forth by scipy.minimize (default 15000)
            iprint(int): The interval between which the user wants result to be printed. A number less than 0
            indicates never. (default -1)
        
        returns:
            1. fi= -ln(Zi/Z0) (S,)
            2. Fl= -log(pl) free energy at the lth bin (M,)
            3. pl=probability at the lth bin (M,)
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
            self.fi = fi
            self.pl = pl

            return fi, F, pl
        else:
            print("Optimization has not converged")
            return None

    def get_pil(self):
        """
        Function that obtains the matrix of pil which is defined as 
        
        Returns:
            pil(np.ndarray): matrix with shape (S,M)
        """
        if self.fi is None:
            raise RuntimeError("Please run self consistent or Maximum_likelihood first!")

        fi = self.fi
        pl = self.pl
        Wil = self.Wil

        M = Wil.shape[1]
        S = Wil.shape[0]

        pil = np.exp(np.repeat(fi[:,np.newaxis],M,axis=1))*np.exp(-Wil)*np.repeat(pl[np.newaxis,:],S,axis=0)

        return pil

def Bwham_NLL_eq(x,Ni,Ml,Wil):
    """
    Args:
        x: shape (S,)

        Ni: Number of data counts in simulation i (S,)

        Ml: Number of data in from simulation i=1,...,S in bin l (M,)

        Wil: 0.5*k*beta*(n-nstar)**2 (S,M)

    Returns:
        the value of the negative likelihood function
    """
    S = Wil.shape[0]
    M = Wil.shape[1]
    
    x = x - x[-1]
    first_term = -(Ni*x).sum()
    
    log_pl = nup.log(Ml) - \
            autograd_logsumexp(nup.repeat(x[:,nup.newaxis],M,axis=1)-Wil,b=nup.repeat(Ni[:,nup.newaxis],M,axis=1),axis=0)

    second_term = (Ml * log_pl).sum(axis=0)


    return first_term - second_term


