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
        xji(np.ndarray): all the observations inclduing biased and unbiased simulations (Ntot,n)
        k(np.ndarray): the parameter in harmonic potential where U(x)=0.5*k*(x-xstar)**2 array of shape (S,n)
        Ntwiddle(np.ndarray): The Ntwiddle of all the biased simulations of shape (S,n)
        Ni(np.ndarray): Number of observations in each simulations (S,)
        beta(float): 1/kbT, the default is at T=298K
        buji(np.ndarray): The energy matrix, is zero for unbiased simulation (S,Ntot)
        fi0(np.ndarray): The initial guess of fi (-ln(Zi/Z0)) for optimization (S,) 
        initialization(str): The way to initialize initial guess fi0 (choice: 'zeros','mbar') (default 'zeros')
    """
    def __init__(self,xji,k,Ntwiddle,Ni,beta=0.4036,initialization='zeros'):
        self.xji = xji
        self.k = k 
        self.beta = beta
        self.lnwji = None
        self.fi = None
        self.Ntwiddle = np.array(Ntwiddle)
        self.Ni = np.array(Ni)
        
        # The total observations Ntot
        Ntot = xji.shape[0] 

        # Length of Ntwiddle has to be 1 less than Ni
        assert len(Ntwiddle) == len(Ni), "Length of Ntwiddle must be equal to length of Ni"

        # The sum of Ni has to be Ntot, this is used for extra security
        assert Ntot == Ni.sum()

        self.buji,self.fi0 = self.initialize(initialization)
        
    def initialize(self,initialization):
        """
        initialize parameters for the class (Uji and fi0) 

        Return: 
            1. buji= beta*k*0.5*(n-nstar)**2 (S,Ntot)
            2. fi0=initial guesses for Uwham (S,)
        """ 
        Ntwiddle = self.Ntwiddle
        beta = self.beta
        k = self.k
        
        # xji is of shape (Ntot,n), Ni is shape (S,)
        xji = self.xji
        Ni = self.Ni
        S,Ntot = Ni.shape[0],xji.shape[0]
        
        # xji:(Ntot,n) -> (S,n,Ntot), Ntwiddle(S,n) -> (S,n,Ntot) --> (S,n,Ntot)
        squared = (np.repeat(np.expand_dims(xji.T,axis=0),S,axis=0) - np.repeat(np.expand_dims(Ntwiddle,axis=-1),Ntot,axis=-1))**2
        buji = 0.5*beta*(np.repeat(np.expand_dims(k,axis=-1),Ntot,axis=-1)*squared).sum(axis=1) # shape (S,Ntot)
        
        # Initialize fi0 according to mbar paper fk0 = 1/Nk*sum_n ln(exp(-beta*Uk))
        if initialization == 'mbar':
            fi0 = 1/Ni*(-buji.sum(axis=1))
            fi0 = fi0 - fi0[-1] 

        if initialization == 'zeros':
            jitter = 1e-8
            fi0 = np.zeros((S,)) + jitter

        return buji,fi0

    def adaptive(self,maxiter=1e5,tol=1e-7,print_every=-1,gamma=0.1):
        """
        performs adaptive optimization of binless Wham that alternates between self consistent iteration and Newton Raphson iteration according to the MBAR paper. The criteria for determining which one is use is based on the gradient norm outputted by the two methods where the method with the lower gradient will be used. 

        Args:
            maxiter(int): specifies the maximum number of self consistent iterations are allowed (default 1e5)
            tol(float): specifies the tolerance of the iteration, max|fn+1-fn|/max|fn| (default 1e-7)
            print_every(int): The frequency at which the programs outputs the result. If the number is less than zero, the program will not output result.(default -1, so nothing will be printed) 
            gamma(float): A float between 0 and 1 where the Newton Raphson update uses. xn+1=xn + gamma*Hinvg

        Return:
            1. lnwji= log of the weights of all the observations in the simulation (Ntot,)
            2. fi=-ln(Zi/Z0) (S,)
        """
        # define variables
        buji = self.buji
        S = buji.shape[0] # number of simulations
        Ntot = buji.shape[1] # total number of observations 
        iter_ = 1
        Ni = self.Ni

        # set several flags
        print_flag = False if print_every == -1 else True
        # This flag is set to true if Newton Raphson step is used
        nr_flag = False
        # This flag is set to true if self consistent step is used
        sc_flag = False
        converged = False
        
 
        fi = self.fi0
        fnr = np.zeros_like(fi)
        nr_count = 0
        fsc = np.zeros_like(fi)
        sc_count = 0
        # Calculate gradient at fi0
        g,lnwji = Uwham.gradient(fi,buji,Ni)
        # set variable fi_prev for the initial iteration
        fi_prev = fi
        
        while not converged:
            # Calculate the Hessian at the current fi
            H,_ = Uwham.Hessian(fi,buji,Ni)

            # Newton Raphson update
            Hinvg = np.linalg.lstsq(H,g,rcond=-1)[0] #Calculates H-1g where xn+1 = xn + H-1g
            fnr = fi - gamma*Hinvg
            fnr = fnr - fnr[-1]             
            g_nr,lnwji_nr = Uwham.gradient(fnr,buji,Ni) #find gradient of Newton Raphson
            gnorm_nr = np.dot(g_nr.T,g_nr) #find the norm fo the gradient of Newton Raphson
 
            # Self consistent update
            fsc = - logsumexp(-buji + np.repeat(lnwji[np.newaxis,:],S,axis=0),axis=1)
            fsc = fsc - fsc[-1] #subtract the unbiased fi 
            g_sc,lnwji_sc = Uwham.gradient(fsc,buji,Ni) #find gradient of self_consistent
            gnorm_sc = np.dot(g_sc.T,g_sc)
            
            if gnorm_sc > gnorm_nr:
                fi = fnr
                nr_flag = True
                sc_flag = False
                lnwji = lnwji_nr
                g = g_nr
                nr_count += 1
            else:
                fi = fsc
                nr_flag = False
                sc_flag = True
                lnwji = lnwji_sc
                g = g_sc
                sc_count += 1

            error = np.max(np.abs(fi-fi_prev)[:-1])/np.max(np.abs(fi_prev[:-1]))

            if print_flag == True:
                if iter_ % print_every == 0:
                    print("Error is {} at iteration {}".format(error,iter_))
                    print("gradient norm for Newton Raphson is {0:.4E} at iteration {1}".format(gnorm_nr,iter_)) 
                    print("gradient norm for Self Consistent is {0:.4E} at iteration {1}".format(gnorm_sc,iter_))
                    if nr_flag:
                        print("Newton Raphson is chosen for step {}".format(iter_))
                    if sc_flag:
                        print("Self Consistent is chosen for step {}".format(iter_))

            iter_ += 1

            if iter_ > maxiter:
                print("UWham did not converge within ", maxiter," iterations")
                break

            if error < tol:
                converged = True

            fi_prev = fi
        
        if converged == True:
            lnwji = - logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 

            self.lnwji = lnwji
            self.fi = fi
            
            return lnwji,fi
        else:
            return None

    
    def Newton_Raphson(self,maxiter=250,tol=1e-7,alpha=0.5,beta=0.5,print_every=-1,verbose=False):
        """
        Function that employs pure Newton Raphson iteration with back-tracking line search, following https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/14-newton.pdf at slide #11 
        stopping_criteria: given in https://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf slide 10-17
        
        Args:
            maxsearch(float): The maximum number performed over for the bisection line search 
            maxiter(int): Maximum number of iterations allowed (default 250)
            tol(float): The tolerance for convergence (default 1e-7)
            alpha(float): The alpha parameter in backtracking line search (default 0.25)
            beta(float): The beta parameter in backtracking line search (default 0.5)
            print_every(int): The frequency at which the program outputs. If -1, then program won't output anything. (default -1)
            verbose(bool): Whether or not to be verbose

        Return:
            1. lnwji= log of the weights of all the observations in the simulation (Ntot,)
            2. fi=-ln(Zi/Z0) (S,)
        """
        fi = self.fi0
        fi_prev = fi
        buji = self.buji
        Ni = self.Ni
        Ntot = Ni.sum()
        iter_ = 0

        while True:
            # Find current value of the function
            func = Uwham_NLL_eq(fi,buji,Ni)

            # Find gradient of the function
            g,_ = Uwham.gradient(fi,buji,Ni)

            # Find Hessian of the function
            H,_ = Uwham.Hessian(fi,buji,Ni)
                
            # Find v where v is the search position, v=H-1g 
            v = -np.linalg.lstsq(H,g,rcond=-1)[0]
            
            # Find the local gradient on alpha del f(x + t*v) <= f(x) + alpha*t*del f(x).v, we call del f(x).v -> m 
            t = 1 
            m = g.dot(v)
            criteria = (v.T).dot(H).dot(v)

            if criteria <= tol:
                fi = fi + t*v
                fi -= fi[-1]
                if verbose:
                    print("optimization has converged in {} iterations.".format(iter_))
                lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) 
                break

            while True:
                # Find the updated value of the function 
                func_updated = Uwham_NLL_eq(fi + t*v,buji,Ni)

                # Find the threshold
                threshold = t*m
                if verbose:
                    print("curr val:{}, updated val:{}, threshold:{}".format(func,func_updated,threshold))
                
                # Performing backtracking algorithm
                if (func_updated - func)/alpha >= threshold:
                    t = beta*t
                else:
                    fi = fi + t*v
                    fi = fi - fi[-1]
                    break
 
            error = np.max(abs(fi - fi_prev)[:-1])/np.max(abs(fi_prev)[:-1])
            iter_ += 1
            
            
            if iter_ % print_every == 0:
                print("At iteration {}, the error is {}".format(iter_, criteria)) 
            fi_prev = fi

        self.fi = fi
        self.lnwji = lnwji

        return lnwji,fi

    def Maximum_likelihood(self,ftol=2.22e-09,gtol=1e-05,maxiter=15000,maxfun=15000,disp=None,iprint=-1,index=-1):
        """
        Optimizes the negative likelihood equation of binless Wham using LBFGS algorithm as implemented by scipy.minimize, the derivatives of the MLE is found by automatic differentiation using autograd 

        Args:
            ftol(float): the tolerance as set forth by scipy.minimize (default 2.22e-09)
            gtol(float): the tolerance as set forth by scipy.minimize (default 1e-05)  
            maxiter(int): the maximum number of iterations as set forth by scipy.minimize. (default 15000)
            maxfun(int): the maximum number of function evaluations as set forth by scipy.minimize. (default 15000)
            iprint(int): the frequency at which the program outputs the result. Will not output if less than 0. (default -1)

        Return:
            1. lnwji= log of the weights of all the observations in the simulation (Ntot,)
            2. fi=-ln(Zi/Z0) (S,)
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
            fi = fi - fi[index]

            lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0) #shape (Ntot,)

            self.lnwji = lnwji
            self.fi = fi

            return lnwji,fi
        else:
            print("Optimization has not converged")
            return None
    
    def compute_betaF_unbiased(self,min_,max_,bins=100):
        """
        Function that calculates the Free energy for Uwham from the observations xji and log of the weights lnwji for unbiased simulation
        
        Args:
            min(float): the minimum of the binned vector (float/int)
            max(float): the maximum of the binned vector (float/int)
            bins(int): number of bins 

        Returns:
            1. bins_vec = binned vector from min to max (bins-1,)
            2. F = The free energy in the binned vectors from min to max for all the simulations performed(S,bins-1)
            3. logp = log of the probability in each bin
        """
        xji = self.xji.squeeze(-1)
        lnwji = self.lnwji

        if self.lnwji is None:
            raise RuntimeError("Please run Maximum_likelihood or self_consistent first to obtain weights lnwji")

        bins_vec = np.linspace(min_,max_,bins)

        # the bin size dl 
        dl = bins_vec[1] - bins_vec[0]
        
        # log(sum(exp(lnpji)))
        lnsum_ = logsumexp(lnwji)

        # This will be a vector such as np.array([1,2,3,2,..]) indicating which bin each element falls into
        digitized = np.digitize(xji,bins_vec)
        indices = [np.argwhere(digitized == j) for j in range(1,bins)]
        logpi = np.array([logsumexp(lnwji[idx]) for idx in indices]) - lnsum_ 

        Fi = np.log(dl)-logpi
        Fi = Fi - Fi.min()
        
        return bins_vec,Fi,logpi

    def compute_betaF_sim(self,min_,max_,bins=100):
        """
        Function that calculates the Free energy for Uwham from the observations xji and log of the weights lnwji
        
        Args:
            min(float): the minimum of the binned vector (float/int)
            max(float): the maximum of the binned vector (float/int)
            bins(int): number of bins 

        Returns:
            1. bins_vec = binned vector from min to max (bins-1,)
            2. F = The free energy in the binned vectors from min to max for all the simulations performed(S,bins-1)
            3. logp = log of the probability in each bin
        """
        xji = self.xji.squeeze(-1)
        buji = self.buji

        if self.lnwji is None:
            raise RuntimeError("Please run Maximum_likelihood or self_consistent first to obtain weights lnwji")

        S = self.buji.shape[0]
        bins_vec = np.linspace(min_,max_,bins)

        # the bin size dl 
        dl = bins_vec[1] - bins_vec[0]
        
        # find lnpji from all simulations (S,Ntot)
        lnpji_k = Uwham.get_lnpjik(self.lnwji,buji,self.fi)

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
    
    @staticmethod
    def get_lnpjik(lnwji,buji,fi):
        """
        Function that obtains all the weights for unbiased as well as biased simulations following the equation lnpji_k = fi-buji_k+lnwji where wji is the unbiased weights 

        Args:
            lnwji(numpy.ndarray): log of the unbiased weights (Ntot,)
            buji(numpy.ndarray): The energy matrix passed in shape (S,Ntot)
            fi(numpy.ndarray): -log(Zi/Z0) (S,)

        Return:
            lnpji_k(np.ndarray): log of the pji matrix with shape (S,Ntot)
        """
        S = buji.shape[0]
        Ntot = buji.shape[1]

        lnpjik = np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji+ np.repeat(lnwji[np.newaxis,:],S,axis=0)

        return lnpjik
    
    @staticmethod
    def gradient(fi,buji,Ni):
        """
        Function that calculates the gradient of the Negative likelihood function. The function form is as follows: gk(f) = Nk - Nk*sum_n Wnk(f) where Wnk=exp(fk)exp(-beta Uk(xn))wn 

        Args:
            fi(numpy.ndarray): fi=-log(Zi/Z0) passed in shape (S,)
            buji(numpy.ndarray): Energy matrix equal to beta*Uji passed in shape (S,Ntot)
            Ni(numpy.ndarray): Number of observations in each simulation passed in shape (S,)

        Return:
            1. g = The gradient of the negative likelihood function (S,)
            2. lnwji = Log of the unbiased weights
        """
        Ntot = Ni.sum()
        
        lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0)
        lnpjik = Uwham.get_lnpjik(lnwji,buji,fi)
 
        # Calculate the gradient of the NLL equation
        lnpk = logsumexp(lnpjik,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=1)

        return -1/Ntot*(Ni - np.exp(lnpk)),lnwji
    
    @staticmethod
    def Hessian(fi,buji,Ni):
        """
        Calculates the Hessian matrix.

        Args:
            fi(numpy.ndarray): fi=-log(Zi/Z0), the weights of each of the system (S,)
            buji(numpy.ndarray): biased energy in each of the biased simulation (S,Ntot)
            Ni(numpy.ndarray): The number of observations from each of the simulations (S,)

        Return:
            1. Hessian = The Hessian matrix (S,S)
            2. lnwji = Log of the unbiased weights (S,Ntot)
        """
        Ntot = Ni.sum()

        Nitensor = np.outer(Ni,Ni)

        lnwji = -logsumexp(np.repeat(fi[:,np.newaxis],Ntot,axis=1)-buji,b=np.repeat(Ni[:,np.newaxis],Ntot,axis=1),axis=0)
        lnpjik = Uwham.get_lnpjik(lnwji,buji,fi) 

        pjik = np.exp(lnpjik)

        H = Nitensor*pjik.dot(pjik.T)
        H -= np.diag(pjik.sum(axis=1)*Ni)

        return -1/Ntot*(H),lnwji
    
    @staticmethod
    def check_posdef(mat):
        """
        Function that checks whether a matrix is positive definite(whether all the eigenvalues are larger than 0)

        Args:
            mat(numpy.ndarray): The matrix to be checked

        Return:
            True or False based on whether the matrix is positive definite 
        """
        if np.array_equal(mat,mat.T):
            try:
                np.linalg.cholesky(mat)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False
    
def Uwham_NLL_eq(fi,buji,Ni):
    """
    Negative likilihood equation for Unbinned wham 

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
