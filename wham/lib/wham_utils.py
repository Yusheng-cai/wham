import numpy as np

def read_dat(file_path):
    """
    Function that reads the .dat from INDUS simulations

    Args:
        file_path(str): the path to the file 

    Return:
        an numpy array that contains the N and Ntilde from the INDUS simulation (N, Ntilde) -> where both are of shape (nobs,)
    """
    f = open(file_path)
    lines = f.readlines()
    lines = [line for line in lines if line[0]!="#"]

    lines = [[float(num) for num in line.rstrip("\n").split()] for line in lines if line[0]!="#"]
    lines = np.array(lines)

    N = lines[:,1]
    Ntilde = lines[:,2]

    return (N,Ntilde,lines)

def make_bins(data,min,max,bins=101):
    """
    A function that bins some data within min and max
    
    Args:
        data(np.ndarray): the data that you want to bin, pass in numpy array (shape(N,))
        min(float): minimum of the bins
        max(float): maximum of the bins
        bins(int): number of bins to make
    
    returns:
        tuple of (bins,binned_vec)
    """
    bin_ = np.linspace(min,max,bins)

    # right = False implies bins[i-1]<=x<bins[i]
    digitized = np.digitize(data,bin_,right=False)

    binned_vec = np.array([(digitized == i).sum() for i in range(1,bins)])


    return (bin_[:-1],binned_vec)

def cov_fi(wji,Ni):
    """
    This is a function that calculates the covariance matrix of fi where
    fi = -ln(Qi/Q0).

    Args:
        wji(np.ndarray): the weight matrices of different simulations (N,k) where N=number of observations 
        in total, k=number of simulations

        Ni(np.ndarray): the number of observations in each simulation

    returns 
        covariance of fi in shape (k,k)
    """
    # obtain shape information
    N,k = wji.shape

    # create a diagonal matrix of Ni
    N = np.diag(Ni)
    
    # define identity matrix 
    I = np.eye(k)

    # Perform SVD on wji matrix
    U, s , Vt = np.linalg.svd(wji,full_matrices=False)
    s = np.diag(s)


    # Inner part of covariance
    inner = I - s.dot(Vt).dot(N).dot(Vt.T).dot(s)
    pseudo_inverse = np.linalg.pinv(inner)

    # find covariance of ln(Qi)
    theta = (Vt.T).dot(s).dot(pseudo_inverse).dot(s).dot(Vt)

    # Using the above to find the covariance of -ln(Qi/Q0) where Q0 is at the 
    # end of the array Cov_i = Theta_{-1,-1} - 2*Theta_{-1,i} + Theta_{i,i}
    
    cov = np.zeros((k,))
    for i in range(k):
        cov[i] = theta[-1,-1] - 2*theta[-1,i] + theta[i,i]


    return cov
