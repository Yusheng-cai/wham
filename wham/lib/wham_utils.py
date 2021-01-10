import numpy as np

def read_dat(file_path):
    """
    Function that reads the .dat from INDUS simulations

    file_path: the path to the file 

    outputs:
        an numpy array that contains the N and Ntilde from the INDUS simulation (N, Ntilde) -> where both are of shape (nobs,)
    """
    f = open(file_path)
    lines = f.readlines()

    lines = [[float(num) for num in line.rstrip("\n").split()] for line in lines if line[0]!="#"]
    lines = np.array(lines)

    N = lines[:,1]
    Ntilde = lines[:,2]

    return (N,Ntilde)


def weighted_hist_Uwham(xji,wji,min_,max_,bins=100):
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
        F: The free energy in the binned vectors from min_ to max_ (bins-1,)
    """
    Ntot = xji.shape[0]
    bins_vec = np.linspace(min_,max_,bins)

    # The weighted probability for each bin
    p = np.zeros((bins-1,))

    for i in range(bins-1):
        mini = bins_vec[i]
        maxi = bins_vec[i+1]
        idx_ = np.argwhere((xji>=mini) & (xji<maxi))
        for index in idx_:
            p[i] += wji[index]


    return (bins_vec[:-1],p)

def Bwham_preprocess(xji,Ntwiddle,min_,max_,k,beta=0.4036,nbins=101,unbiased=True):
    """
    xji: All the observations from the umbrella sampling simulations (Ntot,)

    Ntwiddle: The Ntwiddle in the biased simulations --> exclude the unbiased

    min_: minimum of the bins

    max_: maximum of the bins

    k: the harmonic constant for the bias potential (in kJ/mol)

    beta: 1/KbT (in kJ/mol), default value is for T=298K

    nbins: number of bins

    unbiased: a boolean value signifying whether or not to include the unbiased simulation, 
              if True, a list of zero's will be included for Wil
    
    returns:
        Ml: Number of observations in bin l (M,)
        Wil: beta*k*0.5*(n-nstar)**2 (S,M)
        fi0: The initial guesses for the Bwham (array of 0's)
    """
    if unbiased == True:
        S = len(Ntwiddle)+1
    else:
        S = len(Ntwiddle)
    M = nbins-1

    fi0 = np.zeros((S,))
    
    # Find Ml
    bins_ = np.linspace(min_,max_,nbins)
    digitized = np.digitize(xji,bins_,right=False)
    Ml = np.array([(digitized == i).sum() for i in range(1,nbins)]) # shape nbins-1 (M,)
    
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

def Uwham_preprocess(xji,Ntwiddle,k,beta=0.4036,unbiased=True):
    """
    xji: All the observations from the umbrella sampling simulations (Ntot,)

    Ntwiddle: The Ntwiddle in the biased simulations --> exclude the unbiased

    k: the harmonic constant for the bias potential (in kJ/mol)

    beta: 1/KbT (in kJ/mol), default value is for T=298K

    unbiased: a boolean, if True, the unbiased energy will be added ( a list of 0's)
    
    returns:
        uji: beta*k*0.5*(n-nstar)**2 (S,Ntot)
        fi0: initial guesses for Uwham (S,)
    """ 
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

def make_bins(data,min_,max_,bins=101):
    """
    A function that bins some data within min_ and max_

    data: the data that you want to bin pass in numpy array (shape(N,))
    min_: minimum of the bins
    max_: maximum of the bins
    bins: number of bins to make
    
    returns:
        tuple of (bins,binned_vec)
    """
    bin_ = np.linspace(min_,max_,bins)

    # right = False implies bins[i-1]<=x<bins[i]
    digitized = np.digitize(data,bin_,right=False)

    binned_vec = [(digitized == i).sum() for i in range(1,bins)]


    return (bin_[:-1],binned_vec)
