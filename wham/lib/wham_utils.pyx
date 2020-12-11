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
    U = lines[:,3]

    return (N,Ntilde,U)


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
    Ntwiddle: The Ntwiddle in the biased simulations (S-1,) --> exclude the unbiased
    min_: minimum of the bins
    max_: maximum of the bins
    k: the harmonic constant for the bias potential (in kJ/mol)
    beta: 1/KbT (in kJ/mol), default value is for T=298K
    nbins: number of bins
    
    returns:
        Ml: Number of observations in bin l (M,)
        Wil: beta*k*0.5*(n-nstar)**2 (S,M)
    """
    Ntot = len(xji)
    if unbiased == True:
        S = len(Ntwiddle)+1
    else:
        S = len(Ntwiddle)
    M = nbins-1
    
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
    
    return Ml, Wil,bins_[:-1]

