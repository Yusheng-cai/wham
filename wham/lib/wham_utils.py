import numpy as np
from scipy import integrate

def read_dat_gen(file_path):
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

    return lines

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

def ss_umbrella(qstar,qavg,qvar,kappa):
    """
    Sparse sampling performed for umbrella potentials k/2(q-q*)^2 

    Args:
        qstar(numpy.ndarray): A numpy array of the q* where q is the order parameter
        qavg(numpy.ndarray): A numpy array of the mean values at every simulation <q> where q is the Order Parameter
        qvar(numpy.ndarray): A numpy array of the variance values at every simulation <(q-<q>)^2>
        kappa(float): The kappa parameter in the potential

    Returns:
        F(numpy.ndarray): The unbiased free energy calculated from sparse sampling
    """
    FvkN = np.log(2*np.pi*qvar)

    integrand=kappa*(qstar - qavg)
    FkN = np.zeros((len(integrand),))
    for i in range(2,len(integrand)+1):
        FkN[i-1] = integrate.trapz(integrand[:i],qstar[:i])

    UkN = kappa/2*(qstar-qavg)**2

    F = FvkN - UkN + FkN
    F = F - F.min()
    return (FvkN,FkN,UkN),F

def generateCombineDataInput(file:list, colnums:list, skip:int, skipfrombeginning:list, filename='Combine.dat'):
    """
    Generate the combined data input
    """
    # find out the dimensions of the input data
    dimension = len(colnums[0])

    # write timeseries
    f = open(filename, "w")
    for (i,fi) in enumerate(file):
        f.write("timeseries = {\n")
        f.write("\tpath = {}\n".format(fi))
        f.write("\tcolumns = [ ")
        for col in colnums[i]:
            f.write("{} ".format(col))
        f.write("]\n")
        f.write("\tskip = {}\n".format(skip))
        if len(skipfrombeginning) == 1:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[0]))
        else:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[i]))
        f.write("}\n")
        f.write("\n")
    f.write("\n")
    f.write("tsoperation = {\n")
    f.write("\ttype = combine_data \n")
    f.write("\toutputs = [ totaldata ]\n")
    f.write("\toutputNames = [ ]\n")
    f.write("}\n")
    
    f.close()

def generateHistogramInput(file:list, colnums:list, skip:int, skipfrombeginning:list, filename='Combine.dat'):
    """
    Generate the combined data input
    """
    # find out the dimensions of the input data
    dimension = len(colnums[0])

    # write timeseries
    f = open(filename, "w")
    for (i,fi) in enumerate(file):
        f.write("timeseries = {\n")
        f.write("\tpath = {}\n".format(fi))
        f.write("\tcolumns = [ ")
        for col in colnums[i]:
            f.write("{} ".format(col))
        f.write("]\n")
        f.write("\tskip = {}\n".format(skip[i]))
        if len(skipfrombeginning) == 1:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[0]))
        else:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[i]))
        f.write("}\n")
        f.write("\n")
    f.write("\n")
    f.write("tsoperation = {\n")
    f.write("\ttype = histogram \n")
    f.write("\toutputs = [ histogram totaldata ]\n")

    for i in range(dimension):
        f.write("\tbins = {\n")
        f.write("\t\tdimension = {}\n".format(i+1))
        f.write("\t\trange = [ ]\n")
        f.write("\t\tnumbins = \n")
        f.write("\t}\n")
    f.write("\toutputNames = [ ]\n")
    f.write("}\n")
    
    f.close()

def generateWhamInputCombined(file:str, kappa:list, xstar:list, Nvec=[], temperature=320, filename="input.dat", reweightPhi=[]):
    """
    Performing Input for combined wham input

    Args:
    -----
        file(str)   : One file that corresponds to the combined input data
        kappa(list) : The list of kappas (list[list])
        xstar(list) : The list of xstars (list[list])
        Nvec(list)  : Optional argument that inputs the number of data per simulation
        temperature(float)  : The temperature at which the calculations are performed at 
    """
    # find out the dimensions of the input data
    dimension = len(xstar[0])

    # write timeseries
    f = open(filename, "w")
    f.write("timeseries = {\n")
    f.write("\tpath = {}\n".format(file))
    f.write("\tcolumns = [ ")
    for i in range(dimension):
        f.write("{} ".format(i+1))
    f.write("]\n")
    f.write("\tskip = 0\n")
    f.write("\tskipfrombeginning = 0\n")
    f.write("}\n")
    f.write("\n")

    # write bias
    for (i,x) in enumerate(xstar):
        f.write("bias = {\n")
        f.write("\tdimension = {}\n".format(dimension))
        f.write("\txstar = [ ")
        for xmore in x:
            f.write("{} ".format(xmore))
        f.write("]\n")
        f.write("\tkappa = [ ")
        if len(kappa) > 1:
            for k in kappa[i]:
                f.write("{} ".format(k))
        else:
            for k in kappa[0]:
                f.write("{} ".format(k))
        f.write("]\n")
        f.write("\ttemperature = {}\n".format(temperature))
        f.write("}\n")
        f.write("\n")
    
    f.write("wham = {\n")
    f.write("\tname = w\n")
    f.write("\ttype = Uwham\n")
    f.write("\tUwhamstrategy = {\n")
    f.write("\t\ttype = LBFGS\n")
    f.write("\t\tname = l\n")
    f.write("\t\tmax_iterations = 100\n")
    f.write("\t\tprintevery = 1\n")
    f.write("\t}\n")

    f.write("\tUwhamstrategy = {\n")
    f.write("\t\ttype = adaptive\n")
    f.write("\t\tname = a\n")
    f.write("\t\tprintevery = 1\n")
    f.write("\t}\n")
    f.write("\tstrategyNames = [ l a ]\n")

    for i in range(dimension):
        f.write("\tbins = {\n")
        f.write("\t\tdimension = {}\n".format(i+1))
        f.write("\t\trange = [ ] \n")
        f.write("\t\tnumbins = \n")
        f.write("\t}\n")

    if len(Nvec) > 0: 
        f.write("\tNvec = [ ")
        for n in Nvec:
            f.write("{} ".format(n))
        f.write("]\n")
    else:
        f.write("\tN = \n")

    f.write("\toutputs = [ pji histogram Averages ]\n")
    f.write("\toutputFile = [ p.out h.out avg.out ]\n")
    f.write("}\n")
    f.write("\n")

    if len(reweightPhi) != 0:
        f.write("Reweight = {\n")
        for phi in reweightPhi:
            f.write("\tbias = {\n")
            f.write("\t\tdimension = {}\n".format(dimension))
            f.write("\t\tphi = [ ") 
            assert len(phi) == dimension , "Same dimension in phi please!"
            for p in phi:
                f.write("{} ".format(p))
            f.write(" ]\n")
            f.write("\t}\n")
        
        f.write("\toutputs = [ ReweightAverages ]\n")
        f.write("\toutputNames = [ ReweightAvg.out ]\n")
        f.write("\twham = w\n")
        f.write("\ttype = \n")
        f.write("}\n")
    f.close()



def generateWhamInput(file:list, colnums:list, skip:list, skipfrombeginning:list, xstar=None, kappa=None, phi=None, reweightPhi=[],\
     temperature=295, filename="input.dat" , Nvec=[]):
    """
    A function that writes the input file for Wham calculations

    Args:
        file(list) : list of string of input file names 
        colnums(list) : The column numbers for each of the simulations output, assumes the same for all time series 
        skip(int) : How many number to skip for each timeseries
        skipfrombeginning(list) : How much to skip from beginning of data
        kappa(list(list)) : list of list of kappa 
        xstart(list(list)) : Does not assume the same for every simulation, usually an (N,d) list 
    """
    # find out the dimensions of the input data
    dimension = len(colnums[0])

    # write timeseries
    f = open(filename, "w")
    for (i,fi) in enumerate(file):
        f.write("timeseries = {\n")
        f.write("\tpath = {}\n".format(fi))
        f.write("\tcolumns = [ ")
        for col in colnums[i]:
            f.write("{} ".format(col))
        f.write("]\n")
        f.write("\tskip = {}\n".format(skip[i]))
        if len(skipfrombeginning) == 1:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[0]))
        else:
            f.write("\tskipfrombeginning = {}\n".format(skipfrombeginning[i]))
        f.write("}\n")
        f.write("\n")

    # write bias
    for i in range(len(file)):
        f.write("bias = {\n")
        f.write("\tdimension = {}\n".format(dimension))
        
        # write kappa * 0.5 * (x - xstar)**2 potential
        if xstar is not None:
            f.write("\txstar = [ ")
            for xmore in xstar[i]:
                f.write("{} ".format(xmore))
            f.write("]\n")
        if kappa is not None:
            f.write("\tkappa = [ ")
            if len(kappa) > 1:
                for k in kappa[i]:
                    f.write("{} ".format(k))
            else:
                for k in kappa[0]:
                    f.write("{} ".format(k))
        
        # write phi * x potential
        if phi is not None:
            f.write("\tphi = [ ")
            for p in phi[i]:
                f.write("{} ".format(p))
        f.write("]\n")
        f.write("\ttemperature = {}\n".format(temperature))
        f.write("}\n")
        f.write("\n")
    
    f.write("wham = {\n")
    f.write("\tname = w\n")
    f.write("\ttype = Uwham\n")
    f.write("\tUwhamstrategy = {\n")
    f.write("\t\ttype = LBFGS\n")
    f.write("\t\tname = l\n")
    f.write("\t\tmax_iterations = 100\n")
    f.write("\t\tprintevery = 1\n")
    f.write("\t}\n")

    f.write("\tUwhamstrategy = {\n")
    f.write("\t\ttype = adaptive\n")
    f.write("\t\tname = a\n")
    f.write("\t\tprintevery = 1\n")
    f.write("\t}\n")

    f.write("\tstrategyNames = [ l a ]\n")

    for i in range(dimension):
        f.write("\tbins = {\n")
        f.write("\t\tdimension = {}\n".format(i+1))
        f.write("\t\trange = [ ] \n")
        f.write("\t\tnumbins = \n")
        f.write("\t}\n")

    if len(Nvec) > 0: 
        f.write("\tNvec = [ ")
        for n in Nvec:
            f.write("{} ".format(n))
        f.write("]\n")
    f.write("\toutputs = [ pji histogram Averages forces ]\n")
    f.write("\toutputFile = [ p.out h.out avg.out f.out ]\n")
    f.write("}\n")
    f.write("\n")

    if len(reweightPhi) != 0:
        f.write("Reweight = {\n")
        for phi in reweightPhi:
            f.write("\tbias = {\n")
            f.write("\t\tdimension = {}\n".format(dimension))
            f.write("\t\tphi = [ ") 
            for p in phi:
                f.write("{} ".format(p))
            f.write(" ]\n")
            f.write("\t}\n")
        
        f.write("\toutputs = [ ReweightAverages ]\n")
        f.write("\toutputNames = [ ReweightAvg.out ]\n")
        f.write("\twham = w\n")
        f.write("\ttype = \n")
        f.write("}\n")
    f.close()
