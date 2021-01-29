import sys
sys.path.insert(0,'..')
sys.path.insert(0,"../lib/")
from Uwham import *
from Bwham import *
import numpy as np
from wham_utils import read_dat
import matplotlib.pyplot as plt
import os


def gather_data():
    nlist = ["-5.0","0.0","5.0","10.0","15.0","20.0","25.0","unbiased"]
    Ntwiddle = [-5,0,5,10,15,20,25]
    data_list = []
    beta = 1000/(8.314*300)
    k = 0.98

    for n in nlist:
        if n != "unbiased":
            N,Ntilde = read_dat(os.getcwd()+"/data/nstar_{}/plumed.out".format(n))
            data_list.append(Ntilde[200:])
        else:
            N,Ntilde = read_dat(os.getcwd()+"/data/{}/time_samples.out".format(n))
            data_list.append(Ntilde)

    Ni = np.ones((len(nlist)-1,))*1800
    Ni = np.concatenate([Ni,np.array([1500])])
    xji = np.concatenate(data_list)
    
    return xji,Ni,Ntwiddle,beta,k

def correct_data():
    f = open("data/F_Ntilde_WHAM.out")
    lines = f.readlines()
    lines = [line for line in lines if line[0]!='#']
    lines = np.array([float(line.rstrip("\n").lstrip().split()[1]) for line in lines])
    f.close()

    return lines

def test_u_nll():
    xji,Ni,Ntwiddle,beta,k = gather_data()
    correct = correct_data()
    
    u = Uwham(xji,k,Ntwiddle,Ni,beta=beta)
    a,b = u.Maximum_likelihood(ftol=1e-15)
    ubins,F,f = u.compute_betaF_profile(0,35,bins=36)
    print("Free energy of MLE is ",F[-1])
    F = F[-1]
    NLL = Uwham_NLL_eq(b,u.buji,Ni)

    print("MLE error with reference is:{}".format(np.linalg.norm(F - correct,2)/len(F))) 
    print("MLE NLL is {}".format(NLL))

    assert np.linalg.norm(F - correct,2)/len(F) < 0.05 
    return ubins,F

def test_u():
    xji,Ni,Ntwiddle,beta,k = gather_data()
    correct = correct_data()
    
    u = Uwham(xji,k,Ntwiddle,Ni,beta=beta)
    a,b = u.self_consistent(tol=1e-8)
    ubins,F,f = u.compute_betaF_profile(0,35,bins=36)
    print("Free energy of binless self consistent is ",F[-1])
    NLL = Uwham_NLL_eq(b,u.buji,Ni)

    F = F[-1]
    print("self consistent error with reference is:{}".format(np.linalg.norm(F - correct,2)/len(F))) 
    print("self consistent NLL is {}".format(NLL))

    assert np.linalg.norm(F - correct,2)/len(F) < 0.05  

    return ubins,F

if __name__ == '__main__':
    ubins_nll,F_nll = test_u_nll()
    ubins,F = test_u()
    correct = correct_data()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ubins_nll,F_nll,'r',label="Negative likelihood")
    ax.plot(ubins,correct,label='Sean reference')
    ax.plot(ubins,F,'b--',label='Self iterative result')
    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F$")
    ax.legend(fontsize=15)

    plt.savefig("Binlesstest.png")
