import sys
sys.path.insert(0,"../wham")
from Bwham import Bwham
from wham.lib.wham_utils import *
import matplotlib.pyplot as plt
import os
import numpy as np


def gather_data():
    nlist = ["-5.0","0.0","5.0","10.0","15.0","20.0","25.0","unbiased"]
    Ntwiddle = [-5,0,5,10,15,20,25]
    data_list = []
    beta = 1000/(8.314*300)
    k = 0.98

    for n in nlist:
        if n != "unbiased":
            N,Ntilde = read_dat(os.getcwd()+"/data/nstar_{}/plumed.out".format(n))
            data_list.append(Ntilde[500:])
        else:
            N,Ntilde = read_dat(os.getcwd()+"/data/{}/time_samples.out".format(n))
            data_list.append(Ntilde)

    Ni = np.ones((len(nlist),))*1500
    xji = np.concatenate(data_list)
    
    return xji,beta,Ni,Ntwiddle,k

def correct_data():
    f = open("data/F_Ntilde_WHAM.out")
    lines = f.readlines()
    lines = [line for line in lines if line[0]!='#']
    lines = np.array([float(line.rstrip("\n").lstrip().split()[1]) for line in lines])
    f.close()

    return lines

def test_binned():
    xji,beta,Ni,Ntwiddle,k = gather_data()
    min_ = 0
    max_ = 35
    nbins = 36
    correct = correct_data()
    
    b = Bwham(xji,Ni,Ntwiddle,k,min_,max_,bins=nbins,beta=beta,unbiased=True)
    _,F,_ = b.self_consistent()
    print(np.linalg.norm(F - correct,2)/len(F)) 

    assert np.linalg.norm(F - correct,2)/len(F) < 0.01 
    return np.linspace(min_,max_,nbins)[:-1],F

def test_binned_nll():
    xji,beta,Ni,Ntwiddle,k = gather_data()
    min_ = 0
    max_ = 35
    nbins = 36
    correct = correct_data()
    
    b = Bwham(xji,Ni,Ntwiddle,k,min_,max_,bins=nbins,beta=beta,unbiased=True)
    _,F,_ = b.Maximum_likelihood()
    
    print(np.linalg.norm(F - correct,2)/len(F))
    assert np.linalg.norm(F - correct,2)/len(F) < 0.01  
    return np.linspace(min_,max_,nbins)[:-1],F

if __name__ == "__main__":
    bbins,F = test_binned()
    bbins_NLL,F_NLL = test_binned_nll()
    correct = correct_data() 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bbins,F,label="Self consistent solution")
    ax.plot(bbins_NLL,F_NLL,label='Maximum likelihood solution')
    ax.plot(bbins,correct,label='Sean reference')
    ax.legend(fontsize=15)
    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("Binnedtest")
