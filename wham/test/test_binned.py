import sys
sys.path.insert(0,"..")
from Bwham import Bwham
from wham.lib.wham_utils import *
import matplotlib.pyplot as plt
import os
import numpy as np


def gather_data():
    nlist = ["-5.0","0.0","5.0","10.0","15.0","20.0","25.0","unbiased"]
    Ntwiddle = np.array([-5,0,5,10,15,20,25])
    data_list = []
    beta = 1000/(8.314*300)
    k = 0.98

    for n in nlist:
        if n != "unbiased":
            N,Ntilde,_ = read_dat(os.getcwd()+"/data/nstar_{}/plumed.out".format(n))
            data_list.append(Ntilde[200:])
        else:
            N,Ntilde,_ = read_dat(os.getcwd()+"/data/{}/time_samples.out".format(n))
            data_list.append(Ntilde)

    Ni = np.ones((len(nlist)-1,))*1800
    Ni = np.concatenate([Ni,np.array([1500])])
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
    
    b = Bwham(xji,Ntwiddle,Ni,k,min_,max_,bins=nbins,beta=beta)
    _,F,_ = b.self_consistent()
    print("Free energy of self consistent binned wham is ", F)
    print("The norm difference between self consistent wham calculated and reference is ",np.linalg.norm(F - correct,2)/len(F)) 

    assert np.linalg.norm(F - correct,2)/len(F) < 0.05 
    return np.linspace(min_,max_,nbins)[:-1],F

def test_binned_nll():
    xji,beta,Ni,Ntwiddle,k = gather_data()
    min_ = 0
    max_ = 35
    nbins = 36
    correct = correct_data()
    
    b = Bwham(xji,Ntwiddle,Ni,k,min_,max_,bins=nbins,beta=beta)
    _,F,_ = b.Maximum_likelihood()
    print("Free energy of Maximum likelihood calculation wham is ",F)
    
    print("The norm difference between Maximum likelihood wham calculated and reference is ", np.linalg.norm(F - correct,2)/len(F))
    assert np.linalg.norm(F - correct,2)/len(F) < 0.05 
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
