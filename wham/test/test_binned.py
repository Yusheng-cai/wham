from wham.wham.Bwham import *
from wham.lib.wham_utils import *
from wham.wham.Uwham import *
import os
import numpy as np


def gather_data():
    nlist = ["-5.0","0.0","5.0","10.0","15.0","20.0","25.0","unbiased"]
    nnlist = [-5,0,5,10,15,20,25]
    data_list = []
    beta = 1000/(8.314*300)

    for n in nlist:
        if n != "unbiased":
            N,Ntilde = read_dat(os.getcwd()+"/data/nstar_{}/plumed.out".format(n))
            data_list.append(Ntilde[500:])
        else:
            N,Ntilde = read_dat(os.getcwd()+"/data/{}/time_samples.out".format(n))
            data_list.append(Ntilde)

    data = np.concatenate(data_list)

    Ml,Wil,fi0_b,bbins = Bwham_preprocess(data,nnlist,0,35,0.98,beta=beta,nbins=36,unbiased=True)
    Ni = np.ones((len(nlist),))*1500
    

    return Ml,Wil,fi0_b,Ni,data

def correct_data():
    f = open("data/F_Ntilde_WHAM.out")
    lines = f.readlines()
    lines = [line for line in lines if line[0]!='#']
    lines = np.array([float(line.rstrip("\n").lstrip().split()[1]) for line in lines])
    f.close()

    return lines

def test_binned():
    Ml,Wil,fi0_b,Ni,data = gather_data()
    correct = correct_data()

    _,F_b,_ = Bwham(fi0_b,Ni,Ml,Wil)
    F_b = F_b - F_b.min()

    assert np.linalg.norm(F_b - correct,2)/len(F_b) < 0.01 

def test_binned_nll():
    Ml,Wil,fi0_b,Ni,data = gather_data()
    correct = correct_data()

    _, F_b_nll, _ = Bwham_NLL(fi0_b,Ni,Ml,Wil)
    F_b_nll = F_b_nll - F_b_nll.min()

    assert np.linalg.norm(F_b_nll - correct,2)/len(F_b_nll) < 0.01 
