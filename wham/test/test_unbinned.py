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

    uji,fi0_u = Uwham_preprocess(data,nnlist,0.98,beta=beta)
    Ni = np.ones((len(nlist),))*1500


    return uji,fi0_u,Ni,data

def correct_data():
    f = open("data/F_Ntilde_WHAM.out")
    lines = f.readlines()
    lines = [line for line in lines if line[0]!='#']
    lines = np.array([float(line.rstrip("\n").lstrip().split()[1]) for line in lines])
    f.close()

    return lines

def test_u_nll():
    uji,fi0_u,Ni,data = gather_data()
    correct = correct_data()

    wji_u_nll, fi_u_nll = Uwham_NLL(fi0_u,uji,Ni)
    _, pl_u_nll = weighted_hist_Uwham(data,wji_u_nll,0,35,bins=36)
    F_u_nll = -np.log(pl_u_nll)

    F_u_nll = F_u_nll - F_u_nll.min()
    assert np.linalg.norm(F_u_nll - correct,2)/len(F_u_nll) < 0.05 

def test_u():
    uji,fi0_u,Ni,data = gather_data()
    correct = correct_data()

    wji_u, fi_u = Uwham(fi0_u,uji,Ni)
    _, pl_u = weighted_hist_Uwham(data,wji_u,0,35,bins=36)
    F_u = -np.log(pl_u)

    F_u = F_u - F_u.min()
    assert np.linalg.norm(F_u - correct,2)/len(F_u) < 0.05 
