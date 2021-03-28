import sys
sys.path.insert(0,'..')
sys.path.insert(0,"../lib/")
from Uwham import Uwham_NLL_eq,Uwham
from Bwham import *
import numpy as np
from wham_utils import read_dat
import os
from autograd import jacobian,hessian 

def gather_data():
    nlist = ["-5.0","0.0","5.0","10.0","15.0","20.0","25.0","unbiased"]
    Ntwiddle = [-5,0,5,10,15,20,25,0]
    data_list = []
    beta = 1000/(8.314*300)
    k = 0.98*np.ones((len(nlist),1))
    k[-1] = 0

    for n in nlist:
        if n != "unbiased":
            N,Ntilde,_ = read_dat(os.getcwd()+"/data/nstar_{}/plumed.out".format(n))
            data_list.append(Ntilde[200:])
        else:
            N,Ntilde,_ = read_dat(os.getcwd()+"/data/{}/time_samples.out".format(n))
            data_list.append(Ntilde)

    Ni = np.ones((len(nlist)-1,))*1801
    Ni = np.concatenate([Ni,np.array([1501])])
    xji = np.concatenate(data_list)
    U = Uwham(xji,k,Ntwiddle,Ni,beta)
    return U
    


def test_jacobian():
    U = gather_data()
    j = jacobian(Uwham_NLL_eq)
    Ni = U.Ni

    fi0 = U.fi0
    autograd_result = j(fi0,U.buji,Ni)
    my_result,_ = U.gradient(fi0,U.buji,U.Ni)
    
    # gradient from autograd gives negative of my result
    assert np.allclose(autograd_result,my_result)

def test_hessian():
    U = gather_data()
    h = hessian(Uwham_NLL_eq)
    Ni = U.Ni

    fi0 = U.fi0
    autograd_result = h(fi0,U.buji,Ni)
    my_result,_ = U.Hessian(fi0,U.buji,Ni)
    
    # hessian from autograd gives negative of my result
    assert np.allclose(autograd_result,my_result)
