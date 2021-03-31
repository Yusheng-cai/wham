import sys
sys.path.insert(0,'..')
sys.path.insert(0,"../lib/")
from Uwham import *
from Bwham import *
import numpy as np
from wham_utils import *
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
    
    return xji,Ni,Ntwiddle,beta,k,data_list

def run():
    xji,Ni,Ntwiddle,beta,k,data_list = gather_data()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for d in data_list:
        bins,bin_val = make_bins(d,d.min(),d.max(),bins=100)
        ax.plot(bins,bin_val/bin_val.sum())

    plt.savefig("Histogram.png")

if __name__ == "__main__":
    run()
