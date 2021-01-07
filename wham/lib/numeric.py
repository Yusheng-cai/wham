import numpy as np
import autograd.numpy as nup

def safe_divide(a,b):
    assert type(a) == type(b)
    
    if isinstance(a,float):
        if b != 0:
            return a/b
        else:
            return 0
    if isinstance(a,int):
        if b != 0:
            return a/b
        else:
            return 0
        
    if isinstance(a,np.ndarray):
        ans = np.divide(a,b, out=np.zeros_like(a), where=b!=0)

        return ans

def safe_log(a):
    ans = np.log(a, out=np.zeros_like(a), where=(a!=0))

    return ans

def autograd_logsumexp(a,b=1,axis=0):
    return nup.log(nup.sum(b*nup.exp(a),axis=axis))
