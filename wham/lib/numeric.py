import numpy as np
import autograd.numpy as nup

def safe_divide(a,b):
    """
    Performs divide dividing a 0 in the denominator would be treated as 0
    
    Args:
        a(np.ndarray): a vector
        b(np.ndarray): a vector
    Return:
        A vector which is equal to a/b that ignores 0 
    """
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
    """
    Safely performs logarithms
    """
    ans = np.log(a, out=np.zeros_like(a), where=(a!=0))

    return ans

def alogsumexp(a,b=None,axis=None,keepdims=False):
    """
    Performs logsumexp using the numpy from autograd
    np.log(np.sum(a*np.exp(b)))

    Args:
        a(np.ndarray): The matrix/vector to be exponentiated  (shape (N,..))
        b(np.ndarray): The number at which to multiply exp(a) (shape (N,)) (default None)
        axis(int): the axis at which to sum over (defaul None)
        keepdims(bool): whether to keep the result as the same shape (default False)

    Return:
        a matrix that is the logsumexp result of a & b
    """
    if b is not None:
        if nup.any(b==0):
            a = a + 0. # promote to at least float
            a[b == 0] = -nup.inf

    # find maximum of a along the axis provided 
    a_max = nup.amax(a,axis=axis,keepdims=True)

    if b is not None:
        b = nup.asarray(b)
        tmp = b * nup.exp(a-a_max)
    else:
        tmp = nup.exp(a-a_max)

    
    #suppress warnings about log of zero
    with nup.errstate(divide='ignore'):
        s = nup.sum(tmp,axis=axis,keepdims=keepdims)

    out = nup.log(s)

    if not keepdims:
        a_max = nup.squeeze(a_max,axis=axis)

    out += a_max

    return out 
