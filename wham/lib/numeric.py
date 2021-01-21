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

def autograd_logsumexp(a,b=1,axis=0):
    """
    Performs logsumexp using the numpy from autograd
    np.log(np.sum(a*np.exp(b)))

    Args:
        a(np.ndarray): The matrix/vector to be exponentiated
        b(np.ndarray): The number at which to multiply exp(a)
        axis(int): the axis at which to sum over

    Return:
        a matrix that is the logsumexp result of a & b
    """
    return nup.log(nup.sum(b*nup.exp(a),axis=axis))
