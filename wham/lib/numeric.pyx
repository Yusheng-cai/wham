import numpy as np

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
