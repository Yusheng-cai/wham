import sys
sys.path.insert(0,"../lib/")
from numeric import *
import numpy as np
from scipy.special import logsumexp

def test_lsp():
    test_data = np.random.randn(10000)+100
    
    assert np.allclose(logsumexp(test_data),alogsumexp(test_data))
