# Wham
Code where I developed during my phD studies at Amish Patel's lab from 2019-2024. This code contains implementation of binned wham from the paper "
J Comput Chem. 2012 Feb 5;33(4):453-65. doi: 10.1002/jcc.21989. Epub 2011 Nov 23." and binless wham from the paper "J. Chem. Phys. 136, 144102 (2012); 
https://doi.org/10.1063/1.3701175.".

documentation is at: https://yusheng-cai.github.io/wham/

## Installation

Download the source code and run 

```bash
python setup.py install
```

## Usage

```python
from wham.Uwham import *
import numpy as np

Ntwiddle = np.array([-5,0,5,10,15,20,25,0]) #Where 0 just means unbiased
k = np.ones((len(Ntwiddle),1))*0.98
k[-1] = 0 #last one is unbiased
Ni = np.ones((len(Ntwiddle),1))*1500

# Do some operation to load xji
xji = load_data()

# instantiate Uwham
U = Uwham(xji,k,Ntwiddle,Ni)

# get lnwji using either of the three methods
lnwji,fi = U.adaptive()
lnwji,fi = U.Newton_Raphson()
lnwji,fi = U.Maximum_likelihood()
```

## Test

In wham/test/ folder run the following command 

```bash
pytest 
```

Comparison to other codes
![Comparison](/wham/test/Binlesstest.png)




## License
[MIT](https://choosealicense.com/licenses/mit/)
