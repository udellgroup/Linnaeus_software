# Linnaeus

Linnaeus is a python package for the classification of iterative algorithms and 
identification of relations between algorithms. 
Algorithms are described with variables, 
parameters, functions, oracles and update equations. 
Linnaeus can check algorithm relations including

* Oracle equivalence (algorithms generate identical sequence of oracle calls)
* Linear state transform (states are identical up to a linear map)
* Permutation (shift equivalence, algorithms generate identical sequence of oracle calls up to a shift)
* Repetition (one algorithm repeats the updates of another algorithm)
* Conjugation (algorithm calls conjugate function oracles, i.e. Fenchel conjugate)
* Duality (conjugation on all algorithm oracles)

For more information about algorithm equivalence and relations, see our paper. 

## Installation

To install using pip, run:

```python
pip install git+https://github.com/QCGroup/linnaeus
```

To install from source, first make sure setuptools has been installed. Then,

1. Clone the Linnaeus git repo.
2. Navigate to the top-level of the directory and run:

```python
python setup.py install
```

To test the installation with nose2, first make sure nose2 has been installed. Then run:

```python
nose2 linnaeus
```

The requirements are:

* SymPy >= 1.6.2
* NumPy >= 1.16
* SciPy >= 1.2.1
* Python 3.x


## Algorithm library

Linnaeus include an algorithm library to serve as a reference. Users can check the 
potential relations between the input algorithm and reference algorithms in the library. 
Currently, the algorithm library includes:

* Gradient descent method
* Nesterov's accelerated gradient method (with generalization)
* Heavy-ball method 
* Proximal point method (with generalization)
* Proximal gradient method (ISTA, FISTA)
* Douglas-Rachford splitting method (with different versions)
* Peaceman-Rachford splitting method
* Alternating direction method of multipliers (ADMM)
* Chambolle-Pock method
* Davis-Yin splitting method
* Extragradient method (with different versions)


