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

For more information about algorithm equivalence and relations, see [our paper](https://arxiv.org/abs/2105.04684). 

The documentation is available at [linnaeus-doc.github.io](https://linnaeus-doc.github.io)

## Installation

To install using pip, run:

```python
pip install git+https://github.com/udellgroup/Linnaeus_software
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


## Quick start

In this quick tutorial, we define two algorithms and check oracle equivalence. 

The first algorithm, 

<img src="/docs/Figures/algo1.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Import `linnaeus` and `Algorithm` class,
```python 
import linnaeus as lin
from linnaeus import Algorithm
```
Define the first algorithm as `algo1` with name `Algoritm 1`,
```python
algo1 = Algorithm("Algorithm 1")
```
Add variables `x1`, `x2`, and `x3` to `algo1`,
```python
x1, x2, x3 = algo1.add_var("x1", "x2", "x3")
```
Add oracle `gradf`, gradient of `f`:
```python
gradf = algo1.add_oracle("gradf")
```
Add update equations:
```python
# x3 <- 2x1 - x2 
algo1.add_update(x3, 2*x1 - x2)  
# x2 <- x1
algo1.add_update(x2, x1)  
# x1 <- x3 - 1/10*gradf(x3)
algo1.add_update(x1, x3 - 1/10*gradf(x3))  
```
Parse `algo1`:
```python
algo1.parse()
```
Systerm returns the state-space realization of `algo1`, 

<img src="/docs/Figures/title1.svg?invert_in_darkmode" align=middle width="140" height="140"/>

<img src="/docs/Figures/output1.svg?invert_in_darkmode" align=middle width="320" height="320"/>

The second algorithm,

<img src="/docs/Figures/algo2.svg?invert_in_darkmode" align=middle width="150" height="150"/>

Define the second algorithm as `algo2` and parse it:
```python
algo2 = Algorithm("Algorithm 2")
xi1 = algo2.add_var("xi1")
gradf = algo2.add_oracle("gradf")

# xi3 <- xi1
algo2.add_update(xi3, xi1)
# xi1 <- xi1 - xi2 - 1/5*gradf(xi1)
algo2.add_update(xi1, xi1 - xi2 - 1/5*gradf(xi3))  
# xi2 <- xi2 + 1/10*gradf(xi3)
algo2.add_update(xi2, xi2 + 1/10*gradf(xi3))  

algo2.parse()
```
System returns the state-space realization of `algo2`,

<img src="/docs/Figures/title2.svg?invert_in_darkmode" align=middle width="140" height="140"/>

<img src="/docs/Figures/output2.svg?invert_in_darkmode" align=middle width="320" height="320"/>

Check oracle equivalence : 
```python
lin.is_equivalent(algo1, algo2)
```

System returns 
```python
True
```
which means that `algo1` and `algo2` are oracle-equivalent. 

## Citing Linnaeus

If you use Linnaeus for published work,
we encourage you to cite the software

Use the following BibTeX citation:

```bibtex
@misc{linnaeus,
      title={An automatic system to detect equivalence between iterative algorithms}, 
      author={Shipu Zhao and Laurent Lessard and Madeleine Udell},
      year={2021},
      eprint={2105.04684},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
