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
Systerm parses `algo1` and returns the update equations, 

<img src="/docs/Figures/title1.svg?invert_in_darkmode" align=middle width="75" height="75"/>

<img src="/docs/Figures/output1.svg?invert_in_darkmode" align=middle width="170" height="170"/>

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
System parses `algo2` and returns the update equations,

<img src="/docs/Figures/title2.svg?invert_in_darkmode" align=middle width="75" height="75"/>

<img src="/docs/Figures/output2.svg?invert_in_darkmode" align=middle width="175" height="175"/>

Check oracle equivalence : 
```python
lin.is_equivalent(algo1, algo2)
```

System returns 
```python
True
```
which means that `algo1` and `algo2` are oracle-equivalent. 

## Add new algorithm to library

If you believe that an algorithm is important and not included in the library of Linnaeus, 
you are welcome to submit a GitHub pull request named as `Adding new algorithm XXX`, 
where `XXX` is the name of the new algorithm. 
To add a new algorithm, you need to edit the `algorithms_library.py` file under `linnaeus/` directory. 
New algorithm should be added at the end of the file. 

The procedures to add a new algorithm are generally the same 
as the steps to input and parse an algorithm in Linnaeus as stated in [quick tutorial](https://linnaeus-doc.github.io/quick_tutorial/). 

The following code shows to add Chambolle-Pock method to the library. 

```python
# define name "Chambolle-Pock method" and name string "Cp"
Cp = Algorithm("Chambolle-Pock method", "Cp")

# define functions 
f, g = Cp.add_function("f", "g")
# define parameters
sigma, tau, theta = Cp.add_parameter("sigma", "tau", "theta")
# define varaibles
x1, x2, x3, x4 = Cp.add_var("x1", "x2", "x3", "x4")
# define update equations
Cp.add_update(x1, prox(f, tau)(x3 - tau * x4))  
Cp.add_update(x2, prox(g, sigma)(x4 + sigma * (2 * x1 - x3)))
Cp.add_update(x3, x3 + theta * (x1 - x3)) 
Cp.add_update(x4, x4 + theta * (x2 - x4)) 

# parse the algorithm without showing the update equations
Cp._parse()
# add latex representation of Chambolle-Pock method
Cp.equation_string = r"""
 x_1^+ = \text{prox}_{\tau f} (x_3 - \tau x_4) \\
 x_2^+ = \text{prox}_{\sigma g^*} (x_4 + \sigma (2x_1^+ - x_3)) \\
 x_3^+ = x_3 + \theta (x_1^{+} - x_3) \\ 
 x_4^+ = x_4 + \theta (x_2^+ - x_4) 
    """
# insert Chambolle-Pock method to library
algo_library.library_insert(Cp)

```

To avoid overlap in algorithms, we highly recommand you to [check equivalence and relations](https://linnaeus-doc.github.io/detection/#detection)
between the new algorithm and existing algorithms in the library to ensure uniqueness. 

## Citing Linnaeus

If you use Linnaeus for published work,
we encourage you to cite the software. 

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
