Equivalence Detection
===========

Parse an algorithm
-------------------

Suppose an algorithm is already defined and variables, parameters, oracles, and 
update equations are added to the algorithm, the next step is to parse the algorithm. 
Syntax `parse` is used to translate the input algorithm into a canonical form (transfer function) and use 
the canonical form to perform subsequent analysis such as equivalence detection. 
When parsing an algorithm, the update equations are provided by the system. 

With example of gradient descent algorithm, 

<img src="/Figures/gradient_algo.svg?invert_in_darkmode" align=middle width="130" height="130"/>


The following code defines the gradient descent algorithm as `algo3` and parses it into canonical form. 
```python
import linnaeus as lin
from linnaeus import Algorithm

algo3 = Algorithm("gradient descent algorithm")
x1 = algo3.add_var("x1")
t = algo3.add_parameter("t")
gradf = algo3.add_oracle("gradf")

algo3.add_update(x1, x1 - t*gradf(x1))
algo3.parse()
```
<img src="/Figures/gradient_title.svg?invert_in_darkmode" align=middle width="210" height="210"/>

<img src="/Figures/gradient_eq.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Detection
-----------

Linnaeus provides a set of functions to check algorithm equivalence and relations. 

* Oracle equivalence and linear transform, `is_equivalent()` and `test_equivalence()` 
* Permutation, `is_permutation()` and `test_permutation()`
* Repetition, `is_repetition()` and `test_repetition()`
* Conjugation, `is_conjugation()` and `test_conjugation()`
* Duality, `is_duality()` and `test_duality()`

Here is a [brief description](https://linnaeus-doc.github.io/#linnaeus) of algorithm equivalence and relations. 
For more details, see [our paper](https://arxiv.org/abs/2105.04684). 
These functions take two arguments, the first one is the target algorithm to parse and the second is the 
reference algorithm or a list of reference algorithms. 

For each type of algorithm equivalence or relations, the `is_` function returns boolean or a list of boolean, simply 
indicating whether the input algorithm and the reference algorithm are equivalent/related or not. The `test_` function returns more details. 
Specifically, if two algorithms are equivalent/related only when the parameters satisfy a certain condition, the `test_` function 
will return the condition. 

For illustration, we define the triple momentum algorithm as `algo4` and check oracle equivalence between `algo3` and `algo4`. 

<img src="/Figures/triple_algo.svg?invert_in_darkmode" align=middle width="310" height="310"/>


```python
algo4 = Algorithm("triple momentum algorithm")
x1, x2, x3 = algo4.add_var("x1", "x2", "x3")
alpha, beta, eta = algo4.add_parameter("alpha", "beta", "eta")
gradf = algo4.add_oracle("gradf")

algo4.add_update(x3, x1)
algo4.add_update(x1, (1 + beta)*x1 - beta*x2 - alpha*gradf((1 + eta)*x1 - eta*x2))
algo4.add_update(x2, x3)
algo4.parse()

lin.is_equivalent(algo3, algo4)
```
<img src="/Figures/triple_title.svg?invert_in_darkmode" align=middle width="210" height="210"/>

<img src="/Figures/triple_eq.svg?invert_in_darkmode" align=middle width="340" height="340"/>

The system returns 
```python
True
```

To return the conditions for parameters that yield equivalence, we can use the `test_equivalent()` function as follows.
```python
lin.test_equivalent(algo3, algo4)
```
<img src="/Figures/parameter.svg?invert_in_darkmode" align=middle width="600" height="600"/>

#### Double check

Under very occasional cases, Linnaeus may not be able to identify equivalence and relations between algorithms. 
This is due to limitations of SymPy equation solver. 
In such cases, we recommand users to do double check with the detection functions while exchange the input algorithm 
and the reference algorithm. 

For example, the following code shows to do double check for equivalence between `algo3` and `algo4`.
```python
lin.is_equivalent(algo3, algo4)
# exchange the input algorithm and the reference algorithm
lin.is_equivalent(algo4, algo3)
```

#### Multiple relations

It is possible that algorithms are related with multiple relations. 
In such cases, one algorithm can be transformed into another by sequentially performing 
multiple aforementioned transformations. 

Here is an [example](https://linnaeus-doc.github.io/examples/#check-conjugation-and-permutation-douglas-rachford-splitting-and-admm) 
to detect conjugation and permutation between Douglas-Rachford splitting and ADMM. 
In this example, Douglas-Rachford splitting can be transformed to ADMM, by performing conjugation and permutation. 
(Since conjugation and permutation commute, the sequence does not matter.)

Linnaeus provides two functions `is_conjugate_permutation()` and `test_conjugate_permutation()` 
to detect conjugation and permutation together. 
The input and output for these functions are the same as the aforementioned detection functions. 


LaTeX support
--------------

We recommand users to use an IPython console that LaTeX is installed to check the parameter maps returned by the `test_` functions. 
An IPython notebook with LaTeX support is also recommanded. 


Access state-space realization and transfer function
-----------------------------------

#### State-space realization

Function `get_ss(verbose)` returns the state-space realization of an algorithm. 
If `verbose` is `False`, it will return a big matrix containing state-space matrices A, B, C, and D. 
If `verbose` is `True`, it will explicitly indicate the A, B, C, and D matrices. 
The default setting of `verbose` is `False`.

The following example shows to get the state-space realization of the triple momentum algorithm, `algo4`.

```python
algo4.get_ss()
```
<img src="/Figures/whole_ss.svg?invert_in_darkmode" align=middle width="150" height="150"/>

```python
algo4.get_ss(verbose = True)
```
<img src="/Figures/ss_title.svg?invert_in_darkmode" align=middle width="160" height="160"/>

<img src="/Figures/triple_ss.svg?invert_in_darkmode" align=middle width="350" height="350"/>

Specifically, function `get_canonical_ss(ss_type, verbose)` returns the canonical form of 
the state-space realization. 

If `ss_type` is `'c'`, it will return the controllable canonical form; 
and if `ss_type` is `'o'`, it will return the observable canonical form. 
The default setting is `'c'`.
More details about controlable and observable canonical forms can be found [here](https://faculty.washington.edu/chx/teaching/me547/1-6_ss_realization_slides.pdf). 

`verbose` is the same as function `get_ss()`. 

The following example shows to get the observable canonical form of state-space realization of the triple momentum algorithm, `algo4`.

```python
algo4.get_canonical_ss(ss_type = 'o', verbose = True)
```
<img src="/Figures/observable_title.svg?invert_in_darkmode" align=middle width="220" height="220"/>

<img src="/Figures/observable_ss.svg?invert_in_darkmode" align=middle width="350" height="350"/>

#### Transfer function

Function `get_tf()` returns the transfer function of an algorithm.

The following code shows to get the transfer function of the triple momentum algorithm, `algo4`.

```python
algo4.get_tf()
```
<img src="/Figures/tf.svg?invert_in_darkmode" align=middle width="130" height="130"/>

Transform an algorithm
------------------------

To detect complex relations, Linnaeus provides a set of functions to transform an algorithm. 

* `permute(step)`, returns a permutation of an algorithm with specific steps set by argument `step`. The default setting of `step` is 0, which returns the one-step permutation. `step` should be an integer and less than the total oracle number of the algorithm. 
* `conjugate(conjugate_oracle)`, returns the conjugate of an algorithm with respect to the specific oracle set by argument `conjugate_oracle`. The default setting of `conjugate_oracle` is 0, which returns the conjugate with respect to the first oracle. `conjugate_oracle` should be an integer and within the set of oracle indices of the algorithm. If an algorithm contains `n` oracles, its set of oracle indices is `[0, ..., n - 1]`.
* `dual()`, returns the conjugate of an algorithm with respect to all the oracles. 
* `repeat(times)`, returns a repetition of an algorithm that repeats the algorithm with `times` of times. The default setting of `times` is 2, which repeats the algorithm twice. `times` should be an positive integer. 

The following code transforms `new_algo` in different ways. 
```python
new_algo.permute()

new_algo.conjugate(1)

new_algo.dual()

new_algo.repeat(3)
```

For example, we can get the state-space realization for the permutation of `new_algo` as follows,
```python
new_algo.permute().get_ss()
```
