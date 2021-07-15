Detection
===========

Parse an algorithm
-------------------

Suppose an algorithm is already defined and variables, parameters, oracles, and 
update equations are added to the algorithm, the next step is to parse the algorithm. 
Syntax `parse` is used to translate the input algorithm into the canonical form and use 
the canonical form to perform subsequent analysis such as equivalence detection. 
When parsing an algorithm, the state-space realization is provided by the system. 

With example algorithm (gradient descent algorithm), 

<img src="/Figures/gradient_algo.svg?invert_in_darkmode" align=middle width="130" height="130"/>


The example code defines the algorithm as `algo3` and parses it into canonical form. 
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
<img src="/Figures/gradient_title.svg?invert_in_darkmode" align=middle width="200" height="200"/>

<img src="/Figures/gradient_ss.svg?invert_in_darkmode" align=middle width="240" height="240"/>

Detection
-----------

Linnaeus provides a set of functions to check algorithm equivalence and relations. 

* Oracle equivalence and linear transform, `is_equivalent()` and `test_equivalence()` 
* Permutation, `is_permutation()` and `test_permutation()`
* Repetition, `is_repetition()` and `test_repetition()`
* Conjugation, `is_conjugation()` and `test_conjugation()`
* Duality, `is_duality()` and `test_duality()`

Here is a [brief description](#linnaeus) of algorithm equivalence and relations. For more details, see our paper. 
These functions take two arguments, one is the target algorithm to parse and the other is the 
reference algorithm or a list of reference algorithms. 

For each type of algorithm equivalence or relations, the `is_` function returns boolean or a list of boolean, simply 
indicating whether two algorithms are equivalent/related or not. The `test_` function returns more details. 
Specifically, if two algorithms are equivalent/related only under specific parameter maps, the `test_` function 
will return the parameter maps. 

As an example, we define the triple momentum algorithm as `algo4` and check oracle equivalence between `algo3` and `algo4`. 

<img src="/Figures/triple_algo.svg?invert_in_darkmode" align=middle width="310" height="310"/>


```python
algo4 = Algorithm("triple momentum algorithm")
x1, x2, x3 = algo4.add_var("x1", "x2", "x3")
alpha, beta, eta = algo4.add_var("alpha", "beta", "eta")
gradf = algo4.add_oracle("gradf")

algo4.add_update(x3, x1)
algo4.add_update(x1, (1 + beta)*x1 - beta*x2 - alpha*gradf((1 + eta)*x1 - eta*x2))
algo4.add_update(x2, x3)
algo4.parse()

lin.is_equivalent(algo3, algo4)
```
<img src="/Figures/triple_title.svg?invert_in_darkmode" align=middle width="200" height="200"/>

<img src="/Figures/triple_ss.svg?invert_in_darkmode" align=middle width="340" height="340"/>

The system returns 
```python
True
```

To return the parameter maps, we can use the `test_equivalent()` function.
```python
lin.test_equivalent(algo3, algo4)
```
<img src="/Figures/parameter.svg?invert_in_darkmode" align=middle width="600" height="600"/>



LaTeX support
--------------

We recommand users to use an IPython console that LaTeX is installed to check the parameter maps returned by the `test_` functions. 
An IPython notebook also has LaTeX support. 


Transform an algorithm
------------------------

To detect complex relations, Linnaeus provides a set of functions to transform an algorithm. 

* `permute(step)`, returns a permutation of an algorithm with specific steps set by argument `step`. The default setting of `step` is 0, which returns the one-step permutation. `step` should be an integer and less than the total oracle number of the algorithm. 
* `conjugate(conjugate_oracle)`, returns the conjugate of an algorithm with respect to the specific oracle set by argument `conjugate_oracle`. The default setting of `conjugate_oracle` is 0, which returns the conjugate with respect to the first oracle. `conjugate_oracle` should be an integer and within the set of oracle indices of the algorithm. If an algorithm contains `n` oracles, its set of oracle indices is `[0, ..., n - 1]`.
* `dual()`, returns the conjugate of an algorithm with repect to all the oracles. 
* `repeat(number)`, returns a repetition of an algorithm that repeats the algorithm with `number` of times. The default setting of `number` is 2, which repeats the algorithm twice. 

### Commutative property

* Conjugation of different oracles commutes. 
* Conjugation and permutation also commute. 

The example code transforms `new_algo` in different ways. 
```python
new_algo.permute()

new_algo.conjugate(1)

new_algo.dual()

new_algo.repeat(3)
```
Here is the [example](#check-conjugation-and-permutation-douglas-rachford-splitting-and-admm) to detect complex relation with algorithm transformation. 

