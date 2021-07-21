Basic Types
===========

The major building block is called an algorithm. An algorithm is defined with 
variables, parameters, functions, oracles, and update equations. 
All the expressions in Linnaeus are defined symbolically using SymPy. 

Algorithms
----------

Algorithm is defined using keyword `Algorithm`
along with the name of the algorithm. 

The following code defines an `Algorithm` as `new_algo` named `my new algorithm`.
```python
new_algo = Algorithm("my new algorithm")
```

Expressions such as variables, parameters, functions, oracles, and update equations 
are all attributes to an algorithm object. 
An algorithm must be defined before other expressions are defined and added to it. 

Variables
---------

Variables are algorithm states. They are declared and added to an algorithm
using syntax `add_var` along with the names of the variables. 

In the following example, we add variables `x1` and `x2` to `new_algo`.
```python
x1, x2 = new_algo.add_var("x1", "x2")
```

Each algorithm state (each variable) must be updated at each iteration. The
updated variables can be accessed by syntax `update`. 

Here we access the updated `x1` with `x1u` and updated `x2` with `x2u` in the following code. 
```python
x1u, x2u = new_algo.update(x1, x2)
```

Parameters
---------

Parameters of an algorithm can be declared as scalar (commutative) or vector or matrix (noncommutative). 
Parameters are declared and added to an algorithm using syntax `add_parameter` along with the names of parameters. 
To add a vector or matrix, the argument `commutative` should be set as `False`. 
The default setting of argument `commutative` is `True`. 
There is no need to specialize the dimensions of vectors or matrices, since they are symbolic. 

The following code shows how to add scalar `t` and matrix `L` to `new_algo`.
```python
t = new_algo.add_parameter("t")
L = new_algo.add_parameter("L", commutative = False)
```

#### Warning for noncommutative parameters

Due to limited support for noncommutative symbol calculation in SymPy, 
Linnaeus may not correctly parse extremely complex algorithms including noncommutative symbols. 
It is also possible that Linnaeus may not get the parameter mapping for noncommutative symbols (matrices or vectors) 
when detecting equivalence for such algorithms. 
In such cases, we recommand users to declare all the parameters as commutative symbols! 

Oracles
-----------

Linnaeus provides two approaches to declare and add oracles 
to an algorithm. 

#### Black-box approach

The black-box approach is to define oracles as black boxes. 
When parsing the algorithm, the system treats each oracle as a distinct entity unrelated to any other oracle. 
An oracle declared using syntax `add_oracle` uses the black-box approach

For example, we declare oracles gradient of `f` and proximal operator of `g` with the following code.
```python
gradf = new_algo.add_oracle("gradf")
proxg = new_algo.add_oracle("proxg")
```

#### Functional approach

The functional approach is to define oracles in terms of the (sub)gradient of a function. 
When parsing an algorithm, all the oracles will be decomposed into (sub)gradients. 
This approach is important to allow detection algorithm conjugation. 
This [example](https://linnaeus-doc.github.io/examples/#check-conjugation-and-permutation-douglas-rachford-splitting-and-admm) shows the details. 
With the functional approach, functions must be defined and added to the algorithm 
using syntax `add_function` before defining oracles.

The following example shows to define function `f` and add it to `new_algo`
```python
f = new_algo.add_function("f")
```

Linnaeus provides four types of oracles that can be decomposed into (sub)gradients, 

* Gradient with syntax `grad`
* Proximal operator with syntax `prox`
* Projection with syntax `proj`
* Argmin operator with syntax `argmin`
```python
import linnaeus as lin
# gradient of f with respect to x1 
lin.grad(f)(x1)
# proximal operator of f with respect to x2 and parameter t 
lin.prox(f,t)(x2)
# projection x1 onto set C
lin.proj(C)(x1)
# argmin of f(x) + g(x)
lin.argmin(x, f(x) + g(x))
```
To use the projection oracle, a set must be declared in advance with syntax `add_set`. 


Update equations
-----------

Update equations define the iterative procedures of an algorithm. All the update equations have common 
structure that an updated variable equals an expression of variables, parameters, and oracles. 
An update equation is defined with syntax `add_update`. 
```python
new_algo.add_update(x1, x2 - gradf(x2)) 
new_algo.add_update(x2, x1 + proxg(x1)) 
```
The update equations in the above example are interpreted as 

<img src="/Figures/interpretation1.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Or interpreted as equalities, 

<img src="/Figures/interpretation2.svg?invert_in_darkmode" align=middle width="140" height="140"/>

By default setting, variables which have already been updated will be substituted with their
updated versions. Thus, `x1` in the second update equation is considered as the updated `x1`, as the `x1^+` in 
the second equality interpretation. 


For some cases, we can change the default setting by syntax `set_auto`, such as 
```python
new_algo.set_auto(False)
new_algo.add_update(x1, x2 - gradf(x2)) 
new_algo.add_update(x2, x1 + proxg(x1)) 
```
Under this setting, we have a different interpretation of the update equations, 

<img src="/Figures/interpretation3.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Warning!
---------
To avoid overlap and misleading in expressions, it is highly recommanded that each algorithm should have its own variables, parameters, etc, and those expressions should only be used within this algorithm!

For example, 
```python
algo1 = Algorithm("my first algorithm")
x1, x2 = algo1.add_var("x1", "x2")
t, h = algo1.add_parameter("t", "h")
algo1.add_update(x1, x2 - t*x2)
algo1.add_update(x2, x1 - h*x2)

algo2 = Algorithm("my second algorithm")
x1, x2 = algo2.add_var("x1", "x2")
t, h = algo2.add_parameter("t", "h")
algo2.add_update(x1, x2 - h*x1)
algo2.add_update(x2, x1 - t*x2)
```
Variables `x1` and `x2`, parameters `t` and `h` are declared within the first algorithm `algo1`. They should only be used within `algo1`, such as the update equations of it. 
For the second algorithm `algo2`, if we still want to use variables `x1`, `x2`, and parameters `t`, `h`, they should be declared again within `algo2`.  


