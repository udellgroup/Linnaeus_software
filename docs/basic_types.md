Basic Types
===========

The major building block is called an algorithm. An algorithm consists of expressions 
including variables, parameters, functions, oracles, and update equations. Linnaeus is 
developed based on python symbolic package sympy, and all the expressions are defined symbolically.

Algorithms
----------

Algorithm is defined using the keyword `Algorithm`
along with the name of the algorithm. 

For example, we define an `Algorithm` as `new_algo` named `my new algorithm`.
```python
new_algo = Algorithm("my new algorithm")
```

Expressions such as variables, parameters, functions, oracles, and update equations 
are all attributes to an algorithm object. 
An algorithm must be defined before other expressions are defined and added to it. 

Variables
---------

Variables are algorithm states. Variables are declared and added to an algorithm
using syntax `add_var` along with the names of variables. 

For example, we add variables `x1` and `x2` to `new_algo`.
```python
x1, x2 = new_algo.add_var("x1", "x2")
```

Each algorithm state must be updated at each iteration. The
updated variables can be accessed by syntax `update`. 

For example, we access the updated `x1` with `x1u` and updated `x2` with `x2u`. 
```python
x1u, x2u = new_algo.update(x1, x2)
```

Parameters
---------

Parameters can be a scalar, a vector, or a matrix. The main difference between a scalar and
a vector or a matrix is whether it is commutative. Parameters can be declared and added 
to an algorithm using syntax `add_parameter` along with the names of parameters. 
To add a vector or a matrix, the argument `commutative` should be set as `False`. 
The default setting of argument `commutative` is `True`. 
It is unnecessary to specify the size of the vector or the matrix, since they are all 
regarded as abstract expressions. 

For example, we add a scalar `t` and matrix `L` to `new_algo`.
```python
t = new_algo.add_parameter("t")
L = new_algo.add_parameter("L", commutative = False)
```

Oracles
-----------

Linnaeus provides two approaches to declare and add oracles 
to an algorithm. 

#### Black-box approach

The first approach is to define oracles as black boxes. 
When parsing the algorithm, the system treats each oracle as a whole and does not care what 
happens inside each oracle. 
Under this approach, an oracle can be declared using syntax `add_oracle`.

For example, we declare oracles gradient of `f` and proximal operator of `g`.
```python
gradf = new_algo.add_oracle("gradf")
proxg = new_algo.add_oracle("proxg")
```

#### (Sub)gradient approach

The second approach is to define oracles in the level of (sub)gradients. 
This approach is important especially when detecting conjugation between algorithms. 
When parsing an algorithm, all the oracles will be decomposed into (sub)gradients. 
This [example](#check-conjugation-and-permutation-douglas-rachford-splitting-and-admm) shows the details. 
Under this approach, functions must be defined and added to the algorithm before defining 
oracles. Syntax `add_function` is used to declare and add functions to an algorithm. 

For example, we define function `f` and add it to `new_algo`
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
The update equations in the example are interpreted as 

<img src="/docs/Figures/interpretation1.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Or using equalities, 

<img src="/docs/Figures/interpretation2.svg?invert_in_darkmode" align=middle width="140" height="140"/>

By default setting, variables which have already been updated will be substituted with their
updated versions. Thus, `x1` in the second update equation is interpreted as the updated `x1`, as the `x1^+` in 
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
To avoid overlap and misleading in expressions, it is highly recommanded that each algorithm should have its own expressions such as own variables, parameters, etc, and those expressions should only be used within this algorithm!

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


