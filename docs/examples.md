Examples
===========

Check permutation
---------

`algo5`

<img src="/Figures/algo5.svg?invert_in_darkmode" align=middle width="160" height="160"/>

`algo6`

<img src="/Figures/algo6.svg?invert_in_darkmode" align=middle width="240" height="240"/>


```python
import linnaeus as lin
from linnaeus import Algorithm

algo5 = Algorithm("algo5")
x1, x2, x3 = algo3.add_var("x1", "x2", "x3")
proxf, proxg = algo3.add_oracle("proxf", "proxg")
algo5.add_update(x1, proxf(x3))
algo5.add_update(x2, proxg(2*x1 - x3))
algo5.add_update(x3, x3 + x2 - x1)

algo6 = Algorithm("algo6")
xi1, xi2 = algo6.add_var("xi1", "xi2")
proxf, proxg = algo6.add_oracle("proxf", "proxg")
algo6.add_update(xi1, proxg(-xi1 + 2*xi2) + xi1 - xi2)
algo6.add_update(xi2, proxf(xi1))

algo5.parse()
algo6.parse()
lin.is_permutation(algo5, algo6)
```
System returns
```python
True
```
Check repetition
-------------------

`algo7`

<img src="/Figures/algo7.svg?invert_in_darkmode" align=middle width="140" height="140"/>

`algo8`

<img src="/Figures/algo8.svg?invert_in_darkmode" align=middle width="140" height="140"/>

```python
algo7 = Algorithm("algo7")
t = algo7.add_parameter("t")
x1 = algo7.add_var("x1")
f = algo7.add_function("f")
algo7.add_update(x1, x1 - t*lin.grad(f)(x1))

algo8 = Algorithm("algo8")
xi1, xi2 = algo8.add_var("xi1", "xi2")
t = algo8.add_parameter("t")
f = algo8.add_function("f")

algo8.add_update(xi2, xi1 - t*lin.grad(f)(xi1))
algo8.add_update(xi1, xi2 - t*lin.grad(f)(xi2))

algo7.parse()
algo8.parse()
lin.is_repetition(algo7, algo8)
```

System returns 
```python
True
```

Check conjugation and permutation (Douglas-Rachford splitting and ADMM)
-------------------------------

For problem, 

<img src="/Figures/problem.svg?invert_in_darkmode" align=middle width="140" height="140"/>

Douglas-Rachford splitting

<img src="/Figures/dr.svg?invert_in_darkmode" align=middle width="170" height="170"/>

ADMM

<img src="/Figures/admm.svg?invert_in_darkmode" align=middle width="300" height="300"/>


```python
DR = Algorithm("Douglas-Rachford splitting")
x1, x2, x3 = DR.add_var("x1", "x2", "x3")
t = DR.add_parameter("t")
f, g = DR.add_function("f", "g")

DR.add_update(x1, lin.prox(f, t)(x3)) 
DR.add_update(x2, lin.prox(g, t)(2*x1 - x3)) 
DR.add_update(x3, x3 + x2 - x1)

ADMM = Algorithm("ADMM")
f, g = ADMM.add_function("f", "g")
rho = ADMM.add_parameter("rho")
xi1, xi2, xi3 = ADMM.add_var("xi1", "xi2", "xi3")

ADMM.add_update(xi1, lin.argmin(xi1, g(xi1) + 1/2*rho*lin.norm_square(xi1 + xi2 + xi3))) 
ADMM.add_update(xi2, lin.argmin(xi2, f(xi2) + 1/2*rho*lin.norm_square(xi1 + xi2 + xi3))) 
ADMM.add_update(xi3, xi3 + xi1 + xi2) 

DR.parse()
ADMM.parse()
# get a permutation of DR
test_algo = DR.permute()
lin.test_duality(test_algo, ADMM)
```
<img src="/Figures/dr_title.svg?invert_in_darkmode" align=middle width="200" height="200"/>

<img src="/Figures/dr_ss.svg?invert_in_darkmode" align=middle width="340" height="340"/>

<img src="/Figures/admm_title.svg?invert_in_darkmode" align=middle width="140" height="140"/>

<img src="/Figures/admm_ss.svg?invert_in_darkmode" align=middle width="330" height="330"/>

<img src="/Figures/dr_admm.svg?invert_in_darkmode" align=middle width="550" height="550"/>


Check conjugation (Douglas-Rachdford splitting and Chambolle-Pock method)
----------------

Chambolle-Pock method

<img src="/Figures/cp.svg?invert_in_darkmode" align=middle width="220" height="220"/>

```python
CP = Algorithm("Chambolle-Pock method")
f, g = CP.add_function("f", "g")
x1, x2, x3 = CP.add_var("x1", "x2", "x3")
tau, sigma = CP.add_parameter("tau", "sigma")

CP.add_update(x3, x1)
CP.add_update(x1, lin.prox(f, tau)(x1 - tau*x2))
CP.add_update(x2, lin.prox(g, sigma)(x2 + sigma*(2*x1 - x3))) 

CP.parse()
lin.test_conjugation(DR, CP)
```

<img src="/Figures/cp_title.svg?invert_in_darkmode" align=middle width="180" height="180"/>

<img src="/Figures/cp_ss.svg?invert_in_darkmode" align=middle width="350" height="350"/>

<img src="/Figures/dr_cp.svg?invert_in_darkmode" align=middle width="550" height="550"/>


