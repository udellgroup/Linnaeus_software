Quick Tutorial
==============

In this quick tutorial, we define two algorithms and show that they are oracle-equivalent. 

The first algorithm, 

<img src="/Figures/algo1.svg?invert_in_darkmode" align=middle width="140" height="140"/>

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
Add oracle `gradf`, gradient of `f` to `algo1`,
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
System parses `algo1` and returns the update equations, 

<img src="/Figures/title1.svg?invert_in_darkmode" align=middle width="75" height="75"/>

<img src="/Figures/output1.svg?invert_in_darkmode" align=middle width="170" height="170"/>

The second algorithm,

<img src="/Figures/algo2.svg?invert_in_darkmode" align=middle width="150" height="150"/>

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

<img src="/Figures/title2.svg?invert_in_darkmode" align=middle width="75" height="75"/>

<img src="/Figures/output2.svg?invert_in_darkmode" align=middle width="175" height="175"/>

Check oracle equivalence : 
```python
lin.is_equivalent(algo1, algo2)
```

System returns 
```python
True
```
which shows that `algo1` and `algo2` are oracle-equivalent. 
