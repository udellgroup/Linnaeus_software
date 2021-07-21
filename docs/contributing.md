Contributing
===========

Linnaeus is an open-source project open to any academic or commercial applications. 
Contributions are welcome as [GitHub pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) in any part of the project. 


Add new algorithm to library
---------

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