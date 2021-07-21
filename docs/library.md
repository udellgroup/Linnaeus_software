Algorithm Library
===========

Linnaeus includes an algorithm library to serve as a reference. 
Users can check the potential relations between the input algorithm and reference algorithms in the library. 
The algorithms can be accessed through name abbreviations. 
Currently, the algorithm library includes:

* Gradient descent method, accessed by `Gr`
* [Nesterov's accelerated gradient method](https://epubs.siam.org/doi/abs/10.1137/15M1009597), accessed by `Ng`
* [Heavy-ball method](https://epubs.siam.org/doi/abs/10.1137/15M1009597), accessed by `Hb`
* [Triple momentum algorithm](https://ieeexplore.ieee.org/document/7967721), accessed by `Tm`
* [Proximal point method](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf), accessed by `Pp`
* [Proximal point method with relaxation](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf), accessed by `Pp_r`
* [Proximal gradient method](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf), accessed by `Pg`
* [Douglas-Rachford splitting](http://www.seas.ucla.edu/~vandenbe/236C/lectures/dr.pdf), accessed by `Dr`
* [Permutation of Douglas-Rachford splitting](http://www.seas.ucla.edu/~vandenbe/236C/lectures/dr.pdf), accessed by `Dr_p`
* [Peaceman-Rachford splitting method](https://stanford.edu/class/ee364b/lectures/monotone_split_slides.pdf), accessed by `Pr`
* [Alternating direction method of multipliers (ADMM)](https://stanford.edu/~boyd/admm.html), accessed by `Admm`
* [Chambolle-Pock method](https://hal.archives-ouvertes.fr/hal-00490826/document), accessed by `Cp`
* [Davis-Yin splitting method](https://arxiv.org/abs/1504.01032), accessed by `Dy`
* [Extragradient method](https://arxiv.org/abs/1609.08177), accessed by `Eg`
* [Extragradient method by Korpelevich and Antipin](https://link.springer.com/article/10.3103/S0278641910030039), accessed by `Eg_ka`
* [Extragradient method by Tseng](https://epubs.siam.org/doi/abs/10.1137/S0363012998338806?casa_token=RML0YD1nSUAAAAAA:sFhDEPYjlpkR2Nv5EjzDawca_yST1_qkn0QkWKVuqwwkbJ2Ig1XIT8exbADL3wnSZkrb6a93f0A3), accessed by `Eg_t`
* [Reflected gradient method by Malitsky](https://arxiv.org/abs/1502.04968), accessed by `Rl`

The following code checks equivalence between 
Gradient descent method and Triple momentum method in the library. 

```python
import linnaeus as lin
lin.is_equivalent(lin.Gr, lin.Tm)
```

The results are the same as the [previous example](https://linnaeus-doc.github.io/detection/#detection). 
