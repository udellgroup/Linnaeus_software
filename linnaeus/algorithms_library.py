from .algorithm import *

class library:
    
    def __init__(self, libraryname = "new library"):
        self.name = libraryname
        self.library = []
    
    def library_insert(self, algo):
        if algo.__class__ == Algorithm:
            if algo.in_library == False:
                self.library.append(algo)
                algo.in_library = True
            else:
                raise ValueError("Algorithm already in library!")
        else:
            raise ValueError("Invalid algorithm!")
        
    def library_remove(self, algo):
        if algo in self.library:
            self.library.remove(algo)
            algo.in_library = False
        else:
            raise ValueError("Invalid remove!")
            
 
algo_library = library("Algorithm library")

Admm = Algorithm("ADMM", "Admm")
f, g = Admm.add_function("f", "g")
rho = Admm.add_parameter("rho")
A, B, c = Admm.add_parameter("A", "B", "c")
x1, x2, x3 = Admm.add_var("x1", "x2", "x3")

def L(x, zz, y):
    return f(x) + g(zz) + 1/2 * rho * norm_square(A * x + B * zz + y - c)
    
x = add_generalvar("x")  
Admm.add_update(x1, argmin(x, L(x, x2, x3))) 
Admm.add_update(x2, argmin(x, L(x1, x, x3))) 
Admm.add_update(x3, x3 + A*x1 + B*x2 - c)  
Admm._parse()
Admm.equation_string = r"""
 x_1^+ = \text{argmin}_x \{f(x) + \frac{\rho}{2} 
\left\| Ax + Bx_2 - c + x_3 \|\right \} \\
x_2^+ = \text{argmin}_x \{g(x) + \frac{\rho}{2} 
\left\| Ax_1^+ + Bx - c + x_3 \|\right \} \\
x_3^+ = x_3 + Ax_1^+ + Bx_2^+ - c"""
algo_library.library_insert(Admm)


Dr_p = Algorithm("Douglas-Rachford splitting method permutation", "Dr_p")
f, g = Dr_p.add_function("f", "g")
t = Dr_p.add_parameter("t")
x1, x2, x3 = Dr_p.add_var("x1", "x2", "x3")
Dr_p.add_update(x1, prox(f, t)(x2 - x3)) 
Dr_p.add_update(x2, prox(g, t)(x1 + x3))
Dr_p.add_update(x3, x3 + x1 - x2)
Dr_p._parse()
Dr_p.equation_string = r"""
 x_1^+ = \text{prox}_{tf} (x_2 - x_3) \\
 x_2^+ = \text{prox}_{tg} (x_1^{+} + x_3) \\
 x_3^+ = x_3 + x_1^{+} - x_2^{+} """
algo_library.library_insert(Dr_p)

Cp = Algorithm("Chambolle-Pock method", "Cp")
f, g = Cp.add_function("f", "g")
sigma, tau, theta = Cp.add_parameter("sigma", "tau", "theta")
x1, x2, x3, x4 = Cp.add_var("x1", "x2", "x3", "x4")
Cp.add_update(x1, prox(f, tau)(x3 - tau * x4))  
Cp.add_update(x2, prox(g, sigma)(x4 + sigma * (2 * x1 - x3)))
Cp.add_update(x3, x3 + theta * (x1 - x3)) 
Cp.add_update(x4, x4 + theta * (x2 - x4)) 
Cp._parse()
Cp.equation_string = r"""
 x_1^+ = \text{prox}_{\tau f} (x_3 - \tau x_4) \\
 x_2^+ = \text{prox}_{\sigma g^*} (x_4 + \sigma (2x_1^+ - x_3)) \\
 x_3^+ = x_3 + \theta (x_1^{+} - x_3) \\ 
 x_4^+ = x_4 + \theta (x_2^+ - x_4) 
    """
algo_library.library_insert(Cp)

Gr = Algorithm("Gradient method", "Gr")
f = Gr.add_function("f")
alpha = Gr.add_parameter("alpha")
x = Gr.add_var("x")
Gr.add_update(x, -alpha * grad(f)(x) + x)
Gr._parse()
Gr.equation_string = r"""
x^+ = x - \alpha \nabla f(x)
"""
algo_library.library_insert(Gr)

Ng = Algorithm("Nesterov’s accelerated gradient method", "Ng")
f = Ng.add_function("f")
alpha, beta = Ng.add_parameter("alpha", "beta")
x1, x2, y = Ng.add_var("x1", "x2", "y")
Ng.add_update(y, (1 + beta) * x1 - beta * x2)
Ng.add_update(x2, x1)
Ng.add_update(x1, y - alpha * grad(f)(y))
Ng._parse()
Ng.equation_string = r"""
x_1^+ = y - \alpha \nabla f(y) \\
x_2^+ = x_1 \\
y = (1 + \beta) x_1 - \beta x_2
"""
algo_library.library_insert(Ng)

Tm = Algorithm("Triple momentum method", "Tm")
f = Tm.add_function("f")
alpha, beta, eta = Tm.add_parameter("alpha", "beta", "eta")
x1, x2, y = Tm.add_var("x1", "x2", "y")
Tm.set_auto(False)
yu = Tm.update(y)
Tm.add_update(y, (1 + eta) * x1 - eta * x2)
Tm.add_update(x2, x1)
Tm.add_update(x1, (1 + beta) * x1 - beta * x2 - alpha * grad(f)(yu))
Tm._parse()
Tm.equation_string = r"""
x_1^+ = (1 + \beta) * x1 - \beta * x2 - \alpha \nabla f(y) \\
x_2^+ = x_1 \\
y = (1 + \eta) x_1 - \eta x_2
"""
algo_library.library_insert(Tm)

Pp = Algorithm("Proximal point method", "Pp")
f = Pp.add_function("f")
t = Pp.add_parameter("t")
x = Pp.add_var("x")
Pp.add_eq(Pp.update(x), prox(f, t)(x))
Pp._parse()
Pp.equation_string = r"""
x^+ = \text{prox}_{tf} (x)
"""
algo_library.library_insert(Pp)

Hb = Algorithm("Heavy ball method", "Hb")
f = Hb.add_function("f")
x, p = Hb.add_var("x", "p")
alpha, beta = Hb.add_parameter("alpha", "beta")
Hb.add_update(p, -grad(f)(x) + beta * p)
Hb.add_update(x, x + alpha * p)
Hb._parse()
Hb.equation_string = r"""
p^+ = \beta p - \nabla f (x) \\
x^+ = x + \alpha p
"""
algo_library.library_insert(Hb)

Pg = Algorithm("Proximal gradient method", "Pg")
f, g = Pg.add_function("f", "g")
x = Pg.add_var("x")
t = Pg.add_parameter("t")
Pg.add_update(x, prox(g, t)(x - t * grad(f)(x)))
Pg._parse()
Pg.equation_string = r"""
x^+ = \text{prox}_{tg}(x - t\nabla f(x))
"""
algo_library.library_insert(Pg)

Dr = Algorithm("Douglas-Rachford splitting method","Dr")
x1, x2, x3 = Dr.add_var("x1", "x2", "x3")
t = Dr.add_parameter("t")
Dr.add_update(x1, prox(f, t)(x3))
Dr.add_update(x2, prox(g, t)(2 * x1 - x3)) 
Dr.add_update(x3, x3 + x2 - x1) 
Dr._parse()
Dr.equation_string = r"""
 x_1^+ = \text{prox}_{tf} (x_3) \\
 x_2^+ = \text{prox}_{tg} (2x_1^+ - x_3) \\
 x_3^+ = x_3 + x_2^{+} - x_1^{+} """
algo_library.library_insert(Dr)

Pp_r = Algorithm("Proximal point method relaxation", "Pp_r")
theta, t = Pp_r.add_parameter("theta", "t")
x1, x2 = Pp_r.add_var("x1", "x2")
Pp_r.add_update(x2, prox(f, t)(x1))
Pp_r.add_update(x1, (1 - theta) * x1 + theta * x2)
Pp_r._parse()
Pp_r.equation_string = r"""
x^+ = (1 - \theta) x + \theta \text{prox}_{tf} (x)
"""
algo_library.library_insert(Pp_r)

Pr = Algorithm("Peaceman-Rachford splitting method", "Pr")
x1, x2, x3 = Pr.add_var("x1", "x2", "x3")
t = Pr.add_parameter("t")
Pr.add_update(x1, prox(f, t)(x3))
Pr.add_update(x2, prox(g, t)(2 * x1 - x3)) 
Pr.add_update(x3, x3 + 2 * (x2 - x1)) 
Pr._parse()
Pr.equation_string = r"""
 x_1^+ = \text{prox}_{tf} (x_3) \\
 x_2^+ = \text{prox}_{tg} (2x_1^+ - x_3) \\
 x_3^+ = x_3 + 2 (x_2^{+} - x_1^{+})
"""
algo_library.library_insert(Pr)

Dy = Algorithm("Davis-Yin splitting method", "Dy")
f, g, h = Dy.add_function("f", "g", "h")
t, gamma = Dy.add_parameter("t", "gamma")
xB, xA, x = Dy.add_var("xB", "xA", "x")
Dy.add_update(xB, prox(g, gamma)(x))
Dy.add_update(xA, prox(f, gamma)(2 * xB - x - gamma * grad(h)(xB)))
Dy.add_update(x, x + t * (xA - xB))
Dy._parse()
Dy.equation_string = r"""
x_B^+ = \text{prox}_{\gamma g} (x) \\
x_A^+ = \text{prox}_{\gamma f} (2 x_B^+ - x - \gamma \nabla h (x_B^+)) \\
x^+ = x + t (x_A^+ - x_B^+)
"""
algo_library.library_insert(Dy)

Eg = Algorithm("Extragradient method for two functions", "Eg")
f, g = Eg.add_function("f", "g")
alpha, beta = Eg.add_parameter("alpha", "beta")
y, x = Eg.add_var("y", "x")
Eg.add_update(y, prox(g, alpha)(x - alpha * grad(f)(x)))
Eg.add_update(x, prox(g, beta)(x - beta * grad(f)(y)))
Eg._parse()
Eg.equation_string = r"""
y^+ = \text{prox}_{\alpha g} (x - \alpha \nabla f(x)) \\
x^+ = \text{prox}_{\beta g} (x - \beta \nabla f(y^+)) 
"""
algo_library.library_insert(Eg)

Eg_ka = Algorithm("Extragradient method by Korpelevich and Antipin", "Eg_ka")
f = Eg_ka.add_function("f")
C = Eg_ka.add_set("C")
t = Eg_ka.add_parameter("t")
x, y = Eg_ka.add_var("x", "y")
Eg_ka.add_update(y, proj(C)(x - t*grad(f)(x)))
Eg_ka.add_update(x, proj(C)(x - t*grad(f)(y)))
Eg_ka._parse()
Eg_ka.equation_string = r"""
y^+ = P_C (x - t \nabla f(x)) \\
x^+ = P_C (x - t \nabla f(y^+))
"""
algo_library.library_insert(Eg_ka)

Eg_t = Algorithm("Extragradient method by Tseng", "Eg_t")
f = Eg_t.add_function("f")
C = Eg_t.add_set("C")
t = Eg_t.add_parameter("t")
x, y = Eg_t.add_var("x", "y")
Eg_t.add_update(y, proj(C)(x - t*grad(f)(x)))
Eg_t.add_update(x, y + t*(grad(f)(x) - grad(f)(y)))
Eg_t._parse()
Eg_t.equation_string = r"""
y^+ = P_C (x - t \nabla f(x)) \\
x^+ = y^+ + t (\nabla f(x) - \nabla f(y^+))
"""
algo_library.library_insert(Eg_t)

Eg_rl = Algorithm("Extragradient method (reflected gradient method) by Malitsky", "Eg_rl")
f = Eg_rl.add_function("f")
C = Eg_rl.add_set("C")
t = Eg_rl.add_parameter("t")
x1, x2, y = Eg_rl.add_var("x1", "x2", "y")
Eg_rl.add_update(y, 2*x1 - x2)
Eg_rl.add_update(x2, x1)
Eg_rl.add_update(x1, proj(C)(x1 - t*grad(f)(y)))
Eg_rl._parse()
Eg_rl.equation_string = r"""
x_1^+ = P_C (x_1^+ - t \nabla f(2x_1 - x_2)) \\
x_2^+ = x_1
"""
algo_library.library_insert(Eg_rl)
