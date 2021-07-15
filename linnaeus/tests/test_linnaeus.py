import unittest
import linnaeus as lin
from linnaeus import Algorithm
from sympy.matrices import Matrix
from sympy import simplify
from sympy import symbols
from sympy import zeros

class LinTest(unittest.TestCase):
    
    alpha = symbols("alpha")
    z = symbols("z")
    t = symbols("alpha")
    
    def test_input(self):
        test_algo = Algorithm("test_algorithm")
        x1, x2, x3 = test_algo.add_var("x1", "x2", "x3")
        t, alpha = test_algo.add_parameter("t", "alpha")
        f, g = test_algo.add_function("f", "g")
        test_algo.add_update(x1, lin.prox(f, t)(x3))
        test_algo.add_update(x2, lin.prox(g, t)(2 * x1 - x3))
        test_algo.add_update(x3, x3 + 2 * alpha * (x2 - x1)) 
        test_algo._parse()
        tf = Matrix([[t * (-2 * alpha - z + 1) / (z - 1), -2 * alpha * t / (z - 1)], [2 * t * (-alpha - z + 1) / (z - 1), t * (-2 * alpha - z + 1) / (z - 1)]])
        self.assertTrue(simplify(tf - test_algo.tf) == zeros(2))

    def test_statespace(self):
        ss = Matrix([[0, -1, -1, -1 / t, 0], [0, 1, 0, 1 / t, -1 / t], [0, 0, 0, 0, -1 / t], [0, -1, -1, -1 / t, 0], [0, 1, 0, 1 / t, -1 / t]])
        self.assertTrue(simplify(ss - lin.Admm.ss) == zeros(5, 5))

    def test_transferfunction(self):
        tf = Matrix([[t * (-2 * alpha - z + 1) / (z - 1), -2 * alpha * t / (z - 1)], [2* t * (-alpha - z + 1) / (z - 1), t*(-2 * alpha - z + 1) / (z - 1)]])
        self.assertTrue(simplify(tf - lin.Dr.tf) == zeros(2))
        
    def test_equivalent(self):
        self.assertTrue(lin.is_equivalent(lin.Pp, lin.Pp2))
        self.assertTrue(lin.is_equivalent(lin.Dr, lin.Pr))
        self.assertTrue(lin.is_equivalent(lin.Dr.permute(0), lin.Dr2))
        
    def test_duality(self):
        self.assertTrue(lin.is_duality(lin.Admm, lin.Dr2))
        
    def test_permutation(self):
        self.assertTrue(lin.is_permutation(lin.Dr, lin.Dr2))
        self.assertTrue(lin.is_permutation(lin.Cp.permute(), lin.Cp))
        
    def test_repetition(self):
        self.assertTrue(lin.is_repetition(lin.Dr, lin.Dr.repeat()))
        
    def test_conjugation(self):
        self.assertTrue(lin.is_conjugation(lin.Dr, lin.Cp))
