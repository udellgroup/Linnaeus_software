from __future__ import division
from sympy import core
import sympy.matrices.matrices
from sympy.matrices import Matrix, PermutationMatrix, MatrixPermute, matrices
from sympy.combinatorics import Permutation
from sympy import degree
from sympy import Poly
from sympy import ratsimp
from sympy import *
import numpy as np
from itertools import permutations
import copy
from IPython.display import display, Latex

"""Normsq: the square norm of its argument"""

class norm_square(Function):
    is_commutative = True

    @classmethod
    def eval(self, x):
        if x.is_Number:
            return x**2
        if x.is_Vector:
            return SciPy.norm(x, 2)**2

    def fdiff(self, argindex=1):
        return 2*T(self.args[0])

    
"""Grad: the transpose of the derivative"""

# internal gradient calculation
def _grad(expr, var):   
    return T(expr.diff(var))

# gradient calculation
def grad(expr, num = 1):
    if expr.__class__ == sympy.core.function.UndefinedFunction:
        def F(*var):
            return T(expr(*var).diff(var[num - 1]))
        return F
    else: 
        def G(*var):
            return T(expr.diff(var[num - 1]))
        return G

def add_parameter(varname, commutative = True):  
    var = symbols(varname, commutative = commutative)
    return var

        
def add_function(functionname, commutative = True):
    function = Function(functionname, commutative = commutative)
    return function


# general variables, used for x in x1+ = argmin(x,f(x)+g(x)) eg.
generalvariable_set = []
def add_generalvar(varname, commutative = False):
    var = symbols(varname, commutative = commutative)
    generalvariable_set.append(var)
    return var

proxvar = add_generalvar("proxvar")

# proximal
# def prox(expr, var1, var2, comp = 1, var3 = proxvar):
# expr1 = expr(comp*var3) + 1/(2*var1)*norm_square(var3 - var2)
# eq = Eq(grad_0(expr1, var3), 0)
# return eq

equation_oracle_set = []

def prox(expr, var1, comp = 1):
    if expr.__class__ == sympy.core.function.UndefinedFunction:
        def F(var2, var3 = proxvar):
            expr1 = expr(comp*var3) + 1/(2*var1)*norm_square(var3 - var2)
            eq = Eq(_grad(expr1, var3), 0)
            if comp == 1:
                prox_str = 'prox' + '_' + '{' + latex(var1) + ' ' + latex(expr) + '}' + '(' + latex(var2) + ')' 
            else:
                prox_str = 'prox' + '_' + '{' + latex(var1) + '(' + latex(expr) + '\circ ' + latex(comp) + ')' + '}' + '(' + latex(var2) + ')' 
            equation_oracle_set.append(prox_str)
            return eq
        return F
    else:
        raise ValueError("Invalid proximal operator")

argminvar = add_generalvar("argminvar")

# minimization, argmin
def argmin(var, expr):
    if var not in generalvariable_set:
        argmin_str = 'argmin' + '_' + '{' + latex(var) + '}' + '\{' + latex(expr) + '\}'
        expr = expr.subs(var, argminvar)
        eq = Eq(_grad(expr, argminvar), 0)
        equation_oracle_set.append(argmin_str)
    else:
        eq = Eq(_grad(expr, var), 0)
        argmin_str = 'argmin' + '_' + '{' + latex(var) + '}' + '\{' + latex(expr) + '\}'
        equation_oracle_set.append(argmin_str)
    return eq

# projection 

def proj(expr):
    if expr.__class__ == sympy.core.function.UndefinedFunction:
        def F(var2, var3 = proxvar):
            expr1 = expr(var3) + 1/2 * norm_square(var3 - var2)
            eq = Eq(_grad(expr1, var3), 0)
            proj_str = 'proj' + '_' + '{' + latex(expr) + '}' + '(' + latex(var2) + ')'
            equation_oracle_set.append(proj_str)
            return eq
        return F
    else:
        raise ValueError("Invalid projection")

"""Transpose"""

class T(Function):
    is_commutative = False
        
    def __new__(cls,arg):
        try:
            # T(T(x)) = x
            if arg.__class__ == T:
                return arg.args[0]
            # T(x+y) = T(x) + T(y)
            elif arg.is_Add:
                return Add(*[T(A) for A in arg.args])
            # T(x*y) = T(y)*T(x)
            elif arg.is_Mul:
                L = len(arg.args)
                return Mul(*[T(arg.args[L-i-1]) for i in range(L)])
            # assume all commutative terms are scalars, no need to transpose
            elif arg.is_commutative:
                return arg
            else:
                return Function.__new__(cls,arg)
        except:
            return arg

    @classmethod
    def eval(self, x):
        if x.is_Number:
            return x
        if x.is_Vector:
            return x.T
        if x.is_Matrix:
            return x.T

    def fdiff(self, argindex=1):
        return
    
"""solve matrix equations"""

def _matrix_solve(eqs, var1, var2):
    sol = []
    newsol = []
    i = 0
    neweq = eqs
    varindex = -1*(np.ones(len(var1)))
    
    for i in range(len(neweq)):
        for j in range(len(var1)):
            if var1[j] in neweq[i].free_symbols and varindex[j] == -1:
                sol.append(solve(neweq[i], var1[j]))
                varindex[j] = i
                for m in range(len(neweq)):
                    if m > i and var1[j] in neweq[m].free_symbols:
                        neweq[m] = neweq[m].subs(var1[j], sol[i][0])
                break
    
    newsol.append(sol[len(neweq) - 1][0])
    n = 0
    l = len(neweq) - 2  
    while l in range(len(neweq)):
        for m in range(len(neweq) - 1):
            for k in range(len(neweq)):
                if varindex[k] == l + 1:
                    indexvar = k
            sol[m][0] = sol[m][0].subs(var1[indexvar], newsol[n])
        newsol.append(sol[l][0])
        l = l - 1
        n = n + 1
        
    eqindex = -1 * (np.ones(len(neweq)))
    newsol2 = []
    for j in range(len(neweq)):
        for k in range(len(neweq)):
            if varindex[k] == len(neweq) - 1 - j:
                vindex = k
        eqindex[j] = vindex
                
    for i in range(len(neweq)):
        for j in range(len(neweq)):
            if eqindex[j] == i:
                newsol2.append(newsol[j])
    
    newsol3 = []
    for k in range(len(var1)):
        if var1[k] in var2:
            newsol3.append(newsol2[k])
    return newsol3

""" AbstractAlgorithm class """

z = add_parameter("z")
class AbstractAlgorithm:

    NONLINEARITIES = [Derivative]
    
    @classmethod
    def from_tf(cls, var, tf, parameterset, matrixset = [], algoname = "new algorithm", algostr = "new"):
        new_algo = cls(algoname)
        if len(parameterset) + len(matrixset) == len(tf.free_symbols) - 1 and var not in parameterset and var not in matrixset and var in tf.free_symbols:
        
            if tf.__class__ != Matrix:
                tf = Matrix([transferfunction])
            new_algo.tf = tf.subs(var, z)
            new_algo.parameter_set = parameterset
            new_algo.matrix_set = matrixset
            new_algo.compare_map = {}
            for var in parameterset:
                comparevar_name = str(var) + '_c'
                comparevar = symbols(comparevar_name, commutative = True)
                new_algo.compare_map[var] = comparevar
            for var in matrixset:
                comparevar_name = str(var) + '_c'
                comparevar = symbols(comparevar_name, commutative = True)
                new_algo.compare_map[var] = comparevar
                noncommutvar = symbols(str(var) + '_n', commutative = True)
                self.noncommut_map[var] = noncommutvar 
            new_algo.oraclenumber = tf.shape[0]
            new_algo.is_parsed = True
            new_algo.namestr = algostr
            return new_algo
        else:
            raise ValueError("Invalid transfer function")
    
    def __init__(self, algoname = "new algorithm", algostr = "new", eq_set = equation_oracle_set):
        self.name = algoname
        self.namestr = algostr
        self.state_vars = []    # state variables
        self.update_map = {}
        self.compare_map = {}
        self.update_eqs = []
        self.update_vars = []
    
        self.input_eqs = []
        self.output_eqs = []
        self.input_vars = []   # input variables
        self.output_vars = []  # output variables here refer to gradients
        self.linear_eqs = []
        self.all_eqs = []
        self.left_vars = []    # combine updated state variables and input variables
        self.right_vars = []   # combine state variables and output variables
        self.tf = None
        self.ss = None
        self.ss_const = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.oraclenumber = None
        self.is_parsed = False
        self.auto = True
        self.parameter_set = []
        self.matrix_set = []
        self.noncommut_map = {}
        self.function_set = []
        self.symbol_set = []
        # fix different oracle issue
        self.nonlinearityset = []
        # second way to represent oracle eg. gradf(x1)
        self.oracle_set = []
        self.in_library = False
        self.equation_string = []
        self.set_set = []
        eq_set.clear()
            

    def __str__(self):
        return self.namestr
    def __repr__(self):
        return self.namestr
    def _repr_latex_(self):
        if self.in_library:
            print(self.name)
            return r'${%s}$' % self.equation_string
        else:
            try:
                self._parse()
                print(self.name)
                print_string = ''
                for i in range(len(self.equation_string)):
                    print_string = print_string + self.equation_string[i]
                    if i != len(self.equation_string):
                        print_string = print_string + '\\\\'
                return r'${%s}$' % print_string
            except:
                return self.name
    
    def add_parameter(self, *varname, commutative = True): 
        if varname.__class__ == str:
            var = symbols(varname, commutative = commutative)
            comparevar_name = varname + '_c'
            comparevar = symbols(comparevar_name, commutative = True)
            self.compare_map[var] = comparevar
            if commutative == True:
                self.parameter_set.append(var)
            else:
                self.matrix_set.append(var)
                noncommutvar_name = varname + '_n'
                noncommutvar = symbols(noncommutvar_name, commutative = True)
                self.noncommut_map[var] = noncommutvar
            self.symbol_set.append(var)
            return var
        else:
            varlist = []
            if commutative.__class__ == bool:
                for i in range(len(varname)):
                    var = symbols(varname[i], commutative = commutative)
                    comparevar_name = varname[i] + '_c'
                    comparevar = symbols(comparevar_name, commutative = True)
                    self.compare_map[var] = comparevar
                    if commutative == True:
                        self.parameter_set.append(var)
                    else:
                        self.matrix_set.append(var)
                        noncommutvar_name = varname[i] + '_n'
                        noncommutvar = symbols(noncommutvar_name, commutative = True)
                        self.noncommut_map[var] = noncommutvar
                    self.symbol_set.append(var)
                    varlist.append(var)
                if len(varlist) == 1:
                    return varlist[0]
                else:
                    return varlist 
            elif commutative.__class__ == list:
                for i in range(len(varname)):
                    var = symbols(varname[i], commutative = commutative[i])
                    comparevar_name = varname[i] + '_c'
                    comparevar = symbols(comparevar_name, commutative = True)
                    self.compare_map[var] = comparevar
                    if commutative[i] == True:
                        self.parameter_set.append(var)
                    else:
                        self.matrix_set.append(var)
                        noncommutvar_name = varname[i] + '_n'
                        noncommutvar = symbols(noncommutvar_name, commutative = True)
                        self.noncommut_map[var] = noncommutvar
                    self.symbol_set.append(var)
                    varlist.append(var)
                if len(varlist) == 1:
                    return varlist[0]
                else:
                    return varlist

    def add_function(self, *functionname, commutative = True):
        if functionname.__class__ == str:
            function = Function(functionname, commutative = commutative)
            self.function_set.append(function)
            return function
        else:
            functionlist = []
            if commutative.__class__ == bool:
                for i in range(len(functionname)):
                    function = Function(functionname[i], commutative = commutative)
                    self.function_set.append(function)
                    functionlist.append(function)
                if len(functionlist) == 1:
                    return functionlist[0]
                else:
                    return functionlist
            elif commutative.__class__ == list:
                for i in range(len(functionname)):
                    function = Function(functionname[i], commutative = commutative[i])
                    self.function_set.append(function)
                    functionlist.append(function)
                if len(functionlist) == 1:
                    return functionlist[0]
                else:
                    return functionlist
            
    def add_oracle(self, *oraclename, commutative = False):
        if oraclename.__class__ == str:
            oracle = Function(oraclename, commutative = commutative)
            self.oracle_set.append(oracle)
            return oracle
        else:
            oraclelist = []
            if commutative.__class__ == bool:
                for i in range(len(oraclename)):
                    oracle = Function(oraclename[i], commutative = commutative)
                    self.oracle_set.append(oracle)
                    oraclelist.append(oracle)
                if len(oraclelist) == 1:
                    return oraclelist[0]
                else:
                    return oraclelist
            elif commutative.__class__ == list:
                for i in range(len(oraclename)):
                    oracle = Function(oraclename[i], commutative = commutative[i])
                    self.oracle_set.append(oracle)
                    oraclelist.append(oracle)
                if len(oraclelist) == 1:
                    return oraclelist[0]
                else:
                    return oraclelist
    
    def add_set(self, *setname):
        if setname.__class__ == str:
            cset = Function(setname, commutative = True)
            self.set_set.append(cset)
            return cset
        else:
            setlist = []
            for i in range(len(setname)):
                cset = Function(setname[i], commutative = True)
                self.set_set.append(cset)
                setlist.append(cset)
            if len(setlist) == 1:
                return setlist[0]
            else:
                return setlist
    
    def add_var(self, *varname, commutative = False):
        if varname.__class__ == str:
            var = symbols(varname, commutative = commutative) 
            self.state_vars.append(var)
            self.symbol_set.append(var)
            updatedvarname = varname + '^+'
            updatedvar = symbols(updatedvarname, commutative = commutative)
            self.update_vars.append(updatedvar)
            self.update_map[var] = updatedvar
            self.symbol_set.append(updatedvar)
            return var
        else:
            varlist = []
            if commutative.__class__ == bool:
                for i in range(len(varname)):
                    var = symbols(varname[i], commutative = commutative)
                    self.state_vars.append(var)
                    self.symbol_set.append(var)
                    updatedvarname = varname[i] + '^+'
                    updatedvar = symbols(updatedvarname, commutative = commutative)
                    self.update_vars.append(updatedvar)
                    self.update_map[var] = updatedvar
                    self.symbol_set.append(updatedvar)
                    varlist.append(var)
                if len(varlist) == 1:
                    return varlist[0]
                else:
                    return varlist
            elif commutative.__class__ == list:
                for i in range(len(varname)):
                    var = symbols(varname[i], commutative = commutative[i])
                    self.state_vars.append(var)
                    self.symbol_set.append(var)
                    updatedvarname = varname[i] + '^+'
                    updatedvar = symbols(updatedvarname, commutative = commutative[i])
                    self.update_vars.append(updatedvar)
                    self.update_map[var] = updatedvar
                    self.symbol_set.append(updatedvar)
                    varlist.append(var)
                if len(varlist) == 1:
                    return varlist[0]
                else:
                    return varlist
    
    def update(self, *var):
        update_list = []
        for i in range(len(var)):
            if var[i] in self.state_vars:
                update_list.append(self.update_map[var[i]])
            else:
                raise ValueError("Invalid variable!")
        if len(update_list) == 1:
            return update_list[0]
        else:
            return update_list
    
    def add_update(self, var, expr):
            
        update_eq_len1 = len(self.update_eqs)
        if var.__class__ != list:
            if var in self.update_vars or var in self.state_vars:
                var = [var]
                expr = [expr]
            else:
                raise ValueError("Invalid update equations!")
                    
        num_eqs1 = len(var)
        num_eqs2 = len(expr)
        for i in range(len(var)):
            if var[i] in self.state_vars:
                var[i] = self.update_map[var[i]]
            if var[i] in self.update_vars:
                if expr[i].__class__ == Eq:
                    if var[i] in expr[i].free_symbols:
                        self.update_eqs.append(expr[i])
                    else:
                        for var1 in expr[i].free_symbols:
                            if var1 in generalvariable_set and var1 not in self.symbol_set:
                                expr1 = expr[i].subs(var1, var[i])
                                self.update_eqs.append(expr1)
                                
                    self.equation_string.append(var[i])
                        
                else:
                    eq = Eq(var[i], expr[i])
                    self.update_eqs.append(eq)
                    
                    if self.auto == True:
                        for un_update_var in self.state_vars:
                            if self.update_map[un_update_var] == var[i]:
                                self.equation_string.append("%s & \gets %s" % (latex(un_update_var), latex(expr[i])))
                                break
                    else:
                        self.equation_string.append("%s & = %s" % (latex(var[i]), latex(expr[i])))
            else:
                raise ValueError("Invalid update equations!")
                    
        update_eq_len2 = len(self.update_eqs)
        
        if num_eqs1 != update_eq_len2 - update_eq_len1 or num_eqs2 != update_eq_len2 - update_eq_len1:
            raise ValueError("Invalid update equations!")
        
    def add_eq(self, expr1, expr2):  # add update equations; 1. direct equation expr2; 2. expr1 = expr2
        update_eq_len1 = len(self.update_eqs)
        if expr1.__class__ != list:
            expr1 = [expr1]
            expr2 = [expr2]
            
        num_eqs1 = len(expr1)
        num_eqs2 = len(expr2)
        
        for i in range(len(expr1)):
            if expr2[i].__class__ == Eq:
                if expr1[i].__class__ != Symbol:
                    raise ValueError("Invalid update equations!")
                    
                if expr1[i] in expr2[i].free_symbols:
                    self.update_eqs.append(expr2[i])
                else:
                    for var in expr2[i].free_symbols:
                        if var in generalvariable_set and var not in self.symbol_set:
                            expr3 = expr2[i].subs(var,expr1[i])
                            self.update_eqs.append(expr3)
            else:
                eq = Eq(expr1[i], expr2[i])
                self.update_eqs.append(eq)
        
        update_eq_len2 = len(self.update_eqs)
        
        if num_eqs1 != update_eq_len2 - update_eq_len1 or num_eqs2 != update_eq_len2 - update_eq_len1:
            raise ValueError("Invalid update equations!")
    
    def set_auto(self, status = True):
        if status.__class__ == bool:
            self.auto = status
        else:
            raise ValueError("Status should be boolean!")
            
    def _eq_update(self):
        new_eqs = []
        num_eqs = len(self.update_eqs)
        num_vars = len(self.state_vars)
        if num_eqs < num_vars:
            raise ValueError("Free variables not used!")
        elif num_eqs > num_vars:
            raise ValueError("Invaild update equations!")
        elif num_eqs == num_vars:
            var_index = np.zeros(len(self.state_vars))
            while np.sum(var_index) < len(self.state_vars):
                eq = self.update_eqs[0]
                self.update_eqs.remove(eq)
                new_eqs.append(eq)
                for i in range(len(self.state_vars)):
                    if self.update_map[self.state_vars[i]] in eq.free_symbols and var_index[i] == 0:
                        var_index[i] = 1
                        for j in range(len(self.update_eqs)):
                            if self.state_vars[i] in self.update_eqs[j].free_symbols:
                                self.update_eqs[j] = self.update_eqs[j].subs(self.state_vars[i], self.update_map[self.state_vars[i]])
            
            self.update_eqs = new_eqs
    
    def _update_string(self, eq_set = equation_oracle_set):
        len_eq_oracle = len(eq_set)
        eq_oracle = 0
        for i in range(len(self.equation_string)):
            if self.equation_string[i] in self.update_vars:
                if self.auto:
                    for no_update_var in self.state_vars:
                        if self.update_map[no_update_var] == self.equation_string[i]:
                            eq_string = "%s & \gets %s" % (latex(no_update_var), eq_set[eq_oracle])
                            eq_oracle = eq_oracle + 1
                            self.equation_string[i] = eq_string
                else:
                    eq_string = "%s & = %s" % (latex(self.equation_string[i]), eq_set[eq_oracle])
                    eq_oracle = eq_oracle + 1
                    self.equation_string[i] = eq_string
        eq_set.clear()          
                        
    
    def _linearize(self):
        self._update_string()
        
        self.input_eqs = []
        self.output_eqs = []
        self.input_vars = []
        self.output_vars = []
        self.linear_eqs = []
        self.all_eqs = []
        self.right_vars = []
        self.left_vars = []
        self.nonlinearityset = []
        
        if self.auto:
            self._eq_update()
        
        for eq in self.update_eqs:
            self.linear_eqs.append(self._linearize_eq(eq))
        
        curr_inputsize = len(self.input_vars)
        curr_input_eqs = copy.deepcopy(self.input_eqs)
        for eq in curr_input_eqs:
            expr = self._linearize_eq(eq.args[1])
            eq1 =  Eq(eq.args[0], expr)
            if len(self.input_vars) > curr_inputsize:
                self.linear_eqs.append(eq1)
                curr_inputsize = len(self.input_vars)
                self.input_eqs.remove(eq)
            
            
    def _linearize_eq(self, expr):
        if expr.__class__ in self.oracle_set:
            if expr in self.nonlinearityset:
                for i in range(len(self.nonlinearityset)):
                    if expr == self.nonlinearityset[i]:
                        return self.output_vars[i]
            else:
                inputvar = Dummy('y_%i' % len(self.input_vars), commutative=False)
                inputvar.dummy_index = hash(expr.args) 
                self.input_eqs.append(Eq(inputvar, expr.args[0])) 
                self.input_vars.append(inputvar)
                nonlinearity = Subs(expr, expr.args[0], inputvar).doit() 

                outputvar = Dummy('u_%i' % len(self.output_vars), commutative=False)
                outputvar.dummy_index = hash(nonlinearity) 
                self.output_vars.append(outputvar)
                self.output_eqs.append(Eq(outputvar, nonlinearity)) 
                self.nonlinearityset.append(expr)
            
                return outputvar
                
            
        elif expr.__class__ in self.NONLINEARITIES:

            if expr in self.nonlinearityset:
                for i in range(len(self.nonlinearityset)):
                    if expr == self.nonlinearityset[i]:
                        return self.output_vars[i]
            else:
                inputvar = Dummy('y_%i' % len(self.input_vars), commutative=False)
                inputvar.dummy_index = hash(expr.args[1]) 
                self.input_eqs.append(Eq(inputvar, expr.args[1][0])) # should be expr.args[1][0] instead of expr.args[1]
                self.input_vars.append(inputvar)
                nonlinearity = Subs(expr, expr.args[1][0], inputvar).doit() # should be expr.args[1][0] instead of expr.args[1]

                outputvar = Dummy('u_%i' % len(self.output_vars), commutative=False)
                outputvar.dummy_index = hash(nonlinearity) 
                self.output_vars.append(outputvar)
                self.output_eqs.append(Eq(outputvar, nonlinearity)) 
                self.nonlinearityset.append(expr)
            
                #linearize step, same oracle if use twice, then considered as different oracles, have already fixed. 
            
                return outputvar

        elif expr.__class__ == Subs:
           
            if expr in self.nonlinearityset:
                for i in range(len(self.nonlinearityset)):
                    if expr == self.nonlinearityset[i]:
                        return self.output_vars[i]
            
            else:
                if expr.args[0].__class__ in self.NONLINEARITIES:
                
                    inputvar = Dummy('y_%i' % len(self.input_vars), commutative=False)
                    inputvar.dummy_index = hash(expr.args[2]) 
                    self.input_eqs.append(Eq(inputvar, expr.args[2][0]))
                    self.input_vars.append(inputvar)
                    nonlinearity = Subs(expr, expr.args[2][0], inputvar).doit()

                
                    outputvar = Dummy('u_%i' % len(self.output_vars), commutative=False)
                    outputvar.dummy_index = hash(nonlinearity) 
                    self.output_vars.append(outputvar)
                    self.output_eqs.append(Eq(outputvar, nonlinearity)) 
                    self.nonlinearityset.append(expr)

                    return outputvar

                else:
                    print(expr)
                    raise ValueError("Invalid substitution!")

        elif expr.__class__ == Integer or expr.__class__ == Float: # update linearization step, float or integer number
            return expr
        
        elif expr.__class__ == T: # update linearization step, transpose
            return expr
        
        elif expr.__class__ == Symbol: # update linearization step, symbol
            if expr in self.symbol_set:
                return expr
            else:
                raise ValueError("Invalid symbols!")
        
        elif expr.__class__ == Rational: # update linearization step for rational numbers
            return expr
        
        else:
            newargs = []
            for i in range(len(expr.args)):
                newargs.append(self._linearize_eq(expr.args[i]))
            return expr.func(*newargs)

    """Transform to descriptor state space representation
    
    Forms a system of equations satisfying
        self.statespace * self.statespacevars = 0
    """
    def _check(self):
        
        varindex1 = np.zeros(len(self.state_vars))
        varindex2 = np.zeros(len(self.state_vars))
        
        for i in range(len(self.state_vars)):
            for eq in self.linear_eqs:
                if self.state_vars[i] in eq.free_symbols:
                    varindex1[i] = varindex1[i] + 1
                if self.update_vars[i] in eq.free_symbols:
                    varindex2[i] = varindex2[i] + 1
            for eq in self.input_eqs:
                if self.state_vars[i] in eq.free_symbols:
                    varindex1[i] = varindex1[i] + 1
                if self.update_vars[i] in eq.free_symbols:
                    varindex2[i] = varindex2[i] + 1
        for i in range(len(self.state_vars)):
            if varindex1[i] == 0 and varindex2[i] == 0:
                print(self.state_vars[i])
                raise ValueError("Variable not used in equations!")
            if varindex1[i] != 0 and varindex2[i] == 0:
                print(self.state_vars[i])
                raise ValueError("Variable should be updated!")
    
    def _merge_eqs_vars(self):
        for i in range(len(self.linear_eqs)):
            self.all_eqs.append(self.linear_eqs[i])
        for i in range(len(self.input_eqs)):
            self.all_eqs.append(self.input_eqs[i])
        for i in range(len(self.update_vars)):
            self.left_vars.append(self.update_vars[i])
        for i in range(len(self.input_vars)):
            self.left_vars.append(self.input_vars[i])
        for i in range(len(self.state_vars)):
            self.right_vars.append(self.state_vars[i])
        for i in range(len(self.output_vars)):
            self.right_vars.append(self.output_vars[i])
    
    def _split_matrix(self, statematrix):  # split a state space matrix into A, B, C, D
        A = statematrix[0:len(self.update_vars), 0:len(self.state_vars)]
        B = statematrix[0:len(self.update_vars), len(self.state_vars):len(self.right_vars)]
        C = statematrix[len(self.update_vars):, 0:len(self.state_vars)]
        D = statematrix[len(self.update_vars):, len(self.state_vars):len(self.right_vars)]
        return A,B,C,D
    
    
    def _get_ss(self): # calculate state space matrix
        
        self._linearize()
        self._check()
        self._merge_eqs_vars()
        
        # solve for left variables, updated state variables and input variables
        selfstate = _matrix_solve(list(map(expand, self.all_eqs)), self.left_vars, self.left_vars)
        selfupeqs = []
        for i in range(len(self.left_vars)):
            selfupeqs.append(Eq(selfstate[i], self.left_vars[i]))

        selfuplinmap, selfupconst = linear_eq_to_matrix(selfupeqs, self.right_vars)
        
        # get constant
        selfupconst2 = []
        for i in range(len(self.left_vars)):
            selfupconst2.append(-selfupconst[i] + self.left_vars[i])
        
        selfupconstantmatrix = Matrix(selfupconst2)
        
        # final state space matrix
        # to print use printmatrix(selfupstatematrix, self.leftvars, self.uprightvars)
        selfupstatematrix = matrices.MatrixBase.hstack(selfuplinmap, selfupconstantmatrix)
        
        return selfupstatematrix, selfuplinmap
    
    
    def _get_tf(self, var = z ):
        pre_state_space_const, pre_state_space = self._get_ss()
        A, B, C, D = self._split_matrix(pre_state_space_const)
        
        if A.__class__ == Matrix: 
            tf = C * (var * eye(A.shape[0]) - A)**(-1) * B + D
        else:
            tf = block_collapse(block_collapse(C * block_collapse((var * Identity(A.shape[0]) - A)**(-1)) * B) + D)
        
        tf = simplify(tf)
        return tf
    
    def _print_ss(self):
        pre_state_space_const, pre_state_space = self._get_ss()
        A, B, C, D = self._split_matrix(pre_state_space_const)
        left1 = []
        left2 = []
        right1 = []
        right2 = []
        for i in range(len(self.update_vars)):
            left1.append(self.left_vars[i])
            right1.append(self.right_vars[i])
        for i in range(len(self.output_eqs)):
            left2.append(self.left_vars[i + len(self.update_vars)])
            right2.append(self.output_eqs[i].args[1])
        
        eq1_left = Matrix(left1)
        eq1_right1 = MatMul(A, Matrix(right1)) 
        eq1_right2 = MatMul(B, Matrix(right2))
        eq2_left = Matrix(left2) 
        eq2_right1 = MatMul(C, Matrix(right1)) 
        eq2_right2 = MatMul(D, Matrix(right2))
        
        return eq1_left, eq1_right1, eq1_right2, eq2_left, eq2_right1, eq2_right2
    
    def _realize(self, expr):
    
        for var in self.matrix_set:
            expr = expr.subs(var, self.noncommut_map[var])
        expr = simplify(expr)
        
        # realize a constant, i.e. de_degree = 0, update 

        numerator = expr.as_numer_denom()[0]
        denominator = expr.as_numer_denom()[1]
        nu_degree = degree(numerator, gen = z)
        de_degree = degree(denominator, gen = z)
        nu_poly = Poly(numerator, z)
        de_poly = Poly(denominator, z)
        de_coeff = de_poly.all_coeffs()
        nu_coeff = nu_poly.all_coeffs()
        if nu_degree > de_degree:
            raise ValueError("Transfer function is not realizable!")
        else:
            if de_degree == 0:
                d = numerator
                A = Matrix([[0]])
                B = Matrix([[1]])
                C = Matrix([[0]])
                D = Matrix([[d]])
            else:
                if nu_degree == de_degree:
                    d = nu_poly.LC()/de_poly.LC()
                else:
                    d = 0
                    full_nu_coeff = []
                    for i in range(de_degree - nu_degree):
                        full_nu_coeff.append(0)
                    for i in range(nu_degree + 1):
                        full_nu_coeff.append(nu_coeff[i])
                    nu_coeff = full_nu_coeff

                A = zeros(de_degree, de_degree)
                B = zeros(de_degree, 1)
                C = zeros(1, de_degree)
                D = Matrix([[d]])

                B[de_degree - 1, 0] = 1

                for i in range(de_degree):
                    A[de_degree - 1, i] = -de_coeff[de_degree - i] / de_coeff[0] 
                    if i >= 1:
                        A[i - 1, i] = 1
                for i in range(de_degree):
                    C[0, i] = nu_coeff[de_degree - i] / de_coeff[0]  - de_coeff[de_degree - i] / de_coeff[0] * d

                for var in self.matrix_set:
                    A = A.subs(self.noncommut_map[var], var)
                    B = B.subs(self.noncommut_map[var], var)
                    C = C.subs(self.noncommut_map[var], var)
                    D = D.subs(self.noncommut_map[var], var)
                A = simplify(A)
                B = simplify(B)
                C = simplify(C)
                D = simplify(D)

        return A, B, C, D
    
    def _tf_to_ss(self, ss_type = 'c'):
        
        if not (ss_type == 'o' or ss_type == 'c'):
            raise ValueError("Invalid state-space realization type! \n\
            Type should be controllable 'c' or observable 'o'.")
        
        if self.oraclenumber == 1:
            c_A, c_B, c_C, c_D = self._realize(self.tf[0, 0])
            if ss_type == 'o':
                A = c_A.T
                B = c_C.T
                C = c_B.T
                D = c_D
            elif ss_type == 'c':
                A = c_A
                B = c_B
                C = c_C
                D = c_D
        else:
            A_list = []
            B_list = []
            C_list = []
            D_list = []
            dim_list = []
            num = self.oraclenumber
            for i in range(num):
                A_row_list = []
                B_row_list = []
                C_row_list = []
                D_row_list = []
                for j in range(num):
                    c_A, c_B, c_C, c_D = self._realize(self.tf[i, j])
                    dim_list.append(c_A.shape[0])
                    if ss_type == 'c':
                        A_row_list.append(c_A)
                        B_row_list.append(c_B)
                        C_row_list.append(c_C)
                        D_row_list.append(c_D)
                    elif ss_type == 'o':
                        A_row_list.append(c_A.T)
                        B_row_list.append(c_C.T)
                        C_row_list.append(c_B.T)
                        D_row_list.append(c_D)
                A_list.append(A_row_list)
                B_list.append(B_row_list)
                C_list.append(C_row_list)
                D_list.append(D_row_list)

            A_dim = sum(dim_list)
            A = zeros(A_dim, A_dim)
            B = zeros(A_dim, num)
            C = zeros(num, A_dim)
            D = zeros(num, num)

            dim = 0
            for i in range(num):
                for j in range(num):
                    A[dim : dim + dim_list[i * num + j], dim : dim + dim_list[i * num + j]] = A_list[i][j]
                    B[dim : dim + dim_list[i * num + j], j] = B_list[i][j]
                    C[i, dim : dim + dim_list[i * num + j]] = C_list[i][j]
                    D[i, j] = D_list[i][j]
                    dim = dim + dim_list[i * num + j]
        
        return A, B, C, D


"""  Algorithm class  """


class Algorithm(AbstractAlgorithm):

    def parse(self):
        print("--------------------------------------------------------------")
        if self.is_parsed == False:
            print("Parse " + self.name + ":")
            self.ss_const, self.ss = self._get_ss()
            self.A, self.B, self.C, self.D = self._split_matrix(self.ss_const) 
            self.tf = self._get_tf()
            self.oraclenumber = self.tf.shape[0] # matrix version
            self.is_parsed = True
           
            result = r"\begin{align} %s \end{align}" % r" \\ ".join(self.equation_string)
            
            display(Latex(result))
        else:
            print(self.name + " has already been parsed!")
        print("--------------------------------------------------------------")
    
    def _parse(self):  # private _parse() function for internal transfer function calculation 
        if self.is_parsed == False:
            self.ss_const, self.ss = self._get_ss()
            self.A, self.B, self.C, self.D = self._split_matrix(self.ss_const) 
            self.tf = self._get_tf()
            self.oraclenumber = self.tf.shape[0] # matrix version
            self.is_parsed = True
    
    def get_ss(self, verbose = False):
        if self.is_parsed == False:
            self._parse()
            
        if self.ss == None:
            self.A, self.B, self.C, self.D = self._tf_to_ss()
            self.ss = Matrix([[self.A, self.B], [self.C, self.D]])
            self.state_vars = []
            self.update_vars = []
            self.left_vars = []
            self.right_vars = []
            
            for i in range(self.A.shape[0]):
                var = self.add_var('x_%i' %i, commutative = False)
                self.left_vars.append(self.update_map[var])
                self.right_vars.append(var)
            for i in range(len(self.output_eqs)):
                self.left_vars.append(self.input_vars[i])
        
        if verbose == True:
            print("--------------------------------------------------------------")
            print("State-space realization:")
            left1 = []
            left2 = []
            right1 = []
            right2 = []
            for i in range(len(self.update_vars)):
                left1.append(self.left_vars[i])
                right1.append(self.right_vars[i])
            for i in range(len(self.output_eqs)):
                left2.append(self.left_vars[i + len(self.update_vars)])
                right2.append(self.output_eqs[i].args[1])

            eq1_left = Matrix(left1)
            eq1_right1 = MatMul(self.A, Matrix(right1)) 
            eq1_right2 = MatMul(self.B, Matrix(right2))
            eq2_left = Matrix(left2) 
            eq2_right1 = MatMul(self.C, Matrix(right1)) 
            eq2_right2 = MatMul(self.D, Matrix(right2))
            
            print_string = []
            print_string.append("%s & = %s" % (latex(eq1_left), latex(eq1_right1) + '+' + latex(eq1_right2)))
            print_string.append("%s & = %s" % (latex(eq2_left), latex(eq2_right1) + '+' + latex(eq2_right2)))
            result = r"\begin{align} %s \end{align}" % r" \\ ".join(print_string)
            
            display(Latex(result))
            print("--------------------------------------------------------------")
        else:
            return self.ss
            
    def get_canonical_ss(self, ss_type = 'c', verbose = False):
        
        A, B, C, D = self._tf_to_ss(ss_type)
        ss = Matrix([[A, B], [C, D]])
        
        if verbose == True:
            print("--------------------------------------------------------------")
            if ss_type == 'c':
                print("Controllable state-space realization:")
            else:
                print("Observable state-space realization:")
                
            left1 = []
            left2 = []
            right1 = []
            right2 = []
            for i in range(A.shape[0]):
                left1.append(symbols('x_%i^+' %i, commutative = False))
                right1.append(symbols('x_%i' %i, commutative = False))
                
            for i in range(len(self.output_eqs)):
                left2.append(self.input_vars[i])
                right2.append(self.output_eqs[i].args[1])

            eq1_left = Matrix(left1)
            eq1_right1 = MatMul(A, Matrix(right1)) 
            eq1_right2 = MatMul(B, Matrix(right2))
            eq2_left = Matrix(left2) 
            eq2_right1 = MatMul(C, Matrix(right1)) 
            eq2_right2 = MatMul(D, Matrix(right2))
            
            print_string = []
            print_string.append("%s & = %s" % (latex(eq1_left), latex(eq1_right1) + '+' + latex(eq1_right2)))
            print_string.append("%s & = %s" % (latex(eq2_left), latex(eq2_right1) + '+' + latex(eq2_right2)))
            result = r"\begin{align} %s \end{align}" % r" \\ ".join(print_string)
            
            display(Latex(result))
            print("--------------------------------------------------------------")
        else:
            return ss
        
    def get_tf(self, verbose = False):
        if self.is_parsed == False:
            self._parse()
            
        if verbose == True:
            print("--------------------------------------------------------------")
            print("Transfer function:")
            display(self.tf)
            print("--------------------------------------------------------------")
        else:
            return self.tf
        
    
    def duplicate(self):
        self._parse()
        new_algo = Algorithm("duplication " + "of " + self.name, "cp_" + self.namestr)
        new_algo.tf = copy.deepcopy(self.tf)
        new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
        new_algo.parameter_set = copy.deepcopy(self.parameter_set)
        new_algo.matrix_set = copy.deepcopy(self.matrix_set)
        new_algo.compare_map = copy.deepcopy(self.compare_map)
        new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
        new_algo.input_vars = copy.deepcopy(self.input_vars)
        for i in range(new_algo.oraclenumber):
            new_algo.output_eqs.append(self.output_eqs[i])
        new_algo.is_parsed = True
        return new_algo
    
    def _commutative_algo(self):
        self._parse()
        new_algo = Algorithm(self.name, self.namestr)
        new_algo.tf = copy.deepcopy(self.tf)
        new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
        new_algo.parameter_set = copy.deepcopy(self.parameter_set)
        new_algo.matrix_set = copy.deepcopy(self.matrix_set)
        new_algo.compare_map = copy.deepcopy(self.compare_map)
        new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
        
        for var in new_algo.matrix_set:
            new_algo.tf = new_algo.tf.subs(var, new_algo.noncommut_map[var])
        new_algo.tf = simplify(new_algo.tf)
        new_algo.input_vars = copy.deepcopy(self.input_vars)
        for i in range(new_algo.oraclenumber):
            new_algo.output_eqs.append(self.output_eqs[i])
        new_algo.is_parsed = True
        return new_algo
        
    
    def permute(self, step = 0):
        self._parse()
        if step + 1 > 0 and step + 1 <= self.oraclenumber - 1 and self.oraclenumber > 1:
            new_algo = Algorithm("permutation " + "of " + self.name, "p_" + self.namestr)
            new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
            m = self.oraclenumber - step - 1       
            new_algo.tf = zeros(self.oraclenumber)
            new_algo.tf[0:m, 0:m] = self.tf[1:m + 1, 1:m + 1]
            new_algo.tf[m,m] = self.tf[0, 0]
            new_algo.tf[0:m,m] = simplify((self.tf[m, 0:m] / z).T)
            new_algo.tf[m,0:m] = simplify((self.tf[0:m, m] * z).T)
            new_algo.parameter_set = copy.deepcopy(self.parameter_set)
            new_algo.matrix_set = copy.deepcopy(self.matrix_set)
            new_algo.compare_map = copy.deepcopy(self.compare_map)
            new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
            
            for i in range(step + 1, new_algo.oraclenumber):
                new_algo.input_vars.append(self.input_vars[i])
                new_algo.output_eqs.append(self.output_eqs[i])
            for i in range(0, step + 1):
                new_algo.input_vars.append(self.input_vars[i])
                new_algo.output_eqs.append(self.output_eqs[i])
            
            new_algo.is_parsed = True
            return new_algo
        else:
            raise ValueError("Invalid permutation!")


    def repeat(self, times = 2):
        self._parse()
        if self.A != None and self.B != None and self.C != None and self.D != None:
            new_algo = Algorithm("repetition " + "of " + self.name, "r_" + self.namestr)
            if times == 1:
                new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
                new_algo.A = copy.deepcopy(self.A)
                new_algo.B = copy.deepcopy(self.B)
                new_algo.C = copy.deepcopy(self.C)
                new_algo.D = copy.deepcopy(self.D)
                new_algo.tf = copy.deepcopy(self.tf)
            elif times == 2:
                new_algo.oraclenumber = times * self.oraclenumber
                new_algo.A = self.A * self.A
                new_algo.B = Matrix(BlockMatrix([[self.A * self.B, self.B]]))
                new_algo.C = Matrix(BlockMatrix([[self.C], [self.C * self.A]]))
                new_algo.D = Matrix(BlockMatrix([[self.D, zeros(self.C.shape[0], self.B.shape[1])], [self.C * self.B, self.D]]))
                new_algo.tf = simplify(new_algo.C * (z * eye(new_algo.A.shape[0]) - new_algo.A)**(-1) * new_algo.B + new_algo.D)
            elif times > 2:
                new_algo.oraclenumber = times * self.oraclenumber
                new_algo.A = (self.A)**times
                new_algo.B = zeros(self.A.shape[0], times*self.D.shape[0])
                new_algo.C = zeros(times*self.D.shape[0], self.A.shape[0])
                new_algo.D = zeros(times*self.D.shape[0], times*self.D.shape[0])
                for i in range(times):
                    new_algo.B[:, (times - i - 1)*self.D.shape[0] : (times - i)*self.D.shape[0]] = (self.A**i)*self.B
                    new_algo.C[i*self.D.shape[0] : (i + 1)*self.D.shape[0], :] = self.C*(self.A**i)
                    
                    if i == times - 1:
                        for j in range(times):
                            new_algo.D[(times - i - 1 + j)*self.D.shape[0] : (times - i - 1 + j + 1)*self.D.shape[0], 
                                       j*self.D.shape[0] : (j + 1)*self.D.shape[0]] = self.D
                    else:
                        for j in range(i + 1):
                            new_algo.D[(times - i - 1 + j)*self.D.shape[0] : (times - i - 1 + j + 1)*self.D.shape[0], 
                                       j*self.D.shape[0] : (j + 1)*self.D.shape[0]] = self.C*(self.A**(times - i - 2))*self.B  
                    
                new_algo.tf = simplify(new_algo.C * (z * eye(new_algo.A.shape[0]) - new_algo.A)**(-1) * new_algo.B + new_algo.D) 
            else:
                raise ValueError("Invalid repeat!")
                
            new_algo.parameter_set = copy.deepcopy(self.parameter_set)
            new_algo.matrix_set = copy.deepcopy(self.matrix_set)
            new_algo.compare_map = copy.deepcopy(self.compare_map)
            new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
            new_algo.is_parsed = True
            return new_algo
        else:
            raise ValueError("Invalid repeat!")
    
    
    def conjugation_check(self, conjugate_oracle):
        m = conjugate_oracle
        if conjugate_oracle + 1 > self.oraclenumber or conjugate_oracle < 0:
            return False
        else:
            if simplify(self.tf[m, m]) == 0: 
                return False
            else:
                expr = self.tf[m, m]
                for var in self.matrix_set:
                    expr = expr.subs(var, self.noncommut_map[var])
                expr = simplify(expr)
                
                nu_degree = degree(expr.as_numer_denom()[0], gen = z)
                de_degree = degree(expr.as_numer_denom()[1], gen = z)
                
                if nu_degree == de_degree:
                    return True
                else:
                    return False
    
    def conjugate(self, conjugate_oracle = 0):
        self._parse()
        m = conjugate_oracle
        if self.conjugation_check(m):
            new_algo = Algorithm("conjugate " + "of " + self.name, "c_" + self.namestr)
            new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
            rowlist = list(range(self.oraclenumber))
            rowlist.remove(m)
            columnlist = list(range(self.oraclenumber))
            columnlist.remove(m)
            new_operated_tf = zeros(self.oraclenumber)
            for i in rowlist:
                for j in columnlist:
                    new_operated_tf[i, j] = self.tf[i, j] - self.tf[i, m] * self.tf[m, m]**(-1) * self.tf[m, j]
            for i in rowlist:
                new_operated_tf[i, m] = self.tf[i, m] * self.tf[m, m]**(-1)
            for j in columnlist:
                new_operated_tf[m, j] = - self.tf[m, m]**(-1) * self.tf[m, j]
            new_operated_tf[m, m] = self.tf[m, m]**(-1)
            new_algo.tf = simplify(new_operated_tf)
            new_algo.parameter_set = copy.deepcopy(self.parameter_set)
            new_algo.matrix_set = copy.deepcopy(self.matrix_set)
            new_algo.compare_map = copy.deepcopy(self.compare_map)
            new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
            new_algo.input_vars = copy.deepcopy(self.input_vars)
            for i in range(new_algo.oraclenumber):
                new_algo.output_eqs.append(self.output_eqs[i])
            new_algo.is_parsed = True
            return new_algo
        else:
            raise ValueError("Invalid conjugate oracle!")
    
    def dual(self):
        self._parse()
        dual_check = True
        for i in range(self.oraclenumber):
            dual_check = dual_check and self.conjugation_check(i)
        
        if dual_check == False:
            raise ValueError("Invalid duality!")
        else:
            try: 
                new_algo = Algorithm("duality " + "of " + self.name, "d_" + self.namestr)
                new_algo.oraclenumber = copy.deepcopy(self.oraclenumber)
                new_algo.tf = simplify(self.tf**(-1))
                new_algo.parameter_set = copy.deepcopy(self.parameter_set)
                new_algo.matrix_set = copy.deepcopy(self.matrix_set)
                new_algo.compare_map = copy.deepcopy(self.compare_map)
                new_algo.noncommut_map = copy.deepcopy(self.noncommut_map)
                new_algo.input_vars = copy.deepcopy(self.input_vars)
                for i in range(new_algo.oraclenumber):
                    new_algo.output_eqs.append(self.output_eqs[i])
                new_algo.is_parsed = True
                return new_algo
            except:
                raise ValueError("Invalid duality!")