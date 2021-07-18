from .algorithm import *
from .algorithms_library import *

from contextlib import contextmanager
import threading
import _thread
import time

class TimeoutException(Exception):
    def __init__(self, msg = ''):
        self.msg = msg

@contextmanager
def _time_limit(seconds, msg = ''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        timer.cancel()

def _is_solve(test_equation_set, algo_parameter_set):
    if len(test_equation_set) == 1:
        test_equation_set = test_equation_set[0]
    if algo_parameter_set == []:
        return False
    elif algo_parameter_set != []:
        try:
            with _time_limit(10, 'sleep'):
                test_sol = solve(test_equation_set, algo_parameter_set)
        except:
            test_sol = []
        if test_sol == []:
            return False
        elif type(test_sol) == dict:
            if len(test_sol) == len(algo_parameter_set):
                para_index = np.zeros(len(algo_parameter_set))
                para_i = 0
                for para in algo_parameter_set:
                    test_para = simplify(test_sol[para])
                    if z in test_para.free_symbols:
                        para_index[para_i] = 1
                        para_i = para_i + 1
                        
                if np.sum(para_index) == 0:
                    return True
                else:
                    return False
            else:
                return False
                
        elif type(test_sol) == list:
            para_index2 = np.zeros(len(test_sol))
            for i in range(len(test_sol)):
                if len(algo_parameter_set) == len(test_sol[i]):
                    para_index3 = np.zeros(len(algo_parameter_set))
                    for j in range(len(algo_parameter_set)):
                        test_para3 = simplify(test_sol[i][j])
                        if z in test_para3.free_symbols:
                            para_index3[j] = 1
                    if np.sum(para_index3) == 0:
                        para_index2[i] = 1 
                else:
                    return False
            if np.sum(para_index2) > 0:
                return True
            else:
                return False

            
def _algo_test(algo1, algo_list):
    algo_index = np.zeros(len(algo_list))
    test_tf1 = algo1.tf
    perms = permutations(range(algo1.oraclenumber)) 
    permlist = list(perms)
    for var in algo1.parameter_set + algo1.matrix_set:
        test_tf1 = test_tf1.subs(var, algo1.compare_map[var])
    for var in algo1.matrix_set:
        test_tf1 = test_tf1.subs(algo1.noncommut_map[var], algo1.compare_map[var])
    test_tf1 = simplify(test_tf1)
    compare_para_set = [algo1.compare_map[var] for var in algo1.parameter_set + algo1.matrix_set]
    
    for i in range(len(algo_list)):
        for perm in permlist:
            p = Permutation(perm)
            rowperm = MatrixPermute(test_tf1, p, axis=0)
            test_tf = MatrixPermute(rowperm, p, axis=1) 
            algo = algo_list[i]
            if algo1.oraclenumber == algo.oraclenumber:
                test_tf2 = algo.tf
                for var in algo.matrix_set:
                    test_tf2 = test_tf2.subs(var, algo.noncommut_map[var])
                if simplify(algo1.tf - algo.tf) == zeros(algo1.oraclenumber):
                    algo_index[i] = 1
                    break
                else:
                    test_equation_set = []
                    for k in range(algo1.oraclenumber):
                        for j in range(algo1.oraclenumber):
                            test_expr = test_tf[k, j] - test_tf2[k, j]
                            test_equation_set.append(test_expr)
                    if _is_solve(test_equation_set, algo.parameter_set + [algo.noncommut_map[var] for var in algo.matrix_set]):
                        algo_index[i] = 1
                        break
                    elif _is_solve(test_equation_set, compare_para_set):
                        algo_index[i] = 1
                        break
    return algo_index

def _is_solve_detailed(test_equation_set, algo_parameter_set):
    solve_tag = 0
    if len(test_equation_set) == 1:
        test_equation_set = test_equation_set[0]
    if algo_parameter_set == []:
        return solve_tag
    elif algo_parameter_set != []:
        try:
            with _time_limit(10, 'sleep'):
                test_sol = solve(test_equation_set, algo_parameter_set)
        except:
            test_sol = []
        if test_sol == []:
            return solve_tag, test_sol
        elif type(test_sol) == dict:
            if len(test_sol) == len(algo_parameter_set):
                para_index = np.zeros(len(algo_parameter_set))
                para_i = 0
                for para in algo_parameter_set:
                    test_para = simplify(test_sol[para])
                    if z in test_para.free_symbols:
                        para_index[para_i] = 1
                        para_i = para_i + 1
                        
                if np.sum(para_index) == 0:
                    solve_tag = 1
                    return solve_tag, test_sol
                else:
                    return solve_tag, test_sol
            else:
                return solve_tag, test_sol
                
        elif type(test_sol) == list:
            para_index2 = np.zeros(len(test_sol))
            for i in range(len(test_sol)):
                if len(algo_parameter_set) == len(test_sol[i]):
                    para_index3 = np.zeros(len(algo_parameter_set))
                    for j in range(len(algo_parameter_set)):
                        test_para3 = simplify(test_sol[i][j])
                        if z in test_para3.free_symbols:
                            para_index3[j] = 1
                    if np.sum(para_index3) == 0:
                        para_index2[i] = 1 
                else:
                    return solve_tag, test_sol
            if np.sum(para_index2) > 0:
                solve_tag = 2
                return solve_tag, test_sol
            else:
                return solve_tag, test_sol

def _algo_test_detailed(algo1, algo_list):
    algo_index = np.zeros(len(algo_list))
    test_tf1 = algo1.tf
    perms = permutations(range(algo1.oraclenumber)) 
    permlist = list(perms)
    for var in algo1.parameter_set + algo1.matrix_set:
        test_tf1 = test_tf1.subs(var, algo1.compare_map[var])
        
    for var in algo1.matrix_set:
        test_tf1 = test_tf1.subs(algo1.noncommut_map[var], algo1.compare_map[var])
    
    test_tf1 = simplify(test_tf1)
    
    compare_para_set = [algo1.compare_map[var] for var in algo1.parameter_set + algo1.matrix_set]
    
    sol_list = []
    sol_index = []
    
    for i in range(len(algo_list)):
        for perm in permlist:
            p = Permutation(perm)
            rowperm = MatrixPermute(test_tf1, p, axis=0)
            test_tf = MatrixPermute(rowperm, p, axis=1)
            algo = algo_list[i]
            if algo1.oraclenumber == algo.oraclenumber:
                test_tf2 = algo.tf
                for var in algo.matrix_set:
                    test_tf2 = test_tf2.subs(var, algo.noncommut_map[var])
                if simplify(algo1.tf - algo.tf) == zeros(algo1.oraclenumber):
                    algo_index[i] += 1
                    sol_list.append(None)
                    sol_index.append(3)
                    break
                else:
                    test_equation_set = []
                    for k in range(algo1.oraclenumber):
                        for j in range(algo1.oraclenumber):
                            test_expr = test_tf[k, j] - test_tf2[k, j]
                            test_equation_set.append(test_expr)
                    
                    solution = []
                    solve_tag1, sol1 = _is_solve_detailed(test_equation_set, algo.parameter_set + [algo.noncommut_map[var] for var in algo.matrix_set])
                    if solve_tag1 == 1:
                        algo_index[i] += 1
                        for para1 in algo.parameter_set + [algo.noncommut_map[var] for var in algo.matrix_set]:
                            solution1 = sol1[para1]
                            for var1 in algo1.parameter_set + algo1.matrix_set:
                                solution1 = solution1.subs(algo1.compare_map[var1], var1)
                                
                            for var1 in algo.matrix_set:
                                solution1 = solution1.subs(algo.noncommut_map[var1], var1)
                            
                            solution.append(simplify(solution1))
                        sol_list.append(solution)
                        sol_index.append(1)
                        break
                    elif solve_tag1 == 2:
                        for s_i in range(len(sol1)):
                            algo_index[i] += 1
                            for s_j in range(len(algo.parameter_set + [algo.noncommut_map[var] for var in algo.matrix_set])):
                                solution1 = sol1[s_i][s_j]
                                for var1 in algo1.parameter_set + algo1.matrix_set:
                                    solution1 = solution1.subs(algo1.compare_map[var1], var1)
                                    
                                for var1 in algo.matrix_set:
                                    solution1 = solution1.subs(algo.noncommut_map[var1], var1)
                                
                                solution.append(simplify(solution1))                        
                            sol_list.append(solution)
                            sol_index.append(1)
                        break
                    else: 
                        solve_tag2, sol2 = _is_solve_detailed(test_equation_set, compare_para_set)
                        if solve_tag2 == 1:
                            algo_index[i] += 1
                            for para1 in compare_para_set:
                                solution2 = sol2[para1]
                                for var1 in algo.matrix_set:
                                    solution2 = solution2.subs(algo.noncommut_map[var1], var1)
                                
                                for var1 in algo1.parameter_set + algo1.matrix_set:
                                    solution2 = solution2.subs(algo1.compare_map[var1], var1)
                                    
                                solution.append(simplify(solution2))
                            sol_list.append(solution)
                            sol_index.append(2)
                            break
                        elif solve_tag2 == 2:
                            for s_i in range(len(sol2)):
                                algo_index[i] += 1
                                for s_j in range(len(compare_para_set)):
                                    solution2 = sol2[s_i][s_j]
                                    for var1 in algo.matrix_set:
                                        solution2 = solution2.subs(algo.noncommut_map[var1], var1)
                                    
                                    for var1 in algo1.parameter_set + algo1.matrix_set:
                                        solution2 = solution2.subs(algo1.compare_map[var1], var1)
                                    
                                    solution.append(simplify(solution2))                        
                                sol_list.append(solution)
                                sol_index.append(2)
                            break
    return algo_index, sol_list, sol_index

def _print_parameters(algo1, algo2):
    overlap = []
    for var in algo1.parameter_set + algo1.matrix_set:
        if var in algo2.parameter_set + algo2.matrix_set:
            overlap.append(var)
    print("Parameters of " + algo1.name + ":")
    parameters = ''
    n_para = 0
    for var in algo1.parameter_set + algo1.matrix_set:
        n_para = n_para + 1
        if var in overlap:
            parameters = parameters + latex(var) + '_1'
        else:
            parameters = parameters + latex(var)
        if n_para != len(algo1.parameter_set + algo1.matrix_set):
            parameters = parameters + ","
            parameters = parameters + "\;"
    result1 = "$${}$$".format(parameters)
    display(Latex(result1))
    print("Parameters of " + algo2.name + ":")
    parameters = ''
    n_para = 0
    for var in algo2.parameter_set + algo2.matrix_set:
        n_para = n_para + 1
        if var in overlap:
            parameters = parameters + latex(var) + '_2'
        else:
            parameters = parameters + latex(var)
        if n_para != len(algo2.parameter_set + algo2.matrix_set):
            parameters = parameters + ","
            parameters = parameters + "\;"
    result2 = "$${}$$".format(parameters)
    display(Latex(result2))
    return overlap

def _print_test_result(algo_index, sol_list, sol_index, test_algo, algo_list, algo1, string):
    index = 0
    for i in range(len(algo_index)):
        if algo_index[i] > 0:
            print("==============================================================")
            for j in range(int(algo_index[i])):
                for k in range(index, index + int(algo_index[i])):
                    if sol_index[k] == 3:
                        print(algo1.name + string + algo_list[i].name + ".")
                        print("Algorithm parameters are the same.")
                        print("==============================================================")
                    elif sol_index[k] == 2:
                        m = 0
                        overlap = _print_parameters(algo1, algo_list[i])
                        print(algo1.name + string + algo_list[i].name + ", if the parameters satisfy:")
                        for var in algo1.parameter_set + algo1.matrix_set:
                            expr = sol_list[k][m]
                            for var2 in overlap:
                                expr = expr.subs(var2, symbols(str(var2) + '_2', commutative = var2.is_commutative))
                            expr = simplify(expr)
                    
                            if var in overlap:
                                result = "$${}$$".format(latex(var) + '_1' + "=" + latex(expr))
                            else:
                                result = "$${}$$".format(latex(var) + "=" + latex(expr))
                            display(Latex(result))
                            m = m + 1
                        print("==============================================================")
                    elif sol_index[k] == 1:
                        m = 0
                        overlap = _print_parameters(algo1, algo_list[i])
                        print(algo1.name + string + algo_list[i].name + ", if the parameters satisfy:")
                        for var in algo_list[i].parameter_set + algo_list[i].matrix_set:
                            expr = sol_list[k][m]
                            for var2 in overlap:
                                expr = expr.subs(var2, symbols(str(var2) + '_1', commutative = var2.is_commutative))
                            expr = simplify(expr)
                            
                            if var in overlap:
                                result = "$${}$$".format(latex(var) + '_2' + "=" + latex(expr))
                            else:
                                result = "$${}$$".format(latex(var) + "=" + latex(expr))
                            display(Latex(result))
                            m = m + 1
                        print("==============================================================")
            index = index + int(algo_index[i])

def test_equivalent(algo1, algo_list = algo_library.library):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        algo_index, sol_list, sol_index = _algo_test_detailed(algo1, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
    print("--------------------------------------------------------------")
    if np.sum(algo_index) == 0:
        print("No equivalent algorithm found.")
    else:
        _print_test_result(algo_index, sol_list, sol_index, algo1, algo_list, algo1, " is equivalent to ")
    print("--------------------------------------------------------------")
            
def is_equivalent(algo1, algo_list = algo_library.library, verbose = False):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        test_algo = algo1
        algo_index = _algo_test(test_algo, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
    
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_list)):
            if algo_index[i] > 0:
                print(algo1.name + " is equivalent to " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")
    
    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist
    
def test_duality(algo1, algo_list = algo_library.library):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        test_algo = algo1.dual()
        algo_index, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
    
    print("--------------------------------------------------------------")
    if np.sum(algo_index) == 0:
        print("No dual algorithm found.")
    else:
        _print_test_result(algo_index, sol_list, sol_index, test_algo, algo_list, algo1, " is dual to ")
    print("--------------------------------------------------------------")

def is_duality(algo1, algo_list = algo_library.library, verbose = False):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        test_algo = algo1.dual()
        algo_index = _algo_test(test_algo, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
        
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_index)):
            if algo_index[i] > 0:
                print(algo1.name + " is dual to " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")
            
    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist
    
        
def test_permutation(algo1, algo_list = algo_library.library):
    print("--------------------------------------------------------------")
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    permutation_index = np.zeros(algo1.oraclenumber - 1)
    for j in range(algo1.oraclenumber - 1):
        try:            
            test_algo = algo1.permute(j)
            algo_index, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
        except:
            algo_index = np.zeros(len(algo_list))    
        if np.sum(algo_index) > 0:
            permutation_index[j] = 1
            _print_test_result(algo_index, sol_list, sol_index, test_algo, algo_list, algo1, " is a permutation of ")
    if np.sum(permutation_index) == 0:
        print("No permutation algorithm found.")
    print("--------------------------------------------------------------")

def is_permutation(algo1, algo_list = algo_library.library, verbose = False):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    algo_index = np.zeros(len(algo_list))
    for j in range(algo1.oraclenumber - 1):
        try:            
            test_algo = algo1.permute(j)
            algo_index2 = _algo_test(test_algo, algo_list)
        except:
            algo_index2 = np.zeros(len(algo_list))
        for i in range(len(algo_list)):
            algo_index[i] = algo_index[i] + algo_index2[i]
            
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_index)):
            if algo_index[i] > 0:
                print(algo1.name + " is a permutation of " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")

    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist

def test_conjugation(algo1, algo_list = algo_library.library):
    print("--------------------------------------------------------------")
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    conjugation_index = np.zeros(algo1.oraclenumber)
    for j in range(algo1.oraclenumber):
        try:            
            test_algo = algo1.conjugate(j)
            algo_index, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
        except:
            algo_index = np.zeros(len(algo_list))
        if np.sum(algo_index) > 0:
            conjugation_index[j] = 1
            _print_test_result(algo_index, sol_list, sol_index, test_algo, algo_list, algo1, " is conjugate to ")
    if np.sum(conjugation_index) == 0:
        print("No conjugate algorithm found.")
    print("--------------------------------------------------------------")

    
def is_conjugation(algo1, algo_list = algo_library.library, verbose = False):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    algo_index = np.zeros(len(algo_list))
    for j in range(algo1.oraclenumber):
        try:            
            test_algo = algo1.conjugate(j)
            algo_index2 = _algo_test(test_algo, algo_list)
        except:
            algo_index2 = np.zeros(len(algo_list))
        for i in range(len(algo_list)):
            algo_index[i] = algo_index[i] + algo_index2[i]
     
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_index)):
            if algo_index[i] > 0:
                print(algo1.name + " is conjugate to " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")

    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist        
    
def test_repetition(algo1, algo_list = algo_library.library, times = 2):
    repeat_time = times
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        test_algo = algo1.repeat(repeat_time)
        algo_index, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
    print("--------------------------------------------------------------")
    if np.sum(algo_index) == 0:
        print("No repetition algorithm found.")
    else:
        _print_test_result(algo_index, sol_list, sol_index, test_algo, algo_list, algo1, " is a repetition of ")
    print("--------------------------------------------------------------")  

def is_repetition(algo1, algo_list = algo_library.library, times = 2, verbose = False):
    repeat_time = times
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    try:
        test_algo = algo1.repeat(repeat_time)
        algo_index = _algo_test(test_algo, algo_list)
    except:
        algo_index = np.zeros(len(algo_list))
        
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_index)):
            if algo_index[i] > 0:
                print(algo1.name + " is a repetition of " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")
    
    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist


def is_conjugate_permutation(algo1, algo_list = algo_library.library, verbose = False):
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    algo_index = np.zeros(len(algo_list))
    test_algo = algo1._commutative_algo()
    for j in range(algo1.oraclenumber - 1):
        for k in range(algo1.oraclenumber):    
            try:
                test_algo = test_algo.permute(j).conjugate(k)
                algo_index2 = _algo_test(test_algo, algo_list)
            except:
                algo_index2 = np.zeros(len(algo_list))
            for i in range(len(algo_list)):
                algo_index[i] = algo_index[i] + algo_index2[i]
        try:
            test_algo = algo1.permute(j).dual()
            algo_index2 = _algo_test(test_algo, algo_list)
        except:
            algo_index2 = np.zeros(len(algo_list))
        for i in range(len(algo_list)):
            algo_index[i] = algo_index[i] + algo_index2[i]    
            
    if verbose == True:
        print("--------------------------------------------------------------")
        for i in range(len(algo_index)):
            if algo_index[i] > 0:
                print(algo1.name + " is a conjugate permutation " + algo_list[i].name + ".")
        print("--------------------------------------------------------------")

    boollist = []
    for i in range(len(algo_list)):
        if algo_index[i] > 0:
            boollist.append(True)
        else:
            boollist.append(False)
    if len(boollist) == 1:
        return boollist[0]
    else:
        return boollist
    
def test_conjugate_permutation(algo1, algo_list = algo_library.library):
    print("--------------------------------------------------------------")
    if algo_list.__class__ == Algorithm:
        algo_list = [algo_list]
    algo_index = np.zeros(len(algo_list))
    commutative_algo1 = algo1._commutative_algo()
    for j in range(algo1.oraclenumber - 1):
        for k in range(algo1.oraclenumber):
            try:
                test_algo = commutative_algo1.permute(j).conjugate(k)
                algo_index2, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
            except:
                algo_index2 = np.zeros(len(algo_list))    
            if np.sum(algo_index2) > 0:
                _print_test_result(algo_index2, sol_list, sol_index, test_algo, algo_list, algo1, " is a conjugate permutation of ")
            for i in range(len(algo_list)):
                algo_index[i] = algo_index[i] + algo_index2[i]
        try:
            test_algo = commutative_algo1.permute(j).dual()
            algo_index2, sol_list, sol_index = _algo_test_detailed(test_algo, algo_list)
        except:
            algo_index2 = np.zeros(len(algo_list))    
        if np.sum(algo_index2) > 0:
            _print_test_result(algo_index2, sol_list, sol_index, test_algo, algo_list, algo1, " is a conjugate permutation of ")
        for i in range(len(algo_list)):
            algo_index[i] = algo_index[i] + algo_index2[i]
    
    if np.sum(algo_index) == 0:
        print("No conjugate permutation algorithm found.")
    print("--------------------------------------------------------------")

def analyze(algo1, algo_list = algo_library.library):
    test_equivalent(algo1, algo_list)
    test_duality(algo1, algo_list)
    test_permutation(algo1, algo_list)
    test_repetition(algo1, algo_list)
    test_conjugation(algo1, algo_list)
    test_conjugate_permutation(algo1, algo_list)
