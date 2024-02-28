#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:17:38 2024

@author: hagedorn
"""

import numpy as np
from min_f import min_f
# (make sure, the min_f.py file is in the same folder than this examples.py
# file or your file in which you want to use the function min_f())

''' Also translated from
    http://www.opt.uni-duesseldorf.de/en/forschung-fs.html '''




#%% Definition of funcions

def f1(x, w = 1):
    ''' test function with minimizer in [0,1] '''
    #disp( ' call of f1 ' ); % to check how often this routine is being called

    if w == 1:
        y = -np.sin(3*x)+1
        # min at pi/6

    elif w == 2:
        y = np.exp(10*x) - np.exp(10)*x + 14752.0380362796
        # min near 0.7697414907

    elif w == 3:
        y = -np.arctan(10*x) + np.arctan(10)*x + 0.822937744880092
        # min near 0.24078011

    elif w == 4:
        y = x**4 - x + 0.472470393710578
        # min at 0.25^(1/3)

    elif w == 5:
        y = (x - 0.1)**4
        # min at 0.1

    elif w == 6:
        y = 1 + (x - 0.1)**4
        # small interval of minimizers near 0.1

    elif w == 7:
        sv = 1e4
        y = abs(x - 2/7)*2*sv - np.cos(sv*(x-2/7)) + 1
        # min at 2/7

    elif w == 8:
        y = 1 + (x - 1/7)**2

    elif w == 9:
        y = (x - 1/7)**2

    else:
        y = max(1/7 - x, 100*x - 100/7)

    return y




def f_crosen(x):
    ''' Chained Rosenbrock function with dimension specified by the input.
        The optimal value is zero suggested starting point: (-1,1,...1)^T '''

    #err = 0.0001
    err = 0

    n = len(x)
    y = (x[0] - 1)**2
    for i in range(2,n+1):
        y = y + 100*(x[i-1]-x[i-2]**2)**2

    y = y + err*(2*np.random.rand(1) - 1)
    return y




def f_stmod(x):
    ''' F_STYBLINSKI_TANG modified '''
    n = len(x)
    y = 0
    M = np.triu((n+1)*np.ones((n,n))-np.kron(np.arange(1,n+1),np.ones((n,1))))
    M = M + np.eye(n)
    x = M @ x
    for i in range(0,n):
        y = y + x[i]**4 - 16*x[i]**2 + 35*x[i]

    y = y + n*171.19854821721323

    return y




def f_cx(x):
    ''' F_CX Convex test function with minimum value zero,
        input x a real vector with at least one component '''

    const = 4
    oc    = 2*np.sqrt(const)

    n = len(x)
    y = 0

    for i in range(1,n+1):
        y = y + i**2*((np.exp(i*np.sum(x[:i])) +\
                       const*np.exp(-i*np.sum(x[:i])))-oc)

    return y


#%% Possible function calls

#x, fx, g, H, out = min_f(f1)
#x, fx, g, H, out = min_f(f1, 1) # different initial value

#opts = {'lb': -5, 'ub': 5, 'par_f': 3} # different bounds and function
#opts = {'par_f': 2} # different objective function
#x, fx, g, H, out = min_f(f1, options = opts)

options = {'lb': -5*np.ones((2,1)), 'ub': 5*np.ones((2,1))}
x, fx, g, H, out = min_f(f_crosen, np.zeros((2 ,1)), options)

#x, fx, g, H, out = min_f(f_stmod, np.zeros((8 ,1)))
#x, fx, g, H, out = min_f(f_cx, np.zeros((1 ,1)))

print(x)
print(fx)
print(g)
print(H)
print(out)
