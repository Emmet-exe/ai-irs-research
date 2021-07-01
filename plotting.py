import numpy as np
import scipy.optimize as op

# auxiliary base function for curve fitting
def func(x, a, b, c):
     return a * np.exp(-b * x) + c
    
# scipy curve fitting
def curve_fit(x, y):
    params, pcov = op.curve_fit(func, x, y)
    return func(np.array(x), *params)

# numpy linear regression
def line_fit(x, y):
     m, b = np.polyfit(x, y, 1)
     return m * np.array(x) + b