import numpy as np
import csdl_alpha as csdl

# x data is B, y-data is H
def fit_dep_B(x, a, b, c):
    f = (a * csdl.exp(b*x+c) + 200) * x**1.4
    return f

# x data is H, y-data is B
def fit_dep_H(x, a):
    f = a * csdl.tanh(x/300 - .25) + .4
    return f