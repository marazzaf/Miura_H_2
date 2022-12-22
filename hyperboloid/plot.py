#coding: utf-8

import numpy as np

a = np.array([9450, 35838, 139482, 550242])
b = np.array([1.916e-04, 2.377e-05, 3.005e-06, 3.817e-07])


res = 2*np.log(b[:-1]/b[1:]) / np.log(a[1:]/a[:-1])
print(res)
