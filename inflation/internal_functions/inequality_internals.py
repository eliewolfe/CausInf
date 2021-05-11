#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 01:19:39 2021

@author: boraulu
"""
from __future__ import absolute_import
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import warnings

def ValidityCheck(y, SpMatrix):
    checkY = csr_matrix(y.ravel()).dot(SpMatrix)
    if checkY.min() <= -10**4:
        warnings.warn('The rounding of y has failed: checkY.min()='+str(checkY.min())+'')
    return checkY.min() >= -10 ** -10

def IntelligentRound(y, SpMatrix):
    #y = np.asanyarray(y, dtype=np.float_)
    scale = np.abs(np.amin(y))
    n = 1
    # yt=np.rint(y*n)
    # yt=y*n
    y2 = np.rint(n * y / scale).astype(np.int)  # Can I do this with sparse y?

    while n<=720:
        vcheck = ValidityCheck(y2, SpMatrix)
        if vcheck:
            break
        else:
            n = n * (n + 1)
            y2 = np.rint(np.asanyarray(n * y / scale, dtype=np.float_)).astype(np.int)
    if vcheck:
        return y2
    else:
        warnings.warn('Unable to round y. Try again with simplex method instead of interior point?', RuntimeWarning, stacklevel=2)
        return y
    # while n <= 720 and not ValidityCheck(y2, SpMatrix):
    #     n = n * (n + 1)
    #     # yt=np.rint(y*n)
    #     # yt=yt*n
    #     # yt=yt/n
    #     y2 = np.rint(np.asanyarray(n * y / scale, dtype=np.float_)).astype(np.int)
    #     # y2=y2/(n*10)
    #     # if n > 10**6:
    #     #   y2=y
    #     #  print("RoundingError: Unable to round y")
    #     # yt=np.rint(yt*100)
    #
    # return y2

def indextally(y):
    
    indextally = defaultdict(list)
    [indextally[str(val)].append(i) for i, val in enumerate(y) if val != 0]
    
    return indextally

def symboltally(indextally,symbolic_b):
    
    symboltally = defaultdict(list)
    for i, vals in indextally.items():
        symboltally[i] = np.take(symbolic_b, vals).tolist()
        
    return symboltally

def inequality_as_string(y, symbolic_b):

    final_ineq_WITHOUT_ZEROS = list(map(''.join, np.vstack((y, symbolic_b)).T[np.flatnonzero(y)]))
    Inequality_as_string = '0<=' + "+".join(final_ineq_WITHOUT_ZEROS).replace('+-', '-').replace('+1P', '+P').replace('-1P', '-P')

    return Inequality_as_string.replace('=1P', '=P')