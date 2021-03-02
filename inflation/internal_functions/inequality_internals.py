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

def ValidityCheck(y, SpMatrix):
    checkY = csr_matrix(y.ravel()).dot(SpMatrix)
    if checkY.min() <= -10**4:
        raise RuntimeError('The rounding of y has failed: checkY.min()='+str(checkY.min())+'')
    return checkY.min() >= -10 ** -10

def IntelligentRound(y, SpMatrix):
    scale = np.abs(np.amin(y))
    n = 1
    # yt=np.rint(y*n)
    # yt=y*n
    y2 = np.rint(n * y / scale).astype(np.int)  # Can I do this with sparse y?

    while not ValidityCheck(y2, SpMatrix):
        n = n * (n + 1)
        # yt=np.rint(y*n)
        # yt=yt*n
        # yt=yt/n
        y2 = np.rint(n * y / scale).astype(np.int)
        # y2=y2/(n*10)
        # if n > 10**6:
        #   y2=y
        #  print("RoundingError: Unable to round y")
        # yt=np.rint(yt*100)

    return y2

def indextally(y):
    
    indextally = defaultdict(list)
    [indextally[str(val)].append(i) for i, val in enumerate(y) if val != 0]
    
    return indextally

def symboltally(indextally,symbolic_b):
    
    symboltally = defaultdict(list)
    for i, vals in indextally.items():
        symboltally[i] = np.take(symbolic_b, vals).tolist()
        
    return symboltally

def inequality_as_string(y,symbolic_b):
    #import sympy as sy
    #final_ineq_WITHOUT_ZEROS = np.multiply(y[np.nonzero(y)],sy.symbols(' '.join(np.take(symbolic_b, np.nonzero(y))[0])))

    final_ineq_WITHOUT_ZEROS = [str(val)+str(symbolic_b) for val,scal in zip(
        np.take(y,np.flatnonzero(y)),np.take(y,np.flatnonzero(symbolic_b)))]
                                                   
    #Inequality_as_string = '0<=' + "+".join([str(term) for term in final_ineq_WITHOUT_ZEROS]).replace('*P','P')
    Inequality_as_string = '0<=' + "+".join(final_ineq_WITHOUT_ZEROS).replace('+-', '-')
    #Inequality_as_string = Inequality_as_string.replace('+-', '-')

    return Inequality_as_string