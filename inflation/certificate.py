#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup solver output to yield a readable inequality.
"""
from __future__ import absolute_import
import json
from collections import defaultdict
import numpy as np
#import sympy as sy
from numba import njit
from scipy.sparse import csr_matrix

#from .inflationmatrix import Generate_b_and_counts
#from .quickgraph import LearnSomeInflationGraphParameters


def WitnessDataTest(y, b, tol):
    IncompTest = (np.amin(y) < 0) and (np.dot(y, b) < tol)
    if IncompTest:
        print('Distribution Compatibility Status: INCOMPATIBLE')
    else:
        print('Distribution Compatibility Status: COMPATIBLE')
    return IncompTest


@njit
def Epsilon(y, n):
    ymax = np.amax(np.abs(y))
    denom = 1000 * n
    return ymax / denom


def ValidityCheck(y, SpMatrix):
    # Smatrix=SpMatrix.toarray()    #DO NOT LEAVE SPARSITY!!
    checkY = csr_matrix(y.ravel()) * SpMatrix
    return checkY.min() >= 0


@njit
def yZeroFilter(yRaw, n):
    y = yRaw.copy()
    y[np.abs(y) < Epsilon(y, n)] = 0
    return y


def yRoundFilter(y, n):
    order = -np.floor(np.log10(Epsilon(y, n)))
    y2 = np.rint(y * (10 ** order)).astype(np.int)
    uy = np.unique(np.abs(y2))
    GCD = np.gcd.reduce(uy)
    y2 = y2 / GCD
    return y2.astype(np.int)


# @njit
# def IntelligentRound(y,n):
#    denom = n * np.abs(np.amin(y))
#    return np.rint(y/denom).astype(np.int)


def IntelligentRound(y, SpMatrix):
    scale = np.abs(np.amin(y))
    n = 1
    y2 = np.rint(n * y / scale).astype(np.int)  # Can I do this with sparse y?
    while not ValidityCheck(y, SpMatrix):
        n = n * (n + 1)
        y2 = np.rint(n * y / scale).astype(np.int)
    return y2


# def Inequality(Graph, inflation_order, card, SpMatrix, b, Sol):
#     #Modifying Feb 2, 2021 to pass B_symbolic as an argument for Inequality
#     yRaw = np.array(Sol['x']).ravel()
#     tol = 1 / (np.linalg.norm(b, np.inf) * np.linalg.norm(yRaw, np.inf) * (10 ** 6))
#     if WitnessDataTest(yRaw, b, tol):
#         y = IntelligentRound(yRaw, SpMatrix)
#         obs_count, num_vars, names = LearnSomeInflationGraphParameters(Graph, inflation_order)
#
#         symbolnames = ['P(' + ''.join([''.join(str(i)) for i in idx]) + ')' for idx in
#                        np.ndindex(tuple(np.full(obs_count, card, np.uint8)))]
#         symbols = np.array(sy.symbols(symbolnames))
#         symb, counts = Generate_b_and_counts(symbols, inflation_order)
#         symbtostring = np.array([str(term).replace('*P', 'P') for term in symb])
#
#         yRaw = np.multiply(yRaw, counts)
#
#         y3 = np.multiply(y, counts)
#         uy = np.unique(np.abs(y3))
#         GCD = np.gcd.reduce(uy)
#         y3 = y3 / GCD
#         y3 = y3.astype(np.int)
#         # ySparse=coo_matrix(y3)
#
#         # print('Now to make things human readable...')
#
#         indextally = defaultdict(list)
#         # [indextally[str(y3[i])].append(i) for i in np.nonzero(y3)]
#         [indextally[str(val)].append(i) for i, val in enumerate(y3) if val != 0]
#
#         symboltally = defaultdict(list)
#         for i, vals in indextally.items():
#             symboltally[i] = symbtostring.take(vals).tolist()
#
#         final_ineq_WITHOUT_ZEROS = np.multiply(y3[np.nonzero(y3)], symb[np.nonzero(y3)])
#         Inequality_as_string = '0≤' + "+".join([str(term) for term in final_ineq_WITHOUT_ZEROS]).replace('*P', 'P')
#         Inequality_as_string = Inequality_as_string.replace('+-', '-')
#
#         print("Writing to file: 'inequality_output.json'")
#
#         returntouser = {
#             'Order of variables': names,
#             'Raw rolver output': yRaw.tolist(),
#             'Inequality as string': Inequality_as_string,
#             'Coefficients grouped by index': indextally,
#             'Coefficients grouped by symbol': symboltally,
#             # 'b_vector_position': idx.tolist(),
#             'Clean solver output': y3.tolist(),
#             'Symolic association': symbtostring.tolist()
#         }
#         f = open('inequality_output.json', 'w')
#         print(json.dumps(returntouser), file=f)
#         f.close()
#         return returntouser
#     else:
#         return print('Compatibility Error: The input distribution is compatible with given inflation order test.')

def Inequality(SpMatrix, B_numeric, B_symbolic, Sol):
    #Modified Feb 2, 2021 to pass B_symbolic as an argument for Inequality
    yRaw = np.array(Sol['x']).ravel()
    tol = 1 / (np.linalg.norm(B_numeric, np.inf) * np.linalg.norm(yRaw, np.inf) * (10 ** 6)) #TODO: Choose better tolerance function. This is yielding false incompatibility claims.
    if WitnessDataTest(yRaw, B_numeric, tol):
        y = IntelligentRound(yRaw, SpMatrix)
        # print('Now to make things human readable...')
        indextally = defaultdict(list)
        [indextally[str(val)].append(i) for i, val in enumerate(y) if val != 0]

        symboltally = defaultdict(list)
        for i, vals in indextally.items():
            symboltally[i] = np.take(B_symbolic,vals).tolist()

        #final_ineq_WITHOUT_ZEROS = np.multiply(y[np.nonzero(y)], sy.symbols(np.take(B_symbolic,np.nonzero(y))))
        #Inequality_as_string = '0≤' + "+".join([str(term) for term in final_ineq_WITHOUT_ZEROS]).replace('*P', 'P')
        #Inequality_as_string = Inequality_as_string.replace('+-', '-')

        print("Writing to file: 'inequality_output.json'")

        returntouser = {
            #'Order of variables': names,
            'Raw rolver output': yRaw.tolist(),
            #'Inequality as string': Inequality_as_string,
            'Coefficients grouped by index': indextally,
            'Coefficients grouped by symbol': symboltally,
            # 'b_vector_position': idx.tolist(),
            'Clean solver output': y.tolist()#,
            #'Symbolic association': symbtostring.tolist()
        }
        f = open('inequality_output.json', 'w')
        print(json.dumps(returntouser), file=f)
        f.close()
        return returntouser
    else:
        return print('Compatibility Error: The input distribution is compatible with given inflation order test.')