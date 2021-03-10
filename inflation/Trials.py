#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:04:18 2021

@author: boraulu
"""

from __future__ import absolute_import
from igraph import Graph
from collections import Counter
import numpy as np
from itertools import product, combinations, permutations, chain
# import time
# from numba import njit
from scipy.sparse import coo_matrix, dok_matrix, csr_matrix, save_npz, load_npz
#from classes import *
import time
from internal_functions.inequality_internals import *
from linear_program_options.moseklp import InfeasibilityCertificate
# from linear_program_options.mosekinfeas import InfeasibilityCertificateAUTO
# from linear_program_options.inflationlp import InflationLP
# from classes import InflationProblem
from infgraph import InflationProblem

if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


# def delete_rows_csr(mat, indices):
#     """
#     Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
#     """
#     indices = list(indices)
#     mask = np.ones(mat.shape[0], dtype=bool)
#     mask[indices] = False
#     return mat[mask]
#
# def MixedCardinalityBaseConversion(cardinality, string):
#     card=[]
#     cardinality=np.array(cardinality)
#     for i in range(len(cardinality)):
#         if len(cardinality)-i-1 != 0:
#             card.append(np.product(cardinality[np.arange(len(cardinality)-i-1)+i+1]))
#         else:
#             card.append(1)
#     card=np.array(card)
#     str_to_array=np.array([int(i) for i in string])
#     return np.dot(card,str_to_array)

def arr(n, htimes, vtimes):
    if vtimes == 1:
        return (np.zeros((vtimes, htimes), np.uint8) + n)[0]
    else:
        return np.zeros((vtimes, htimes), np.uint8) + n


def AllG(mB, mA, g, h):
    if g != 0 and h != 0:
        M1 = np.hstack((arr(0, mB - g, 1), arr(3, g, 1)))
        M2 = np.hstack((np.array([0, 1]), arr(5, mB - 2 - g, 1), arr(3, g, 1)))
        M3 = np.hstack((arr(0, 1, mA - 2 - h), arr(5, mB - 1 - g, mA - 2 - h), arr(3, g, mB - 2 - h)))
        M4 = np.hstack((arr(2, mB - g, h), arr(4, g, h)))

        G = np.vstack((M1, M2, M3, M4))
    elif g != 0 and h == 0:
        M1 = np.hstack((arr(0, mB - g, 1), arr(3, g, 1)))
        M2 = np.hstack((np.array([0, 1]), arr(5, mB - 2 - g, 1), arr(3, g, 1)))
        M3 = np.hstack((arr(0, 1, mA - 2), arr(5, mB - 1 - g, mA - 2), arr(3, g, mB - 2)))

        G = np.vstack((M1, M2, M3))

    elif h != 0 and g == 0:
        M1 = np.hstack((arr(0, mB, 1)))
        M2 = np.hstack((np.array([0, 1]), arr(5, mB - 2, 1)))
        M3 = np.hstack((arr(0, 1, mA - 2 - h), arr(5, mB - 1, mA - 2 - h)))

        G = np.vstack((M1, M2, M3))

    else:
        M1 = arr(0, mB, 1)
        M2 = np.hstack((np.array([0, 1]), arr(5, mB - 2, 1)))
        M3 = np.hstack((arr(0, 1, mA - 2), arr(5, mB - 1, mA - 2)))

        G = np.vstack((M1, M2, M3))

    G = G.ravel()
    fives = np.where(G == 5)[0]

    dif = list(product('01', repeat=len(fives)))
    dif = [np.array([int(k) for k in ''.join(i)]) for i in dif]

    Gs = []

    for i in dif:
        G1 = G.copy()
        G1[fives.tolist()] = i
        G1 = G1.reshape((mA, mB))
        Gs.append(G1)
    return Gs


def UniformDistributionFromSupport(list_of_strings, cardinality_list):
    numvar = len(cardinality_list)
    prod_cardinality = np.prod(cardinality_list)
    cardinality_converter = np.flip(np.multiply.accumulate(np.hstack((1, np.flip(cardinality_list))))[:-1])
    data = np.zeros(prod_cardinality)
    array_of_integers = np.fromiter(map(int, ''.join(list_of_strings)), dtype=np.int).reshape((-1, numvar))
    list_of_integers = np.unique(np.dot(array_of_integers, cardinality_converter))
    numevents = len(list_of_integers)
    #print(list_of_integers)
    if numevents>0:
        data[list_of_integers] = 1 / numevents
    return data


def FindData(elem, Gs, cardinality):
    H = Gs[elem]
    Initial_Data = []
    To_Be_Removed = []
    To_Be_Added = []
    for i in range(len(H)):
        for j in range(len(H[i])):
            if H[i][j] == 0:
                Initial_Data.append('00' + str(i) + str(j))
                Initial_Data.append('11' + str(i) + str(j))
            elif H[i][j] == 1:
                Initial_Data.append('10' + str(i) + str(j))
                Initial_Data.append('01' + str(i) + str(j))
            elif H[i][j] == 2:
                Initial_Data.append('00' + str(i) + str(j))
                Initial_Data.append('01' + str(i) + str(j))
            elif H[i][j] == 3:
                Initial_Data.append('00' + str(i) + str(j))
                Initial_Data.append('10' + str(i) + str(j))
            elif H[i][j] == 4:
                Initial_Data.append('00' + str(i) + str(j))
                Initial_Data.append('11' + str(i) + str(j))
                To_Be_Removed.append('11' + str(i) + str(j))
                To_Be_Added.append('00' + str(i) + str(j))

    BellData = Initial_Data
    print(BellData)

    # original_card_product=np.prod(cardinality)
    # data = np.zeros(original_card_product)
    # data[list(map(lambda s: MixedCardinalityBaseConversion(cardinality, s),BellData))] = 1/len(BellData)
    # print(To_Be_Added)
    data = UniformDistributionFromSupport(BellData, cardinality)
    data[np.flatnonzero(UniformDistributionFromSupport(To_Be_Added, cardinality))] = 1 / np.prod(cardinality[-2:])
    data[np.flatnonzero(UniformDistributionFromSupport(To_Be_Removed, cardinality))] = 0

    # if To_Be_Added != []:
    #     for i in range(len(To_Be_Added)):
    #         data[MixedCardinalityBaseConversion(cardinality,To_Be_Added[i] )]=1/(cardinality[-1]*cardinality[-2])
    #         data[MixedCardinalityBaseConversion(cardinality,To_Be_Removed[i] )]=0
    return data


def WitnessDataTest(tol, numeric_b, y):
    IncompTest = (np.amin(y) < 0) and (np.dot(y, numeric_b) < tol)
    # if IncompTest:
    # print('Distribution Compatibility Status: INCOMPATIBLE')
    # else:
    #   print('Distribution Compatibility Status: COMPATIBLE')
    return IncompTest


def BellFacet(yRaw, InMat, tol, numeric_b):
    #InfMat = csr_matrix(InMat)
    if WitnessDataTest(tol, numeric_b, yRaw):
        y = IntelligentRound(yRaw, InfMat.asformat('csr', copy=False))
        checkY = csr_matrix(y.ravel()).dot(InfMat.asformat('csr', copy=False))
        # print(np.nonzero(y))
        # print(y[np.nonzero(y)[0]])
        # print(np.unique(checkY.toarray()))
        ZeroColumns = np.where(np.abs(checkY.toarray().ravel()) <= 10 ** -10)[0]
        # SmallerInfMat=InfMat[:,ZeroColumns.tolist()]-csr_matrix(np.repeat(InfMat[:,ZeroColumns.tolist()[0]].toarray().ravel()[:,np.newaxis],len(InfMat[:,ZeroColumns.tolist()].toarray()[0]),1))
        SmallerInfMat = InfMat[:, ZeroColumns.tolist()]

        SmallerRank = np.linalg.matrix_rank(SmallerInfMat.todense())
        print("----------------------")
        print(OriginalRank - SmallerRank)
        if (OriginalRank - SmallerRank) == 1:
            print('FOUND IT!')
            Facet = True
        elif (OriginalRank - SmallerRank) == 0:
            print('Yikes, this is an equality constraint.')
            Facet = True
        else:
            # print(yRaw)
            Facet = False

        return y, Facet, ZeroColumns, checkY.toarray().ravel()
    else:
        print('WARNING: Solver was unable to witness incompatibility!')
        return yRaw, False, np.asarray([]), csr_matrix(yRaw.ravel()).dot(InfMat.asformat('csr', copy=False))



def v_prim(vv):
    v = vv.copy()
    v = v.toarray().ravel()
    if not np.all(v == 0):
        min_v = np.amin(v[v != np.amin(v)])
        min_v_pos = np.where(v == min_v)[0]
        v_prime = v.copy()
        v_prime[min_v_pos.tolist()] = 1

        return v_prime
    else:
        return v


def IfSame(v_prime, old_v_primes):
    Same = False
    if old_v_primes != []:
        for old_v_prime in old_v_primes:
            check1 = np.all(v_prime == old_v_prime)
            Same = Same | check1
            if not check1:
                check2 = Counter(v_prime) == Counter(old_v_prime)
                Same = Same | check2
    return Same


mB = 3
mA = 3
g = 0
h = 0
n = 0
Gs = AllG(mB, mA, g, h)
FacetYs = []
# old_v_primes=[]
rawgraph = Graph.Formula("L->A:B,Ux->X,Uy->Y,X->A,Y->B")
# for n in range(len(Gs)):
card = [2, 2, mA, mB]
rawdata = FindData(n, Gs, card)

inflation_order = [1, 2, 2]
"""
-----------------------------------------------------------------
"""
InfProb = InflationProblem(rawgraph, rawdata, card, inflation_order)

numeric_b = InfProb.numeric_b
symbolic_b = InfProb.symbolic_b
InfMat = InfProb.inflation_matrix

# numeric_b,symbolic_b=InfProb.numeric_and_symbolic_b()
# InfMat=csr_matrix(InfProb.InflationMatrix())
"""
------------------------------------------------------------------
"""

outcomes = np.vstack(
    np.unravel_index(InfProb.expressible_sets[0].which_rows_to_keep, InfProb.expressible_sets[0].shape_of_eset)).T
meanings = InfProb.expressible_sets[0].from_inflation_indices[InfProb.expressible_sets[0].flat_eset]
# positions_to_check = list(
#     chain.from_iterable(combinations(np.flatnonzero(meanings == i), 2) for i in np.unique(meanings)))
# smart_to_check = tuple(np.asarray(positions_to_check).T)
# rows_to_keep_old = np.flatnonzero(
#     np.logical_not(np.any(outcomes[:, smart_to_check[0]] == outcomes[:, smart_to_check[1]], axis=-1)))
rows_to_keep = np.flatnonzero(np.all(np.vstack(tuple(
    list(np.unique(v,return_counts=True)[1].max()==1 for v in outcomes[:,meanings==m]) for
    m,c in zip(*np.unique(meanings, return_counts=True)) if c>1)), axis=0))
# print(rows_to_keep_old, rows_to_keep)

numeric_b = numeric_b[rows_to_keep]
symbolic_b = symbolic_b[rows_to_keep]
InfMat = InfMat[rows_to_keep]
InfMat = InfMat[:, InfMat.getnnz(0) > 0]
OriginalRank = np.linalg.matrix_rank(InfMat.todense())

# save_npz('Huge_Bell_InfMat',InfMat)
# save_npz('Huge_Bell_b',csr_matrix(numeric_b))

# InfMat=load_npz("Huge_Bell_InfMat.npz")
# numeric_b=load_npz("Huge_Bell_b.npz")
# numeric_b=numeric_b.toarray().ravel()


solve = InfeasibilityCertificate(InfMat, numeric_b)
yRaw = np.array(solve['x']).ravel()
tol = solve['gap']
y, Fcondition, t, v = BellFacet(yRaw, InfMat, tol, numeric_b)

print(inequality_as_string(y, symbolic_b))
# v_prime=v_prim(v)
# Scondition=IfSame(v_prime,old_v_primes)

# if Fcondition and not Scondition:
"""
if Fcondition:
    FacetYs.append(y)

epsilon = (2 / 3) / 4
All_kl = list(permutations(t, 2))
for i in range(len(All_kl)):
    k = All_kl[i][0]
    l = All_kl[i][1]
    Pk = InfMat[:, k].toarray().ravel()
    Pl = InfMat[:, l].toarray().ravel()
    # Pk=np.nonzero(Pk)[0]
    # Pl=np.nonzero(Pl)[0]
    # PPk=np.zeros(len(numeric_b))
    # PPl=np.zeros(len(numeric_b))
    # PPk[Pk.tolist()]=numeric_b[Pk.tolist()]
    # PPl[Pl.tolist()]=numeric_b[Pl.tolist()]
    b_prime = (1 - (3 * epsilon / 2)) * numeric_b + epsilon * Pk + Pl * epsilon / 2

    solve = InfeasibilityCertificate(InfMat, b_prime)
    yRaw = np.array(solve['x']).ravel()
    tol = solve['gap']
    y, Fcondition, t, v = BellFacet(yRaw, InfMat, tol, numeric_b)
    if Fcondition:
        FacetYs.append(y)
"""
"""
    InflatedGraph(rawgraph,inflation_order).print_assessment()
    Solution=InflationLP(rawgraph, rawdata, card, inflation_order,extra_ex,solver).Inequality()
end_time = time.time()
print('It took '+str(end_time-start_time)+' second to run.')
"""
"""
rawgraph = Graph.Formula("L->A:B,Ux->X,Uy->Y,X->A,Y->B")
A=list(product('0123',repeat=4))
B=[''.join(i) for i in A]
del B[0]
BellData=['0000','0010','0001','0020','0002','1011','0111','1100','1110','1101','1120','1102','0021','1121','0012','1112','0022','1122','0031','0131','0032','0132','0013','1013','0023','1023','0030','0130','0003','1003','0033','0133']
cardinality=[2,2,4,4]
original_card_product=np.prod(cardinality)
data = np.zeros(original_card_product)
data[list(map(lambda s: MixedCardinalityBaseConversion(cardinality, s),BellData))] = 1/len(BellData)
data[MixedCardinalityBaseConversion(cardinality, '0033')]=1/16
data[MixedCardinalityBaseConversion(cardinality, '0133')]=0


rawgraph=g
rawdata=data

card=[2,2,4,4]
inflation_order=[1,2,2]
extra_ex=True
solver='moseklp'

InflatedGraph(rawgraph,inflation_order).print_assessment()

Solution=InflationLP(rawgraph, rawdata, card, inflation_order,extra_ex,solver).Inequality()

"""
