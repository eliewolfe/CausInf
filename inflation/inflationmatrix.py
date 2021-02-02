#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:16:04 2020

@author: boraulu
"""
from __future__ import absolute_import
import numpy as np
from scipy.sparse import coo_matrix
from functools import lru_cache
from itertools import permutations

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from inflation.graphs import LearnInflationGraphParameters
from inflation.strategies import ValidColumnOrbits
from inflation.utilities import PositionIndex, MoveToBack, GenShapedColumnIntegers


@lru_cache(maxsize=16)
def GenerateEncodingMonomialToRow(original_cardinality_product,
                                  inflation_order):  # I should make this recursive, as called by both A and b construction.
    monomial_count = int(original_cardinality_product ** inflation_order)
    permutation_count = int(np.math.factorial(inflation_order))
    MonomialIntegers = np.arange(0, monomial_count, 1, np.uint)
    new_shape = np.full(inflation_order, original_cardinality_product)
    MonomialIntegersPermutations = np.empty([permutation_count, monomial_count], np.uint)
    IndexPermutations = list(permutations(np.arange(inflation_order)))
    MonomialIntegersPermutations[0] = MonomialIntegers
    MonomialIntegers = MonomialIntegers.reshape(new_shape)
    for i in np.arange(1, permutation_count):
        MonomialIntegersPermutations[i] = np.transpose(MonomialIntegers, IndexPermutations[i]).flat
    return PositionIndex(np.amin(
        MonomialIntegersPermutations, axis=0))


def GenerateEncodingColumnToMonomial(card, num_var, expr_set):
    #Can be used for off-diagonal expressible sets with no adjustment!
    initialshape = np.full(num_var, card, np.uint)
    ColumnIntegers = GenShapedColumnIntegers(tuple(initialshape))
    ColumnIntegers = ColumnIntegers.transpose(MoveToBack(num_var, np.array(expr_set))).reshape(
        (-1, card ** len(expr_set)))
    EncodingColumnToMonomial = np.empty(card ** num_var, np.uint32)
    EncodingColumnToMonomial[ColumnIntegers] = np.arange(card ** len(expr_set))
    return EncodingColumnToMonomial


def EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    original_product_cardinality = card ** obs_count
    EncodingMonomialToRow = GenerateEncodingMonomialToRow(original_product_cardinality, inflation_order)
    EncodingColumnToMonomial = GenerateEncodingColumnToMonomial(card, num_vars, np.array(expr_set))
    result = EncodingMonomialToRow.take(EncodingColumnToMonomial).take(valid_column_orbits)
    # Once the encoding is done, the order of the columns can be tweaked at will!
    result.sort(axis=0)  # in-place sort
    return result

def EncodeA_ExtraExpressible(obs_count, num_vars, valid_column_orbits, expr_set, other_expressible_sets, inflation_order, card):
    ######WORK IN PROGRESS, READY FOR BETA TESTING######
    original_product_cardinality = card ** obs_count
    row_blocks_count=len(other_expressible_sets)+1
    results = np.empty(np.hstack((row_blocks_count,valid_column_orbits.shape)), np.uint32)
    EncodingMonomialToRow = GenerateEncodingMonomialToRow(original_product_cardinality, inflation_order)
    EncodingColumnToMonomial = GenerateEncodingColumnToMonomial(card, num_vars, np.array(expr_set))
    results[0] = EncodingMonomialToRow.take(EncodingColumnToMonomial).take(valid_column_orbits)
    for i in np.arange(1,row_blocks_count):
        extra_expr_set_flattened = np.hstack(other_expressible_sets[i-1])
        #It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
        #In order for names to make sense, I am electing to pass a SORTED version of the flat set.
        extra_expr_set_flattened.sort()
        results[i] = GenerateEncodingColumnToMonomial(card, num_vars, extra_expr_set_flattened)
    accumulated = np.add.accumulate(np.amax(results, axis=0)+1)
    offsets = np.hstack(([0], accumulated[:-1]))
    # Once the encoding is done, the order of the columns can be tweaked at will!
    #result.sort(axis=0)  # in-place sort
    return np.hstack(results+offsets)


def SciPyArrayFromOnesPositions(OnesPositions, sort_columns=True):
    columncount = OnesPositions.shape[-1]
    if sort_columns:
        ar_to_broadcast = np.lexsort(OnesPositions)
    else:
        ar_to_broadcast = np.arange(columncount)
    columnspec = np.broadcast_to(ar_to_broadcast, (len(OnesPositions), columncount)).ravel()
    return coo_matrix((np.ones(OnesPositions.size, np.uint), (OnesPositions.ravel(), columnspec)),
                      (int(np.amax(OnesPositions) + 1), columncount), dtype=np.uint)


# def SciPyArrayFromOnesPositionsWithSort(OnesPositions):
#    columncount=OnesPositions.shape[-1]
#    columnspec=np.broadcast_to(np.lexsort(OnesPositions), (len(OnesPositions), columncount)).ravel()
#    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

def SparseInflationMatrix(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    return SciPyArrayFromOnesPositions(
        EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))


def InflationMatrixFromGraph(g, inflation_order, card, extra_expressible=False):
    #Needs documentation!
    learned_parameters = LearnInflationGraphParameters(g, inflation_order, extra_expressible=True)
    (obs_count, num_vars, expr_set, group_elem, det_assumptions, names) = learned_parameters[:-1]
    print(names)  # REMOVE THIS PRINTOUT after accepting fixed order of variables.
    valid_column_orbits = ValidColumnOrbits(card, num_vars, group_elem, det_assumptions)
    if extra_expressible:
        other_expressible_sets = learned_parameters[-1]
        return SciPyArrayFromOnesPositions(EncodeA_ExtraExpressible(obs_count,
                                                                    num_vars,
                                                                    valid_column_orbits,
                                                                    expr_set,
                                                                    other_expressible_sets,
                                                                    inflation_order,
                                                                    card))
    else:
        return SciPyArrayFromOnesPositions(EncodeA(obs_count,
                                                   num_vars,
                                                   valid_column_orbits,
                                                   expr_set,
                                                   inflation_order,
                                                   card))


#NEW FUNCTION (Feb 2 2021)
def Numeric_and_Symbolic_b_block_DIAGONAL(data, inflation_order, obs_count, card):
    EncodingMonomialToRow = GenerateEncodingMonomialToRow(len(data), inflation_order)
    s, idx, counts = np.unique(EncodingMonomialToRow, return_index=True, return_counts=True)
    pre_numeric_b = np.array(data)
    numeric_b = pre_numeric_b.copy()
    pre_symbolic_b = np.array(['P(' + ''.join([''.join(str(i)) for i in idx]) + ')' for idx in
                   np.ndindex(tuple(np.full(obs_count, card, np.uint8)))])
    symbolic_b = pre_symbolic_b.copy()
    for i in range(1, inflation_order):
        numeric_b = np.kron(pre_numeric_b, numeric_b)
        symbolic_b = np.add.outer(pre_symbolic_b, symbolic_b).ravel()
    numeric_b_block = np.multiply(numeric_b.take(idx), counts)
    string_multipliers = np.fromiter(('' if i == 1 else str(i)+'*' for i in counts),np.str_)
    symbolic_b_block = np.add(string_multipliers, symbolic_b.take(idx))
    return numeric_b_block, symbolic_b_block

#NEW FUNCTION (Feb 2 2021)
def Numeric_and_Symbolic_b_block_NON_AI_EXPR(data, other_expressible_set_original, obs_count, card, names):
    first_indices = np.arange(obs_count)
    Y = list(other_expressible_set_original[0])
    X = list(other_expressible_set_original[1])
    Z = list(other_expressible_set_original[2])
    # It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
    # In order for names to make sense, I am electing to pass a SORTED version of the flat set.
    YXZ = sorted(Y + X + Z)
    lenY = len(Y)
    lenX = len(X)
    lenZ = len(Z)
    lenYXZ = len(YXZ)

    numeric_b_block = np.einsum(np.einsum(data, first_indices, sorted(X + Y)), sorted(X + Y),
                                np.einsum(data, first_indices, sorted(X + Z)), sorted(X + Z),
                                1 / np.einsum(data, first_indices, sorted(X)), sorted(X),
                                YXZ).ravel()

    lowY = np.arange(lenY).tolist()
    lowX = np.arange(lenY, lenY + lenX).tolist()
    lowZ = np.arange(lenY + lenX, lenY + lenX + lenZ).tolist()
    newshape = tuple(np.full(lenYXZ, card, np.uint8))
    symbolic_b_block = [
        'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowY))).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowZ))]) + ')' + \
        'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowZ))).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowZ))]) + ')' +
        '/P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
        for idYXZ in np.ndindex(newshape)]

    return numeric_b_block, symbolic_b_block


def Generate_b_and_counts(Data, inflation_order):
    """
    Parameters
    ----------
    Data : array_like
        The probability distribution for the original scenario's observable variables.
    inflation_order : int
        The order of the inflation matrix.

    Returns
    -------
    b : vector_of_integers
        A numerical vector computered from `Data` to be evaluated with linear programming.
    counts : vector_of_integers
        For each probability in `b`, an integer counting how many distinct monomials were summed to obtain that probability.


    Notes
    -----
    The distribution is only compatible with the inflations test if there exists some positive x
    such that

    .. math:: A \dot x = b

    For :math:`x \geq 0`.


    Examples
    --------
    Some example code to illustrate function usage.

    >>> Data = None
    >>> Inflation_order = None
    >>> Generate_b_and_counts(Data, inflation_order)
    ***TO FILL IN***
    """
    EncodingMonomialToRow = GenerateEncodingMonomialToRow(len(Data), inflation_order)
    s, idx, counts = np.unique(EncodingMonomialToRow, return_index=True, return_counts=True)
    preb = np.array(Data)
    b = preb
    for i in range(1, inflation_order):
        b = np.kron(preb, b)
    return b[idx], counts


def FindB(Data, inflation_order):
    return np.multiply(*Generate_b_and_counts(Data, inflation_order))

# def MarginalOn(data, card, obs_count, marginal):
#     initial_shape = np.full(obs_count, card, np.uint)
#     reshaped_data = np.reshape(data,tuple(initial_shape))
#     return reshaped_data.transpose(MoveToBack(obs_count, np.array(marginal))).reshape(
#            (-1, card ** len(marginal))).sum(axis=0)

def MarginalVectorFromGraph(g, data, inflation_order, card, extra_expressible=False):
    ###WORK IN PROGRESS; still only returns b-vector associated with the diagonal expressible set.###
    if not extra_expressible:
        return FindB(data, inflation_order)
    else:
        b_diagonal_exp_set = FindB(data, inflation_order)
        names, parents_of, roots_of, screening_off_relationships = LearnParametersFromGraph(g, hasty=False)
        obs_count = len(list(filter(None, parents_of)))
        latent_count = len(parents_of) - obs_count
        obs_names = names[latent_count:]
        other_expressible_sets_original = [tuple(map(lambda orig_node_indices: np.array(orig_node_indices) - latent_count,
                                screening[1:4])) for screening in filter(lambda screening: len(screening[-1]) > 0, screening_off_relations)]
        #How should we compute the marginal probability?
        #Given P(ABC) how do we obtain P(AB)P(BC)/P(B) as a vector of appropriate length?
        for eset in other_expressible_sets_original:
            print(tuple(np.take(obs_names, indices).tolist() for indices in eset))
        return b_diagonal_exp_set

#def ReshapedSymbolicProbabilities(names, card, indices):
#    import sympy as sy
#    newshape = tuple(np.full(len(indices), card, np.uint8))
#    return np.reshape(sy.symbols(['P['+''.join(np.take(names, indices).tolist())+'](' + ''.join([''.join(str(i)) for i in idx]) + ')' for idx in np.ndindex(newshape)]),newshape)

if __name__ == '__main__':
    # In our test example let's compute index [2] separated from indices [4] by indices [1,3]
    normalize = lambda v: v / np.linalg.norm(v, ord=1)
    data = normalize(np.arange(1, 2 ** 5 + 1)).reshape(np.full(5, 2, np.uint))
    obs_count = 5
    card = 2
    names = ['A','B','C','D','E']
    other_expressible_set_original = [{2},{1,3},{4}]
    numeric_b_block, symbolic_b_block = Numeric_and_Symbolic_b_block_NON_AI_EXPR(data, other_expressible_set_original, obs_count, card, names)
    # first_indices = np.arange(obs_count)
    # #second_indices = first_indices + obs_count
    # #third_indices = second_indices + obs_count
    # Y = [2]
    # X = [1,3]
    # Z = [4]
    # # It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
    # # In order for names to make sense, I am electing to pass a SORTED version of the flat set.
    # YXZ = sorted(Y + X + Z)
    # lenY = len(Y)
    # lenX = len(X)
    # lenZ = len(Z)
    # lenYXZ = len(YXZ)
    # #np.put(second_indices, X, X)
    # #np.put(second_indices, Z, Z)
    # #np.put(third_indices, X, X)
    # #after_summation = np.einsum(data,first_indices,
    # #                            data,second_indices,
    # #                            data,third_indices,
    # #                            YXZ)
    #
    # numeric_b_block = np.einsum(np.einsum(data, first_indices, sorted(X + Y)), sorted(X + Y),
    #                             np.einsum(data, first_indices, sorted(X + Z)), sorted(X + Z),
    #                             1/np.einsum(data, first_indices, sorted(X)), sorted(X),
    #                             YXZ).ravel()
    #
    # lowY = np.arange(lenY).tolist()
    # lowX = np.arange(lenY,lenY + lenX).tolist()
    # lowZ = np.arange(lenY + lenX, lenY + lenX + lenZ).tolist()
    # newshape = tuple(np.full(lenYXZ, card, np.uint8))
    # symbolic_b_block = [
    #  'P[' + ''.join(np.take(names, np.take(YXZ,sorted(lowX + lowY))).tolist()) + '](' +
    #  ''.join([''.join(str(i)) for i in np.take(idYXZ,sorted(lowX + lowZ))]) + ')' + \
    #  'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowZ))).tolist()) + '](' +
    #  ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowZ))]) + ')' +
    #  '/P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
    #  ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
    #  for idYXZ in np.ndindex(newshape)]

    #SymbolicXY = ReshapedSymbolicProbabilities(names, card, sorted(X + Y))

    #NEW PLAN: Just manually create symbols using ndindex and rescaling everything

    #after_summation_symbolic = np.einsum(ReshapedSymbolicProbabilities(names, card, sorted(X + Y)), sorted(X + Y),
    #                                     ReshapedSymbolicProbabilities(names, card, sorted(X + Z)), sorted(X + Z),
    #                                     1/ReshapedSymbolicProbabilities(names, card, sorted(X)), sorted(X),
    #                                     Y + X + Z).ravel()
