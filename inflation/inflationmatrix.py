#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:16:04 2020
@author: Bora Ulu & Elie Wolfe
"""
from __future__ import absolute_import
import numpy as np
#from scipy.sparse import coo_matrix
from functools import lru_cache
from itertools import permutations

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from inflation.graphs import LearnInflationGraphParameters
from inflation.quickgraph import LearnOriginalGraphParameters
from inflation.strategies import ValidColumnOrbits
from inflation.utilities import PositionIndex, MoveToBack, GenShapedColumnIntegers, SparseMatrixFromRowsPerColumn


@lru_cache(maxsize=16)
def GenerateEncodingMonomialToRow(original_cardinality_product,
                                  inflation_order):  # Cached in memory, as this function is called by both inflation matrix and inflation vector construction.    
    """
    Parameters
    ----------
    original_cardinality_product : int
        The number of permutations of observable variable configurations given by :math:`(\mbox{cardinality})^{(\mbox{number of observable variables})}`
         
    inflation_order : int
        The order of the inflation matrix.

    Returns
    -------
    EncodingMonomialToRow : vector_of_integers
        A numerical vector where each element represents a row and the value of each element represents the index of that row under the symmetry conditions exerted by the cannonical expressible set of the inflated graph.
    
    Notes
    -----
    
    For each row there is a set of symmetric rows produced by the interchange of the copy indecies of the observable variables that create the expressible set. While computing the value of each element of EncodingMonomialToRow, the function chooses the row inside the symmetric set with the smallest index.
    
    Examples
    --------
    For a graph of 3 observable variables, each with cardinality 4:
    
    >>> obs_count=3    
    >>> card=4    
    >>> original_cardinality_product=card**obs_count
    
    With an inflation order of 2:
        
    >>> inflation_order=2    
    >>> EncodingMonomialToRow=GenerateEncodingMonomialToRow(original_cardinality_product,inflation_order)
    
    There will be :math:`4^{3}=4096` rows
    
    >>> print(len(EncodingMonomialToRow))
    >>> 4096
    
    The intechange of the copy indecies of the variables that make up the 6 element cannonical expressible set will produce a 2-fold symmetry and therefore the largest row index inside EncodingMonomialToRow will be :math:`(4096/2)-1=2079`.
    
    >>> print(EncodingMonomialToRow)
    >>> [   0    1    2 ... 2076 2078 2079]
    """
    
    monomial_count = int(original_cardinality_product**inflation_order)    
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

def EncodeA_ExtraExpressible(obs_count, num_vars, valid_column_orbits, expr_set, inflated_expressible_sets, inflation_order, card):
    ######WORK IN PROGRESS, READY FOR BETA TESTING######
    original_product_cardinality = card ** obs_count
    row_blocks_count=len(inflated_expressible_sets)+1
    results = np.empty(np.hstack((row_blocks_count,valid_column_orbits.shape)), np.uint32)
    EncodingMonomialToRow = GenerateEncodingMonomialToRow(original_product_cardinality, inflation_order)
    EncodingColumnToMonomial = GenerateEncodingColumnToMonomial(card, num_vars, np.array(expr_set))
    results[0] = EncodingMonomialToRow.take(EncodingColumnToMonomial).take(valid_column_orbits)
    for i in np.arange(1,row_blocks_count):
        #It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
        #In order for names to make sense, I am electing to pass a SORTED version of the flat set, see InflateOneExpressibleSet
        results[i] = GenerateEncodingColumnToMonomial(card, num_vars, inflated_expressible_sets[i-1]).take(valid_column_orbits)
    accumulated = np.add.accumulate(np.amax(results, axis=(1,2))+1)
    offsets = np.hstack(([0], accumulated[:-1]))
    # Once the encoding is done, the order of the columns can be tweaked at will!
    #result.sort(axis=0)  # in-place sort
    return np.hstack(results+offsets[:, np.newaxis, np.newaxis])

# def SciPyArrayFromOnesPositions(OnesPositions, sort_columns=True):
#     columncount = OnesPositions.shape[-1]
#     if sort_columns:
#         ar_to_broadcast = np.lexsort(OnesPositions)
#     else:
#         ar_to_broadcast = np.arange(columncount)
#     columnspec = np.broadcast_to(ar_to_broadcast, (len(OnesPositions), columncount)).ravel()
#     return coo_matrix((np.ones(OnesPositions.size, np.uint), (OnesPositions.ravel(), columnspec)),
#                       (int(np.amax(OnesPositions) + 1), columncount), dtype=np.uint)


# def SciPyArrayFromOnesPositionsWithSort(OnesPositions):
#    columncount=OnesPositions.shape[-1]
#    columnspec=np.broadcast_to(np.lexsort(OnesPositions), (len(OnesPositions), columncount)).ravel()
#    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

#def SparseInflationMatrix(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
#    return SciPyArrayFromOnesPositions(
#        EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))


def InflationMatrixFromGraph(g, inflation_order, card, extra_expressible=False):
    """
    Parameters
    ----------
    g : igraph.Graph
        The causal graph containing all of the observable and latent variables.
         
    inflation_order : int
        The order of the inflation matrix.

    card : int
        The cardinality of every observable variable.
        
    extra_expressible : bool,optional
        If True the rows representing the non-cannonical expressible sets are included in the marginal description matrix (default set to False)
    
    Returns
    -------
    InflationMatrix : scipy.sparse.coo.coo_matrix
        The marginal description matrix in the form of a sparse matrix in COOrdinate format.
    
    Notes
    -----
    
    The columns of the marginal description matrix correspond to different strategies (permutations of the inflated observable variable values) such as:
    
    .. math:: P(A_{1},A_{2},...,A_{N},B_{1},...,B_{N},C_{1},...,C_{N})=P(1,0,1,1,...,0) 
    
    For the triangle scenario with cardinality 2 where :math:`N` is the inflation order. The rows of this matrix correspond to the marginal distributions containing the inflated observable variables in the expressible sets such as:
    
    .. math:: P(A_{1},B_{1},C_{1},A_{4},B_{4},C_{4})=P(0,1,1,0,1,1)
    
    Which is a cannonical expressible set of the same scenario where :math:`N=2`. It can be used in a linear program for an infeasibility certificate (see ``inflation.moseklp.InfeasibilityCertificate``) and/or for a set of inequalities that must be satisfied for the compatibility of the distribution (see ``inflation.certificate.Inequality``).
    
    Examples
    --------
    For the triangle scenario with cardinality 4 and an inflation order of 2:
    
    >>> g=igraph.Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    >>> card=4
    >>> inflation_order=2
    
    We would expect to obtain a marginal description matrix with 2123776 columns (reduced from :math:`4^{12}=16777216` by imposing the symmetry conditions of the copy indecies on different strategies) and 2080 rows (reduced from :math:`4^{6}=4096` by imposing the same symmetry conditions on the expressible sets):
        
    >>> InfMat = InflationMatrixFromGraph(g, inflation_order, card)
    >>> print(InfMat.shape)
    >>> (2080, 2123776)
    
    """
    
    learned_parameters = LearnInflationGraphParameters(g, inflation_order, extra_expressible=True)
    (obs_count, num_vars, expr_set, group_elem, det_assumptions, names) = learned_parameters[:-1]
    #print(names)  # REMOVE THIS PRINTOUT after accepting fixed order of variables.
    valid_column_orbits = ValidColumnOrbits(card, num_vars, group_elem, det_assumptions)
    if extra_expressible:
        other_inflated_expressible_sets = learned_parameters[-1]
        return SparseMatrixFromRowsPerColumn(EncodeA_ExtraExpressible(obs_count,
                                                                    num_vars,
                                                                    valid_column_orbits,
                                                                    expr_set,
                                                                    other_inflated_expressible_sets,
                                                                    inflation_order,
                                                                    card))
    else:
        return SparseMatrixFromRowsPerColumn(EncodeA(obs_count,
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
        symbolic_b = [s1+s2 for s1 in pre_symbolic_b for s2 in symbolic_b]
    numeric_b_block = np.multiply(numeric_b.take(idx), counts)
    string_multipliers = ('' if i == 1 else str(i)+'*' for i in counts)
    symbolic_b_block = [s1+s2 for s1,s2 in zip(string_multipliers, np.take(symbolic_b,idx))]
    return numeric_b_block, symbolic_b_block

#NEW FUNCTION (Feb 2 2021)
def Numeric_and_Symbolic_b_block_NON_AI_EXPR(data, other_expressible_set_original, obs_count, card, all_names):
    latent_count = len(all_names) - obs_count
    names = all_names[latent_count:]
    all_original_indices = np.arange(obs_count)
    Y = list(other_expressible_set_original[0])
    X = list(other_expressible_set_original[1])
    Z = list(other_expressible_set_original[2])
    # It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
    # In order for names to make sense, I am electing to pass a SORTED version of the flat set.
    YXZ = sorted(Y + X + Z)  #see InflateOneExpressibleSet in graphs.py
    lenY = len(Y)
    lenX = len(X)
    lenZ = len(Z)
    lenYXZ = len(YXZ)

    data_reshaped = np.reshape(data,tuple(np.full(obs_count, card, np.uint8)))

    np.seterr(divide='ignore')
    marginal_on_XY = np.einsum(data_reshaped, all_original_indices, X + Y)
    marginal_on_XZ = np.einsum(data_reshaped, all_original_indices, X + Z)
    marginal_on_X = np.einsum(marginal_on_XY, X + Y, X)
    # Y_conditional_on_X = np.einsum(marginal_on_XY, X + Y,
    #                             1 / marginal_on_X, X,
    #                             Y + X)
    # numeric_b_block = np.einsum(marginal_on_XZ, X + Z,
    #                             Y_conditional_on_X, Y + X,
    #                             YXZ).ravel()
    numeric_b_block = np.einsum(marginal_on_XY, X + Y,
                                marginal_on_XZ, X + Z,
                                np.divide(1.0, marginal_on_X), X,
                                YXZ).ravel()
    numeric_b_block[np.isnan(numeric_b_block)] = 0 #Conditioning on zero probability events
    np.seterr(divide='warn')

    lowY = np.arange(lenY).tolist()
    lowX = np.arange(lenY, lenY + lenX).tolist()
    lowZ = np.arange(lenY + lenX, lenY + lenX + lenZ).tolist()
    newshape = tuple(np.full(lenYXZ, card, np.uint8))
    # symbolic_b_block = [
    #     'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowY))).tolist()) + '](' +
    #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowY))]) + ')' +
    #     'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowZ))).tolist()) + '](' +
    #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowZ))]) + ')' +
    #     '/P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
    #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
    #     for idYXZ in np.ndindex(newshape)]
    symbolic_b_block = [
        'P[' + ''.join(np.take(names, np.take(YXZ, lowY)).tolist()) + '|' +
        ''.join(np.take(names, np.take(YXZ, lowX)).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, lowY)]) + '|' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, lowX)]) + ')' +
        'P[' + ''.join(np.take(names, np.take(YXZ, lowZ)).tolist()) + '|' +
        ''.join(np.take(names, np.take(YXZ, lowX)).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, lowZ)]) + '|' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, lowX)]) + ')' +
        'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
        ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
        for idYXZ in np.ndindex(newshape)]

    return numeric_b_block, symbolic_b_block

def NumericalAndSymbolicVectorsFromGraph(g, data, inflation_order, card, extra_expressible=False):
    ###WORK IN PROGRESS; still only returns b-vector associated with the diagonal expressible set.###
    if not extra_expressible:
        names, parents_of, roots_of = LearnOriginalGraphParameters(g, hasty=True)
        obs_count = len(list(filter(None, parents_of)))
        return Numeric_and_Symbolic_b_block_DIAGONAL(data, inflation_order, obs_count, card)
    else:
        names, parents_of, roots_of, determinism_checks, extra_expressible_sets = LearnOriginalGraphParameters(g, hasty=False)
        obs_count = len(list(filter(None, parents_of)))
        latent_count = len(parents_of) - obs_count
        obs_names = names[latent_count:]
        numeric_b, symbolic_b = Numeric_and_Symbolic_b_block_DIAGONAL(data, inflation_order, obs_count, card)
        other_expressible_sets_original = [
            tuple(map(lambda orig_node_indices: np.array(orig_node_indices) - latent_count,
                      e_set[:-1])) for e_set in extra_expressible_sets]
        #How should we compute the marginal probability?
        #Given P(ABC) how do we obtain P(AB)P(BC)/P(B) as a vector of appropriate length?
        for eset in other_expressible_sets_original:
            #print(tuple(np.take(obs_names, indices).tolist() for indices in eset))
            numeric_b_block, symbolic_b_block = Numeric_and_Symbolic_b_block_NON_AI_EXPR(data, eset, obs_count, card, names)
            numeric_b.resize(len(numeric_b) + len(numeric_b_block))
            numeric_b[-len(numeric_b_block):] = numeric_b_block
            symbolic_b.extend(symbolic_b_block)
        return numeric_b, symbolic_b


def Generate_b_and_counts(Data, inflation_order):
    #TO BE DEPRECATED
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
    # TO BE DEPRECATED
    return np.multiply(*Generate_b_and_counts(Data, inflation_order))




#def ReshapedSymbolicProbabilities(names, card, indices):
#    import sympy as sy
#    newshape = tuple(np.full(len(indices), card, np.uint8))
#    return np.reshape(sy.symbols(['P['+''.join(np.take(names, indices).tolist())+'](' + ''.join([''.join(str(i)) for i in idx]) + ')' for idx in np.ndindex(newshape)]),newshape)

if __name__ == '__main__':
    # In our test example let's compute index [2] separated from indices [4] by indices [1,3]
    normalize = lambda v: v / np.linalg.norm(v, ord=1)
    data = normalize(np.arange(1, 2 ** 5 + 1)).reshape(np.full(5, 2, np.uint)).ravel().tolist()
    obs_count = 5
    card = 2
    names = ['A','B','C','D','E']
    other_expressible_set_original = [{2},{1,3},{4}]
    numeric_b_block, symbolic_b_block = Numeric_and_Symbolic_b_block_NON_AI_EXPR(data, other_expressible_set_original, obs_count, card, names)

    from igraph import Graph


    def ListOfBitStringsToListOfIntegers(list_of_bitstrings):
        return list(map(lambda s: int(s, 2), list_of_bitstrings))


    def UniformDistributionFromSupport(list_of_bitstrings):
        numvar = max(map(len, list_of_bitstrings))
        numevents = len(list_of_bitstrings)
        data = np.zeros(2 ** numvar)
        data[ListOfBitStringsToListOfIntegers(list_of_bitstrings)] = 1 / numevents
        return data


    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")

    inflation_order = 2
    card = 4


    TriData = UniformDistributionFromSupport(['000', '111'])
    InstrumentalData = UniformDistributionFromSupport(['000', '101'])
    BiconfoundingInstrumentalData = UniformDistributionFromSupport(['0000', '0100', '1011', '1111'])


    g = TriangleGraph
    data = TriData

    #InfMat = InflationMatrixFromGraph(g, inflation_order, card)
    #b = FindB(data, inflation_order)

    InfMat = InflationMatrixFromGraph(g, inflation_order, card, extra_expressible=True)
    B_numeric, B_symbolic = NumericalAndSymbolicVectorsFromGraph(g, data, inflation_order, card, extra_expressible=True)


