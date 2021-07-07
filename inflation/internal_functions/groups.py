#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm to compute group elements from group generators. 
Input as permutations lists in numpy array format.
"""

import numpy as np
#import numba
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
#from numba.np.unsafe.ndarray import to_fixed_tuple


#@numba.njit
def is_vec_in_mat(vec, mat):
    assume = True
    for elem in mat:
        if np.array_equal(vec, elem):
            assume = False
            break
    return assume

def dimino_sympy(group_generators):

    gens=[Permutation(list(gen)) for gen in group_generators]
    group=PermutationGroup(gens)
    group_elements=list(group.generate_dimino(af=True))
    return group_elements


 #@numba.njit
# def dimino_wolfe(group_generators):
#     gens = group_generators
#     degree = np.max(gens) + 1
#     idn = np.arange(degree)
#     order = 0
#     element_list = numba.typed.List()
#     element_list.append(idn)
#     for i in np.arange(len(gens)):
#         # D elements of the subgroup G_i generated by gens[:i]
#         D = element_list[:]
#         N = [idn]
#         while N:
#             A = N
#             N = []
#             for a in A:
#                 for g in gens[:i + 1]:
#                     ag = a[g]
#                     if is_vec_in_mat(ag, element_list):
#                         for d in D:
#                             order += 1
#                             ap = d[ag]
#                             element_list.append(ap)
#                             N.append(ap)
#     #print(element_list)
#     return element_list



#@numba.njit
def indexed_tensor(dims):
    return np.arange(np.prod(np.array(dims))).reshape(dims)

#@numba.njit
def symmetrize_implicit_tensor(dims, group, skip=0):
    """
    Parameters
    ----------
    dims: a tuple of integers, specifying the shape of the implicit tensor
    
    group: a 2d numpy.ndarray, each row being a permutation list representation of a group element

    Returns
    -------
    A symmetrized version of the indexed tensor.
    """
    rank = len(dims)
    tensor = indexed_tensor(dims)
    #rank = tensor.ndim
    #tensor = tensor.copy()
    for index_permutation in group[skip:]:
        tensor = np.minimum(
            tensor,
            tensor.transpose(to_fixed_tuple(index_permutation,rank)))
    return tensor



def symmetrize_tensor(tensor, group, skip=0):
    """
    Parameters
    ----------
    tensor: a numpy.ndarray, with the number of dimension matching the support of the group

    group: a 2d numpy.ndarray, each row being a permutation list representation of a group element

    Returns
    -------
    Null, the tensor is symmetrized IN PLACE.
    """
    for index_permutation in group[skip:]:
        np.minimum(
            tensor,
            tensor.transpose(index_permutation),
            out = tensor)

def minimize_object_under_group_action(object, group, action = lambda M, g: M.transpose(g), skip=0):
    """
    Parameters
    ----------
    object: the object to be minimized under the group action. Must be a numpy array.

    group: a list of group elements, each row being a permutation list representation of a group element

    action: a 2-argument function which describes how the object is transformed under the group action

    Returns
    -------
    Null, the object is minimized under the group action IN PLACE.
    """
    for group_element in group[skip:]:
        np.minimum(
            object,
            action(object,group_element),
            out = object)





# #@numba.njit
# def orbits_of_implicit_tensor_super_slow(dims, group):
#     groupT = np.transpose(group)
#     searched_already = np.full(dims, False)
#     rank = len(dims)
#     discovered_orbits = numba.typed.List()
#     for i, s in np.ndenumerate(searched_already):
#         if not s:
#             orbit = np.take(i,groupT)
#             searched_already[tuple(orbit)] = True
#             discovered_orbits.append(orbit)
#     return np.ravel_multi_index(np.transpose(discovered_orbits,(1,0,2)),dims)

def orbits_of_implicit_tensor(dims, group):
    group_order = len(group)
    tensor = indexed_tensor(dims)
    results = np.empty((group_order, tensor.size), np.int)
    for i, index_permutation in enumerate(group):
        results[i] = np.transpose(tensor, index_permutation).flat
    mask = np.amin(results, axis=0) == results[0]
    return results.compress(mask, axis = 1).T

def orbits_of_object_under_group_action(object, group, action = lambda M, g: M.transpose(g)):
    """
    Parameters
    ----------
    object: the object to collect into orbits (with multiplicity) under the group action. MUST be a numpy array.

    group: a list of group elements, each row being a permutation list representation of a group element

    action: a 2-argument function which describes how the object is transformed under the group action

    Returns
    -------
    A list of orbits as a 2d numpy array. Each array is t
    """
    group_order = len(group)
    results = np.empty_like(object, shape = (group_order, object.size))
    for i, group_element in enumerate(group):
        results[i] = action(object, group_element).flat
    mask = np.amin(results, axis=0) == results[0]
    return results.compress(mask, axis = 1).T






if __name__ == '__main__':
    dims = (4,4,4,4,4,4,4,4,4,4,4,4)
    test_tensor = indexed_tensor(dims)
    group_generators = np.array([[ 2,  3,  0,  1,  4,  5,  6,  7, 10, 11,  8,  9], [ 1,  0,  3,  2,  6,  7,  4,  5,  8,  9, 10, 11], [ 0,  1,  2,  3,  5,  4,  7,  6,  9,  8, 11, 10]])
    group_elements=dimino_wolfe(group_generators)
    import timeit
    
    #print(group_generators)
    print(orbits_of_object_under_group_action(
        indexed_tensor(dims),
        group_elements,
        lambda M,g : M.transpose(g)
    ).shape)

    dims = (3,3,3,3)
    test_tensor = indexed_tensor(dims)
    group_generators = np.array([[1,2,3,0]])
    group_elements = dimino_wolfe(group_generators)
    print(orbits_of_object_under_group_action(
        indexed_tensor(dims),
        group_elements))
