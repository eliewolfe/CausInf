#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning all the relevant properties of the inflation graph.
"""
from __future__ import absolute_import
import numpy as np

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from inflation.dimino import dimino_wolfe
#from igraph import *
from inflation.quickgraph import LearnOriginalGraphParameters
from inflation.utilities import MoveToFront
from functools import reduce #for permutation composition


def GenerateCanonicalExpressibleSet(inflation_order, inflation_depths, offsets):
    # offsets=GenerateOffsets(inflation_order,inflation_depths)
    obs_count = len(inflation_depths)
    order_range = np.arange(inflation_order)
    cannonical_pos = np.empty((obs_count, inflation_order), dtype=np.int) #uint would give rise to type casting error
    for i in np.arange(obs_count):
        cannonical_pos[i] = np.sum(np.outer(inflation_order ** np.arange(inflation_depths[i]), order_range), axis=0) + \
                            offsets[i]
    return cannonical_pos.T.ravel()


def GenerateInflationGroupGenerators(inflation_order, latent_count, root_structure, inflation_depths, offsets):
    inflationcopies = inflation_order ** inflation_depths
    num_vars = inflationcopies.sum()
    # offsets=GenerateOffsets(inflation_order,inflation_depths)
    globalstrategyflat = list(np.add(*stuff) for stuff in zip(list(map(np.arange, inflationcopies.tolist())), offsets))
    obs_count = len(inflation_depths)
    reshapings = np.ones((obs_count, latent_count), np.uint8)
    contractings = np.zeros((obs_count, latent_count), np.object)
    for idx, elem in enumerate(root_structure):
        reshapings[idx][elem] = inflation_order
        contractings[idx][elem] = np.s_[:]
    reshapings = list(map(tuple, reshapings))
    contractings = list(map(tuple, contractings))
    globalstrategyshaped = list(np.reshape(*stuff) for stuff in zip(globalstrategyflat, reshapings))
    fullshape = tuple(np.full(latent_count, inflation_order))
    if inflation_order == 2:
        inflation_order_gen_count = 1
    else:
        inflation_order_gen_count = 2
    group_generators = np.empty((latent_count, inflation_order_gen_count, num_vars), np.int) #uint would give rise to type casting error
    for latent_to_explore in np.arange(latent_count):
        for gen_idx in np.arange(inflation_order_gen_count):
            initialtranspose = MoveToFront(latent_count, np.array([latent_to_explore]))
            inversetranspose = np.hstack((np.array([0]), 1 + np.argsort(initialtranspose)))
            label_permutation = np.arange(inflation_order)
            if gen_idx == 0:
                label_permutation[np.array([0, 1])] = np.array([1, 0])
            elif gen_idx == 1:
                label_permutation = np.roll(label_permutation, 1)
            global_permutation = np.array(list(
                np.broadcast_to(elem, fullshape).transpose(tuple(initialtranspose))[label_permutation] for elem in
                globalstrategyshaped))
            global_permutation = np.transpose(global_permutation, tuple(inversetranspose))
            global_permutation = np.hstack(
                tuple(global_permutation[i][contractings[i]].ravel() for i in np.arange(obs_count)))
            group_generators[latent_to_explore, gen_idx] = global_permutation
    return group_generators



def InflateDeterminismAssumptions(determinism_checks, latent_count, group_generators, exp_set):
    """
    Recall that a determinism check is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
    Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
    Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
    """
    def InflateOneDeterminismAssumption(screening):
        U1s = screening[0]
        XsY = np.array(list(screening[2])+list(screening[1])) - latent_count
        flatset_original_world = np.take(exp_set, XsY)
        symops = group_generators[U1s, 0]  # Now 2d array
        flatset_new_world = np.take(reduce(np.take, symops), flatset_original_world)
        rule = np.vstack((flatset_original_world, flatset_new_world)).T.astype('uint32')
        rule = rule[:-1, :].T.tolist() + rule[-1, :].T.tolist()
        return rule

    return list(map(InflateOneDeterminismAssumption, determinism_checks))

#TODO: Explore larger sets being screened off. What about Ys PLURAL being screened off from U1s? Isn't that worth looking into?
def InflateOtherExpressibleSets(extra_expressible_sets, latent_count, group_generators, exp_set):
    """
    New function to identify extra expressible sets.
    Recall that a screening relation is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
    Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
    Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
    """

    def InflateOneExpressibleSet(screening):
        U3s = screening[-1]
        (Ys, Xs, Zs) = tuple(map(lambda orig_node_indices: np.take(exp_set, np.array(orig_node_indices) - latent_count),
                                screening[:-1])) #Feb 2 2021: reindexed as U1s no longer passed to e_eset
        symops = group_generators[U3s, 0]  # Now 2d array
        Zs_new_world = np.take(reduce(np.take, symops), Zs)
        #The ordering of variables here must reflect the ordering used by Find_b
        #We can return it as a flat array.
        variable_ordering = np.argsort(np.hstack(screening[:-1]))
        #nonai_exp_set = (Ys.tolist(),Xs.tolist(),Zs_new_world.tolist()) #Changed ordering
        nonai_exp_set = np.hstack((Ys, Xs, Zs_new_world)).take(variable_ordering)  #Changed ordering
        return nonai_exp_set

    return list(map(InflateOneExpressibleSet, extra_expressible_sets))








def LearnInflationGraphParameters(g, inflation_order, extra_expressible=False, debug=False):
    names, parents_of, roots_of, determinism_checks, extra_expressible_sets = LearnOriginalGraphParameters(g)
    # print(names)
    graph_structure = list(filter(None, parents_of))
    obs_count = len(graph_structure)
    latent_count = len(parents_of) - obs_count
    root_structure = roots_of[latent_count:]
    inflation_depths = np.array(list(map(len, root_structure)))
    inflationcopies = inflation_order ** inflation_depths
    num_vars = inflationcopies.sum()
    accumulated = np.add.accumulate(inflation_order ** inflation_depths)
    offsets = np.hstack(([0], accumulated[:-1]))
    exp_set = GenerateCanonicalExpressibleSet(inflation_order, inflation_depths, offsets)
    group_generators = GenerateInflationGroupGenerators(inflation_order, latent_count, root_structure, inflation_depths,
                                                        offsets)
    group_elem = np.array(dimino_wolfe(group_generators.reshape((-1, num_vars))))
    det_assumptions = InflateDeterminismAssumptions(determinism_checks, latent_count, group_generators, exp_set)
    other_expressible_sets = InflateOtherExpressibleSets(extra_expressible_sets, latent_count, group_generators, exp_set)
    if debug:
        print("For the graph who's parental structure is given by:")
        print([':'.join(np.take(names, vals)) + '->' + np.take(names, idx) for idx, vals in enumerate(graph_structure)])
        print("We have "+str(num_vars)+" total inflation graph observable variables.")
        print("The diagonal expressible set is given by:")
        print(exp_set)
        print("And we count "+str(len(other_expressible_sets))+" other expressible sets, namely:")
        for nonai_exp_set in other_expressible_sets:
            print(nonai_exp_set)
        print('\u2500' * 80 + '\n')
    #TODO: Return exp_sets plural instead of only diagonal instance.
    if extra_expressible:
        return obs_count, num_vars, exp_set, group_elem, det_assumptions, names[latent_count:], other_expressible_sets
    else:
        return obs_count, num_vars, exp_set, group_elem, det_assumptions, names[latent_count:]



if __name__ == '__main__':
    from igraph import Graph
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    [LearnInflationGraphParameters(g,2, debug=False, extra_expressible=True) for g in (InstrumentalGraph,Evans14a,Evans14b,Evans14c,IceCreamGraph,BiconfoundingInstrumental, TriangleGraph)]

