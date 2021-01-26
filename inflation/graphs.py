#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning all the relevant properties of the inflation graph.
"""

import numpy as np
from .dimino import dimino_wolfe
#from igraph import *
from .quickgraph import LearnParametersFromGraph
from .utilities import MoveToFront


def GenerateCanonicalExpressibleSet(inflation_order, inflation_depths, offsets):
    # offsets=GenerateOffsets(inflation_order,inflation_depths)
    obs_count = len(inflation_depths)
    order_range = np.arange(inflation_order)
    cannonical_pos = np.empty((obs_count, inflation_order), dtype=np.uint32)
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
    group_generators = np.empty((latent_count, inflation_order_gen_count, num_vars), np.uint)
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
            # global_permutationOLD=Deduplicate(np.ravel(global_permutation))   #Deduplication has been replaced with intelligent extraction.
            # print(np.all(global_permutation==global_permutationOLD))
            group_generators[latent_to_explore, gen_idx] = global_permutation
    return group_generators


def GenerateDeterminismAssumptions(determinism_checks, latent_count, group_generators, exp_set):
    one_generator_per_root = group_generators[:, 0]
    det_assumptions = list();
    for pair in determinism_checks:
        flatset = exp_set[list(np.array(pair[1]) - latent_count)] #TODO: change to np.take
        symop = one_generator_per_root[pair[0]]
        rule = np.vstack((flatset, symop[flatset])).T.astype('uint32')
        rule = rule[:-1, :].T.tolist() + rule[-1, :].T.tolist()
        det_assumptions.append(rule)
    return det_assumptions

# We should add a new function to give expressible sets. Ideally with symbolic output.




def LearnInflationGraphParameters(g, inflation_order):
    names, parents_of, roots_of, determinism_checks, filtered_determinism_checks = LearnParametersFromGraph(g)
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
    det_assumptions = GenerateDeterminismAssumptions(determinism_checks, latent_count, group_generators, exp_set)
    return obs_count, num_vars, exp_set, group_elem, det_assumptions, names[latent_count:]

