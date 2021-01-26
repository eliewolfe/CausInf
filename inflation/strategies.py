#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finding all valid deterministic outcome assignments of the inflation graph, collected as orbits under the inflation symmetry.
"""

import numpy as np
from .graphs import LearnInflationGraphParameters
from .utilities import MoveToFront, GenShapedColumnIntegers


def MarkInvalidStrategies(card, num_var, det_assumptions):
    initialshape = np.full(num_var, card, np.uint8)
    ColumnIntegers = GenShapedColumnIntegers(tuple(initialshape))
    for detrule in det_assumptions:
        initialtranspose = MoveToFront(num_var, np.hstack(tuple(detrule)))
        inversetranspose = np.argsort(initialtranspose)
        parentsdimension = card ** len(detrule[1])
        intermediateshape = (parentsdimension, parentsdimension, card, card, -1);
        ColumnIntegers = ColumnIntegers.transpose(tuple(initialtranspose)).reshape(intermediateshape)
        for i in np.arange(parentsdimension):
            for j in np.arange(card - 1):
                for k in np.arange(j + 1, card):
                    ColumnIntegers[i, i, j, k] = -1
        ColumnIntegers = ColumnIntegers.reshape(initialshape).transpose(tuple(inversetranspose))
    return ColumnIntegers


def ValidColumnOrbits(card, num_vars, group_elem, det_assumptions=[]):
    ColumnIntegers = MarkInvalidStrategies(card, num_vars, det_assumptions)
    group_elements = group_elem  # GroupElementsFromGenerators(GroupGeneratorsFromSwaps(num_var,anc_con))
    group_order = len(group_elements)
    AMatrix = np.empty([group_order, card ** num_vars], np.int32)
    AMatrix[0] = ColumnIntegers.flat  # Assuming first group element is the identity
    for i in np.arange(1, group_order):
        AMatrix[i] = np.transpose(ColumnIntegers, group_elements[i]).flat
    minima = np.amin(AMatrix, axis=0)
    AMatrix = np.compress(minima == np.abs(AMatrix[0]), AMatrix, axis=1)
    # print(AMatrix.shape)
    return AMatrix


def ValidColumnOrbitsFromGraph(g, inflation_order, card):
    obs_count, num_vars, exp_set, group_elem, det_assumptions, names = LearnInflationGraphParameters(g, inflation_order)
    print(names)
    return ValidColumnOrbits(card, num_vars, group_elem, det_assumptions)
