#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 01:33:53 2021

@author: boraulu
"""

import numpy as np
from itertools import chain

def _identify_determinism_check(parents_of, descendants_of , root_indices, observed_index):
    """
    root_indices is a list of root nodes (integers) which can be screened off.
    observed_index is a single node (integer).
    The output will be a tuple of 3 lists
    (U1s,Ys,Xs) with the following meaning: Ys are screened off from U1s by Xs.
    """
    list_extract_and_union = lambda list_of_lists, indices: set().union(
        chain.from_iterable(list_of_lists[v] for v in indices))
    parents_of_observed = set(parents_of[observed_index])
    # descendants_of_roots = [self.descendants_of[v] for v in root_indices]
    # descendants_of_roots = set().union(*descendants_of_roots)
    descendants_of_roots = list_extract_and_union(descendants_of, root_indices)
    U1s = list(root_indices)
    Y = observed_index
    Xs = list(parents_of_observed.intersection(descendants_of_roots))
    return (U1s, [Y], Xs)


