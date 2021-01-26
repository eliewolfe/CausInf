#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions.
"""

import numpy as np
from numba import njit


def Deduplicate(
        ar):  # Alternatives include unique_everseen and panda.unique, see https://stackoverflow.com/a/15637512 and https://stackoverflow.com/a/41577279
    (vals, idx) = np.unique(ar, return_index=True)
    return vals[np.argsort(idx)]


@njit
def MoveToFront(num_var, ar):
    return np.hstack((ar, np.delete(np.arange(num_var), ar)))


@njit
def MoveToBack(num_var, ar):
    return np.hstack((np.delete(np.arange(num_var), ar), ar))


@njit
def GenShapedColumnIntegers(range_shape):
    return np.arange(0, np.prod(np.array(range_shape)), 1, np.int32).reshape(range_shape)


# def PositionIndex(arraywithduplicates):
#    arraycopy=np.empty_like(arraywithduplicates)
#    u=np.unique(arraywithduplicates)
#    arraycopy[u]=np.arange(len(u))
#    return arraycopy.take(arraywithduplicates)

def PositionIndex(arraywithduplicates):
    # u,inv,idx=np.unique(arraywithduplicates,return_inverse=True)[1]
    return np.unique(arraywithduplicates, return_inverse=True)[1]


@njit
def reindex_list(ar):
    seenbefore = np.full(np.max(ar) + 1, -1)
    newlist = np.empty(len(ar), np.uint)
    currentindex = 0
    for idx, val in enumerate(ar):
        if seenbefore[val] == -1:
            seenbefore[val] = currentindex
            newlist[idx] = currentindex
            currentindex += 1
        else:
            newlist[idx] = seenbefore[val]
    return newlist
