#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from __future__ import absolute_import
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix


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
    #return np.arange(np.array(range_shape).prod()).reshape(range_shape)


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

def SparseMatrixFromRowsPerColumn(OnesPositions, sort_columns=True):
    #Assumes that OnesPositions is a 2d numpy array of integers.
    #First dimension indicates which ORBIT we are considering.
    #Second dimension indicates which COLUMN we are listing rows for.
    columncount = OnesPositions.shape[-1]
    rowcount = int(np.amax(OnesPositions)) + 1
    if sort_columns:
        ar_to_broadcast = np.lexsort(OnesPositions)
    else:
        ar_to_broadcast = np.arange(columncount)
    columnspec = np.broadcast_to(ar_to_broadcast, OnesPositions.shape).ravel()
    return coo_matrix((np.ones(OnesPositions.size, np.uint), (OnesPositions.ravel(), columnspec)),
                      (rowcount, columncount), dtype=np.uint)


if __name__ == '__main__':
    print(SparseMatrixFromRowsPerColumn(np.array([[1,1,3],[2,5,3]])).tocsr())