#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting up the inflation linear program.
"""

import numpy as np
from scipy.sparse import coo_matrix
from .utilities import reindex_list


def optimize_inflation_matrix(A):
    coo = A.asformat('coo', copy=False)
    rowsorting = np.argsort(coo.row)
    newrows = coo.row[rowsorting].tolist()
    newcols = reindex_list(coo.col[rowsorting])
    # return spmatrix(coo.data[rowsorting].tolist(),newcols,newrows, (A.shape[1],A.shape[0]))
    return coo_matrix((coo.data[rowsorting], (newrows, newcols)), (A.shape[0], A.shape[1]), dtype=np.uint)
