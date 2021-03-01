#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
Setting up the inflation linear program.
"""

# TODO: Export LP in standard format.
# TODO: Save entire solver output.
# TODO: Modify to pass entire solver output forward.


import sys
import mosek
import numpy as np
from cvxopt import matrix, solvers, spmatrix


# def scipy_sparse_to_spmatrix(A):
#     coo = A.asformat('coo', copy=False)
#     SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
#     return SP


def InflationLP(A, b):  # A is expected as coo_matrix. Hmm, I wish we could compile this.
    print('Setting up LP in CVXOPT...', flush=True)
    sys.stdout.flush()
    MCVXOPT = spmatrix(A.data.tolist(), A.col.tolist(), A.row.tolist(), size=A.shape[::-1])
    # MCVXOPT=scipy_sparse_to_spmatrix(SparseInflationMatrix.T)
    rowcount = MCVXOPT.size[0];
    colcount = MCVXOPT.size[1];
    CVXOPTb = matrix(np.atleast_2d(b).T)
    CVXOPTh = matrix(np.zeros((rowcount, 1)))
    CVXOPTA = matrix(np.ones((1, colcount)))
    solvers.options['show_progress'] = True
    solvers.options['mosek'] = {mosek.iparam.log: 10,
                                mosek.iparam.presolve_use: mosek.presolvemode.off,
                                mosek.iparam.presolve_lindep_use: mosek.onoffkey.off,
                                mosek.iparam.optimizer: mosek.optimizertype.free,
                                mosek.iparam.presolve_max_num_reductions: 0,
                                mosek.iparam.intpnt_solve_form: mosek.solveform.free,
                                mosek.iparam.sim_solve_form: mosek.solveform.free,
                                mosek.iparam.bi_clean_optimizer: mosek.optimizertype.primal_simplex,
                                mosek.iparam.intpnt_basis: mosek.basindtype.always,
                                mosek.iparam.bi_max_iterations: 1000000
                                }
    # Other options could be: {mosek.iparam.presolve_use:    mosek.presolvemode.on,      mosek.iparam.presolve_max_num_reductions:   -1, mosek.iparam.presolve_lindep_use:   mosek.onoffkey.on,                       mosek.iparam.optimizer:   mosek.optimizertype.free_simplex,        mosek.iparam.intpnt_solve_form:   mosek.solveform.dual, mosek.iparam.intpnt_basis:    mosek.basindtype.always,
    # iparam.sim_switch_optimizer: mosek.onoffkey.on}
    print('Initiating LP...')
    return solvers.lp(CVXOPTb, -MCVXOPT, CVXOPTh, CVXOPTA, matrix(np.ones((1, 1))), solver='mosek')

