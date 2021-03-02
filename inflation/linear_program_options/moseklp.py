import sys
import mosek
import numpy as np

# from scipy.sparse import csr_matrix


# Since the actual value of Infinity is ignores, we define it solely
# for symbolic purposes:
inf = 0.0


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def InfeasibilityCertificate(A, b):
    # A is presumed to by a SciPy sparse matrix, b is a numpy array.
    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)

        # Create a task
        with env.Task(0, 0) as task:
            # Attach a printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            (numvar, numcon) = A.shape

            streamprinter('Inferred primal variable count: ' + str(numvar) + '\n')
            streamprinter('Inferred dual variable count: ' + str(numcon) + '\n')

            # Bounds for variables (We take the smallest value to be no lower than -1. Seems reasonable, right?)
            bkx = numvar * [mosek.boundkey.lo]
            blx = numvar * [-1.0]
            # blx = [0.] * numvar
            bux = numvar * [+inf]

            # Bounds for constraints
            bkc = numcon * [mosek.boundkey.ra]
            blc = numcon * [0.0]
            buc = numcon * [1.0]

            # Solver output is still unstable. We need to add a new constraint pertaining to sum of coefficients.

            # A matrix parameters
            csr = A.asformat('csr', copy=False)
            ptrb = csr.indptr[:-1]
            ptre = csr.indptr[1:]

            task.inputdata(numcon,  # number of constraints
                           numvar,  # number of variables
                           b.tolist(),  # linear objective coefficients
                           0.0,  # objective fixed value
                           list(ptrb),
                           list(ptre),
                           list(csr.indices),
                           list(csr.data),
                           bkc,
                           blc,
                           buc,
                           bkx,
                           blx,
                           bux)

            # rangecon=np.arange(numcon).astype(np.int_).tolist()
            rangevar = np.arange(numvar).astype(np.int_).tolist()
            counts = np.ones(numvar).tolist()  # We should accept this as solver input eventually.
            task.appendcons(1)
            task.putarow(numcon, rangevar, counts)
            task.putconbound(numcon, mosek.boundkey.fx, 1, 1)  # Sum of dual is 1.
            numcon += 1

            task.putobjsense(mosek.objsense.minimize)
            task.putintparam(mosek.iparam.bi_clean_optimizer, mosek.optimizertype.primal_simplex)
            task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
            task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
            task.putintparam(mosek.iparam.presolve_max_num_reductions, 0)
            #task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.dual_simplex)
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            streamprinter('LP constructed, initiated optimizer.\n')
            # task.writetask('mosekLPinput.tar')
            task.writedata('mosek_prob.jtask')
            # Optimize the task
            task.optimize()
            task.writejsonsol('mosek_sol.json')

            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            # Use basic if available. (It won't be.)
            if task.solutiondef(mosek.soltype.bas) == 1:
                soltype = mosek.soltype.bas
            elif task.solutiondef(mosek.soltype.itr) > 0:
                soltype = mosek.soltype.itr
            else:
                raise ValueError("No valid soltype for mosek detected.")

            prosta = task.getprosta(soltype)
            solsta = task.getsolsta(soltype)

            # Output a solution
            xx = numvar * [0.0]
            y = numcon * [0.0]
            skc = numcon * [mosek.stakey.unk]
            skx = numvar * [mosek.stakey.unk]
            skn = 0 * [mosek.stakey.unk]
            xc = numcon * [0.0]
            slc = numcon * [0.0]
            suc = numcon * [0.0]
            slx = numvar * [0.0]
            sux = numvar * [0.0]
            snx = numvar * [0.0]
            task.getsolution(soltype, skc, skx, skn, xc, xx, y, slc, suc, slx, sux, snx)
            gap = np.linalg.norm(np.subtract(np.array(suc), np.array(slc)), np.inf)
            Sol = {'x': np.array(xx), 'y': np.array(y), 'xc': np.array(xc), 'gap': gap}

            # task.dispose()
        # env.dispose()
    streamprinter('Problem status: ' + str(prosta) + '\n')
    streamprinter('Solution status: ' + str(solsta) + '\n')
    streamprinter('\nCoefficient Range: ' + str([min(Sol['x']), max(Sol['x'])]) + '\n')

    return Sol


if __name__ == '__main__':
    from igraph import Graph

    TriangleData = [0.12199995751046305, 0.0022969343799089472, 0.001748319476328954, 3.999015242496535e-05,
                    0.028907881434196828, 0.0005736087488455967, 0.0003924033706699725, 1.1247230369521505e-05,
                    0.0030142577390317635, 0.09234476010282468, 4.373922921480586e-05, 0.0014533921021948346,
                    0.0007798079722868244, 0.024091567451515063, 1.1247230369521505e-05, 0.0003849052170902915,
                    0.020774884184769502, 0.000396152447459813, 0.0003049249122403608, 4.998769053120669e-06,
                    0.10820335492385, 0.0020794879260981982, 0.0015546171755205281, 2.4993845265603346e-05,
                    0.0006260958239033638, 0.020273757587194154, 7.498153579681003e-06, 0.0003374169110856452,
                    0.0028942872817568676, 0.08976414557915113, 2.624353752888351e-05, 0.0012984302615480939,
                    0.002370666223442477, 4.7488306004646356e-05, 0.0999928767540993, 0.001957018084296742,
                    0.0006198473625869629, 8.747845842961171e-06, 0.02636975644747481, 0.0005198719815245496,
                    1.4996307159362007e-05, 0.000403650601039494, 0.0005498645958432735, 0.017359475229224805,
                    7.123245900696953e-05, 0.002346922070440154, 0.0033754188031197316, 0.10295964618712641,
                    0.00038740460161685187, 7.498153579681003e-06, 0.01608353942841575, 0.000306174604503641,
                    0.0021319750011559654, 4.248953695152569e-05, 0.09107007399427891, 0.001860791780024169,
                    5.998522863744803e-05, 0.0018395470115484063, 0.002570616985567304, 0.0766411271224461,
                    1.874538394920251e-05, 0.00048238121362614454, 0.0006410921310627258, 0.020223769896662948]


    def PRBoxF(Ain, Aout, Bin, Bout, CA, CB):
        if Ain == CA and Bin == CB:
            if Ain == 0 or Bin == 0:
                if Aout == Bout:
                    return 1 / 8
                else:
                    return 0
            elif Ain == 1 and Bin == 1:
                if Aout == Bout:
                    return 0
                else:
                    return 1 / 8
        else:
            return 0


    def TsirelsonBoxF(Ain, Aout, Bin, Bout, CA, CB):
        if Ain == CA and Bin == CB:
            if Ain == 0 or Bin == 0:
                if Aout == Bout:
                    return (4 + 2 * np.sqrt(2)) / 64
                else:
                    return (4 - 2 * np.sqrt(2)) / 64
            elif Ain == 1 and Bin == 1:
                if Aout == Bout:
                    return (4 - 2 * np.sqrt(2)) / 64
                else:
                    return (4 + 2 * np.sqrt(2)) / 64
        else:
            return 0


    PRBox = np.array([PRBoxF(*idx) for idx, val in np.ndenumerate(np.arange(64).reshape((2, 2, 2, 2, 2, 2)))])
    TsirelsonBox = np.array(
        [TsirelsonBoxF(*idx) for idx, val in np.ndenumerate(np.arange(64).reshape((2, 2, 2, 2, 2, 2)))])

    InstrumentalData = np.zeros(8);
    InstrumentalData[0] = 0.5;
    InstrumentalData[5] = 0.5;

    inflation_order = 2
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    EvansGraph = Graph.Formula("U3->A:C:D,U2->B:C:D,U1->A:B,A->C,B->D")
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    # We are going to need to get some code to work with variable ordering.

    Graph = InstrumentalGraph
    Data = InstrumentalData
    card = 2
    Graph = TriangleGraph
    Data = PRBox
    card = 4

    from lib.inflationmatrix import InflationMatrixFromGraph, FindB

    A = InflationMatrixFromGraph(Graph, inflation_order, card)
    b = FindB(Data, inflation_order)
    Sol = InfeasibilityCertificate(A, b)
    Sol['x']
    # print([min(Sol['x']),max(Sol['x'])])
