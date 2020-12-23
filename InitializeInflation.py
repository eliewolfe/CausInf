#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:16:04 2020

@author: boraulu
"""

import numpy as np
from itertools import permutations, combinations, chain
import time
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from cvxopt import matrix, solvers, sparse, spmatrix
import mosek
from numba import njit
import functools 
import copy
from igraph import *

@njit
def Is_vec_in_mat(vec,mat):
    assume=True
    for elem in mat:
        if np.array_equal(vec, elem):
            assume=False
            break
    return assume


@njit
def dimino_wolfe(group_generators):
        gens=group_generators
        degree=np.max(gens)+1
        idn = np.arange(degree)
        order = 0
        element_list = [idn]
        #element_list=np.atleast_2d(idn)
        #set_element_list = {tuple(idn)}
        for i in np.arange(len(gens)):
            # D elements of the subgroup G_i generated by gens[:i]
            D = element_list[:]
            N = [idn]
            while N:
                A = N
                N = []
                for a in A:
                    for g in gens[:i + 1]:
                        ag = a[g]
                        if Is_vec_in_mat(ag,element_list):
                        #if not np.any(np.all(ag==np.array(element_list,np.int64),axis=1)):
                        #if ag not in np.array(element_list):
                        #if tuple(ag) not in set_element_list:
                            # produce G_i*g
                            for d in D:
                                order += 1
                                ap = d[ag]
                                element_list.append(ap)
                                #element_list=np.append(element_list, np.atleast_2d(ap), axis=0)
                                #set_element_list.add(tuple(ap))
                                N.append(ap)
        return element_list

def Deduplicate(ar): #Alternatives include unique_everseen and panda.unique, see https://stackoverflow.com/a/15637512 and https://stackoverflow.com/a/41577279
    (vals,idx)=np.unique(ar,return_index=True)
    return vals[np.argsort(idx)]

@njit
def MoveToFront(num_var,ar):
    return np.hstack((ar,np.delete(np.arange(num_var),ar)))

@njit
def MoveToBack(num_var,ar):
    return np.hstack((np.delete(np.arange(num_var),ar),ar))

def ToTopologicalOrdering(g):
    return g.permute_vertices(np.argsort(g.topological_sorting('out')).tolist())

#@functools.lru_cache(maxsize=16)
def LearnParametersFromGraph(origgraph):
    g=ToTopologicalOrdering(origgraph)
    verts=g.vs
    verts["parents"]=g.get_adjlist('in');
    #verts["children"]=g.get_adjlist('out');
    verts["ancestors"]=[g.subcomponent(i,'in') for i in g.vs]
    verts["descendants"]=[g.subcomponent(i,'out') for i in g.vs]
    verts["indegree"]=g.indegree()
    #verts["outdegree"]=g.outdegree() #Not needed
    verts["grandparents"]=g.neighborhood(None, order=2, mode='in', mindist=2)
    #verts["parents_inclusive"]=g.neighborhood(None, order=1, mode='in', mindist=0) #Not needed
    has_grandparents=[idx for idx,v in enumerate(g.vs["grandparents"]) if len(v)>=1]
    verts["isroot"]=[0==i for i in g.vs["indegree"]]
    root_vertices=verts.select(isroot = True).indices
    nonroot_vertices=verts.select(isroot = False).indices
    verts["roots_of"]=[np.intersect1d(anc,root_vertices).tolist() for anc in g.vs["ancestors"]]
    def FindScreeningOffSet(root,observed):
        screeningset=np.intersect1d(root["descendants"],observed["parents"]).tolist()
        screeningset.append(observed.index)
        return screeningset
    determinism_checks=[(root,FindScreeningOffSet(verts[root],v)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"],v["parents"])]
    return verts["name"],verts["parents"],verts["roots_of"],determinism_checks

#inflation_order=2
#card=2
#g=Graph.Formula("X->A:C,Y->A:B,Z->B:C")
#g=Graph.Formula("U3->A:C:D,U2->B:C:D,U1->A:B,A->C,B->D")
#g=Graph.Formula("U1->A,U2->B:C,U1->D,A->B,B->C:D")


def GenerateCanonicalExpressibleSet(inflation_order, inflation_depths, offsets):
    #offsets=GenerateOffsets(inflation_order,inflation_depths)
    obs_count=len(inflation_depths)
    order_range=np.arange(inflation_order)
    cannonical_pos=np.empty((obs_count,inflation_order),dtype=np.uint32)
    for i in np.arange(obs_count):
        cannonical_pos[i]=np.sum(np.outer(inflation_order**np.arange(inflation_depths[i]),order_range),axis=0)+offsets[i]
    return cannonical_pos.T.ravel()

def GenerateInflationGroupGenerators(inflation_order, latent_count, root_structure, inflation_depths, offsets):
    inflationcopies=inflation_order**inflation_depths
    num_vars=inflationcopies.sum()
    #offsets=GenerateOffsets(inflation_order,inflation_depths)
    globalstrategyflat=list(np.add(*stuff) for stuff in zip(list(map(np.arange,inflationcopies.tolist())),offsets))
    obs_count=len(inflation_depths)
    reshapings=np.ones((obs_count,latent_count),np.uint8)
    contractings=np.zeros((obs_count,latent_count),np.object)
    for idx,elem in enumerate(root_structure):
        reshapings[idx][elem]=inflation_order
        contractings[idx][elem]=np.s_[:]
    reshapings=list(map(tuple,reshapings))
    contractings=list(map(tuple,contractings))
    globalstrategyshaped=list(np.reshape(*stuff) for stuff in zip(globalstrategyflat,reshapings))
    fullshape=tuple(np.full(latent_count,inflation_order))
    if inflation_order==2:
        inflation_order_gen_count=1
    else:
        inflation_order_gen_count=2
    group_generators=np.empty((latent_count,inflation_order_gen_count,num_vars),np.uint)
    for latent_to_explore in np.arange(latent_count):
        for gen_idx in np.arange(inflation_order_gen_count):
            initialtranspose=MoveToFront(latent_count,np.array([latent_to_explore]))
            inversetranspose=np.hstack((np.array([0]),1+np.argsort(initialtranspose)))
            label_permutation=np.arange(inflation_order)
            if gen_idx==0:
                label_permutation[np.array([0,1])]=np.array([1,0])
            elif gen_idx==1:
                label_permutation=np.roll(label_permutation, 1)
            global_permutation=np.array(list(np.broadcast_to(elem,fullshape).transpose(tuple(initialtranspose))[label_permutation] for elem in globalstrategyshaped))
            global_permutation=np.transpose(global_permutation,tuple(inversetranspose))
            global_permutation=np.hstack(tuple(global_permutation[i][contractings[i]].ravel() for i in np.arange(obs_count)))
            #global_permutationOLD=Deduplicate(np.ravel(global_permutation))   #Deduplication has been replaced with intelligent extraction.
            #print(np.all(global_permutation==global_permutationOLD))
            group_generators[latent_to_explore,gen_idx]=global_permutation
    return group_generators

def GenDeterminismAssumptions(determinism_checks,latent_count,group_generators,exp_set):
    one_generator_per_root=group_generators[:,0]
    det_assumptions=list();
    for pair in determinism_checks:
        flatset=exp_set[list(np.array(pair[1])-latent_count)]
        symop=one_generator_per_root[pair[0]]
        rule=np.vstack((flatset,symop[flatset])).T.astype('uint32')
        rule=rule[:-1,:].T.tolist()+rule[-1,:].T.tolist()
        det_assumptions.append(rule)
    return det_assumptions
#det_assumptions=GenDeterminismAssumptions(determinism_checks,latent_count,group_generators,exp_set)
#print(det_assumptions)


#group_elem=dimino_wolfe(group_generators.reshape((-1,num_vars)))
#print(np.array(group_elem))

def LearnInflationGraphParameters(g,inflation_order):
    names,parents_of,roots_of,determinism_checks = LearnParametersFromGraph(g)
    #print(names)
    graph_structure=list(filter(None,parents_of))
    obs_count=len(graph_structure)
    latent_count=len(parents_of)-obs_count
    root_structure=roots_of[latent_count:]
    inflation_depths=np.array(list(map(len,root_structure)))
    inflationcopies=inflation_order**inflation_depths
    num_vars=inflationcopies.sum()
    accumulated=np.add.accumulate(inflation_order**inflation_depths)
    offsets=np.hstack(([0],accumulated[:-1]))
    exp_set=GenerateCanonicalExpressibleSet(inflation_order, inflation_depths, offsets)
    group_generators = GenerateInflationGroupGenerators(inflation_order, latent_count, root_structure, inflation_depths, offsets)
    group_elem=np.array(dimino_wolfe(group_generators.reshape((-1,num_vars))))
    det_assumptions=GenDeterminismAssumptions(determinism_checks,latent_count,group_generators,exp_set)
    return obs_count,num_vars,exp_set,group_elem,det_assumptions,names[latent_count:]

#obs_count,num_vars,exp_set,group_elem,det_assumptions = LearnInflationGraphParameters(g,inflation_order)
#print([obs_count,num_vars])
#print(exp_set)
#print(group_elem)
#print(det_assumptions)

"""
names,parents_of,roots_of,determinism_checks = LearnParametersFromGraph(g)
graph_structure=list(filter(None,parents_of))
obs_count=len(graph_structure)
latent_count=len(parents_of)-obs_count
root_structure=roots_of[latent_count:]

inflation_depths=np.array(list(map(len,root_structure)))
inflationcopies=inflation_order**inflation_depths
num_vars=inflationcopies.sum()

#accumulated=np.add.accumulate(inflation_order**inflation_depths)
#offsets = np.hstack(([0],accumulated[:-1]))
def GenerateOffsets(inflation_order,inflation_depths):
    accumulated=np.add.accumulate(inflation_order**inflation_depths)
    return np.hstack(([0],accumulated[:-1]))
offsets=GenerateOffsets(inflation_order,inflation_depths)

exp_set=GenerateCanonicalExpressibleSet(inflation_order,inflation_depths)
"""

#accumulated=np.add.accumulate(inflationcopies)
#offsets = np.hstack(([0],accumulated[:-1]))
#cannonical_pos=np.zeros((obs_count,inflation_order),dtype=np.uint32)
#for i in range(obs_count):
#    
#    depth=inflation_depths[i]
#    number_of_copies=inflation_order**depth
#    step=int(np.floor((number_of_copies)/(inflation_order-1)))
#    if step == number_of_copies:
#        step=step-1
#    cannonical_pos[i]=np.arange(0,number_of_copies,step)
#    
#exp_set=np.ravel(cannonical_pos.T+offsets)
#    
#print(exp_set) 







    

@njit
def GenShapedColumnIntegers(range_shape):    
    return np.arange(0,np.prod(np.array(range_shape)),1,np.int32).reshape(range_shape)

def MarkInvalidStrategies(card,num_var,det_assumptions):
    initialshape=np.full(num_var,card,np.uint8)
    ColumnIntegers=GenShapedColumnIntegers(tuple(initialshape))
    for detrule in det_assumptions:
        initialtranspose=MoveToFront(num_var,np.hstack(tuple(detrule)))
        inversetranspose=np.argsort(initialtranspose)
        parentsdimension=card**len(detrule[1])
        intermediateshape=(parentsdimension,parentsdimension,card,card,-1);
        ColumnIntegers=ColumnIntegers.transpose(tuple(initialtranspose)).reshape(intermediateshape)
        for i in np.arange(parentsdimension):
            for j in np.arange(card-1):
                for k in np.arange(j+1,card):
                    ColumnIntegers[i,i,j,k]=-1
        ColumnIntegers=ColumnIntegers.reshape(initialshape).transpose(tuple(inversetranspose))
    return ColumnIntegers

def ValidColumnOrbits(card, num_vars, group_elem,det_assumptions=[]):
    ColumnIntegers=MarkInvalidStrategies(card,num_vars,det_assumptions)
    group_elements=group_elem#GroupElementsFromGenerators(GroupGeneratorsFromSwaps(num_var,anc_con))
    group_order=len(group_elements)
    AMatrix=np.empty([group_order,card**num_vars],np.int32)
    AMatrix[0]=ColumnIntegers.flat #Assuming first group element is the identity
    for i in np.arange(1,group_order):
        AMatrix[i]=np.transpose(ColumnIntegers,group_elements[i]).flat
    minima=np.amin(AMatrix,axis=0)
    AMatrix=np.compress(minima==np.abs(AMatrix[0]), AMatrix, axis=1)
    #print(AMatrix.shape)
    return AMatrix

def ValidColumnOrbitsFromGraph(g,inflation_order,card):
    obs_count,num_vars,exp_set,group_elem,det_assumptions,names = LearnInflationGraphParameters(g,inflation_order)
    print(names)
    return ValidColumnOrbits(card, num_vars, group_elem, det_assumptions)

def PositionIndex(arraywithduplicates):
    arraycopy=np.empty_like(arraywithduplicates)
    u=np.unique(arraywithduplicates)
    arraycopy[u]=np.arange(len(u))
    return arraycopy[arraywithduplicates]

@functools.lru_cache(maxsize=16)
def GenerateEncodingMonomialToRow(original_cardinality_product,inflation_order): #I should make this recursive, as called by both A and b construction.
    monomial_count=int(original_cardinality_product**inflation_order)
    permutation_count=int(np.math.factorial(inflation_order))
    MonomialIntegers=np.arange(0,monomial_count,1,np.uint)
    new_shape=np.full(inflation_order,original_cardinality_product)
    MonomialIntegersPermutations=np.empty([permutation_count,monomial_count],np.uint)
    IndexPermutations=list(permutations(np.arange(inflation_order)))
    MonomialIntegersPermutations[0]=MonomialIntegers
    MonomialIntegers=MonomialIntegers.reshape(new_shape)
    for i in np.arange(1,permutation_count):
        MonomialIntegersPermutations[i]=np.transpose(MonomialIntegers,IndexPermutations[i]).flat
    return PositionIndex(np.amin(
        MonomialIntegersPermutations,axis=0))

def GenerateEncodingColumnToMonomial(card,num_var,expr_set):
    initialshape=np.full(num_var,card,np.uint)
    ColumnIntegers=GenShapedColumnIntegers(tuple(initialshape))
    ColumnIntegers=ColumnIntegers.transpose(MoveToBack(num_var,np.array(expr_set))).reshape((-1,card**len(expr_set)))
    EncodingColumnToMonomial=np.empty(card**num_var,np.uint32)
    EncodingColumnToMonomial[ColumnIntegers]=np.arange(card**len(expr_set))
    return EncodingColumnToMonomial

def MergeMonomials(bvector,encoding):
    return np.ravel(coo_matrix((bvector, (np.zeros(len(bvector),np.uint8), encoding)),(1, int(np.amax(encoding)+1))).toarray())

#def EncodeA(card, num_var, group_elem, expr_set, inflation_order):
#    original_product_cardinality=(card**np.rint(len(expr_set)/inflation_order)).astype(np.uint)
#    EncodingMonomialToRow=GenerateEncodingMonomialToRow(original_product_cardinality,inflation_order)
#    EncodingColumnToMonomial=GenerateEncodingColumnToMonomial(card,num_var,np.array(expr_set))
#    return EncodingMonomialToRow[EncodingColumnToMonomial][ValidColumnOrbits(card, num_var, group_elem,det_assumptions)]

def EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    #original_product_cardinality=(card**np.rint(len(expr_set)/inflation_order)).astype(np.uint)
    original_product_cardinality=card**obs_count
    EncodingMonomialToRow=GenerateEncodingMonomialToRow(original_product_cardinality,inflation_order)
    EncodingColumnToMonomial=GenerateEncodingColumnToMonomial(card,num_vars,np.array(expr_set))
    result=EncodingMonomialToRow[EncodingColumnToMonomial][valid_column_orbits]
    #Once the encoding is done, the order of the columns can be tweaked at will!
    #result=np.sort(result,axis=0)
    result.sort(axis=0) #in-place sort
    #result=result[np.lexsort(result),:]
    return result
    #return EncodingMonomialToRow[EncodingColumnToMonomial][valid_column_orbits]


def SciPyArrayFromOnesPositions(OnesPositions):
    columncount=OnesPositions.shape[-1]
    columnspec=np.broadcast_to(np.arange(columncount), (len(OnesPositions), columncount)).ravel()
    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

def SciPyArrayFromOnesPositionsWithSort(OnesPositions):
    columncount=OnesPositions.shape[-1]
    columnspec=np.broadcast_to(np.lexsort(OnesPositions), (len(OnesPositions), columncount)).ravel()
    return coo_matrix((np.ones(OnesPositions.size,np.uint), (OnesPositions.ravel(), columnspec)),(int(np.amax(OnesPositions)+1), columncount),dtype=np.uint)

def SparseInflationMatrix(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card):
    return SciPyArrayFromOnesPositionsWithSort(EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))

def InflationMatrixFromGraph(g,inflation_order,card):
    obs_count,num_vars,expr_set,group_elem,det_assumptions,names = LearnInflationGraphParameters(g,inflation_order)
    print(names)
    #valid_column_orbits=ValidColumnOrbitsFromGraph(g,inflation_order,card) #Should be fixed so as not to learn inflation parameters twice.
    valid_column_orbits=ValidColumnOrbits(card, num_vars, group_elem,det_assumptions)
    return SciPyArrayFromOnesPositions(EncodeA(obs_count, num_vars, valid_column_orbits, expr_set, inflation_order, card))

def FindB(Data, inflation_order):
    EncodingMonomialToRow=GenerateEncodingMonomialToRow(len(Data),inflation_order)
    preb=np.array(Data)
    b=preb
    for i in range(1,inflation_order):
        b=np.kron(preb,b)
    b=MergeMonomials(b,EncodingMonomialToRow)
    return b




@njit
def reindex_list(ar):
        seenbefore=np.full(np.max(ar)+1,-1)
        newlist=np.empty(len(ar),np.uint)
        currentindex=0
        for idx,val in enumerate(ar):
            if seenbefore[val]==-1:
                seenbefore[val]=currentindex
                newlist[idx]=currentindex
                currentindex+=1
            else:
                newlist[idx]=seenbefore[val]
        return (newlist)
    
def scipy_sparse_to_spmatrix(A):
    coo = A.asformat('coo', copy=False)
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def optimize_inflation_matrix(A):
    coo = A.asformat('coo', copy=False)
    rowsorting=np.argsort(coo.row)
    newrows=coo.row[rowsorting].tolist()
    newcols=reindex_list(coo.col[rowsorting])
    #return spmatrix(coo.data[rowsorting].tolist(),newcols,newrows, (A.shape[1],A.shape[0]))
    return coo_matrix((coo.data[rowsorting], (newrows, newcols)),(A.shape[0],A.shape[1]),dtype=np.uint)

#def CVXOPTArrayFromOnesPositions(OnesPositions):
#    columncount=OnesPositions.shape[-1]
#    columnspec=np.broadcast_to(np.arange(columncount), (len(OnesPositions), columncount)).ravel()
#    return spmatrix(np.ones(OnesPositions.size), OnesPositions.ravel().tolist(), columnspec.tolist(),(int(np.amax(OnesPositions)+1), columncount))

def InflationLP(SparseInflationMatrix,b):
    print('Preprocessing LP for efficiency boost...')
    #MCVXOPT=CVXOPTArrayFromOnesPositions(EncodedA).T
    #MCVXOPT=scipy_sparse_to_row_optimized_spmatrix_transpose(SciPyArrayFromOnesPositions(EncodedA))
    MCVXOPT=scipy_sparse_to_spmatrix(SparseInflationMatrix.T)
    rowcount=MCVXOPT.size[0];
    colcount=MCVXOPT.size[1];
    CVXOPTb=matrix(np.atleast_2d(b).T)
    CVXOPTh=matrix(np.zeros((rowcount,1)))
    CVXOPTA=matrix(np.ones((1,colcount)))
    solvers.options['show_progress'] = True
    solvers.options['mosek'] = {mosek.iparam.log:   10,
                                   mosek.iparam.presolve_use:    mosek.presolvemode.off,
                                   mosek.iparam.presolve_lindep_use:   mosek.onoffkey.off,
                                   mosek.iparam.optimizer:    mosek.optimizertype.free,
                                   mosek.iparam.presolve_max_num_reductions:   0,
                                   mosek.iparam.intpnt_solve_form:   mosek.solveform.free,
                                   mosek.iparam.sim_solve_form:   mosek.solveform.free,
                                   mosek.iparam.bi_clean_optimizer: mosek.optimizertype.primal_simplex,
                                   mosek.iparam.intpnt_basis:    mosek.basindtype.always,
                                   mosek.iparam.bi_max_iterations:   1000000
                                   }
    #Other options could be: {mosek.iparam.presolve_use:    mosek.presolvemode.on,      mosek.iparam.presolve_max_num_reductions:   -1, mosek.iparam.presolve_lindep_use:   mosek.onoffkey.on,                       mosek.iparam.optimizer:   mosek.optimizertype.free_simplex,        mosek.iparam.intpnt_solve_form:   mosek.solveform.dual, mosek.iparam.intpnt_basis:    mosek.basindtype.always,
    #iparam.sim_switch_optimizer: mosek.onoffkey.on}
    print('Initiating LP')
    sol=solvers.lp(CVXOPTb,-MCVXOPT,CVXOPTh,CVXOPTA,matrix(np.ones((1,1))),solver='mosek')
    return sol['x'],sol['gap']



#start = time.time()
#valid_column_orbits=ValidColumnOrbitsFromGraph(g,inflation_order,card)
#EncodedA = EncodeA(card, num_vars, valid_column_orbits, exp_set, inflation_order)
#print('It took', time.time()-start, 'seconds.')
#print(EncodedA.shape)

#start = time.time()
#b=FindB(Data,inflation_order)
#MCVXOPT=FormCVXOPTArrayFromOnesPositions(EncodedA).T 
#print('It took', time.time()-start, 'seconds.')
#print(MCVXOPT.size)

#solverout=InflationLP(EncodedA,b)


