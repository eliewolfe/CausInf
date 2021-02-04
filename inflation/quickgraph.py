#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""
from __future__ import absolute_import
import numpy as np



def ToTopologicalOrdering(g):
    return g.permute_vertices(np.argsort(g.topological_sorting('out')).tolist())

def ToRootLexicographicOrdering(g):
    """

    Parameters
    ----------
    g: an iGraph graph

    Returns
    -------
    A copy of g with nodes internally ordered such that root nodes are first and non-root nodes are subsequent.
    Within each set (root \& nonroot) the nodes are ordered lexicographically.
    """
    verts = g.vs
    verts["isroot"] = [0 == i for i in g.indegree()]
    root_vertices_igraph = verts.select(isroot=True)
    nonroot_vertices_igraph = verts.select(isroot=False)
    optimal_root_node_indices = np.take(root_vertices_igraph.indices,np.argsort(root_vertices_igraph["name"]))
    optimal_nonroot_node_indices = np.take(nonroot_vertices_igraph.indices, np.argsort(nonroot_vertices_igraph["name"]))
    new_ordering = np.hstack((optimal_root_node_indices,optimal_nonroot_node_indices))
    #print(np.array_str(np.take(verts["name"],new_ordering)))
    return g.permute_vertices(np.argsort(new_ordering).tolist())

#Feb 2 2021 Breaking changes:
#Now outputting determinism_checks and expressible_sets as DIFFERENT lists
def LearnOriginalGraphParameters(origgraph, hasty = False):
    
    """
    Parameters
    ----------
    g : igraph.Graph
        The causal graph containing all of the observable and latent variables.
         
    hasty : bool,optional
        If ``True``, returns a tuple of lists where the first list contains the names of the variables involved, the second list contains the parents of each of these variables in the form of a list-of-lists and the third list contains the roots of each variable again, in the form of a list-of-lists (default set to ``False``)

    Returns
    -------
    verts["name"] : list_of_strings
       A list of strings containing the names of the variables of the graph in topological order.
       
    verts["parents"] : list_of_lists
        A list of lists where each list contains the parents of a variable's node.
        
    verts["roots_of"] : list_of_lists
        A list of lists where each list contains the roots of a variable's node.
        
    determinism_checks : list_of_tuples_of_lists
        Returns the determinism assumptions that could be used to mark invalid strategies when computing the marginal description matrix (see ``inflation.strategies.MarkInvalidStrategies``). In each tuple, the elements of the first list are blocked off from the elements of the second list by the elements of the third list.
        
    extra_expressible_sets : list_of_tuples_of_lists
        Returns the non-ai expressible sets obtained from d-seperation relations. The sets are given as :math:`(Y,\mathbf{X},\mathbf{Z},\mathbf{U_{3}})` where :math:`Y` is screened off from the set of nodes :math:`\mathbf{Z}` by :math:`\mathbf{X}` when the inflated copy of the set :math:`\mathbf{U_{3}}` is set to be different than that for the roots of :math:`\mathbf{X}` and :math:`Y`. 
            
    Notes
    -----
    
    The determinism assumptions mark invalid strategies. For example in the instrumental scenario (defined in the example below) the variable :math:`B` is blocked from :math:`U_{1}` by the variable :math:`A` with which it shares its second root variable :math:`U_{2}`. Therefore, if the variable A has the same value for different copies of :math:`U_{1}` then the same copies of :math:`B` must also have the same values. Hence, any startegy that contradicts this logic must be marked and removed from the marginal description matrix.
    
    The extra expressible sets arise from the following d-seperation relation:
        
    .. math:: \mathbf{Z}^{U_{1}=a, U_{3}=b} \perp_{d} Y^{U_{1}=a, U_{2}=a, U_{3}=a} | \mathbf{X}^{U_{1}=a, U_{2}=a}
    
    
    For :math:`a` :math:`≠` :math:`b`, where the set :math:`\mathbf{X}` screens off :math:`Y` from the set :math:`\mathbf{U_{1}}`, the set :math:`\mathbf{U_{2}}` contains all mutual roots of :math:`\mathbf{X}` and :math:`Y` other than :math:`\mathbf{U_{1}}`, the set :math:`\mathbf{U_{3}}` are all root nodes other than :math:`\mathbf{U_{1}}` and :math:`\mathbf{U_{2}}` and the set :math:`\mathbf{Z}` are all the other nodes which are non-descendants of :math:`\mathbf{U_{2}}`.
    
    Examples
    --------
    For the instrumental scenario we have:
    
    >>> g=Graph.Formula("U1->X->A->B,U2->A:B")
    >>> names, parents, roots_of, determinism_checks, extra_expressible_sets=LearnOriginalGraphParameters(g, hasty=False)
    >>> print(names)
    >>> ['U1', 'U2', 'X', 'A', 'B']
    >>> print(parents)
    >>> [[], [], [0], [1, 2], [1, 3]]
    
    Here the numbers are the indecies of the variables in ``names`` and the lists represent variables in the same order as in ``names``.
    
    >>> print(roots_of)
    >>> [[0], [1], [0], [0, 1], [0, 1]]
    >>> print(determinism_checks)
    >>> [([0], [3], [2]), ([0], [4], [3])]
    >>> print(extra_expressible_sets)
    >>> [([3], [2], [4], [1]), ([4], [2], [3], [1])]
    
    """
    
    g = ToRootLexicographicOrdering(origgraph)
    verts = g.vs
    verts["parents"] = g.get_adjlist('in');
    verts["children"] = g.get_adjlist('out');
    verts["ancestors"] = [g.subcomponent(i, 'in') for i in g.vs]
    verts["descendants"] = [g.subcomponent(i, 'out') for i in g.vs]
    verts["indegree"] = g.indegree()
    # verts["outdegree"]=g.outdegree() #Not needed
    verts["grandparents"] = g.neighborhood(None, order=2, mode='in', mindist=2)
    # verts["parents_inclusive"]=g.neighborhood(None, order=1, mode='in', mindist=0) #Not needed
    has_grandparents = [idx for idx, v in enumerate(verts["grandparents"]) if len(v) >= 1]
    verts["isroot"] = [0 == i for i in verts["indegree"]]
    root_vertices = verts.select(isroot=True).indices
    nonroot_vertices = verts.select(isroot=False).indices
    verts["roots_of"] = [np.intersect1d(anc, root_vertices).tolist() for anc in verts["ancestors"]]
    if hasty:
        return verts["name"], verts["parents"], verts["roots_of"]
    else:
        # def FindScreeningOffSet(root, observed):
        #    screeningset = np.intersect1d(root["children"], observed["ancestors"]).tolist()
        #    screeningset.append(observed.index)
        #    return screeningset

        def Identify_Determinism_Check(roots, observed):
            """roots is a list of root nodes which can be screened off, observed is a single node, in iGraph.VertexSeq format.
            The output will be a tuple of 3 lists
            (U1s,Ys,Xs) with the following meaning: Ys are screened off from U1s by Xs.
            """
            parents_of_observed = set(observed["parents"])
            descendants_of_roots = set().union(*roots["descendants"])
            U1s = roots.indices
            Y = observed.index
            Xs = list(parents_of_observed.intersection(descendants_of_roots))
            return (U1s, [Y], Xs)

        def Identify_Expressible_Set(roots, observed):
            """roots is a list of root nodes which can be screened off, observed is a single node, in iGraph.VertexSeq format.
            The output will be a tuple of 4 lists
            (Ys,Xs,Zs,U3s) with the following meaning:
            Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
            """
            children_of_roots = set().union(*roots["children"])
            screeningset = children_of_roots.intersection(observed["ancestors"])
            Xs = screeningset.copy()
            for sidx in screeningset:
                screeningset_rest = screeningset.copy()
                screeningset_rest.remove(sidx)
                #unblocked_path if screeningset_rest.isdisjoint(directed_path)
                #sidx is redundant if there are not ANY unblocked paths.
                if not any(screeningset_rest.isdisjoint(directed_path) for directed_path in
                           g.get_all_simple_paths(sidx, to=observed)):
                    Xs.remove(sidx)

            U1s = set(roots.indices)
            Y = observed.index
            U2s = set().union(*verts[Xs]["roots_of"]).difference(U1s)

            U2s_descendants = set().union(*verts[U2s]["descendants"])
            observable_nodes_aside_from_Zs = set(Xs)
            observable_nodes_aside_from_Zs.add(Y)
            Zs = set(nonroot_vertices).difference(U2s_descendants).difference(observable_nodes_aside_from_Zs)

            roots_of_Y_aside_from_U1s = set(verts[Y]["roots_of"]).difference(U1s)
            roots_of_Zs = set().union(*verts[Zs]["roots_of"])
            U3YZ = roots_of_Y_aside_from_U1s.intersection(roots_of_Zs)
            #Adding a sanity filter:
            if len(U3YZ) == 0:
                Zs = set()

            return tuple(map(list,([Y], Xs, Zs, U3YZ)))

        from itertools import chain, combinations
        def Root_Subsets(v):
            "v is presumed to be a iGraph vertex object."
            screenable_roots = np.setdiff1d(v["roots_of"], v["parents"])
            return [verts[subroots] for r in np.arange(1, screenable_roots.size + 1) for subroots in combinations(screenable_roots, r)]

        determinism_checks = [Identify_Determinism_Check(roots_subset,v)
                                                       for v in g.vs[has_grandparents]
                                                       for roots_subset in Root_Subsets(v)
                                                       ]

        extra_expressible_sets = list(filter(lambda screening: len(screening[-1]) > 0,
            [Identify_Expressible_Set(roots_subset,v)
                                                       for v in g.vs[has_grandparents]
                                                       for roots_subset in Root_Subsets(v)
                                                       ]))

        return verts["name"], verts["parents"], verts["roots_of"], determinism_checks, extra_expressible_sets


def LearnSomeInflationGraphParameters(g, inflation_order):
    names, parents_of, roots_of = LearnOriginalGraphParameters(g, hasty=True)
    # print(names)
    graph_structure = list(filter(None, parents_of))
    obs_count = len(graph_structure)
    latent_count = len(parents_of) - obs_count
    root_structure = roots_of[latent_count:]
    inflation_depths = np.array(list(map(len, root_structure)))
    inflationcopies = inflation_order ** inflation_depths
    num_vars = inflationcopies.sum()
    return obs_count, num_vars, names[latent_count:]


def QuickGraphAssessment(g):
    list_of_strings_to_string = lambda l: '['+','.join(l)+']'
    tuples_of_strings_to_string = lambda l: '(' + ','.join(l) + ')'
    names, parents_of, roots_of, determinism_checks, extra_expressible_sets = LearnOriginalGraphParameters(g, hasty=False)
    graph_structure = list(filter(None, parents_of))
    latent_count = len(parents_of) - len(graph_structure)
    print("For the graph who's parental structure is given by:")
    print([':'.join(np.take(names, vals)) + '->' + np.take(names, idx+latent_count) for idx, vals in enumerate(graph_structure)])
    print("We utilize the following ordering of variables: "+list_of_strings_to_string(names))
    print("We identify the following screening-off relationships relevant to enforcing determinism:")
    print("Sets given as (U1s,Y,Xs) with the following meaning:\nYs are screened off from U1s by Xs.")
    for screening in determinism_checks:
        print(tuples_of_strings_to_string(tuple(list_of_strings_to_string(np.take(names,indices).tolist()) for indices in screening)))
    print("We identify the following screening-off non-ai expressible sets:")
    print("Sets given as (Y,Xs,Zs,U3s) with the following meaning:\nYs are screened off from Zs by Xs when U3s is different for (Y,Xs) vs Zs.")
    for screening in extra_expressible_sets:
        print(tuples_of_strings_to_string(tuple(list_of_strings_to_string(np.take(names,indices).tolist()) for indices in screening)))
    print('\u2500'*80+'\n')

if __name__ == '__main__':
    from igraph import Graph
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    [QuickGraphAssessment(g) for g in (InstrumentalGraph,Evans14a,Evans14b,Evans14c,IceCreamGraph,BiconfoundingInstrumental)]
