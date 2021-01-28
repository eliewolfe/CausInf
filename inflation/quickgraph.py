#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""

import numpy as np
from igraph import Graph


def ToTopologicalOrdering(g):
    return g.permute_vertices(np.argsort(g.topological_sorting('out')).tolist())


def LearnParametersFromGraph(origgraph, hasty=False):
    g = ToTopologicalOrdering(origgraph)
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

        def DeterminismCheckAndExpressibleSet(roots, observed):
            """roots is a list of root nodes which can be screened off, observed is a single node, in iGraph.VertexSeq format.
            The output will be a tuple of 5 lists (except Y is integer, not list):
            (U1s,Y,Xs,Zs,U3s) with the following meaning:
            Ys are screened off from U1s by Xs.
            Zs are variables appearing in an expressible set with {Xs,Y} when U3s is different for Xs and Zs)
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
            #Xs = screeningset
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

            #return (U1s,Y,Xs,Zs,U3YZ)
            return tuple(map(list,(U1s, [Y], Xs, Zs, U3YZ)))

        from itertools import chain, combinations
        def PossibleScreenings(v):
            "v is presumed to be a iGraph vertex object."
            screenable_roots = np.setdiff1d(v["roots_of"], v["parents"])
            return [DeterminismCheckAndExpressibleSet(verts[subroots], v) for r in np.arange(1, screenable_roots.size + 1) for subroots in combinations(screenable_roots, r)]

        determinism_checks_and_expressible_sets = list(chain.from_iterable([PossibleScreenings(v) for v in g.vs[has_grandparents]]))

        return verts["name"], verts["parents"], verts["roots_of"], determinism_checks_and_expressible_sets


def LearnSomeInflationGraphParameters(g, inflation_order):
    names, parents_of, roots_of = LearnParametersFromGraph(g, hasty=True)
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
    names, parents_of, roots_of, screening_off_relationships = LearnParametersFromGraph(g, hasty=False)
    graph_structure = list(filter(None, parents_of))
    print("For the graph who's parental structure is given by:")
    print([':'.join(np.take(names, vals)) + '->' + np.take(names, idx) for idx, vals in enumerate(graph_structure)])
    print("We identify the following screening-off relationship relevant to enforcing determinism and expressible sets:")
    for screening in screening_off_relationships:
        print(tuple(np.take(names,{
            set: lambda s: list(s),
            int: lambda s: [s],
            list: lambda s: s}[type(indices)](indices)).tolist() for indices in screening))
    print('\u2500'*80+'\n')

if __name__ == '__main__':
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    [QuickGraphAssessment(g) for g in (InstrumentalGraph,Evans14a,Evans14b,Evans14c,IceCreamGraph,BiconfoundingInstrumental)]
