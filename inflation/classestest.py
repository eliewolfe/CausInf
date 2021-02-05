#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""
from __future__ import absolute_import
import numpy as np
from itertools import combinations, chain
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property





class LatentVariableGraph:
    # __slots__ = ['g','latent_count','observed_count']


    def __init__(self, rawgraph):
        g = ToRootLexicographicOrdering(rawgraph)
        verts = g.vs
        verts["isroot"] = [0 == i for i in g.indegree()]
        root_vertices = verts.select(isroot=True).indices
        self.latent_count = len(root_vertices)
        nonroot_vertices = verts.select(isroot=False).indices
        self.observed_count = len(nonroot_vertices)

        verts["parents"] = g.get_adjlist('in');
        verts["children"] = g.get_adjlist('out');
        verts["ancestors"] = [g.subcomponent(i, 'in') for i in verts]
        verts["descendants"] = [g.subcomponent(i, 'out') for i in verts]
        self.parents_of = verts["parents"]
        self.children_of = verts["children"]
        self.ancestors_of = verts["ancestors"]
        self.descendants_of = verts["descendants"]


        verts["grandparents"] = g.neighborhood(None, order=2, mode='in', mindist=2)
        self._has_grandparents = [idx for idx, v in enumerate(verts["grandparents"]) if len(v) >= 1]

        verts["roots_of"] = [np.intersect1d(anc, root_vertices).tolist() for anc in verts["ancestors"]]
        self.roots_of = verts["roots_of"]
        self.g = g #Defined LATE, after attributes have been incoporated into g.
        #self.vs = verts
        self.names = verts["name"]
        #self.verts = verts
        self.latent_indices = np.arange(self.latent_count).tolist()
        self.observed_indices = np.arange(self.latent_count, self.observed_count+self.latent_count).tolist()
        #self.latent_variables = self.vs[self.latent_indices]
        #self.observed_variables = self.vs[self.observed_indices]


    def Root_Subsets(self,v):
        "v is presumed to be a iGraph vertex object."
        screenable_roots = np.setdiff1d(v["roots_of"], v["parents"])
        return [self.verts[subroots] for r in np.arange(1, screenable_roots.size + 1) for subroots in
                combinations(screenable_roots, r)]

    def RootIndices_Subsets(self,v):
        "v is presumed to be an integer specifying some node."
        screenable_roots = np.setdiff1d(self.roots_of[v], self.parents_of[v])
        # return [subroots for r in np.arange(1, screenable_roots.size + 1) for subroots in combinations(screenable_roots, r)]
        return chain.from_iterable(combinations(screenable_roots, r) for r in np.arange(1, screenable_roots.size + 1))

    def _identify_determinism_check(self, root_indices, observed_index):
        """
        root_indices is a list of root nodes (integers) which can be screened off.
        observed_index is a single node (integer).
        The output will be a tuple of 3 lists
        (U1s,Ys,Xs) with the following meaning: Ys are screened off from U1s by Xs.
        """
        parents_of_observed = set(self.parents_of[observed_index])
        [self.descendants_of[v] for v in root_indices]
        descendants_of_roots = [self.descendants_of[v] for v in root_indices]
        descendants_of_roots = set().union(*descendants_of_roots)
        U1s = list(root_indices)
        Y = observed_index
        Xs = list(parents_of_observed.intersection(descendants_of_roots))
        return (U1s, [Y], Xs)

    @cached_property
    def determinism_checks(self):
        return [self._identify_determinism_check(roots_subset, v)
                              for v in self._has_grandparents
                              for roots_subset in self.RootIndices_Subsets(v)
                              ]

    def _identify_expressible_set(self, root_indices, observed):
        """
        root_indices is a list of root nodes (integers) which can be screened off.
        observed_index is a single node (integer).
        The output will be a tuple of 4 lists
        (Ys,Xs,Zs,U3s) with the following meaning:
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        children_of_roots = [self.children_of[v] for v in root_indices]
        children_of_roots = set().union(*children_of_roots)
        screeningset = children_of_roots.intersection(self.ancestors_of[observed])
        Xs = screeningset.copy()
        for sidx in screeningset:
            screeningset_rest = screeningset.copy()
            screeningset_rest.remove(sidx)
            # unblocked_path if screeningset_rest.isdisjoint(directed_path)
            # sidx is redundant if there are not ANY unblocked paths.
            if not any(screeningset_rest.isdisjoint(directed_path) for directed_path in
                    # self.g.get_all_simple_paths(sidx, to=self.vs[observed])):
                    self.g.get_all_simple_paths(sidx, to=observed)):
                Xs.remove(sidx)

        U1s = set(root_indices)
        Y = observed
        roots_of_Xs = [self.roots_of[x] for x in Xs]
        U2s = set().union(*roots_of_Xs).difference(U1s)

        U2s_descendants = [self.descendants_of[u] for u in U2s]
        U2s_descendants = set().union(*U2s_descendants)
        observable_nodes_aside_from_Zs = set(Xs)
        observable_nodes_aside_from_Zs.add(Y)
        Zs = set(self.observed_indices).difference(U2s_descendants).difference(observable_nodes_aside_from_Zs)

        roots_of_Y_aside_from_U1s = set(self.roots_of[Y]).difference(U1s)

        roots_of_Zs = [self.roots_of[z] for z in Zs]
        roots_of_Zs = set().union(*roots_of_Zs)
        U3YZ = roots_of_Y_aside_from_U1s.intersection(roots_of_Zs)
        # Adding a sanity filter:
        if len(U3YZ) == 0:
            Zs = set()

        return tuple(map(list, ([Y], Xs, Zs, U3YZ)))

    @cached_property
    def extra_expressible_sets(self):
        return list(filter(lambda screening: len(screening[-1]) > 0,
                                              [self._identify_expressible_set(roots_subset, v)
                                               for v in self._has_grandparents
                                               for roots_subset in self.RootIndices_Subsets(v)
                                               ]))

    def __str__(self):
        "Convert to string, for str()."
        return str([':'.join(np.take(self.names, vals)) + '->' + np.take(self.names, idx + self.latent_count) for idx, vals in
               enumerate(self.parents_of[-self.observed_count:])])

    def print_assessment(self):
        list_of_strings_to_string = lambda l: '[' + ','.join(l) + ']'
        tuples_of_strings_to_string = lambda l: '(' + ','.join(l) + ')'
        print("For the graph who's parental structure is given by:")
        print(str(self))
        print("We utilize the following ordering of latent variables: " + list_of_strings_to_string(self.names[:self.latent_count]))
        print("We utilize the following ordering of observed variables: " + list_of_strings_to_string(self.names[-self.observed_count:]))
        print("We identify the following screening-off relationships relevant to enforcing determinism:")
        print("Sets given as (U1s,Ys,Xs) with the following meaning:\tYs are screened off from U1s by Xs.")
        for screening in self.determinism_checks:
            print(tuples_of_strings_to_string(
                tuple(list_of_strings_to_string(np.take(self.names, indices).tolist()) for indices in screening)))
        print("We identify the following screening-off non-ai expressible sets:")
        print(
            "Sets given as (Y,Xs,Zs,U3s) with the following meaning:\nYs are screened off from Zs by Xs when U3s is different for (Y,Xs) vs Zs.")
        for screening in self.extra_expressible_sets:
            print(tuples_of_strings_to_string(
                tuple(list_of_strings_to_string(np.take(self.names, indices).tolist()) for indices in screening)))
        print('\u2500' * 80 + '\n')










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






if __name__ == '__main__':
    from igraph import Graph
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    [LatentVariableGraph(g).print_assessment() for g in
     (InstrumentalGraph, Evans14a, Evans14b, Evans14c, IceCreamGraph, BiconfoundingInstrumental)]


