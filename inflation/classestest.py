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
from functools import reduce 
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from inflation.dimino import dimino_wolfe
from inflation.utilities import MoveToFront



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
        list_extract_and_union = lambda list_of_lists, indices: set().union(
            chain.from_iterable(list_of_lists[v] for v in indices))
        parents_of_observed = set(self.parents_of[observed_index])
        #descendants_of_roots = [self.descendants_of[v] for v in root_indices]
        #descendants_of_roots = set().union(*descendants_of_roots)
        descendants_of_roots = list_extract_and_union(self.descendants_of, root_indices)
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
        list_extract_and_union = lambda list_of_lists, indices: set().union(
            chain.from_iterable(list_of_lists[v] for v in indices))
        children_of_roots = list_extract_and_union(self.children_of, root_indices)
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

        U2s_descendants = list_extract_and_union(self.descendants_of, U2s)
        observable_nodes_aside_from_Zs = set(Xs)
        observable_nodes_aside_from_Zs.add(Y)
        Zs = set(self.observed_indices).difference(U2s_descendants).difference(observable_nodes_aside_from_Zs)

        roots_of_Y_aside_from_U1s = set(self.roots_of[Y]).difference(U1s)

        roots_of_Zs = list_extract_and_union(self.roots_of, Zs)
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


class InflatedGraph:
    
    def __init__(self,g ,inflation_order, extra_expressible=False, debug=False):
        
        self.inflation_order=inflation_order
        self.determinism_checks= LatentVariableGraph(g).determinism_checks
        self.extra_expressible_sets =LatentVariableGraph(g).extra_expressible_sets  
        self.obs_count=LatentVariableGraph(g).observed_count 
        self.latent_count = len(LatentVariableGraph(g).parents_of) - self.obs_count
        self.root_structure = LatentVariableGraph(g).roots_of[self.latent_count:]
        self.inflation_depths = np.array(list(map(len, self.root_structure)))
        self.inflationcopies = inflation_order ** self.inflation_depths
        self.num_vars = self.inflationcopies.sum()
        accumulated = np.add.accumulate(inflation_order ** self.inflation_depths)
        self.offsets = np.hstack(([0], accumulated[:-1]))
        
    def CanonicalExpressibleSet(self):
    # offsets=GenerateOffsets(inflation_order,inflation_depths)
        order_range = np.arange(self.inflation_order)
        cannonical_pos = np.empty((self.obs_count, self.inflation_order), dtype=np.int) #uint would give rise to type casting error
        for i in np.arange(self.obs_count):
            cannonical_pos[i] = np.sum(np.outer(self.inflation_order ** np.arange(self.inflation_depths[i]), order_range), axis=0) + \
                                self.offsets[i]
        return cannonical_pos.T.ravel()

    def GroupGenerators(self):
        globalstrategyflat = list(np.add(*stuff) for stuff in zip(list(map(np.arange, self.inflationcopies.tolist())), self.offsets))
        reshapings = np.ones((self.obs_count, self.latent_count), np.uint8)
        contractings = np.zeros((self.obs_count, self.latent_count), np.object)
        for idx, elem in enumerate(self.root_structure):
            reshapings[idx][elem] = self.inflation_order
            contractings[idx][elem] = np.s_[:]
        reshapings = list(map(tuple, reshapings))
        contractings = list(map(tuple, contractings))
        globalstrategyshaped = list(np.reshape(*stuff) for stuff in zip(globalstrategyflat, reshapings))
        fullshape = tuple(np.full(self.latent_count, self.inflation_order))
        if self.inflation_order == 2:
            inflation_order_gen_count = 1
        else:
            inflation_order_gen_count = 2
        group_generators = np.empty((self.latent_count, self.inflation_order_gen_count, self.num_vars), np.int) #uint would give rise to type casting error
        for latent_to_explore in np.arange(self.latent_count):
            for gen_idx in np.arange(inflation_order_gen_count):
                initialtranspose = MoveToFront(self.latent_count, np.array([latent_to_explore]))
                inversetranspose = np.hstack((np.array([0]), 1 + np.argsort(initialtranspose)))
                label_permutation = np.arange(self.inflation_order)
                if gen_idx == 0:
                    label_permutation[np.array([0, 1])] = np.array([1, 0])
                elif gen_idx == 1:
                    label_permutation = np.roll(label_permutation, 1)
                global_permutation = np.array(list(
                    np.broadcast_to(elem, fullshape).transpose(tuple(initialtranspose))[label_permutation] for elem in
                    globalstrategyshaped))
                global_permutation = np.transpose(global_permutation, tuple(inversetranspose))
                global_permutation = np.hstack(
                    tuple(global_permutation[i][contractings[i]].ravel() for i in np.arange(self.obs_count)))
                group_generators[latent_to_explore, gen_idx] = global_permutation
        return group_generators

    def InflatedDeterminismAssumptions(self, group_generators, exp_set):
        """
        Recall that a determinism check is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
        Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        def InflateOneDeterminismAssumption(self,screening):
            U1s = screening[0]
            XsY = np.array(list(screening[2])+list(screening[1])) - self.latent_count
            flatset_original_world = np.take(exp_set, XsY)
            symops = group_generators[U1s, 0]  # Now 2d array
            flatset_new_world = np.take(reduce(np.take, symops), flatset_original_world)
            rule = np.vstack((flatset_original_world, flatset_new_world)).T.astype('uint32')
            rule = rule[:-1, :].T.tolist() + rule[-1, :].T.tolist()
            return rule
    
        return list(map(InflateOneDeterminismAssumption, self.determinism_checks))




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


