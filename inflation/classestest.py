#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""
from __future__ import absolute_import
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from itertools import combinations, chain, permutations
import json
from collections import defaultdict
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
from inflation.utilities import MoveToFront, PositionIndex, MoveToBack, SparseMatrixFromRowsPerColumn
from inflation.moseklp import InfeasibilityCertificate
from inflation.mosekinfeas import InfeasibilityCertificateAUTO
from inflation.inflationlp import InflationLP


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

def MixedCardinalityBaseConversion(cardinality, string):
    card=np.array([cardinality[i]**(len(cardinality)-(i+1)) for i in range(len(cardinality))])
    str_to_array=np.array([int(i) for i in string])
    return np.dot(card,str_to_array)
    


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

    def print_assessment(self, wait_for_more = False):
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
        if not wait_for_more:
            print('\u2500' * 80 + '\n')


class InflatedGraph(LatentVariableGraph):
    
    def __init__(self, rawgraph ,inflation_order):
        
        self.inflation_order=inflation_order
        LatentVariableGraph.__init__(self, rawgraph)
        #self.determinism_checks= LatentVariableGraph(g).determinism_checks
        #self.extra_expressible_sets =LatentVariableGraph(g).extra_expressible_sets
        #self.obs_count=LatentVariableGraph(g).observed_count
        #self.latent_count = len(LatentVariableGraph(g).parents_of) - self.obs_count
        self.root_structure = self.roots_of[self.latent_count:]
        self.inflation_depths = np.array(list(map(len, self.root_structure))) #Counts how many roots each random variable has. Should be deprecated upon upgrading to mixed inflation order.
        self.inflation_copies = self.inflation_order ** self.inflation_depths #Counts how many times each random variable is copied.
        #self.inflated_observed_count = self.inflation_copies.sum()
        accumulated = np.add.accumulate(self.inflation_copies)
        self.inflated_observed_count = accumulated[-1]
        self.offsets = np.hstack(([0], accumulated[:-1]))

    @cached_property
    def diagonal_expressible_set(self):
    # offsets=GenerateOffsets(inflation_order,inflation_depths)
        order_range = np.arange(self.inflation_order)
        cannonical_pos = np.empty((self.observed_count, self.inflation_order), dtype=np.int) #uint would give rise to type casting error
        for i in np.arange(self.observed_count):
            cannonical_pos[i] = np.sum(np.outer(self.inflation_order ** np.arange(self.inflation_depths[i]), order_range), axis=0) + \
                                self.offsets[i]
        return cannonical_pos.T.ravel()

    @cached_property
    def inflation_group_generators(self):
        globalstrategyflat = list(np.add(*stuff) for stuff in zip(list(map(np.arange, self.inflation_copies.tolist())), self.offsets))
        reshapings = np.ones((self.observed_count, self.latent_count), np.uint8)
        contractings = np.zeros((self.observed_count, self.latent_count), np.object)
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
        group_generators = np.empty((self.latent_count, inflation_order_gen_count, self.inflated_observed_count), np.int) #uint would give rise to type casting error
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
                    tuple(global_permutation[i][contractings[i]].ravel() for i in np.arange(self.observed_count)))
                group_generators[latent_to_explore, gen_idx] = global_permutation
        return group_generators

    @cached_property
    def inflation_group_elements(self):
        return np.array(dimino_wolfe(self.inflation_group_generators.reshape((-1, self.inflated_observed_count))))

    def _InflateOneDeterminismAssumption(self, screening):
        U1s = screening[0]
        XsY = np.array(list(screening[2]) + list(screening[1])) - self.latent_count
        flatset_original_world = np.take(self.diagonal_expressible_set, XsY)
        symops = self.inflation_group_generators[U1s, 0]  # Now 2d array
        flatset_new_world = np.take(reduce(np.take, symops), flatset_original_world)
        rule = np.vstack((flatset_original_world, flatset_new_world)).T.astype('uint32')
        rule = rule[:-1, :].T.tolist() + rule[-1, :].T.tolist()
        return rule

    @cached_property
    def inflated_determinism_checks(self):
        """
        Recall that a determinism check is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
        Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        return list(map(self._InflateOneDeterminismAssumption, self.determinism_checks))

    def _InflateOneExpressibleSet(self, screening):
        U3s = screening[-1]
        (Ys, Xs, Zs) = tuple(
            map(lambda orig_node_indices: np.take(self.diagonal_expressible_set, np.array(orig_node_indices) - self.latent_count),
                screening[:-1]))  # Feb 2 2021: reindexed as U1s no longer passed to e_eset
        symops = self.inflation_group_generators[U3s, 0]  # Now 2d array
        Zs_new_world = np.take(reduce(np.take, symops), Zs)
        # The ordering of variables here must reflect the ordering used by Find_b
        # We can return it as a flat array.
        variable_ordering = np.argsort(np.hstack(screening[:-1]))
        # nonai_exp_set = (Ys.tolist(),Xs.tolist(),Zs_new_world.tolist()) #Changed ordering
        nonai_exp_set = np.hstack((Ys, Xs, Zs_new_world)).take(variable_ordering)  # Changed ordering
        return nonai_exp_set

    # TODO: Explore larger sets being screened off. What about Ys PLURAL being screened off from U1s? Isn't that worth looking into?
    @cached_property
    def inflated_offdiagonal_expressible_sets(self):
        """
        New function to identify extra expressible sets.
        Recall that a screening relation is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
        Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        return list(map(self._InflateOneExpressibleSet, self.extra_expressible_sets))

    def print_assessment(self):
        super().print_assessment(wait_for_more = True)
        list_of_strings_to_string = lambda l: '[' + ','.join(l) + ']'
        tuples_of_strings_to_string = lambda l: '(' + ','.join(l) + ')'
        print("For inflation order %s:" % self.inflation_order)
        print("The inflated diagonal expressible set is given by:")
        print(self.diagonal_expressible_set)
        print("And we count " + str(len(self.extra_expressible_sets)) + " other expressible sets, namely:")
        for nonai_exp_set in self.inflated_offdiagonal_expressible_sets:
            print(nonai_exp_set)
        print('\u2500' * 80 + '\n')


class ObservationalData:
    def __init__(self, rawdata, cardinality):
        
        if isinstance(rawdata[0],str):#When the input is in the form ['101','100'] for support certification purposes
            numevents = len(rawdata)
            if isinstance(cardinality, int): #When cardinality is specified as an integer
                self.observed_count = len(rawdata[0])
                self.original_card_product= cardinality ** self.observed_count               
                data = np.zeros(self.original_card_product ** self.observed_count)
                data[list(map(lambda s: int(s,cardinality),rawdata))] = 1/numevents
                self.data_flat=data
                self.size = self.data_flat.size
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else: #When cardinalities are specified as a list
                assert isinstance(cardinality, (list, tuple, np.ndarray)), 'Cardinality not given as list of integers.'
                self.observed_count = len(rawdata[0])
                self.original_card_product=np.prod(cardinality)
                data = np.zeros(self.original_card_product ** self.observed_count)
                data[list(map(lambda s: MixedCardinalityBaseConversion(cardinality, s),rawdata))] = 1/numevents
                self.data_flat=data
                self.size = self.data_flat.size
                assert self.observed_count==len(cardinality), 'Cardinality specification does not match the number of observed variables.' 
                self.cardinalities_array = np.array(cardinality)

        
        elif isinstance(rawdata, int):  # When only the number of observed variables is specified, but no actual data, we fake it.
            if isinstance(cardinality, int): #When cardinality is specified as an integer
                self.observed_count = rawdata
                self.original_card_product=cardinality ** self.observed_count
                self.data_flat = np.full(self.original_card_product,1.0/(self.original_card_product))
                self.size = self.data_flat.size
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else: #When cardinalities are specified as a list
                assert isinstance(cardinality, (list, tuple, np.ndarray)), 'Cardinality not given as list of integers.'
                self.observed_count = rawdata
                self.original_card_product=np.prod(cardinality)
                self.data_flat = np.full(self.original_card_product,1.0/self.original_card_product)
                self.size = self.data_flat.size
                assert self.observed_count ==len(cardinality), 'Cardinality specification does not match the number of observed variables.' 
                self.cardinalities_array = np.array(cardinality)
        else:
            self.data_flat = np.array(rawdata).ravel()
            self.size = self.data_flat.size
            norm = np.linalg.norm(self.data_flat, ord=1)
            if norm == 0:
                self.data_flat = np.full(1.0 / self.size, self.size)
            else: #Manual renormalization.
                self.data_flat = self.data_flat / norm
            if isinstance(cardinality, int): #When cardinality is specified as an integer
                self.observed_count = np.rint(np.divide(np.log(self.size),np.log(cardinality))).astype(np.int)
                self.original_card_product=cardinality ** self.observed_count
                assert self.size == self.original_card_product, 'Cardinality of individual variable could not be inferred.'
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else: #When cardinalities are specified as a list
                assert isinstance(cardinality, (list, tuple, np.ndarray)), 'Cardinality not given as list of integers.'
                self.observed_count = len(cardinality)
                self.original_card_product=np.prod(cardinality)
                assert self.size == self.original_card_product, 'Cardinality specification does not match the data.'
                self.cardinalities_array = np.array(cardinality)
        self.cardinalities_tuple = tuple(self.cardinalities_array.tolist())
        self.data_reshaped = np.reshape(self.data_flat,self.cardinalities_tuple)

class InflationProblem(LatentVariableGraph,ObservationalData):
    
    def __init__(self,rawgraph,rawdata,card,inflation_order):
        InflatedGraph.__init__(self, rawgraph,inflation_order)
        ObservationalData.__init__(self,rawdata,card)
        assert isinstance(self.cardinalities_array, all(i == list(self.cardinalities_array)[0] for i in list(card))), 'This class does not support mixed cardinalities yet.'
        self.cardinality = self.cardinalities_array[0] #To be deprecated on upgrade to mixed cardinality

        self.original_cardinalities_array = self.cardinalities_array
        self.original_cardinalities_tuple = self.cardinalities_tuple
        self.original_size = self.size

        self.inflated_cardinalities_array = np.repeat(self.inflation_copies, self.original_cardinalities_array)
        self.inflated_cardinalities_tuple = tuple(self.inflated_cardinalities_array.tolist())
        self.column_count = self.inflated_cardinalities_array.prod()
        self.shaped_column_integers = np.arange(self.column_count).reshape(self.inflated_cardinalities_tuple)

    @cached_property
    def shaped_column_integers_marked(self):
        column_integers_marked = self.shaped_column_integers.copy()
        for detrule in self.inflated_determinism_checks:
            #det rule comes as a list with four elements
            initialtranspose = MoveToFront(self.inflated_observed_count, np.hstack(tuple(detrule)))
            inversetranspose = np.argsort(initialtranspose)
            parents_card_product = self.inflated_cardinalities_array.take(detrule[1]).prod()
            child_cardinality = np.atleast_1d(self.inflated_cardinalities_array.take(detrule[-1])).prod()
            intermediateshape = (parents_card_product, parents_card_product, child_cardinality , child_cardinality , -1);
            column_integers_marked = column_integers_marked.transpose(tuple(initialtranspose)).reshape(intermediateshape)
            for i in np.arange(parents_card_product):
                for j in np.arange(child_cardinality - 1):
                    for k in np.arange(j + 1, child_cardinality):
                        column_integers_marked[i, i, j, k] = -1
            column_integers_marked = column_integers_marked.reshape(self.inflated_cardinalities_tuple).transpose(tuple(inversetranspose))
        return column_integers_marked


    def ValidColumnOrbits(self):
        group_order = len(self.inflation_group_elements)
        AMatrix = np.empty([group_order, self.column_count], np.int)
        AMatrix[0] = self.shaped_column_integers_marked.flat  # Assuming first group element is the identity
        for i in np.arange(1, group_order):
            AMatrix[i] = np.transpose(self.shaped_column_integers_marked, self.inflation_group_elements[i]).flat
        minima = np.amin(AMatrix, axis=0)
        AMatrix = np.compress(minima == np.abs(AMatrix[0]), AMatrix, axis=1)
        return AMatrix

    def EncodedMonomialToRow(self, product):  # Cached in memory, as this function is called by both inflation matrix and inflation vector construction.    
        """
        Parameters
        ----------
        original_cardinality_product : int
            The number of permutations of observable variable configurations given by :math:`(\mbox{cardinality})^{(\mbox{number of observable variables})}`
             
        inflation_order : int
            The order of the inflation matrix.
    
        Returns
        -------
        EncodingMonomialToRow : vector_of_integers
            A numerical vector where each element represents a row and the value of each element represents the index of that row under the symmetry conditions exerted by the cannonical expressible set of the inflated graph.
        
        Notes
        -----
        
        For each row there is a set of symmetric rows produced by the interchange of the copy indecies of the observable variables that create the expressible set. While computing the value of each element of EncodingMonomialToRow, the function chooses the row inside the symmetric set with the smallest index.
        
        Examples
        --------
        For a graph of 3 observable variables, each with cardinality 4:
        
        >>> obs_count=3    
        >>> card=4    
        >>> original_cardinality_product=card**obs_count
        
        With an inflation order of 2:
            
        >>> inflation_order=2    
        >>> EncodingMonomialToRow=GenerateEncodingMonomialToRow(original_cardinality_product,inflation_order)
        
        There will be :math:`4^{3}=4096` rows
        
        >>> print(len(EncodingMonomialToRow))
        >>> 4096
        
        The intechange of the copy indecies of the variables that make up the 6 element cannonical expressible set will produce a 2-fold symmetry and therefore the largest row index inside EncodingMonomialToRow will be :math:`(4096/2)-1=2079`.
        
        >>> print(EncodingMonomialToRow)
        >>> [   0    1    2 ... 2076 2078 2079]
        """
        
        monomial_count = int(product**self.inflation_order)    
        permutation_count = int(np.math.factorial(self.inflation_order))
        MonomialIntegers = np.arange(0, monomial_count, 1, np.uint)
        new_shape = np.full(self.inflation_order, product)
        MonomialIntegersPermutations = np.empty([permutation_count, monomial_count], np.uint)
        IndexPermutations = list(permutations(np.arange(self.inflation_order)))
        MonomialIntegersPermutations[0] = MonomialIntegers
        MonomialIntegers = MonomialIntegers.reshape(new_shape)
        for i in np.arange(1, permutation_count):
            MonomialIntegersPermutations[i] = np.transpose(MonomialIntegers, IndexPermutations[i]).flat
        return PositionIndex(np.amin(
            MonomialIntegersPermutations, axis=0))
 
    def EncodedColumnToMonomial(self,expr_set):
        # Can be used for off-diagonal expressible sets with no adjustment!
        expr_set_size = self.inflated_cardinalities_array.take(expr_set).prod()

        ColumnIntegers = self.shaped_column_integers.transpose(MoveToBack(self.inflated_observed_count, np.array(expr_set))).reshape(
            (-1, expr_set_size))
        EncodingColumnToMonomial = np.empty(self.column_count, np.int)
        EncodingColumnToMonomial[ColumnIntegers] = np.arange(expr_set_size)
        return EncodingColumnToMonomial
    
    def EncodedA(self):
        result = self.EncodedMonomialToRow(self.original_card_product).take(self.EncodedColumnToMonomial(self.diagonal_expressible_set)).take(self.ValidColumnOrbits)
        # Once the encoding is done, the order of the columns can be tweaked at will!
        result.sort(axis=0)  # in-place sort
        return result
    
    def EncodedA_ExtraExpressible(self):

        row_blocks_count=len(self.inflated_offdiagonal_expressible_sets)+1
        results = np.empty(np.hstack((row_blocks_count,self.ValidColumnOrbits.shape)), np.uint32)
        results[0] = self.EncodedMonomialToRow(self.original_card_product).take(self.EncodedColumnToMonomial(self.diagonal_expressible_set)).take(self.ValidColumnOrbits)
        for i in np.arange(1,row_blocks_count):
            #It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
            #In order for names to make sense, I am electing to pass a SORTED version of the flat set, see InflateOneExpressibleSet
            results[i] = self.EncodedColumnToMonomial(self.inflated_offdiagonal_expressible_sets[i-1]).take(self.ValidColumnOrbits)
        accumulated = np.add.accumulate(np.amax(results, axis=(1,2))+1)
        offsets = np.hstack(([0], accumulated[:-1]))
        # Once the encoding is done, the order of the columns can be tweaked at will!
        #result.sort(axis=0)  # in-place sort
        return np.hstack(results+offsets[:, np.newaxis, np.newaxis])
    
    def InflationMatrix(self, extra_expressible=False):
        """
        Parameters
        ----------
        g : igraph.Graph
            The causal graph containing all of the observable and latent variables.
             
        inflation_order : int
            The order of the inflation matrix.
    
        card : int
            The cardinality of every observable variable.
            
        extra_expressible : bool,optional
            If True the rows representing the non-cannonical expressible sets are included in the marginal description matrix (default set to False)
        
        Returns
        -------
        InflationMatrix : scipy.sparse.coo.coo_matrix
            The marginal description matrix in the form of a sparse matrix in COOrdinate format.
        
        Notes
        -----
        
        The columns of the marginal description matrix correspond to different strategies (permutations of the inflated observable variable values) such as:
        
        .. math:: P(A_{1},A_{2},...,A_{N},B_{1},...,B_{N},C_{1},...,C_{N})=P(1,0,1,1,...,0) 
        
        For the triangle scenario with cardinality 2 where :math:`N` is the inflation order. The rows of this matrix correspond to the marginal distributions containing the inflated observable variables in the expressible sets such as:
        
        .. math:: P(A_{1},B_{1},C_{1},A_{4},B_{4},C_{4})=P(0,1,1,0,1,1)
        
        Which is a cannonical expressible set of the same scenario where :math:`N=2`. It can be used in a linear program for an infeasibility certificate (see ``inflation.moseklp.InfeasibilityCertificate``) and/or for a set of inequalities that must be satisfied for the compatibility of the distribution (see ``inflation.certificate.Inequality``).
        
        Examples
        --------
        For the triangle scenario with cardinality 4 and an inflation order of 2:
        
        >>> g=igraph.Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
        >>> card=4
        >>> inflation_order=2
        
        We would expect to obtain a marginal description matrix with 2123776 columns (reduced from :math:`4^{12}=16777216` by imposing the symmetry conditions of the copy indecies on different strategies) and 2080 rows (reduced from :math:`4^{6}=4096` by imposing the same symmetry conditions on the expressible sets):
            
        >>> InfMat = InflationMatrixFromGraph(g, inflation_order, card)
        >>> print(InfMat.shape)
        >>> (2080, 2123776)
        
        """
        
        if extra_expressible:
            return SparseMatrixFromRowsPerColumn(self.EncodedA_ExtraExpressible)
        else:
            return SparseMatrixFromRowsPerColumn(self.EncodedA)
    
    def Numeric_and_Symbolic_b_block_DIAGONAL(self):
        s, idx, counts = np.unique(self.EncodedMonomialToRow(len(self.data_flat)), return_index=True, return_counts=True)
        pre_numeric_b = np.array(self.data_flat)
        numeric_b = pre_numeric_b.copy()
        pre_symbolic_b = np.array(['P(' + ''.join([''.join(str(i)) for i in idx]) + ')' for idx in
                       np.ndindex(self.inflated_cardinalities_tuple)])
        symbolic_b = pre_symbolic_b.copy()
        for i in range(1, self.inflation_order):
            numeric_b = np.kron(pre_numeric_b, numeric_b)
            symbolic_b = [s1+s2 for s1 in pre_symbolic_b for s2 in symbolic_b]
        numeric_b_block = np.multiply(numeric_b.take(idx), counts)
        string_multipliers = ('' if i == 1 else str(i)+'*' for i in counts)
        symbolic_b_block = [s1+s2 for s1,s2 in zip(string_multipliers, np.take(symbolic_b,idx))]
        return numeric_b_block, symbolic_b_block
    
    def Numeric_and_Symbolic_b_block_NON_AI_EXPR(self):
        latent_count = len(self.names) - self.observed_count
        names = self.names[latent_count:]
        all_original_indices = np.arange(self.observed_count)
        Y = list(self.extra_expressible_sets[0])
        X = list(self.extra_expressible_sets[1])
        Z = list(self.extra_expressible_sets[2])
        # It is critical to pass the same ORDER of variables to GenerateEncodingColumnToMonomial and to Find_B_block
        # In order for names to make sense, I am electing to pass a SORTED version of the flat set.
        YXZ = sorted(Y + X + Z)  #see InflateOneExpressibleSet in graphs.py
        lenY = len(Y)
        lenX = len(X)
        lenZ = len(Z)
        lenYXZ = len(YXZ)
    
        np.seterr(divide='ignore')
        marginal_on_XY = np.einsum(self.data_reshaped, all_original_indices, X + Y)
        marginal_on_XZ = np.einsum(self.data_reshaped, all_original_indices, X + Z)
        marginal_on_X = np.einsum(marginal_on_XY, X + Y, X)
        # Y_conditional_on_X = np.einsum(marginal_on_XY, X + Y,
        #                             1 / marginal_on_X, X,
        #                             Y + X)
        # numeric_b_block = np.einsum(marginal_on_XZ, X + Z,
        #                             Y_conditional_on_X, Y + X,
        #                             YXZ).ravel()
        numeric_b_block = np.einsum(marginal_on_XY, X + Y,
                                    marginal_on_XZ, X + Z,
                                    np.divide(1.0, marginal_on_X), X,
                                    YXZ).ravel()
        numeric_b_block[np.isnan(numeric_b_block)] = 0 #Conditioning on zero probability events
        np.seterr(divide='warn')
    
        lowY = np.arange(lenY).tolist()
        lowX = np.arange(lenY, lenY + lenX).tolist()
        lowZ = np.arange(lenY + lenX, lenY + lenX + lenZ).tolist()

        newshape = tuple(self.inflated_cardinalities_array.take(YXZ))
        #newshape = tuple(np.full(lenYXZ, self.cardinality, np.uint8))
        # symbolic_b_block = [
        #     'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowY))).tolist()) + '](' +
        #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowY))]) + ')' +
        #     'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX + lowZ))).tolist()) + '](' +
        #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX + lowZ))]) + ')' +
        #     '/P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
        #     ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
        #     for idYXZ in np.ndindex(newshape)]
        symbolic_b_block = [
            'P[' + ''.join(np.take(names, np.take(YXZ, lowY)).tolist()) + '|' +
            ''.join(np.take(names, np.take(YXZ, lowX)).tolist()) + '](' +
            ''.join([''.join(str(i)) for i in np.take(idYXZ, lowY)]) + '|' +
            ''.join([''.join(str(i)) for i in np.take(idYXZ, lowX)]) + ')' +
            'P[' + ''.join(np.take(names, np.take(YXZ, lowZ)).tolist()) + '|' +
            ''.join(np.take(names, np.take(YXZ, lowX)).tolist()) + '](' +
            ''.join([''.join(str(i)) for i in np.take(idYXZ, lowZ)]) + '|' +
            ''.join([''.join(str(i)) for i in np.take(idYXZ, lowX)]) + ')' +
            'P[' + ''.join(np.take(names, np.take(YXZ, sorted(lowX))).tolist()) + '](' +
            ''.join([''.join(str(i)) for i in np.take(idYXZ, sorted(lowX))]) + ')'
            for idYXZ in np.ndindex(newshape)]
    
        return numeric_b_block, symbolic_b_block
    
    def numeric_and_symbolic_b(self,extra_expressible=False):
        if not extra_expressible:
            return self.Numeric_and_Symbolic_b_block_DIAGONAL
        else:
            numeric_b, symbolic_b = self.Numeric_and_Symbolic_b_block_DIAGONAL
            other_expressible_sets_original = [
                tuple(map(lambda orig_node_indices: np.array(orig_node_indices) - self.latent_count,
                          e_set[:-1])) for e_set in self.extra_expressible_sets]
            #How should we compute the marginal probability?
            #Given P(ABC) how do we obtain P(AB)P(BC)/P(B) as a vector of appropriate length?
            for eset in other_expressible_sets_original:
                #print(tuple(np.take(obs_names, indices).tolist() for indices in eset))
                numeric_b_block, symbolic_b_block = self.Numeric_and_Symbolic_b_block_NON_AI_EXPR
                numeric_b.resize(len(numeric_b) + len(numeric_b_block))
                numeric_b[-len(numeric_b_block):] = numeric_b_block
                symbolic_b.extend(symbolic_b_block)
            return numeric_b, symbolic_b
   
class InflationLP(InflationProblem):
    
    def __init__(self, rawgraph, rawdata, card, inflation_order,extra_ex,solver):
        
        InflationProblem.__init__(self,rawgraph, rawdata, card, inflation_order)
        
        self.numeric_b, self.symbolic_b=self.numeric_and_symbolic_b(extra_expressible=extra_ex)
        self.InfMat=self.InflationMatrix(self, extra_expressible=extra_ex)
        assert isinstance(solver, ('moseklp', 'CVXOPT')), "The accepted solvers are: 'moseklp' and 'CVXOPT'"
        
        if solver == 'moseklp':
            
            self.solve=InfeasibilityCertificate(self.InfMat, self.numeric_b)
            
        elif solver == 'CVXOPT':
            
            self.solve=InflationLP(self.InfMat, self.numeric_b)
        
        self.tol=self.solve['gap']/10 #TODO: Choose better tolerance function. This is yielding false incompatibility claims.
        self.yRaw=np.array(self.solve['x']).ravel()
        
    def WitnessDataTest(self,y):
        IncompTest = (np.amin(y) < 0) and (np.dot(y, self.numeric_b) < self.tol)
        if IncompTest:
            print('Distribution Compatibility Status: INCOMPATIBLE')
        else:
            print('Distribution Compatibility Status: COMPATIBLE')
        return IncompTest

    def ValidityCheck(y, SpMatrix):
        # Smatrix=SpMatrix.toarray()    #DO NOT LEAVE SPARSITY!!
        checkY = csr_matrix(y.ravel()) * SpMatrix
        return checkY.min() >= 0

    def IntelligentRound(self,y, SpMatrix):
        scale = np.abs(np.amin(y))
        n = 1
        y2 = np.rint(n * y / scale).astype(np.int)  # Can I do this with sparse y?
        while not self.ValidityCheck(y, SpMatrix):
            n = n * (n + 1)
            y2 = np.rint(n * y / scale).astype(np.int)
        return y2
    
    def Inequality(self):
        #Modified Feb 2, 2021 to pass B_symbolic as an argument for Inequality
        if self.WitnessDataTest(self.yRaw):
            y = self.IntelligentRound(self.yRaw, self.InfMat)
            # print('Now to make things human readable...')
            indextally = defaultdict(list)
            [indextally[str(val)].append(i) for i, val in enumerate(y) if val != 0]
    
            symboltally = defaultdict(list)
            for i, vals in indextally.items():
                symboltally[i] = np.take(self.symbolic_b,vals).tolist()
    
            #final_ineq_WITHOUT_ZEROS = np.multiply(y[np.nonzero(y)], sy.symbols(np.take(B_symbolic,np.nonzero(y))))
            #Inequality_as_string = '0≤' + "+".join([str(term) for term in final_ineq_WITHOUT_ZEROS]).replace('*P', 'P')
            #Inequality_as_string = Inequality_as_string.replace('+-', '-')
    
            print("Writing to file: 'inequality_output.json'")
    
            returntouser = {
                #'Order of variables': names,
                'Raw rolver output': self.yRaw.tolist(),
                #'Inequality as string': Inequality_as_string,
                'Coefficients grouped by index': indextally,
                'Coefficients grouped by symbol': symboltally,
                # 'b_vector_position': idx.tolist(),
                'Clean solver output': y.tolist()#,
                #'Symbolic association': symbtostring.tolist()
            }
            f = open('inequality_output.json', 'w')
            print(json.dumps(returntouser), file=f)
            f.close()
            return returntouser
        else:
            return print('Compatibility Error: The input distribution is compatible with given inflation order test.')       

class SupportCertificate(InflationProblem):
    
    def __init__(self,rawgraph,rawdata,card,inflation_order,extra_ex):
        
        InfMat=self.InflationMatrix(self, extra_expressible=extra_ex)
        numeric_b, symbolic_b=self.numeric_and_symbolic_b(extra_expressible=extra_ex)
        
        Rows=InfMat.row
        Cols=InfMat.col
        
        ForbiddenRowIdx=np.where(numeric_b[Rows] == 0)[0]
        ForbiddenColumnIdx=np.unique(Cols[ForbiddenRowIdx])
        
        ColumnTemp=np.ones(Cols.max()+1)
        ColumnTemp[ForbiddenColumnIdx]=0
        
        ForbiddenColsZeroMarked=ColumnTemp[Cols]
        
        IdxToRemove=np.where(ForbiddenColsZeroMarked == 0)[0]
        
        NewRows=np.delete(Rows,IdxToRemove)
        NewCols=np.delete(Cols,IdxToRemove)
        NewData=np.ones(len(NewCols),dtype=np.uint)
        
        NewMatrix=coo_matrix((NewData, (NewRows, NewCols)))
        
        NonzeroRows=np.nonzero(numeric_b)[0]
        self.Check=True
        
        for r in NonzeroRows:
            if self.Check:
                
                self.Check=NewMatrix.getrow(r).toarray().any()
    
        if self.Check:
            
            print("Supported")
        
        else:
            
            print("Not Supported")


   #The following is commented out for the later application of mixed cardinality
    #def MarkInvalidStrategies(self,cards, num_var, det_assumptions):
     #   ColumnIntegers = GenShapedColumnIntegers(self.cardinalities_tuple)
      #  for detrule in det_assumptions:
       #     initialtranspose = MoveToFront(num_var, np.hstack(tuple(detrule)))
        #    inversetranspose = np.argsort(initialtranspose)
         #   parentsdimension1=1
          #  for var in detrule[0]:
           #     parentsdimension1=parentsdimension1*cards[var]
           # parentsdimension2=1
           # for var in detrule[1]:
            #    parentsdimension2=parentsdimension2*cards[var]
           # intermediateshape = (parentsdimension1, parentsdimension2, cards[detrule[2]], cards[detrule[3]], -1);
           # ColumnIntegers = ColumnIntegers.transpose(tuple(initialtranspose)).reshape(intermediateshape)
           # for i in np.arange(min(parentsdimension1,parentsdimension2)):
            #    for j in np.arange(cards[detrule[2]] - 1):
             #       for k in np.arange(j + 1, cards[detrule[3]]):
              #          ColumnIntegers[i, i, j, k] = -1
            #ColumnIntegers = ColumnIntegers.reshape(cards).transpose(tuple(inversetranspose))
        #return ColumnIntegers
        

if __name__ == '__main__':
    from igraph import Graph
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    [
        InflatedGraph(g,2).print_assessment() for g in
     (InstrumentalGraph, Evans14a, Evans14b, Evans14c, IceCreamGraph, BiconfoundingInstrumental)]


