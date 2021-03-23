#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""
from __future__ import absolute_import
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import vstack as sparse_vstack
import itertools
import json
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
from internal_functions.inequality_internals import *
from internal_functions.groups import dimino_sympy,dimino_wolfe, minimize_object_under_group_action, orbits_of_object_under_group_action
from internal_functions.utilities import MoveToFront, MoveToBack, SparseMatrixFromRowsPerColumn
from linear_program_options.moseklp import InfeasibilityCertificate
from linear_program_options.moseklp_dual import InfeasibilityCertificateAUTO
from linear_program_options.cvxopt import InflationLP

#from methodtools import lru_cache


class LatentVariableGraph:

    @staticmethod
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
        optimal_root_node_indices = np.take(root_vertices_igraph.indices, np.argsort(root_vertices_igraph["name"]))
        optimal_nonroot_node_indices = np.take(nonroot_vertices_igraph.indices,
                                               np.argsort(nonroot_vertices_igraph["name"]))
        new_ordering = np.hstack((optimal_root_node_indices, optimal_nonroot_node_indices))
        return g.permute_vertices(np.argsort(new_ordering).tolist())

    def __init__(self, rawgraph):
        g = self.ToRootLexicographicOrdering(rawgraph)
        verts = g.vs
        verts["isroot"] = [0 == i for i in g.indegree()]
        root_vertices = verts.select(isroot=True).indices
        self.latent_count = len(root_vertices)
        nonroot_vertices = verts.select(isroot=False).indices
        self.observed_count = len(nonroot_vertices)

        verts["parents"] = g.get_adjlist('in')
        verts["children"] = g.get_adjlist('out')
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
        self.g = g  # Defined LATE, after attributes have been incoporated into g.
        self.names = verts["name"]
        self.latent_indices = np.arange(self.latent_count).tolist()
        self.observed_indices = np.arange(self.latent_count, self.observed_count + self.latent_count).tolist()
        self.observed_names = self.names[self.latent_count:]
        self.latent_names = self.names[:self.latent_count]

        # self.latent_variables = self.vs[self.latent_indices]
        # self.observed_variables = self.vs[self.observed_indices]

    def RootIndices_Subsets(self, v):
        "v is presumed to be an integer specifying some node."
        screenable_roots = np.setdiff1d(self.roots_of[v], self.parents_of[v])
        return itertools.chain.from_iterable(itertools.combinations(screenable_roots, r) for r in np.arange(1, screenable_roots.size + 1))

    def _identify_determinism_check(self, root_indices, observed_index):
        """
        root_indices is a list of root nodes (integers) which can be screened off.
        observed_index is a single node (integer).
        The output will be a tuple of 3 lists
        (U1s,Ys,Xs) with the following meaning: Ys are screened off from U1s by Xs.
        """
        list_extract_and_union = lambda list_of_lists, indices: set().union(
            itertools.chain.from_iterable(list_of_lists[v] for v in indices))
        parents_of_observed = set(self.parents_of[observed_index])
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
            itertools.chain.from_iterable(list_of_lists[v] for v in indices))
        children_of_roots = list_extract_and_union(self.children_of, root_indices)
        screeningset = children_of_roots.intersection(self.ancestors_of[observed])
        Xs = screeningset.copy()
        for sidx in screeningset:
            screeningset_rest = screeningset.copy()
            screeningset_rest.remove(sidx)
            # sidx is redundant if there are not ANY unblocked paths.
            if not any(screeningset_rest.isdisjoint(directed_path) for directed_path in
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
        """Convert to string, for str()."""
        return str(
            [':'.join(np.take(self.names, vals)) + '->' + np.take(self.names, idx + self.latent_count) for idx, vals in
             enumerate(self.parents_of[-self.observed_count:])])

    def print_assessment(self, wait_for_more=False):
        list_of_strings_to_string = lambda l: '[' + ','.join(l) + ']'
        tuples_of_strings_to_string = lambda l: '(' + ','.join(l) + ')'
        print("For the graph who's parental structure is given by:")
        print(str(self))
        print("We utilize the following ordering of latent variables: " + list_of_strings_to_string(
            self.names[:self.latent_count]))
        print("We utilize the following ordering of observed variables: " + list_of_strings_to_string(
            self.names[-self.observed_count:]))
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

    def __init__(self, rawgraph, inflation_order):
        LatentVariableGraph.__init__(self, rawgraph)

        if isinstance(inflation_order, int):
            self.inflations_orders = np.full(self.latent_count, inflation_order)
        else:  # When inflation_order is specified as a list
            if not isinstance(inflation_order, (list, tuple, np.ndarray)):
                raise TypeError("Inflation orders not given as list of integers.")
            self.inflations_orders = np.array(inflation_order)
        self.min_inflation_order = self.inflations_orders.min()  # Should be deprecated after upgrade to diagonal expressible set
        self.max_inflation_order = self.inflations_orders.max()

        self.determinism_checks = list(
            filter(lambda screening: all(U >= 2 for U in self.inflations_orders[screening[0]]),
                   self.determinism_checks))
        self.extra_expressible_sets = list(
            filter(lambda screening: all(U >= 2 for U in self.inflations_orders[screening[-1]]),
                   self.extra_expressible_sets))

        self.root_structure = self.roots_of[self.latent_count:]

        self._latent_ancestors_cardinalities_of = [self.inflations_orders.take(latent_parents) for latent_parents in
                                                   self.root_structure]
        self.inflation_copies = np.fromiter(map(np.prod, self._latent_ancestors_cardinalities_of), np.int)
        self.inflation_minima = np.fromiter(map(np.amin, self._latent_ancestors_cardinalities_of), np.int)
        self.inflation_depths = np.fromiter(map(len, self._latent_ancestors_cardinalities_of), np.int)

        self.original_observed_indices = np.asanyarray(self.observed_indices) - self.latent_count
        self.original_observed_names = self.observed_names
        self.from_inflation_indices = np.repeat(self.original_observed_indices, self.inflation_copies)
        self.inflated_observed_names = np.repeat(self.original_observed_names, self.inflation_copies)

        accumulated = np.add.accumulate(self.inflation_copies)
        self.inflated_observed_count = accumulated[-1]
        
        self.offsets = np.hstack(([0], accumulated[:-1]))
        self._canonical_pos = [
            np.outer(inflation_minimum ** np.arange(inflation_depth), np.arange(inflation_minimum)).sum(axis=0) + offset
            for inflation_minimum, inflation_depth, offset
            in zip(self.inflation_minima, self.inflation_depths, self.offsets)]
        self.canonical_world = np.fromiter((pos[0] for pos in self._canonical_pos), np.int)

    #I'm about to overhaul the diagonal expressible set code. We will just use the canonical positions. The art, hereafter, will be obtaining symmetry group elements.

    @property
    def partitioned_expressible_set(self):
        return [np.compress(np.add(part, 1).astype(np.bool), part)
                for part in itertools.zip_longest(*self._canonical_pos, fillvalue=-1)]

    #Suppose the canonical positions are [(0,3),(6,10,14),(15,18)]
    #Then, the *paritioned* expressible set will be [(0,6,15),(3,10,18),(14)]
    #The FLAT SORTED eset will be [0,3,6,10,14,15,18]
    #The FLAT SORTED symmetry group will be generated by [(0,3)(1,4)(2,5)] i.e. [3,4,5,0,1,2,6]

    @cached_property
    def diagonal_expressible_set(self):
        temp = np.hstack(self.partitioned_expressible_set)
        temp.sort()
        return temp


    @cached_property
    def diagonal_expressible_set_symmetry_group(self):
        when_sorted = np.argsort(np.hstack(self.partitioned_expressible_set))
        it = iter(when_sorted)
        _canonical_pos2 = [list(itertools.islice(it, size)) for size in self.inflation_minima]

        unfiltered_variants = np.array([np.hstack(np.vstack(perm).T) for perm in
                                        itertools.permutations(itertools.zip_longest(*_canonical_pos2, fillvalue=-1))])
        expressible_set_variants_filter = np.add(unfiltered_variants, 1).astype(np.bool)
        unfiltered_variants = unfiltered_variants.compress(
            (expressible_set_variants_filter == np.atleast_2d(expressible_set_variants_filter[0])).all(axis=1),
            axis=0)
        filtered_variants = np.array([eset.compress(np.add(eset, 1).astype(np.bool)) for eset in unfiltered_variants])
        return np.take_along_axis(np.argsort(filtered_variants,axis=1), np.atleast_2d(when_sorted),   axis=1)

    # @cached_property
    # def expressible_set_variants(self):
    #     unfiltered_variants = np.array([np.hstack(np.vstack(perm).T) for perm in
    #                                     itertools.permutations(zip(*itertools.zip_longest(*self._canonical_pos, fillvalue=-1)))])
    #     expressible_set_variants_filter = np.add(unfiltered_variants, 1).astype(np.bool)
    #     unfiltered_variants = unfiltered_variants.compress(
    #         (expressible_set_variants_filter == np.atleast_2d(expressible_set_variants_filter[0])).all(axis=1),
    #         axis=0)
    #     return np.array([eset.compress(np.add(eset, 1).astype(np.bool)) for eset in unfiltered_variants])
    #
    # @cached_property
    # def diagonal_expressible_set(self):
    #     return self.expressible_set_variants[0]
    #
    # @property
    # def partitioned_expressible_set(self):
    #     return [np.compress(np.add(part, 1).astype(np.bool), part)
    #             for part in itertools.zip_longest(*self._canonical_pos, fillvalue=-1)]
    #
    # @property
    # def diagonal_expressible_set_symmetry_group(self):
    #     core_orders = np.asarray(list(map(PositionIndex,self.expressible_set_variants)))
    #     return np.take_along_axis(core_orders, np.atleast_2d(np.argsort(core_orders[0])),
    #                               axis=1)
    # I know I can do this more efficiently, but I don't care at the moment.
    # return map(lambda v : np.argsort(v).take(core_order), self.expressible_set_variants[1:])

    @cached_property
    def inflation_group_generators(self):
        # Upgrade to mixed inflation order
        globalstrategyflat = list(
            np.add(*stuff) for stuff in zip(list(map(np.arange, self.inflation_copies.tolist())), self.offsets))
        reshapings = np.ones((self.observed_count, self.latent_count), np.uint8)
        contractings = np.zeros((self.observed_count, self.latent_count), np.object)
        for idx, latent_ancestors in enumerate(self.root_structure):
            reshapings[idx][latent_ancestors] = self.inflations_orders[latent_ancestors]
            contractings[idx][latent_ancestors] = np.s_[:]
        reshapings = map(tuple, reshapings)
        contractings = map(tuple, contractings)
        globalstrategyshaped = list(np.reshape(*stuff) for stuff in zip(globalstrategyflat, reshapings))
        gloablstrategybroadcast = np.stack(np.broadcast_arrays(*globalstrategyshaped), axis=0)
        indices_to_extract = np.hstack(tuple(shaped_elem[contraction].ravel() for shaped_elem, contraction in zip(
            np.arange(gloablstrategybroadcast.size).reshape(gloablstrategybroadcast.shape), contractings)))
        group_generators = []
        for latent_to_explore, inflation_order_for_U in enumerate(self.inflations_orders):
            generator_count_for_U = np.minimum(inflation_order_for_U, 3) - 1
            group_generators_for_U = np.empty((generator_count_for_U, self.inflated_observed_count), np.int)
            # Maybe assert that inflation order must be a strictly positive integer?
            for gen_idx in np.arange(generator_count_for_U):
                initialtranspose = MoveToFront(self.latent_count + 1, np.array([latent_to_explore + 1]))
                inversetranspose = np.argsort(initialtranspose)
                label_permutation = np.arange(inflation_order_for_U)
                if gen_idx == 0:
                    label_permutation[:2] = [1, 0]
                elif gen_idx == 1:
                    label_permutation = np.roll(label_permutation, 1)
                group_generators_for_U[gen_idx] = gloablstrategybroadcast.transpose(
                    tuple(initialtranspose))[label_permutation].transpose(
                    tuple(inversetranspose)).flat[indices_to_extract]
            group_generators.append(group_generators_for_U)
        
        #print('group gens:')
        #print(group_generators)
        #print('----------')
        return group_generators
    
    @cached_property
    def inflation_group_elements(self):
        #return np.array(dimino_wolfe(
         #   np.vstack(self.inflation_group_generators)))
         return np.array(dimino_sympy([gen.ravel() for gen in self.inflation_group_generators]))

    @property
    def _InflateOneDeterminismAssumption(self):
        for screening in self.determinism_checks:
            U1s = screening[0]
            XsY = np.array(list(screening[2]) + list(screening[1])) - self.latent_count
            flatset_original_world = np.take(self.canonical_world, XsY)
            symops = [self.inflation_group_generators[U1][0] for U1 in U1s]  # Now 2d array
            flatset_new_world = np.take(reduce(np.take, symops), flatset_original_world)
            rule = np.vstack((flatset_original_world, flatset_new_world)).T.astype('uint32')
            rule = rule[:-1, :].T.tolist() + rule[-1, :].T.tolist()
            yield rule

    @cached_property
    def inflated_determinism_checks(self):
        """
        Recall that a determinism check is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
        Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        return list(self._InflateOneDeterminismAssumption)

    @property
    def _InflateOneExpressibleSet(self):
        for screening in self.extra_expressible_sets:
            U3s = screening[-1]
            (Ys, Xs, Zs) = tuple(
                map(lambda orig_node_indices: np.take(self.canonical_world,
                                                      np.array(orig_node_indices) - self.latent_count),
                    screening[:-1]))
            symops = np.array([self.inflation_group_generators[U3][0] for U3 in U3s])
            Zs_new_world = np.take(reduce(np.take, symops), Zs)
            variable_ordering = np.argsort(np.hstack(screening[:-1]))
            nonai_exp_set = np.hstack((Ys, Xs, Zs_new_world)).take(variable_ordering)
            yield nonai_exp_set

    # TODO: Explore larger sets being screened off. What about Ys PLURAL being screened off from U1s? Isn't that worth looking into?
    @cached_property
    def inflated_offdiagonal_expressible_sets(self):
        """
        New function to identify extra expressible sets.
        Recall that a screening relation is passed in the form of (U1s,Ys,Xs,Zs,U3s) with the following meaning:
        Ys are screened off from U1s by Xs. (Ys is always a list with only one element.)
        Zs are variables appearing in an expressible set with {Xs,Ys} when U3s is different for Xs and Zs)
        """
        return list(self._InflateOneExpressibleSet)

 
    class ExpressibleSet:
        # WORK IN PROGRESS
        def __init__(self, partitioned_eset_inflated_indices, composition_rule, from_inflation_indices, observed_names, original_observed_indices, symmetry_group=[]):
            self.partitioned_eset_inflated_indices = partitioned_eset_inflated_indices
            self.composition_rule = composition_rule
            self.symmetry_group = symmetry_group #NOTE: The symmetry group must act on the FLAT version of the eSET. I'm not sure we have this at the moment.
            self.from_inflation_indices = from_inflation_indices
            self.observed_names = observed_names
            self.all_original_indices = original_observed_indices

            self.partition_eset_original_indices = list(map(self.from_inflation_indices.take, self.partitioned_eset_inflated_indices))

            self.flat_eset, self._low_indices_flat_uncompressed = np.unique(
                np.hstack(self.partitioned_eset_inflated_indices), return_inverse=True)

            self._posts = [str(rule).replace('1', '').replace('-1', '^(-1)') for rule in self.composition_rule]

            self.symbolic_wrappers = ['P[' + ''.join(np.take(self.observed_names, sub_eset).tolist()) + ']{}' + post for
                                      sub_eset, post in zip(self.partition_eset_original_indices, self._posts)]



        #Generating the vector of symbolic probabilities
        @property
        def low_indices(self):
            it = iter(self._low_indices_flat_uncompressed)
            return [list(itertools.islice(it, size)) for size in map(len, self.partitioned_eset_inflated_indices)]

        def _symbolic_element_from_intlist(self, int_list):
            return ''.join(
                wrapper.format(np.take(int_list,low_sub_eset)) for wrapper, low_sub_eset in zip(self.symbolic_wrappers, self.low_indices))
        def Generate_symbolic_b_block(self, data_shape):
            return list(map(self._symbolic_element_from_intlist, np.ndindex(tuple(np.take(data_shape, self.flat_eset)))))



        # Generating the vector of numerical probabilities
        @staticmethod
        def _mult_or_div(rule, tensor):
            if rule==-1:
                np.seterr(divide='ignore')
                newtensor = np.true_divide(1,tensor)
                newtensor[np.isnan(newtensor)] = 0  # Conditioning on zero probability events
                np.seterr(divide='warn')
                return newtensor
            else:
                return tensor
        def Generate_numeric_b_block(self, data_reshaped):
            print('-----------')
            print(data_reshaped)
            print(self.all_original_indices)
            print(self.partition_eset_original_indices)
            print('-----------')
            marginals = (np.einsum(data_reshaped, self.all_original_indices, sub_eset) for sub_eset in self.partition_eset_original_indices)
            marginals = map(self._mult_or_div, self.composition_rule, marginals)

            einsumargs = list(itertools.chain.from_iterable(zip(marginals,self.partitioned_eset_inflated_indices)))
            einsumargs.append(self.flat_eset)
            return np.einsum(*einsumargs)



        # Getting ready to generate the inflation matrix
        def Columns_to_unique_rows(self, shaped_column_integers):
            data_shape = shaped_column_integers.shape
            # Can be used for off-diagonal expressible sets with no adjustment!
            expr_set_size = np.take(data_shape, self.flat_eset).prod()

            reshaped_column_integers = shaped_column_integers.transpose(
                MoveToBack(len(data_shape), self.flat_eset)).reshape(
                (-1, expr_set_size))
            encoding_of_columns_to_monomials = np.empty(shaped_column_integers.size, np.int)
            encoding_of_columns_to_monomials[reshaped_column_integers] = np.arange(expr_set_size)
            return encoding_of_columns_to_monomials

        #I really want a function which allows me to delete certain rows based on symmetry...
        # def Which_rows_to_keep(self, data_shape):
        #     shape_of_eset = np.take(data_shape, self.flat_eset)
        #     size_of_eset = shape_of_eset.prod()
        #     rows_as_integers = np.arange(size_of_eset).reshape(shape_of_eset)
        #     minimize_object_under_group_action(rows_as_integers,
        #     self.symmetry_group, skip=1)
        #     return np.unique(rows_as_integers.ravel(), return_index=True)[1]
        #
        # def Discarded_rows_to_the_back(self, data_shape):
        #     shape_of_eset = np.take(data_shape, self.flat_eset)
        #     size_of_eset = shape_of_eset.prod()
        #     valid = self.Which_rows_to_keep(data_shape)
        #     lenvalid = len(valid)
        #     result = np.full(size_of_eset, lenvalid, dtype=np.int)
        #     np.put(result, valid, np.arange(len(valid)))
        #     return result


    @cached_property
    def expressible_sets(self):
        results = [self.ExpressibleSet(
            self.partitioned_expressible_set,
            np.ones(len(self.partitioned_expressible_set), dtype=np.int),
            self.from_inflation_indices, self.observed_names, self.original_observed_indices,
            symmetry_group = self.diagonal_expressible_set_symmetry_group
        )]

        for non_ai_eset in self.inflated_offdiagonal_expressible_sets:

            X = list([non_ai_eset[1]])
            Y = list([non_ai_eset[0]])
            Z = list([non_ai_eset[2]])
            results.append(self.ExpressibleSet(
                    [sorted(X+Y), sorted(X+Z), sorted(X)],
                    [+1, +1, -1],
            self.from_inflation_indices, self.observed_names, self.observed_indices))
        return results


    def print_assessment(self):
        super().print_assessment(wait_for_more=True)
        list_of_strings_to_string = lambda l: '[' + ','.join(l) + ']'
        tuples_of_strings_to_string = lambda l: '(' + ','.join(l) + ')'
        print("For inflation order %s:" % self.inflations_orders)
        print("The inflated diagonal expressible set is given by:")
        print(self.diagonal_expressible_set)
        print("And we count " + str(len(self.extra_expressible_sets)) + " other expressible sets, namely:")
        for nonai_exp_set in self.inflated_offdiagonal_expressible_sets:
            print(nonai_exp_set)
        print('\u2500' * 80 + '\n')







# class InflationGraph_WithCardinalities(InflatedGraph):
#     # WORK IN PROGRESS
#     # The
#     def __init__(self, rawgraph, inflation_order, cardinalities_list):
#         InflatedGraph.__init__(self, rawgraph, inflation_order)
#
#         self.inflated_cardinalities_array = np.repeat(cardinalities_list, self.inflation_copies)
#         self.inflated_cardinalities_tuple = tuple(self.inflated_cardinalities_array.tolist())
#         self.column_count = self.inflated_cardinalities_array.prod()
#         self.shaped_column_integers = np.arange(self.column_count).reshape(self.inflated_cardinalities_tuple)




class ObservationalData:

    @staticmethod
    def MixedCardinalityBaseConversion(cardinality, string):
        #card = np.array([cardinality[i] ** (len(cardinality) - (i + 1)) for i in range(len(cardinality))])
        card = np.flip(np.multiply.accumulate(np.hstack((1, np.flip(cardinality))))[:-1])
        str_to_array = np.array([int(i) for i in string])
        return np.dot(card, str_to_array)

    def __init__(self, rawdata, cardinality):

        if isinstance(rawdata,
                      int):  # When only the number of observed variables is specified, but no actual data, we fake it.
            if isinstance(cardinality, int):  # When cardinality is specified as an integer
                self.observed_count = rawdata
                self.original_card_product = cardinality ** self.observed_count
                self.data_flat = np.full(self.original_card_product, 1.0 / self.original_card_product)
                self.size = self.data_flat.size
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else:  # When cardinalities are specified as a list
                if not isinstance(cardinality, (list, tuple, np.ndarray)):
                    raise TypeError("Cardinality not given as list of integers.")
                self.observed_count = rawdata
                self.original_card_product = np.prod(cardinality)
                self.data_flat = np.full(self.original_card_product, 1.0 / self.original_card_product)
                self.size = self.data_flat.size
                if self.observed_count !=len(cardinality):
                    raise ValueError("Cardinality specification does not match the number of observed variables.")
                self.cardinalities_array = np.array(cardinality)

        elif isinstance(rawdata[0],
                        str):  # When the input is in the form ['101','100'] for support certification purposes
            numevents = len(rawdata)
            if isinstance(cardinality, int):  # When cardinality is specified as an integer
                self.observed_count = len(rawdata[0])
                self.original_card_product = cardinality ** self.observed_count
                data = np.zeros(self.original_card_product)
                data[list(map(lambda s: int(s, cardinality), rawdata))] = 1 / numevents
                self.data_flat = data
                self.size = self.data_flat.size
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else:  # When cardinalities are specified as a list
                if not isinstance(cardinality, (list, tuple, np.ndarray)):
                    raise TypeError("Cardinality not given as list of integers.")
                self.observed_count = len(rawdata[0])
                self.original_card_product = np.prod(cardinality)
                data = np.zeros(self.original_card_product)
                data[list(map(lambda s: self.MixedCardinalityBaseConversion(cardinality, s), rawdata))] = 1 / numevents
                self.data_flat = data
                self.size = self.data_flat.size
                if self.observed_count !=len(cardinality):
                    raise ValueError("Cardinality specification does not match the number of observed variables.")
                self.cardinalities_array = np.array(cardinality)

        else:
            self.data_flat = np.array(rawdata).ravel()
            self.size = self.data_flat.size
            norm = np.linalg.norm(self.data_flat, ord=1)
            if norm == 0:
                self.data_flat = np.full(1.0 / self.size, self.size)
            else:  # Manual renormalization.
                self.data_flat = self.data_flat / norm
            if isinstance(cardinality, int):  # When cardinality is specified as an integer
                self.observed_count = np.rint(np.divide(np.log(self.size), np.log(cardinality))).astype(np.int)
                self.original_card_product = cardinality ** self.observed_count
                if self.original_card_product != self.size:
                    raise ValueError("Cardinality of individual variable could not be inferred.")
                self.cardinalities_array = np.full(self.observed_count, cardinality)
            else:  # When cardinalities are specified as a list
                if not isinstance(cardinality, (list, tuple, np.ndarray)):
                    raise TypeError("Cardinality not given as list of integers.")
                self.observed_count = len(cardinality)
                self.original_card_product = np.prod(cardinality)
                if self.observed_count !=len(cardinality):
                    raise ValueError("Cardinality specification does not match the data.")
                self.cardinalities_array = np.array(cardinality)
        self.cardinalities_tuple = tuple(self.cardinalities_array.tolist())
        self.data_reshaped = np.reshape(self.data_flat, self.cardinalities_tuple)


class InflationProblem(InflatedGraph, ObservationalData):

    def __init__(self, rawgraph, rawdata, card, inflation_order):
        InflatedGraph.__init__(self, rawgraph, inflation_order)
        ObservationalData.__init__(self, rawdata, card)

        self.original_cardinalities_array = self.cardinalities_array
        self.original_cardinalities_tuple = self.cardinalities_tuple
        self.original_size = self.size

        self.inflated_cardinalities_array = np.repeat(self.original_cardinalities_array, self.inflation_copies)
        self.inflated_cardinalities_tuple = tuple(self.inflated_cardinalities_array.tolist())
        self.column_count = self.inflated_cardinalities_array.prod()
        self.shaped_column_integers = np.arange(self.column_count).reshape(self.inflated_cardinalities_tuple)

        #Now we need to cache the relevant properties of an expressible set.

        for eset in self.expressible_sets:
            eset.shape_of_eset = np.take(self.inflated_cardinalities_array, eset.flat_eset)
            eset.size_of_eset = eset.shape_of_eset.prod()
            eset.which_rows_to_keep = np.arange(eset.size_of_eset).reshape(eset.shape_of_eset)
            minimize_object_under_group_action(
                eset.which_rows_to_keep,
                eset.symmetry_group, skip=1)
            eset.which_rows_to_keep = np.unique(eset.which_rows_to_keep.ravel(), return_index=True)[1]
            eset.size_of_eset_after_symmetry = len(eset.which_rows_to_keep)
            eset.there_are_discarded_rows = (eset.size_of_eset_after_symmetry < eset.size_of_eset)
            eset.discarded_rows_to_the_back = np.full(eset.size_of_eset, eset.size_of_eset_after_symmetry, dtype=np.int)
            np.put(eset.discarded_rows_to_the_back, eset.which_rows_to_keep, np.arange(eset.size_of_eset_after_symmetry))

        print(len(self.expressible_sets), 'expressible sets have been computed. Now constructing objects.')


    @cached_property
    def shaped_column_integers_marked(self):
        column_integers_marked = self.shaped_column_integers.copy()
        for detrule in self.inflated_determinism_checks:
            # det rule comes as a list with four elements
            initialtranspose = MoveToFront(self.inflated_observed_count, np.hstack(tuple(detrule)))
            inversetranspose = np.argsort(initialtranspose)
            parents_card_product = self.inflated_cardinalities_array.take(detrule[1]).prod()
            child_cardinality = np.atleast_1d(self.inflated_cardinalities_array.take(detrule[-1])).prod()
            intermediateshape = (parents_card_product, parents_card_product, child_cardinality, child_cardinality, -1)
            column_integers_marked = column_integers_marked.transpose(tuple(initialtranspose)).reshape(
                intermediateshape)
            for i in np.arange(parents_card_product):
                for j in np.arange(child_cardinality - 1):
                    for k in np.arange(j + 1, child_cardinality):
                        column_integers_marked[i, i, j, k] = -1
            column_integers_marked = column_integers_marked.reshape(self.inflated_cardinalities_tuple).transpose(
                tuple(inversetranspose))
        return column_integers_marked

    @cached_property
    def valid_column_orbits(self):
        AMatrix = orbits_of_object_under_group_action(
            self.shaped_column_integers_marked,
            self.inflation_group_elements).T
        AMatrix = np.compress(AMatrix[0]>=0, AMatrix, axis=1)
        #AMatrix = np.compress(minima == np.abs(AMatrix[0]), AMatrix, axis=1)
        return AMatrix

    @staticmethod
    def csr_vstack(it):
        newdata = np.hstack(tuple(m.data for m in it))
        newindices = np.hstack(tuple(m.indices for m in it))
        newindptrs = np.vstack(tuple(m.indptr for m in it))
        newindptrs = np.hstack(([0],newindptrs[1:,-1]))+newindptrs[:,1:]
        newindptrs = np.hstack(([0],newindptrs.ravel()))
        newlen = np.sum(tuple(len(m) for m in it))
        return csr_matrix((newdata,newindices,newindptrs), shape=(newlen,it[0].shape[1]))


    @property
    def _InflationMatrix(self):
        for eset in self.expressible_sets:
            AMatrix = eset.discarded_rows_to_the_back.take(
                eset.Columns_to_unique_rows(self.shaped_column_integers)).take(self.valid_column_orbits)
            AMatrix = SparseMatrixFromRowsPerColumn(AMatrix).asformat('csr', copy=False)
            if eset.there_are_discarded_rows:
                yield AMatrix[:-1]
            else:
                yield AMatrix
    @cached_property
    def inflation_matrix(self):
        if len(self.expressible_sets)==1:
            return next(self._InflationMatrix)
        else:
            #return self.csr_vstack(list(self._InflationMatrix))
            return sparse_vstack(list(self._InflationMatrix))
    @property
    def _NumericB(self):
        for eset in self.expressible_sets:
            b_block = eset.Generate_numeric_b_block(self.data_reshaped)
            yield np.take(b_block,eset.which_rows_to_keep)
    @cached_property
    def numeric_b(self):
        return np.hstack(list(self._NumericB))

    @property
    def _SymbolicB(self):
        for eset in self.expressible_sets:
            s_block = eset.Generate_symbolic_b_block(self.inflated_cardinalities_array)
            yield np.take(s_block,eset.which_rows_to_keep)
    @cached_property
    def symbolic_b(self):
        return np.hstack(list(self._SymbolicB))




class InflationLP(InflationProblem):

    def __init__(self, rawgraph, rawdata, card, inflation_order, solver = 'moseklp'):

        InflationProblem.__init__(self, rawgraph, rawdata, card, inflation_order)

        #self.numeric_b, self.symbolic_b = self.numeric_and_symbolic_b(extra_expressible=extra_ex)
        #self.InfMat = self.InflationMatrix(extra_expressible=extra_ex)
        
        if not ((solver == 'moseklp') or (solver == 'CVXOPT') or (solver == 'mosekAUTO')):
            raise TypeError("The accepted solvers are: 'moseklp', 'CVXOPT' and 'mosekAUTO'")

        if solver == 'moseklp':

            self.solve = InfeasibilityCertificate(self.inflation_matrix, self.numeric_b)

        elif solver == 'CVXOPT':

            self.solve = InflationLP(self.inflation_matrix, self.numeric_b)

        elif solver == 'mosekAUTO':

            self.solve = InfeasibilityCertificateAUTO(self.inflation_matrix, self.numeric_b)

        self.tol = self.solve[
                       'gap'] / 10  # TODO: Choose better tolerance function. This is yielding false incompatibility claims.
        self.yRaw = np.array(self.solve['x']).ravel()

        self.y = IntelligentRound(self.yRaw, self.inflation_matrix)

        self.checkY = csr_matrix(self.y).dot(self.inflation_matrix)

        self.idxtally=indextally(self.y)

        self.symtally = symboltally(self.idxtally, self.symbolic_b)

        self.ineq_as_str = inequality_as_string(self.y, self.symbolic_b)



    def WitnessDataTest(self, y):
        IncompTest = (np.amin(y) < 0) and (np.dot(y, self.numeric_b) < -self.tol)
        if IncompTest:
            print('Distribution Compatibility Status: INCOMPATIBLE')
        else:
            print('Distribution Compatibility Status: COMPATIBLE')
        return IncompTest




    def Inequality(self,output=[]):
        # Modified Feb 2, 2021 to pass B_symbolic as an argument for Inequality
        # Modified Feb 25, 2021 to accept custom output options from user
        if self.WitnessDataTest(self.yRaw):
            y = IntelligentRound(self.yRaw, self.inflation_matrix)
            
            if output==[]:
            
                idxtally=indextally(y)
                symtally=symboltally(indextally(y),self.symbolic_b)
                ineq_as_str=inequality_as_string(y,self.symbolic_b)

                print("Writing to file: 'inequality_output.json'")

                returntouser = {
                    'Raw solver output': self.yRaw.tolist(),
                    'Inequality as string': ineq_as_str,
                    'Coefficients grouped by index': idxtally,
                    'Coefficients grouped by symbol': symtally,
                    'Clean solver output': y.tolist()
                }
                f = open('inequality_output.json', 'w')
                print(json.dumps(returntouser), file=f)
                f.close()
                
            else:
                returntouser={}
                
                if 'Raw solver output' in output:
                    returntouser['Raw solver output']=self.yRaw.tolist()
                if 'Inequality as string' in output:
                    ineq_as_str=inequality_as_string(y,self.symbolic_b)
                    returntouser['Inequality as string']=ineq_as_str
                if 'Coefficients grouped by index' in output:
                    idxtally=indextally(y)
                    returntouser['Coefficients grouped by index']=idxtally
                if 'Coefficients grouped by symbol' in output:
                    symtally=symboltally(indextally(y),self.symbolic_b)
                    returntouser['Coefficients grouped by symbol']=symtally
                if 'Clean solver output' in output:
                    returntouser['Clean solver output']=y.tolist()
                
                f = open('inequality_output.json', 'w')
                print(json.dumps(returntouser), file=f)
                f.close()
                
            return returntouser
        else:
            return print('Compatibility Error: The input distribution is compatible with given inflation order test.')


class SupportCertificate(InflationProblem):

    def __init__(self, rawgraph, rawdata, card, inflation_order):

        InflationProblem.__init__(self, rawgraph, rawdata, card, inflation_order)

        inflation_matrix = self.inflation_matrix.asformat('coo', copy=False)
        numeric_b = self.numeric_b
        Rows = inflation_matrix.row
        Cols = inflation_matrix.col

        ForbiddenRowIdx = np.where(numeric_b[Rows] == 0)[0]
        ForbiddenColumnIdx = np.unique(Cols[ForbiddenRowIdx])

        ColumnTemp = np.ones(Cols.max() + 1)
        ColumnTemp[ForbiddenColumnIdx] = 0

        ForbiddenColsZeroMarked = ColumnTemp[Cols]

        IdxToRemove = np.where(ForbiddenColsZeroMarked == 0)[0]

        NewRows = np.delete(Rows, IdxToRemove)
        NewCols = np.delete(Cols, IdxToRemove)
        NewData = np.ones(len(NewCols), dtype=np.uint)

        NewMatrix = coo_matrix((NewData, (NewRows, NewCols)))

        NonzeroRows = np.nonzero(numeric_b)[0]
        self.Check = True

        for r in NonzeroRows:
            if self.Check:
                self.Check = NewMatrix.getrow(r).toarray().any()

        if self.Check:

            print("Supported")

        else:

            print("Not Supported")


if __name__ == '__main__':
    from igraph import Graph

    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    #[InflatedGraph(g, [2, 3, 3]).print_assessment() for g in
    #    (TriangleGraph, Evans14a, Evans14b, Evans14c, IceCreamGraph, BiconfoundingInstrumental)]

    testig = InflatedGraph(TriangleGraph, [2, 3, 3])
    testig.print_assessment()
    #print(testig._canonical_pos)
    print(testig.partitioned_expressible_set)
    print(testig.diagonal_expressible_set)
    print(testig.diagonal_expressible_set_symmetry_group)
    print(testig.diagonal_expressible_set.take(testig.diagonal_expressible_set_symmetry_group))
