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
    verts["children"]=g.get_adjlist('out');
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
    #print(verts["parents"])
    #de= g.successors(verts)
    #print(de)
    if hasty:
        return verts["name"], verts["parents"], verts["roots_of"]
    else:
        def FindScreeningOffSet(root, observed):
            screeningset = np.intersect1d(root["children"], observed["ancestors"]).tolist()
            screeningset.append(observed.index)
            return screeningset
        
        def FindBestScreeningOffSet(root, observed, g):
            screeningset = np.intersect1d(root["children"], observed["ancestors"]).tolist()

            for sidx in screeningset:

                screeningset_rest = set(screeningset)
                screeningset_rest.remove(sidx)
                if not all(screeningset_rest.isdisjoint(directed_path) for directed_path in g.get_all_simple_paths(sidx, to=observed)):
                    screeningset.remove(sidx)

            screeningset.append(observed.index)
            return screeningset

        determinism_checks = [(root, FindScreeningOffSet(verts[root], v)) for v in g.vs[has_grandparents] for root in
                              np.setdiff1d(v["roots_of"], v["parents"])]
        
        filtered_determinism_checks = [(root, FindBestScreeningOffSet(verts[root], v,g)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"], v["parents"])]

        #We are going to want to adjust this code to handle screening off of multiple roots at once.
        #Should we be concerned about multiple root screening in terms of determinism checks? YES!!!
        def FindExpressibleSet(determinism_check, dict_of_roots, dict_of_descendants):
            Y = determinism_check[1][-1]
            Xs = determinism_check[1][:-1]
            U1 = determinism_check[0]
            roots = dict_of_roots[Y]

            U2 = set([])
            for X in Xs:
                U2.update(dict_of_roots[X])
            U2.remove(U1)

            Zs = set(nonroot_vertices)
            Zs.remove(Y)
            Zs.difference_update(Xs)
            for u2 in U2:
                Zs.difference_update(dict_of_descendants[u2])

            U3YZ = set([])
            for Z in Zs:
                U3YZ.update(dict_of_roots[Z])
            U3YZ.intersection_update(dict_of_roots[Y])
            U3YZ.remove(U1)
            U3YZ.difference_update(U2)

            return tuple([list(U3YZ),(Y,list(Zs),Xs)])

        expressible_sets = [FindExpressibleSet(determinism_check, verts["roots_of"], verts["descendants"]) for determinism_check in filtered_determinism_checks]
        
    return verts["name"], verts["parents"], verts["roots_of"], determinism_checks, filtered_determinism_checks, expressible_sets


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



if __name__ == '__main__':

    TGraph=Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    #TGraph=Graph.Formula("U1->A:B:C:D,D->C,C->X,A->X,B->X")
    #TGraph=Graph.Formula("U1->X->A->B,U2->A:B")

    g = ToTopologicalOrdering(TGraph)
    verts = g.vs
    verts["parents"] = g.get_adjlist('in');
    verts["children"]=g.get_adjlist('out');
    verts["ancestors"] = [g.subcomponent(i, 'in') for i in g.vs]
    verts["descendants"] = [g.subcomponent(i, 'out') for i in g.vs]
    verts["indegree"] = g.indegree()
    # verts["outdegree"]=g.outdegree() #Not needed
    verts["grandparents"] = g.neighborhood(None, order=2, mode='in', mindist=2)
    # verts["parents_inclusive"]=g.neighborhood(None, order=1, mode='in', mindist=0) #Not needed
    has_grandparents = [idx for idx, v in enumerate(verts["grandparents"]) if len(v) >= 1]
    verts["isroot"] = [0 == i for i in g.vs["indegree"]]
    root_vertices = verts.select(isroot=True).indices
    nonroot_vertices = verts.select(isroot=False).indices
    verts["roots_of"] = [np.intersect1d(anc, root_vertices).tolist() for anc in verts["ancestors"]]


    #filtered_determinism_checks = [(root, FindBestScreeningOffSet(verts[root], v,g)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"], v["parents"])]

    names, parents_dict, roots_dict, determinism_checks, filtered_determinism_checks, expressible_sets = LearnParametersFromGraph(TGraph)
    print(names)
    print(determinism_checks)
    print(filtered_determinism_checks)
    print(expressible_sets)

    inflation_order=2
    ExSets=[]
    for screenoffs in filtered_determinism_checks:

        Y=screenoffs[1][-1]
        Xs=screenoffs[1][:-1]
        U1=screenoffs[0]
        roots=verts["roots_of"][Y]
        notroots=list(np.setdiff1d(roots,root_vertices))

        for X in Xs:

            U2=list(np.setdiff1d(verts[X]["ancestors"],[X,U1]))
            U3=list(np.setdiff1d(roots,[U1]+U2))

            DescendantsU2=[]

            for i in U2:
                DescendantsU2.extend(verts[i]["descendants"])
            DescendantsU2=list(set(DescendantsU2))

            Z=list(np.setdiff1d(nonroot_vertices,DescendantsU2+[Y]))

            copies=list(np.arange(inflation_order)+1)

            for z in Z:
                for cU1U2 in copies:

                    Xtemplate=np.full(len(roots),cU1U2)
                    Xtemplate[U3]=-1

                    Ytemplate=np.full(len(roots),cU1U2)

                    Ztemplate=np.full(len(roots),cU1U2)
                    Ztemplate[U2]=-1

                    if notroots != []:
                        Xtemplate[notroots]=-1
                        Ytemplate[notroots]=-1
                        Ztemplate[notroots]=-1

                    for u3 in U3:
                        for cU3 in copies:
                            if cU3 != cU1U2:

                                Zcopy=Ztemplate
                                Zcopy[u3]=cU3

                                Xcopy=''.join([''.join(str(i)) for i in list(Xtemplate)])
                                Ycopy=''.join([''.join(str(i)) for i in list(Ytemplate)])
                                Zcopy=''.join([''.join(str(i)) for i in list(Zcopy)])

                                Xcopy=Xcopy.replace('-1','_')
                                Ycopy=Ycopy.replace('-1','_')
                                Zcopy=Zcopy.replace('-1','_')

                                Xcopy=verts[X]["name"]+'['+Xcopy+']'
                                Ycopy=verts[Y]["name"]+'['+Ycopy+']'
                                Zcopy=verts[z]["name"]+'['+Zcopy+']'

                                exset=Zcopy+' '+Ycopy+' '+Xcopy
                                ExSets.append(exset)

    print(ExSets)










