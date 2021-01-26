#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning a little bit about the inflation graph from the original graph
"""

import numpy as np
from igraph import Graph

# from igraph import * #Not needed, since we are using methods built into the object...!

def adjlist_find_paths(graph, n, m , path=[]):
  "Find paths from node index n to m using adjacency list a."
  path = path + [n]
  if n == m:
    return [path]
  paths = []
  for child in graph.successors(n):
    if child not in path:
      child_paths = adjlist_find_paths(graph, child, m, path)
      for child_path in child_paths:
        paths.append(child_path)
  return paths

def paths_from_to(graph, source, dest):
  "Find paths in graph from vertex source to vertex dest."
  n = source.index
  m = dest.index
  return adjlist_find_paths(graph, n, m)


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
    has_grandparents = [idx for idx, v in enumerate(g.vs["grandparents"]) if len(v) >= 1]
    verts["isroot"] = [0 == i for i in g.vs["indegree"]]
    root_vertices = verts.select(isroot=True).indices
    nonroot_vertices = verts.select(isroot=False).indices
    verts["roots_of"] = [np.intersect1d(anc, root_vertices).tolist() for anc in g.vs["ancestors"]]
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
            validvars=[]
            for sidx in screeningset:
                        
                p=paths_from_to(g, g.vs[sidx], observed)
                
                        
                for pat in p:
                            
                    pat.remove(sidx)
                    check=False
                            
                    for v in screeningset:
                        check=check | any(np.array(pat) == v)
                                
                if check == False:
                                
                    validvars.append(sidx)
          
            validvars.append(observed.index)
            return validvars

        determinism_checks = [(root, FindScreeningOffSet(verts[root], v)) for v in g.vs[has_grandparents] for root in
                              np.setdiff1d(v["roots_of"], v["parents"])]
        
        filtered_determinism_checks = [(root, FindBestScreeningOffSet(verts[root], v,g)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"], v["parents"])]
        
    return verts["name"], verts["parents"], verts["roots_of"], determinism_checks, filtered_determinism_checks


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

def FindBestScreeningOffSet(root, observed, g):
    screeningset = np.intersect1d(root["children"], observed["ancestors"]).tolist()
    validvars=[]
    for sidx in screeningset:
                        
        p=paths_from_to(g, g.vs[sidx], observed)
                
                        
        for pat in p:
                            
            pat.remove(sidx)
            check=False
                            
            for v in screeningset:
                check=check | any(np.array(pat) == v)
                                
        if check == False:
                                
            validvars.append(sidx)
          
    validvars.append(observed.index)
    return validvars



TGraph=Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
#TGraph=Graph.Formula("U1->A:B:C:D,D->C,C->X,A->X,B->X,D->X")
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
has_grandparents = [idx for idx, v in enumerate(g.vs["grandparents"]) if len(v) >= 1]
verts["isroot"] = [0 == i for i in g.vs["indegree"]]
root_vertices = verts.select(isroot=True).indices
nonroot_vertices = verts.select(isroot=False).indices
verts["roots_of"] = [np.intersect1d(anc, root_vertices).tolist() for anc in g.vs["ancestors"]]


filtered_determinism_checks = [(root, FindBestScreeningOffSet(verts[root], v,g)) for v in g.vs[has_grandparents] for root in np.setdiff1d(v["roots_of"], v["parents"])]

print(filtered_determinism_checks)

"""
p=paths_from_to(g, verts[3], verts[5])
print(p)
"""
"""
names,parents,roots_of,determinism_checks, filtered_determinism_checks=LearnParametersFromGraph(TGraph, hasty=False)


print(names)
print(parents)
print(roots_of)
print(determinism_checks)
print(filtered_determinism_checks)
"""
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










