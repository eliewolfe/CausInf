#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 00:26:31 2021

@author: boraulu
"""

from __future__ import absolute_import
import numpy as np
from scipy.sparse import coo_matrix

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    
from inflation.inflationmatrix import InflationMatrixFromGraph
from inflation.inflationmatrix import FindB



def SupportCertificate(InfMat,b):
    
    Rows=InfMat.row
    Cols=InfMat.col
    
    ForbiddenRowIdx=np.where(b[Rows] == 0)[0]
    ForbiddenColumnIdx=np.unique(Cols[ForbiddenRowIdx])
    
    ColumnTemp=np.ones(Cols.max()+1)
    ColumnTemp[ForbiddenColumnIdx]=0
    
    ForbiddenColsZeroMarked=ColumnTemp[Cols]
    
    IdxToRemove=np.where(ForbiddenColsZeroMarked == 0)[0]
    
    NewRows=np.delete(Rows,IdxToRemove)
    NewCols=np.delete(Cols,IdxToRemove)
    NewData=np.ones(len(NewCols),dtype=np.uint)
    
    NewMatrix=coo_matrix((NewData, (NewRows, NewCols)))
    
    NonzeroRows=np.nonzero(b)[0]
    Check=True
    
    for r in NonzeroRows:
        if Check:
            
            Check=NewMatrix.getrow(r).toarray().any()

    if Check:
        
        print("Supported")
    
    else:
        
        print("Not Supported")

    return Check



if __name__ == '__main__':
    from igraph import Graph
    
    def ListOfBitStringsToListOfIntegers(list_of_bitstrings):
        return list(map(lambda s: int(s,2),list_of_bitstrings))
    
    def UniformDistributionFromSupport(list_of_bitstrings):
        numvar = max(map(len,list_of_bitstrings))
        numevents = len(list_of_bitstrings)
        data = np.zeros(2 ** numvar)
        data[ListOfBitStringsToListOfIntegers(list_of_bitstrings)] = 1/numevents
        return data
    
    InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
    Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
    Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
    Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
    IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
    BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
    TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
    
    inflation_order= 2
    card=2
    g=TriangleGraph
    
    TriData=UniformDistributionFromSupport(['000','111'])
    InstrumentalData=UniformDistributionFromSupport(['000','101'])
    BiconfoundingInstrumentalData=UniformDistributionFromSupport(['0000','0100','1011','1111'])
    Data=TriData

    InfMat=InflationMatrixFromGraph(g, inflation_order, card)
    b=FindB(Data, inflation_order)

    S=SupportCertificate(InfMat,b)

