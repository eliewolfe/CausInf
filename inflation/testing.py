#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:49:49 2021

@author: boraulu
"""
from __future__ import absolute_import
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from itertools import combinations, chain, permutations
from igraph import Graph
from itertools import product
import json
from collections import defaultdict
from sys import hexversion
if hexversion >= 0x3080000:
    from functools import cached_property
elif hexversion >= 0x3060000:
    from backports.cached_property import cached_property
else:
    cached_property = property
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
#from inflation.classes import InflationLP,InflatedGraph
from inflation.infgraph import InflationLP, InflatedGraph, InflationProblem
    
#def ListOfBitStringsToListOfIntegers(list_of_bitstrings):
#    return list(map(lambda s: int(s,4), list_of_bitstrings))



def MixedCardinalityBaseConversion(cardinality, string):
    #card=np.array([cardinality[i]**(len(cardinality)-(i+1)) for i in range(len(cardinality))])
    card=np.flip(np.multiply.accumulate(np.hstack((1, np.flip(cardinality))))[:-1])
    str_to_array=np.array([int(i) for i in string])
    return np.dot(card,str_to_array)
def UniformDistributionFromSupport(list_of_strings, cardinality_list):
    numvar = len(cardinality_list)
    prod_cardinality = np.prod(cardinality_list)
    cardinality_converter = np.flip(np.multiply.accumulate(np.hstack((1, np.flip(cardinality_list))))[:-1])
    data = np.zeros(prod_cardinality)
    array_of_integers = np.fromiter(map(int,''.join(list_of_strings)),dtype=np.int).reshape((-1,numvar))
    list_of_integers = np.unique(np.dot(array_of_integers, cardinality_converter))
    numevents = len(list_of_integers)
    print(list_of_integers)
    data[list_of_integers] = 1/numevents
    return data

InstrumentalGraph = Graph.Formula("U1->X->A->B,U2->A:B")
BiconfoundingGraph = Graph.Formula("U1->A:B,U2->A:C,A->B:C")
Evans14a = Graph.Formula("U1->A:C,U2->A:B:D,U3->B:C:D,A->B,C->D")
Evans14b = Graph.Formula("U1->A:C,U2->B:C:D,U3->A:D,A->B,B:C->D")
Evans14c = Graph.Formula("U1->A:C,U2->B:D,U3->A:D,A->B->C->D")
IceCreamGraph = Graph.Formula("U1->A,U2->B:D,U3->C:D,A->B:C,B->D")
BiconfoundingInstrumental = Graph.Formula("U1->A,U2->B:C,U3->B:D,A->B,B->C:D")
TriangleGraph = Graph.Formula("X->A,Y->A:B,Z->B:C,X->C")
BellGraph = Graph.Formula("L->A:B,Ux->X,Uy->Y,X->A,Y->B")

TriangleData=[0.12199995751046305, 0.0022969343799089472, 0.001748319476328954, 3.999015242496535e-05, 0.028907881434196828, 0.0005736087488455967, 0.0003924033706699725, 1.1247230369521505e-05, 0.0030142577390317635, 0.09234476010282468, 4.373922921480586e-05, 0.0014533921021948346, 0.0007798079722868244, 0.024091567451515063, 1.1247230369521505e-05, 0.0003849052170902915, 0.020774884184769502, 0.000396152447459813, 0.0003049249122403608, 4.998769053120669e-06, 0.10820335492385, 0.0020794879260981982, 0.0015546171755205281, 2.4993845265603346e-05, 0.0006260958239033638, 0.020273757587194154, 7.498153579681003e-06, 0.0003374169110856452, 0.0028942872817568676, 0.08976414557915113, 2.624353752888351e-05, 0.0012984302615480939, 0.002370666223442477, 4.7488306004646356e-05, 0.0999928767540993, 0.001957018084296742, 0.0006198473625869629, 8.747845842961171e-06, 0.02636975644747481, 0.0005198719815245496, 1.4996307159362007e-05, 0.000403650601039494, 0.0005498645958432735, 0.017359475229224805, 7.123245900696953e-05, 0.002346922070440154, 0.0033754188031197316, 0.10295964618712641, 0.00038740460161685187, 7.498153579681003e-06, 0.01608353942841575, 0.000306174604503641, 0.0021319750011559654, 4.248953695152569e-05, 0.09107007399427891, 0.001860791780024169, 5.998522863744803e-05, 0.0018395470115484063, 0.002570616985567304, 0.0766411271224461, 1.874538394920251e-05, 0.00048238121362614454, 0.0006410921310627258, 0.020223769896662948]
InstrumentalData=['000','011']
InstrumentalData2=['000']
BiconfoundingData=['000','011']
BiconfoundingInstrumentalData=['0000','0100','1011','1111']
Evans14aData=['0000','1001','1111']
Evans14aData2=['0000','0010','0101']
Evans14bData=['1000','1001','1111']
Evans14cData=['0000','1101','1011']
IceCreamData=['0000','1111','1010','0011']
A=list(product('0123',repeat=4))
B=[''.join(i) for i in A]
del B[0]
BellDataIncomp=B

BellData=['0000','0010','0001','0020','0002','1011','0111','1100','1110','1101','1120','1102','0021','1121','0012','1112','0022','1122','0031','0131','0032','0132','0013','1013','0023','1023','0030','0130','0003','1003','0033','0133']
cardinality=[2,2,4,4]
original_card_product=np.prod(cardinality)
#data = np.zeros(original_card_product)
#data[list(map(lambda s: MixedCardinalityBaseConversion(cardinality, s),BellData))] = 1/len(BellData)
data = UniformDistributionFromSupport(BellData, cardinality)
data[MixedCardinalityBaseConversion(cardinality, '0033')]=1/16
data[MixedCardinalityBaseConversion(cardinality, '0133')]=0

"""
data[MixedCardinalityBaseConversion(cardinality, '0123')]=0.05
data[MixedCardinalityBaseConversion(cardinality, '1111')]=0.008
data[MixedCardinalityBaseConversion(cardinality, '1130')]=0.03
"""
#rawgraph=InstrumentalGraph 
#rawdata=InstrumentalData2
rawgraph=BellGraph
rawdata=data
card=[2,2,4,4]
inflation_order=[1,3,3]

# extra_ex=True
# solver='moseklp'
#solver='mosekAUTO'

#print(InflatedGraph(rawgraph,[2,1,2]).inflation_group_generators)

#rawgraph=TriangleGraph
#rawdata=TriangleData
#card=4
#inflation_order=2

InflatedGraph(rawgraph,inflation_order).print_assessment()
#PreLP = InflationProblem(rawgraph, rawdata, card, inflation_order)
#print(PreLP.inflation_matrix.shape, PreLP.numeric_b.shape, PreLP.symbolic_b.shape)

InfLP = InflationLP(rawgraph, rawdata, card, inflation_order)

InfLP.symbolic_b
InfLP.numeric_b
InfLP.inflation_matrix

Solution=InfLP.Inequality(['Raw solver output','Inequality as string','Clean solver output'])











