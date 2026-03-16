# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 08:31:08 2026

@author: abhyu
"""
from IntrinsicBeamNL import *
import dimentions as dim
import pandas as pd
V_samples = pd.read_csv("V_samples.csv", header=None).to_numpy()  
NL_samples = pd.read_csv("NL_samples.csv", header=None).to_numpy()
interior = slice(6,-6)
beamNL = BeamNL(dim.n_nodes, dim.span, dim.chord, dim.dm, dim.r_g, dim.mass_mom_per_span, dim.Cinv)
l = 4
r = 6
beamPOD = BeamNL_POD(beamNL, l, r=r)  
beamPOD.POD_offline(V_samples[interior,:])
beamPOD.POD_DEIM_offline(V_samples[interior,:], NL_samples[interior,:])
V_rand = np.random.uniform(-10,10,180)
NL = beamPOD.K_nl_gl(V_rand) @ V_rand
NL_row = beamPOD.NL_FOM_rowwise(V_rand)

P = beamPOD.P
    
ps = np.where(P.T == 1)[1]
ps6 = np.where(P.T == 1)[1] + 6
p2q = {p:q for q,p in enumerate(ps6)}
            
index_to_elem = beamPOD.index_to_elem
node_lookup = {el:[] for el in beamPOD.elems}
for p in ps6:
    for el, ran in index_to_elem.items():
        if p in ran:
            node_lookup[el].append(p)
            
K_nl_DEIM = np.zeros((180, 180))
K_nl_DEIM2 = np.zeros((r, 180))

K_nl_FOM = np.zeros((180, 180))
for el, n in beamPOD.elems.items():
    print(f'---elem {el}---')
    i = n[0] - 1 #global first index of the elem
    V_el = V_rand[i*12:(i+3)*12]
    K_elem = beamPOD.K_nl_el(V_el)
    for p in range(i*12, (i+3)*12):
        elj = p % (12*3) #local index in elem
        q = elj + i*12
        K_nl_FOM[q,i*12:(i+3)*12] += K_elem[elj,:]
        K_FOM_row = K_nl_FOM[q,i*12:(i+3)*12]
        if q in node_lookup[el]:
            p = p2q[q]
            print(f'p = {p}, node {i}, row {q}')

            #elj = p % (12*3) #local index in elem
            K_nl_DEIM[q,i*12:(i+3)*12] += K_elem[elj,:]
            K_nl_DEIM2[p,i*12:(i+3)*12] +=K_elem[elj,:]



NL_row2 = K_nl_FOM @ V_rand
PT_NL_row = P.T @ NL_row2[interior]
NL_DEIM = K_nl_DEIM @ V_rand
PT_NL_DEIM = P.T @ NL_DEIM[interior]
PT_NL_DEIM2 = K_nl_DEIM2 @ V_rand
print('---------------')
print(PT_NL_DEIM2 ==PT_NL_row)
        

