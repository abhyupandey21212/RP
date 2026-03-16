# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 15:57:05 2026

@author: abhyu
"""
import numpy as np
inv = np.linalg.inv

#Dimention
n_nodes = 9
dt = 0.001
time_grid = np.arange(0, 1+dt, dt)
span, chord = [0.55, 0.1] #meters, span, chord, height
mass_per_len = dm = 0.55 #kg/m
r_g = np.array([[0,-3.5e-3,0]]).T
#Moments per span
Ixx = 3.030e-4 #kgm2 / m
Iyy = 1.515e-4 #kgm2 / m
Izz = 3.636e-3 #kgm2 / m

EA = 7.6e6 #N
EIy = 4.67 #Nm2
EIz = 3.31e3 #Nm2
GJ = 7.2 #Nm2
GAy = 1e6 #N
GAz = 3.31e6 #N
#Defining the geo and the material
eta_grid = np.linspace(0, span, n_nodes)
h = eta_grid[1] - eta_grid[0]

#Mass matrix (in continous) per span
mass_mom_per_span = np.diag([Ixx, Iyy, Izz])
M = np.block([[dm*np.eye(3), np.zeros((3,3))],
              [np.zeros((3,3)), mass_mom_per_span]])

e1 = np.array([[1, 0, 0]]).T
k0 = np.array([[0, 0, 0]]).T #inital curvature is zero, initially straight beam


eta_grid = np.linspace(0, span, n_nodes)
h = eta_grid[1] - eta_grid[0]




#Stiffness matrix (in continous) per span
Cinv = np.diag([EA, GAy, GAz, GJ, EIy, EIz])
C = inv(Cinv)
