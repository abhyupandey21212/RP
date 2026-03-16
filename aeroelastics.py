# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:27:02 2026

@author: abhyu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

class AeroStrip:
    def __init__(self, wing, rho = 1.25, g = -10, r_a = np.array([[0,0,0]]).T):
        """

        Parameters
        ----------
        wing : beamNL object
        rho : air density
        g : gravity
        r_g : inertial axis offset vector (> 0  if in front)
        r_a : aero axis offset vector (>0 if in front)
        Returns
        -------
        global load vector for wing

        """
        self.wing = wing
        self.rho = rho
        self.g = g
        self.r_g = self.wing.r_g
        self.r_a = r_a
        self.CL_model()
        
    def CL_model(self, a_STALL = 20, a_0 = -5, a_3 = 25, CL_max = 2, CL_3 = 1.8):
        x = np.array([[a_0, a_STALL, a_3]]).T  * np.pi / 180
        X = np.hstack((x**2, x**1, x**0))
        Y = np.array([[0, CL_max, CL_3]]).T
        self.CLalpha = np.linalg.inv(X) @ Y
        return self.CLalpha
    
    def CL(self, a_RAD):
        """
        Really simple airfoil model

        """
        x = np.array([[a_RAD**2, a_RAD**1, 1]])
        return x @ self.CLalpha
    

    def force_per_elem(self, V, vel, AoA = 0, n = 1):
        """

        Parameters
        ----------
          V : wing state vector
          vel : velocity
          AoA : global angle of attack
          n : Load factor

        Returns
        -------
        list of force and moment on each elemenet, in the format complatible with the BeamNL
        fe = [fx fy fz mx my mz 0 0 0 0 0 0].

        """
        q = 0.5 * self.rho * vel * vel
        load_per_elem = []
        X = self.wing.post(V)
        chord = self.wing.chord
        e = self.r_a[1] #Distance from aero-axis to elastic axis rel chord
        d = self.r_g[1] #Distance from interia axis to elastic axis rel chord
        dm = self.wing.dm
        for el in self.wing.elems.values(): #Going over each part of the wing
            i = el[0]
            X_el = X[(i-1)*6:(i+2)*6]
            thetax_el = X_el[3::6]
            def thetax(s):
                return self.wing.shape_funcs(s) @ thetax_el            
            
            def CL_elem(s):
                AoA_l = AoA + thetax(s)
                return self.CL(float(AoA_l))
            
            def dFdL_aero(s):
                return q * chord * CL_elem(s) # + chord * chord *Cmo)
            
            def dMdL_aero(s):
                return q * chord * CL_elem(s) * e# + chord * chord *Cmo)
            
            def dFdL_g(s):
                return n * dm * self.g
            
            def dMdL_g(s):
                return n * dm * self.g * d
            
            F_aero_elem = scipy.integrate.quad(dFdL_aero, 0, 1)[0] if e == 0 else 0
            M_aero_elem = scipy.integrate.quad(dMdL_aero, 0, 1)[0] if e != 0 else 0
            F_g_elem = scipy.integrate.quad(dFdL_g, 0, 1)[0] if d == 0 else 0
            M_g_elem = scipy.integrate.quad(dMdL_g, 0, 1)[0] if d != 0 else 0

            L_elem = np.array([[0,0,F_aero_elem + F_g_elem, M_aero_elem + M_g_elem, 0, 0, 0, 0, 0, 0, 0, 0]]).T
            load_per_elem.append(L_elem)
        return load_per_elem
        
    