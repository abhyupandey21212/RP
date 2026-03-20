
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 21:18:30 2026

@author: abhyu
"""

import numpy as np
inv = np.linalg.inv
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy
import aeroelastics as ae
import time
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from scipy.sparse import lil_matrix, eye, hstack, vstack, csr_matrix
from scipy.integrate import solve_ivp




#Helper functions
def cross(x, y):
    # reshape to (n_elem, 3)
    X = x.reshape(-1, 3)
    Y = y.reshape(-1, 3)
    return np.cross(X, Y).ravel().reshape(-1, 1)

def cross_mat(v):
    # v: length-3 1D array-like
    x1, x2, x3 = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -x3, x2],
                     [x3, 0, -x1],
                     [-x2, x1, 0]])

def L1(v1):
    v1 = v1.reshape((-1,1))
    v = v1[:3]
    w = v1[3:]
    cross_w = cross_mat(w)
    return np.block([[cross_w, np.zeros_like(cross_w)], 
                     [cross_mat(v), cross_w]])

def L2(v2):
    v2 = v2.reshape((-1,1))
    f = v2[:3]
    m = v2[3:]
    cross_f = cross_mat(f)
    return np.block([[np.zeros_like(cross_f), cross_f], 
                     [cross_f, cross_mat(m)]])

class BeamNL:
    #Helper functions
    def _cross(self, x, y):
        # reshape to (n_elem, 3)
        X = x.reshape(-1, 3)
        Y = y.reshape(-1, 3)
        return np.cross(X, Y).ravel().reshape(-1, 1)

    def _cross_mat(self, v):
        # v: length-3 1D array-like
        x1, x2, x3 = float(v[0]), float(v[1]), float(v[2])
        return np.array([[0, -x3, x2],
                         [x3, 0, -x1],
                         [-x2, x1, 0]])

    def _L1(self, v1):
        v1 = v1.reshape((-1,1))
        v = v1[:3]
        w = v1[3:]
        cross_w = cross_mat(w)
        return np.block([[cross_w, np.zeros_like(cross_w)], 
                         [cross_mat(v), cross_w]])
   
    def _L1T(self, v1):
        v1 = v1.reshape((-1,1))
        v = v1[:3]
        w = v1[3:]
        cross_w = cross_mat(w)
        return np.block([[cross_w, cross_mat(v)], 
                         [np.zeros_like(cross_w), cross_w]])

    def _L2(self, v2):
        v2 = v2.reshape((-1,1))
        f = v2[:3]
        m = v2[3:]
        cross_f = cross_mat(f)
        return np.block([[np.zeros_like(cross_f), cross_f], 
                         [cross_f, cross_mat(m)]])
    
    def _delete_cols(self, A, ranges):
        cols = []
        for r in ranges:
            cols = cols + [_ for _ in r]
        cols = set(cols)
        keep = [i for i in range(A.shape[1]) if i not in cols]
        return A.tocsc()[:,keep]

    def _BC_remover(self, A):
        n,m = A.shape
        del_idx_ranges = [range(0,6*12*self.n_nodes)]

        end = 0
        b = True #begining of a node
        #while end*6 < m - 12*6*5:
        while end*6 < m - 12*6*self.n_nodes:
            if b:
                b = False
                #start = end if not end == 0 else 5*12
                start = end if not end == 0 else self.n_nodes*12
            else:
                b = True
                #start = end + 8
                start = end + (self.n_nodes - 1)*2

            end = start + 1
            del_idx_ranges.append(range(start*6,end*6))
        del_idx_ranges.append(range(m-6*12*self.n_nodes,m))
        #print(del_idx_ranges)
        
        return self._delete_cols(A,del_idx_ranges) 
    
    def __init__(self, n_nodes, span, chord, dm, r_g, I_per_span, Cinv, e1 = np.array([[1, 0, 0]]).T,
k0 = np.array([[0, 0, 0]]).T):
        """
        Parameters
        ----------
        n_nodes : number of nodes in the beam
        span : span lenghth of wing
        chord : chord lenght of wing (assuming constant, might accept function later)
        k0 : pretwist of the beam
        dm : mass per unit span
        r_g : relative lenght from elastic axis to inertial axis (where grav acts), 
                r_g > 0 ==> inertial axis is in front
                r_g < 0 ==> inertial axis is behind
        I_per_span : mass moments PER UNIT SPAN matrix
        Cinv : stiffness matrix 
                Cinv = np.diag([EA, GAy, GAz, GJ, EIy, EIz])
        """
        self.n_nodes = n_nodes
        self.n_joins = ((n_nodes - 3) // 2)
        self.span = span
        self.chord = chord
        self.dm = dm
        self.r_g = r_g
        self.M = np.block([[dm*np.eye(3), -dm*cross_mat(r_g)],
                      [dm*cross_mat(r_g), I_per_span]])
        self.Cinv = Cinv
        self.C = inv(Cinv)
        
        self.eta_grid = np.linspace(0, span, n_nodes)
        self.h = self.eta_grid[1] - self.eta_grid[0]
        
        self.E = np.block([[cross_mat(k0), np.zeros_like(cross_mat(k0))],
                  [cross_mat(e1), cross_mat(k0)]])
        self.ET = self.E.T
        
        #Global matrix
        self.nodes_per_elem = 3
        self.nodes = {i+1: eta for i, eta in enumerate(self.eta_grid)}
        no_elems = (n_nodes - 1)//(self.nodes_per_elem - 1)
        if (self.n_nodes - 1) % (self.nodes_per_elem - 1) != 0:
            raise ValueError
        self.elems = {i+1: [j+1 for j in range(2*i, 2*i + 3)] for i in range(no_elems)}
        
        self.initalize_FEM()
        self.M_gl, self.K_lin_gl = self.MK_gl()
        self.K_nl_gl_ = self.K_nl_gl_kron()
        self.M_gl_inv = inv(self.M_gl)
        
        #BC and init
        self.V0, self.gBC, self.GBC, self.GBC_orth = self.init_and_BC()
        
        #Init aero model
        self.init_aero(r_a=np.array([[0,3.5e-3,0]]).T, rho=1.25, g=-10)
        
        self.index_to_elem = {i+1: range(12*(2*i),12*(2*i)+36) for i in range(len(self.elems))}
        
        self.NL_samples = []

        
        ints = {'111': 39, '112': 20, '113': -3, '122': 16, '123': -8, '133': -3, '222': 192, '223': 16, '233': 20, '333': 39}
        K_ij = {}
        for i in range(1, 4):
            for j in range(1, 4):
                Kij = np.array([ints[''.join(sorted(f'{i}{j}{k}'))] for k in [1, 2, 3]])
                K_ij[int(f'{i}{j}')] = Kij
        self.K_ij = K_ij
        
        self.NL_kronecker_build()

    def initalize_FEM(self):
        """
        Created the precomputed shape function intergral matrices
        """
        h = self.h
        self.phiphi = (h/15)*np.array([[4,2,-1],
                                       [2,16,2],
                                       [-1,2,4]])
        self.phiphiprime = (1/6)*np.array([[-3,4,-1],
                                           [-4,0,4],
                                           [1,-4,3]])
        self.phiphiphi = np.array([
            [39, 20, -3, 20,  16, -8, -3, -8, -3],
            [20, 16, -8, 16, 192, 16, -8, 16, 20],
            [-3, -8, -3, -8,  16, 20, -3, 20, 39]])
    #Element matrix
    def M_el(self):
        """
        VERIFIED
        """
        M, C, h = self.M, self.C, self.h
        z = np.zeros_like(M)
        M_gen = np.block([[M, z], [z, C]])
        return np.kron(self.phiphi, M_gen)
    
    def K_lin_el(self):
        """
        VERIFIED
        """
        E, ET = self.E, self.ET
        h = self.h
        z = np.zeros_like(E)
        I = np.eye(6)
        K_gen1 = np.block([[z, -E], [ET, z]])
        K_gen2 = np.block([[z,I],[I,z]])
        K_lin_el_1 = np.kron(self.phiphi, K_gen1)
        K_lin_el_2 = -np.kron(self.phiphiprime, K_gen2)
        return K_lin_el_1 + K_lin_el_2
    
    def NL_kronecker_build(self):
        """
        VERFIED
        Builds the nonlinear term matrix as well as the nonlinear jacobian term matrix, to be used with the kronecker product.
        """
        M, C, h = self.M, self.C, self.h
        Xcross = np.array([[0, 0, 0, 0, 0, 1, 0, -1, 0],
                           [0, 0, -1, 0, 0, 0, 1, 0, 0],
                           [0, 1, 0, -1, 0, 0, 0, 0, 0]])
        Zcross = np.zeros_like(Xcross)
        
        self.K1 = self.krondelacer2x2(n=3)
        self.K2 = self.krondelacer2x2(n=6)
        self.K3 = self.krondelacer3x3(n=12)
     
        self.L1mat = np.block([[Zcross, Zcross, Xcross, Zcross],
                       [Xcross, Zcross, Zcross, Xcross]]) @ self.K1
        
        self.L2mat = np.block([[Zcross, Xcross, Zcross, Zcross],
                       [Xcross, Zcross, Zcross, Xcross]]) @ self.K1
        
        self.L1Tmat = np.block([[Zcross, Xcross, Xcross, Zcross],
                       [Zcross, Zcross, Zcross, Xcross]]) @ self.K1
        
        zL = np.zeros_like(self.L1mat)
        fNL1 = np.block([[self.L1mat, self.L2mat, zL],
                        [zL, zL, -self.L1Tmat]])
        IkronM = np.kron(np.eye(M.shape[0]), M)
        IkronC = np.kron(np.eye(C.shape[0]), C)
        zkron = np.zeros_like(IkronM)
        fNL2 = np.block([[IkronM, zkron, zkron, zkron],
                         [zkron, zkron, zkron, IkronC],
                         [zkron, IkronC, zkron, zkron]])
        fNL = fNL1 @ fNL2 @ self.K2
        self.fNL_ = fNL
        
        JacfNL1 = np.block([[self.L2mat, zL, self.L1mat],
                            [zL, -self.L1Tmat, zL]])
        JacfNL2 = np.block([[IkronM, zkron, zkron, zkron],
                            [zkron, zkron, IkronC, zkron],
                            [zkron, zkron, zkron, IkronC]])
        
        self.JacfNL = fNL - JacfNL1 @ JacfNL2 @ self.K2
        return fNL #VERIFED!
    
    def krondelacer2x2(self, n, k=2):
        """ VERFIED
        Returns a matrix that turns the kron prod 
            | x1 |      | x1 |
            | x2 | kron | x2 |
        into 
            | x1 kron x1 |
            | x1 kron x2 |
            | x2 kron x1 |
            | x2 kron x2 |
            
        where x1, x2 have n-dimentional
        """
        krondelacer11 = np.zeros((n*n,2*n*n))
        krondelacer12 = np.zeros((n*n,2*n*n))

        for i in range(n):
            krondelacer11[i*n:(i+1)*n,(i*k)*n:(i*k + 1)*n] = np.eye(n)
            krondelacer12[i*n:(i+1)*n,(i*k + 1)*n:(i*k + 2)*n] = np.eye(n)

        krondelacer0s = np.zeros_like(krondelacer11)
        krondelacer = np.block([[krondelacer11, krondelacer0s],
                                [krondelacer12, krondelacer0s],
                                [krondelacer0s, krondelacer11],
                                [krondelacer0s, krondelacer12]])
        return krondelacer
    
    def krondelacer3x3(self, n):
        """ VERFIED
        Returns a matrix that turns the kron prod 
            | x1 |      | x1 |
            | x2 | kron | x2 |
            | x3 |      | x3 |

        into 
            | x1 kron x1 |
            | x1 kron x2 |
            | x1 kron x3 |
            |     ...    |
            | x3 kron x1 |
            | x3 kron x2 |
            | x3 kron x3 |
            
        where x1, x2 have n-dimentional
        """
        krondelacer1 = np.zeros((n*n,n*n*3))
        krondelacer2 = np.zeros((n*n,n*n*3))
        krondelacer3 = np.zeros((n*n,n*n*3))

        for i in range(n):
            krondelacer1[i*n:(i+1)*n,(i*3    )*n:(i*3 + 1)*n] = np.eye(n)
            krondelacer2[i*n:(i+1)*n,(i*3 + 1)*n:(i*3 + 2)*n] = np.eye(n)
            krondelacer3[i*n:(i+1)*n,(i*3 + 2)*n:(i*3 + 3)*n] = np.eye(n)
        krondelacer0s = np.zeros_like(krondelacer1)
        krondelacer = np.block([[krondelacer1, krondelacer0s, krondelacer0s],
                                [krondelacer2, krondelacer0s, krondelacer0s],
                                [krondelacer3, krondelacer0s, krondelacer0s],
                                [krondelacer0s, krondelacer1, krondelacer0s],
                                [krondelacer0s, krondelacer2, krondelacer0s],
                                [krondelacer0s, krondelacer3, krondelacer0s],
                                [krondelacer0s, krondelacer0s, krondelacer1],
                                [krondelacer0s, krondelacer0s, krondelacer2],
                                [krondelacer0s, krondelacer0s, krondelacer3]])
        return krondelacer
    

    def krondelacergl(self, n):
        """
        VERIFEID
        
        Created the delacing matrices K4 and K5 for the global FEM matrix
        Builds krondelacers as sparse matrices for memory management
        """
        k = self.n_nodes
        krondelacers = [lil_matrix((n*n, n*n*k), dtype=np.int8) for _ in range(k)]
        
        for i in range(n):
            for j, kron in enumerate(krondelacers):
                kron[i*n:(i+1)*n, (i*k + j)*n:(i*k + j + 1)*n] = eye(n, dtype=np.int8)
        
        # Convert to csr for faster operations
        krondelacers = [k.tocsr() for k in krondelacers]
        
        kblock = {}
        step = 2
        for nn in [3, 5]:  # renamed to avoid shadowing n
            start = 0
            suffix = 1
            while start + nn <= len(krondelacers):
                key = int(f"{nn}{suffix}")
                kblock[key] = vstack([krondelacers[start + i] for i in range(nn)], format='csr')
                start += step
                suffix += 1
    
        kblock30 = csr_matrix(kblock[31].shape, dtype=np.int8)
        kblock50 = csr_matrix(kblock[51].shape, dtype=np.int8)
        
        krondelacer5 = {}
        for i in range((k - 3) // 2):
            block_key = int(f'5{i+1}')
            key = int(f'{i+1}')
            
            n_before = 2*(i + 1)
            n_after  = k - n_before - 1
            
            blocks = ([kblock50] * n_before) + [kblock[block_key]] + ([kblock50] * n_after)
            krondelacer5[key] = hstack(blocks, format='csr')
    
        krondelacer3 = {}
        for i in range(k//2):
            for j in range(3):
                block_key = int(f'3{i+1}')
                key = int(f'{i+1}{j+1}')
                
                n_before = 2*i + j
                n_after  = k - 2*i - j - 1
                
                blocks = ([kblock30] * n_before) + [kblock[block_key]] + ([kblock30] * n_after)
                krondelacer3[key] = hstack(blocks, format='csr')
                
                
        krondelacer_final = []
        for i in range(((k - 3) // 2) + 1):
            if i == 0:
                krondelacer_final.append(vstack([krondelacer3[int(f'{i+1}1')],krondelacer3[int(f'{i+1}2')]]))
            elif i == ((k - 3) // 2):
                krondelacer_final.append(vstack([krondelacer5[i], krondelacer3[int(f'{i+1}2')],krondelacer3[int(f'{i+1}3')]]))
            else:
                krondelacer_final.append(vstack([krondelacer5[i], krondelacer3[int(f'{i+1}2')]]))

        K4 = vstack(krondelacer_final)
        self.K4 = K4
        #Creating K5 which works the same but on the interor nodes only
        self.K5 = self._BC_remover(K4)
        return K4
        

    def K_nl_el_kron(self):
        self.fNL_el = (self.h/210)*np.kron(self.phiphiphi, self.fNL) @ self.krondelacer3x3(12)
        return self.fNL_el 
    
    
    def K_nl_el(self, V):
        """Legacy bersion of K_nl_el_kron, not using the Kronecker product"""
        L1, L2, L1T, M, C, h = self._L1, self._L2, self._L1T, self.M, self.C, self.h
        #V = (v11, v21, v21, v22, v13, v23)
        v1 = np.array([[V[:6], V[12:18], V[24:30]]]).reshape((-1,3)) #v11, v12, v13
        v2 = np.array([[V[6:12], V[18:24], V[30:36]]]).reshape((-1,3)) #v21, v22, v23
        v11, v12, v13 = V[:6], V[12:18], V[24:30]
        v21, v22, v23 = V[6:12], V[18:24], V[30:36]
        V_stretched = np.vstack((v11,v11,v11,v21,v21,v21,v12,v12,v12,v22,v22,v22,v13,v13,v13,v23,v23,v23))
        z = np.zeros_like(M)
        K = self.phiphiphi
        return (h/210)*np.block([
[K[0,0]*L1(v11)@M + K[0,1]*L1(v12)@M + K[0,2]*L1(v13)@M, K[0,0]*L2(v21)@C + K[0,1]*L2(v22)@C + K[0,2]*L2(v23)@C, K[0,3]*L1(v11)@M + K[0,4]*L1(v12)@M + K[0,5]*L1(v13)@M, K[0,3]*L2(v21)@C + K[0,4]*L2(v22)@C + K[0,5]*L2(v23)@C, K[0,6]*L1(v11)@M + K[0,7]*L1(v12)@M + K[0,8]*L1(v13)@M, K[0,6]*L2(v21)@C + K[0,7]*L2(v22)@C + K[0,8]*L2(v23)@C],
[z, -K[0,0]*L1T(v11)@C -K[0,1]*L1T(v12)@C -K[0,2]*L1T(v13)@C, z, -K[0,3]*L1T(v11)@C -K[0,4]*L1T(v12)@C -K[0,5]*L1T(v13)@C, z, -K[0,6]*L1T(v11)@C -K[0,7]*L1T(v12)@C -K[0,8]*L1T(v13)@C],
[K[1,0]*L1(v11)@M + K[1,1]*L1(v12)@M + K[1,2]*L1(v13)@M, K[1,0]*L2(v21)@C + K[1,1]*L2(v22)@C + K[1,2]*L2(v23)@C, K[1,3]*L1(v11)@M + K[1,4]*L1(v12)@M + K[1,5]*L1(v13)@M, K[1,3]*L2(v21)@C + K[1,4]*L2(v22)@C + K[1,5]*L2(v23)@C, K[1,6]*L1(v11)@M + K[1,7]*L1(v12)@M + K[1,8]*L1(v13)@M, K[1,6]*L2(v21)@C + K[1,7]*L2(v22)@C + K[1,8]*L2(v23)@C],
[z, -K[1,0]*L1T(v11)@C -K[1,1]*L1T(v12)@C -K[1,2]*L1T(v13)@C, z, -K[1,3]*L1T(v11)@C -K[1,4]*L1T(v12)@C -K[1,5]*L1T(v13)@C, z, -K[1,6]*L1T(v11)@C -K[1,7]*L1T(v12)@C -K[1,8]*L1T(v13)@C],
[K[2,0]*L1(v11)@M + K[2,1]*L1(v12)@M + K[2,2]*L1(v13)@M, K[2,0]*L2(v21)@C + K[2,1]*L2(v22)@C + K[2,2]*L2(v23)@C, K[2,3]*L1(v11)@M + K[2,4]*L1(v12)@M + K[2,5]*L1(v13)@M, K[2,3]*L2(v21)@C + K[2,4]*L2(v22)@C + K[2,5]*L2(v23)@C, K[2,6]*L1(v11)@M + K[2,7]*L1(v12)@M + K[2,8]*L1(v13)@M, K[2,6]*L2(v21)@C + K[2,7]*L2(v22)@C + K[2,8]*L2(v23)@C],
[z, -K[2,0]*L1T(v11)@C -K[2,1]*L1T(v12)@C -K[2,2]*L1T(v13)@C, z, -K[2,3]*L1T(v11)@C -K[2,4]*L1T(v12)@C -K[2,5]*L1T(v13)@C, z, -K[2,6]*L1T(v11)@C -K[2,7]*L1T(v12)@C -K[2,8]*L1T(v13)@C]])
    
    
    def MK_gl(self):
        M_gl = np.zeros((self.n_nodes*12, self.n_nodes*12))
        K_lin_gl = np.zeros((self.n_nodes*12, self.n_nodes*12))
        for el in self.elems.values():
            i = el[0]
            M_gl[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += self.M_el()
            K_lin_gl[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += self.K_lin_el()
        return M_gl, K_lin_gl
                
    
    def K_nl_gl(self, V):
        K_nl_global = np.zeros((self.n_nodes*12, self.n_nodes*12))
        end = None
        for el in self.elems.values():
            i = el[0]
            V_el = V[(i-1)*12:(i+2)*12]
            K_el = self.K_nl_el(V_el)
            l = K_el.shape[0]
            start = end - 12 if not end is None else 0
            end = start + l
            K_nl_global[start:end,start:end] += K_el
        return K_nl_global
    
    def K_nl_gl_kron(self):
        """
        VERIFIED
        """
        fNL = self.NL_kronecker_build()
        K_el = (self.h/210)*np.kron(self.phiphiphi, fNL) 
        l = K_el.shape
        K4 = self.krondelacergl(12)
        n_notjoins = self.n_nodes - self.n_joins
        K_nl_global = np.zeros((self.n_nodes*12, 12*12*(5*self.n_joins + 3*n_notjoins)))
        end = None
        end_row = None
        for el in self.elems.values():
            start = end - 12*12 if not end is None else 0
            end = start + l[1]
            start_row = end_row - 12 if not end_row is None else 0
            end_row = start_row + l[0]
            #print(i, start, end, start_row, end_row)
            K_nl_global[start_row:end_row,start:end] += K_el
        return K_nl_global @ K4

    def JacK_nl_gl_kron(self):
        """
        VERIFIED
        """
        JacfNL2 = self.JacfNL
        JacfNL2_el = (self.h/210)*np.kron(self.phiphiphi, JacfNL2) 
        JacfNL1 = self.fNL_
        JacfNL1_el = (self.h/210)*np.kron(self.phiphiphi, JacfNL1) 
        K_el = JacfNL1_el - JacfNL2_el
        l = K_el.shape
        K4 = self.krondelacergl(12)
        K_nl_global = np.zeros((self.n_nodes*12, 12*12*(5*6+3*5+6*2)))
        end = None
        end_row = None
        for el in self.elems.values():
            start = end - 12*12 if not end is None else 0
            end = start + l[1]
            start_row = end_row - 12 if not end_row is None else 0
            end_row = start_row + l[0]
            #print(i, start, end, start_row, end_row)
            K_nl_global[start_row:end_row,start:end] += K_el
        return K_nl_global @ K4
    
    

    def force_templates(self):
        """
        Generates some template forces for the FEM model, like uniform force, tip force along different axis
        """
        fz_tip_ext = np.array([0,0,1])
        mz_tip_ext = np.array([0,0,0])
        fy_tip_ext = np.array([0,1,0])
        my_tip_ext = np.array([0,0,0])
        Fztip_gl = np.block([np.zeros((1,(self.n_nodes-1)*12)), fz_tip_ext,mz_tip_ext,np.zeros((1,6))]).T
        Fytip_gl = np.block([np.zeros((1,(self.n_nodes-1)*12)), fy_tip_ext,my_tip_ext,np.zeros((1,6))]).T

        
        fy_ext = np.array([[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
        fz_ext = np.array([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
        Fy_gl = np.vstack([fy_ext for _ in range(self.n_nodes)])
        Fz_gl = np.vstack([fz_ext for _ in range(self.n_nodes)])
        return Fytip_gl, Fztip_gl, Fy_gl, Fz_gl
        
    def init_and_BC(self):
        #Zero displacement init condt
        V0 = np.zeros((self.n_nodes*12, 1))
        
        #BC matricies (Currently implemented as cantilevered beam)
        G0 = np.block([[np.eye(6), np.zeros((6,6))], [np.zeros((6,6)), np.zeros((6,6))]])
        G0_orth = np.block([[np.zeros((6,6)), np.zeros((6,6))], [np.zeros((6,6)), np.eye(6)]])
    
        #sL will be free hence
        GL = G0_orth
        GL_orth = G0
        g0 = gL = np.zeros((12, 1))
        gBC = np.vstack((g0, np.zeros(((self.n_nodes - 2)*12,1)), gL))
        GBC = scipy.linalg.block_diag(G0, np.zeros(((self.n_nodes - 2)*12,(self.n_nodes - 2)*12)), GL)
        GBC_orth = scipy.linalg.block_diag(G0_orth, np.eye((self.n_nodes - 2)*12), GL_orth)
        return V0, gBC, GBC, GBC_orth
    
    def post(self, Vn):
        """
        Post-processor, computes the displacement and finite rotations
        """
        C, h = self.C, self.h
        if len(Vn.shape) == 1:
            f_int_x = Vn[6::12]
            f_int_y = Vn[7::12]
            f_int_z = Vn[8::12]
            m_int_x = Vn[9::12]
            m_int_y = Vn[10::12]
            m_int_z = Vn[11::12]
            
            Cx, Cy, Cz = C[0,0], C[1,1], C[2,2]
            C_rotx, C_roty, C_rotz = C[3,3], C[4,4], C[5,5]
            
            strainx = f_int_x * Cx
            strainy = f_int_y * Cy        
            strainz = f_int_z * Cz

            curvex = C_rotx*m_int_x
            curvey = C_roty*m_int_y
            curvez = C_rotz*m_int_z
            
            strainx_prime = np.vstack(((np.zeros((1,1))),strainx.reshape((-1,1))[:-1]))
            strainy_prime = np.vstack(((np.zeros((1,1))),strainy.reshape((-1,1))[:-1]))
            strainz_prime = np.vstack(((np.zeros((1,1))),strainz.reshape((-1,1))[:-1]))
            curvex_prime = np.vstack(((np.zeros((1,1))),curvex.reshape((-1,1))[:-1]))
            curvey_prime = np.vstack(((np.zeros((1,1))),curvey.reshape((-1,1))[:-1]))
            curvez_prime = np.vstack(((np.zeros((1,1))),curvez.reshape((-1,1))[:-1]))
        
            
            Alpha = np.identity(n=strainz.shape[0]) - np.diag(np.ones((strainz.shape[0] - 1)), k=-1)
            Al_inv = inv(Alpha)
            
            posx = Al_inv @ (Al_inv @ -curvez_prime*h*h + strainx_prime * h )
            posy = Al_inv @ (Al_inv @ -curvex_prime*h*h + strainy_prime * h )
            posz = Al_inv @ (Al_inv @ -curvey_prime*h*h + strainz_prime * h )
            
            rotx = Al_inv @ curvex_prime*h
            roty = Al_inv @ curvey_prime*h
            rotz = Al_inv @ curvez_prime*h
        
            X = np.vstack([np.array([posx[i], posy[i], posz[i], rotx[i], roty[i], rotz[i]]) for i in range(len(posz))])
        else:
            n_dims, n_samples = Vn.shape
            n_nodes = n_dims // 12
            f_int_x = Vn[6::12,:]
            f_int_y = Vn[7::12,:]
            f_int_z = Vn[8::12,:]
            m_int_x = Vn[9::12,:]
            m_int_y = Vn[10::12,:]
            m_int_z = Vn[11::12,:]
            
            Cx, Cy, Cz = C[0,0], C[1,1], C[2,2]
            C_rotx, C_roty, C_rotz = C[3,3], C[4,4], C[5,5]
            
            strainx = f_int_x * Cx
            strainy = f_int_y * Cy        
            strainz = f_int_z * Cz

            curvex = C_rotx*m_int_x
            curvey = C_roty*m_int_y
            curvez = C_rotz*m_int_z
            
            strainx_prime = np.vstack(((np.zeros((1, n_samples))),strainx[:-1,:]))
            strainy_prime = np.vstack(((np.zeros((1, n_samples))),strainy[:-1,:]))
            strainz_prime = np.vstack(((np.zeros((1, n_samples))),strainz[:-1,:]))
            curvex_prime = np.vstack(((np.zeros((1, n_samples))),curvex[:-1,:]))
            curvey_prime = np.vstack(((np.zeros((1, n_samples))),curvey[:-1,:]))
            curvez_prime = np.vstack(((np.zeros((1, n_samples))),curvez[:-1,:]))
        
            
            Alpha = np.identity(n=strainz.shape[0]) - np.diag(np.ones((strainz.shape[0] - 1)), k=-1)
            Al_inv = inv(Alpha)
            
            posx = Al_inv @ (Al_inv @ -curvez_prime*h*h + strainx_prime * h )
            posy = Al_inv @ (Al_inv @ -curvex_prime*h*h + strainy_prime * h )
            posz = Al_inv @ (Al_inv @ -curvey_prime*h*h + strainz_prime * h )
            
            rotx = Al_inv @ curvex_prime*h
            roty = Al_inv @ curvey_prime*h
            rotz = Al_inv @ curvez_prime*h
        
            X = np.vstack([np.array([posx[i,:], posy[i,:], posz[i,:], rotx[i,:], roty[i,:], rotz[i,:]]) for i in range(n_nodes)])
        return X
    
    def post_dyn(self, soly):
        out = []
        for y in soly:
            out.append(self.post(y))
        return np.array(out)

    def static(self, V_sol, F,  NLinclude):
        interior = slice(6,-6)
        V_sol = V_sol.reshape((-1,1))
        NL = self.K_nl_gl_ @ np.kron(V_sol, V_sol).reshape((-1,1))
        #print(NL.max())
        res_int = self.K_lin_gl@V_sol - F  + NL*NLinclude
        res = self.GBC@V_sol - self.gBC
        res[interior] = res_int[interior]
        return res.reshape((-1,))
    
    def static_legacy(self, V_sol, F, NLinclude):
        interior = slice(6,-6)
        V_sol = V_sol.reshape((-1,1))
        NL = self.K_nl_gl(V_sol) @ V_sol
        self.NL_samples.append(NL.T)
        #print(NL.max())
        res_int = self.K_lin_gl@V_sol + NL*NLinclude - F 
        res = self.GBC@V_sol - self.gBC
        res[interior] = res_int[interior]
        return res.reshape((-1,))
    
    def static_solver(self, F, V0 = None, dt = 0.1, full_output = True, anal_jac = True, legacy = False, NLinclude = 1):
        self.NL_samples = []
        V0 = self.V0 if V0 is None else V0
        #initazlie with current V
        #Impose BC by premultiplying G0_orth and GL_orth to first and last nodes
        V0 = self.GBC_orth @ V0
        if legacy:
            static = self.static_legacy
            jac = self.res_jac_legacy
        else:
            static = self.static
            jac = self.res_jac
        if anal_jac:
            fprime = jac
        else:
            fprime = None
        V_sol = fsolve(lambda V: static(V, F, NLinclude), V0, fprime=fprime, full_output=full_output)
        if full_output:
            return V_sol[0], V_sol[-1]
        else:
            return V_sol
        
    def coupled_solver(self, vel, AoA=0, n=1, V0 = None, full_output = False):
        M_gl, K_lin_gl, K_nl_gl = self.M_gl, self.K_lin_gl, self.K_nl_gl
        GBC, GBC_orth, gBC = self.GBC, self.GBC_orth, self.gBC
        V0 = self.V0 if V0 is None else V0
        #initazlie with current V
        #Impose BC by premultiplying G0_orth and GL_orth to first and last nodes
        V0 = GBC_orth @ V0
        V_sol = fsolve(lambda V: self.static(V, M_gl, K_lin_gl, K_nl_gl, self.elemwise_cst_load(self.aeromodel.force_per_elem(V=V, AoA=AoA, vel=vel, n=n)), GBC, gBC), V0,full_output=full_output)
        if full_output:
            return V_sol[0], V_sol[-1]
        else:
            return V_sol
        
    def elemwise_cst_load(self, F_per_elem):
        """F_e is a list of the general force vectors on each elements, which are to be translated into nodal loads for FEM. The geeneral force vector has the following shape:
            F_e = [fx fy fz mx my mz 0 0 0 0 0 0].T
        as does the output nodal force. 
        The translation is done by using the shape functions of the element. 
        """
        h = self.h
        shape_ints = [1/6, 2/3, 1/6]
        F_nodal = np.zeros((self.n_nodes * 12,1))
        for j, fe in enumerate(F_per_elem):
            i = min(self.elems[j+1])
            F_e_nodal = 3*h*np.vstack([fe * s for s in shape_ints])
            F_nodal[(i-1)*12:(i+2)*12,:] += F_e_nodal
        return F_nodal
    
    def phi1(self, s):
        if s < 0 or s > 1:
            raise
        return 2*(s - 1)*(s - 0.5)

    def phi2(self, s):
        if s < 0 or s > 1:
            raise
        return -4*s*(s-1)

    def phi3(self, s):
        if s < 0 or s > 1:
            raise
        return 2*s*(s-0.5)
    
    def shape_funcs(self, s):
        return np.array([[self.phi1(s), self.phi2(s), self.phi3(s)]])
    
    def init_aero(self, r_a, rho, g):
        self.aeromodel = ae.AeroStrip(self, rho, g, r_a)
        
    def JacNL_elem(self,V):
     L1, L2, M, C, h = self._L1, self._L2, self.M, self.C, self.h
     v1 = np.array([[V[:6], V[12:18], V[24:30]]]).reshape((-1,3)) #v11, v12, v13
     v2 = np.array([[V[6:12], V[18:24], V[30:36]]]).reshape((-1,3)) #v21, v22, v23
     u1 = C @ v2
     u2 = M @ v1
     z = np.zeros_like(M)
     K_ij = self.K_ij
     #---------JacNL2-------------------------------        
     JacNL_elem2 = (h/210)*np.block(
         [[L2(u2@K_ij[11]), L1(u1@K_ij[11]), L2(u2@K_ij[12]), L1(u1@K_ij[12]), L2(u2@K_ij[13]), L1(u1@K_ij[13])],
          [-1*L1(u1@K_ij[11]).T, z, -1*L1(u1@K_ij[12]).T, z, -1*L1(u1@K_ij[13]).T, z],
          [L2(u2@K_ij[21]), L1(u1@K_ij[21]), L2(u2@K_ij[22]), L1(u1@K_ij[22]), L2(u2@K_ij[23]), L1(u1@K_ij[23])],
          [-1*L1(u1@K_ij[21]).T, z, -1*L1(u1@K_ij[22]).T, z, -1*L1(u1@K_ij[23]).T,z],
          [L2(u2@K_ij[31]), L1(u1@K_ij[31]), L2(u2@K_ij[32]), L1(u1@K_ij[32]), L2(u2@K_ij[33]), L1(u1@K_ij[33])],
          [-1*L1(u1@K_ij[31]).T, z, -1*L1(u1@K_ij[32]).T, z, -1*L1(u1@K_ij[32]).T,z]])
     
     #---------JacNL1-------------------------------             
     JacNL_elem1 = (h/210)*np.block([[L1(v1@K_ij[11])@M, L2(v2@K_ij[11])@C, L1(v1@K_ij[12])@M, L2(v2@K_ij[12])@C, L1(v1@K_ij[13])@M, L2(v2@K_ij[13])@C],
                         [z, -1*L1(v1@K_ij[11]).T@C, z, -1*L1(v1@K_ij[12]).T@C, z, -1*L1(v1@K_ij[13]).T@C],
                         [L1(v1@K_ij[21])@M, L2(v2@K_ij[21])@C, L1(v1@K_ij[22])@M, L2(v2@K_ij[22])@C, L1(v1@K_ij[23])@M, L2(v2@K_ij[23])@C],
                         [z, -1*L1(v1@K_ij[21]).T@C, z, -1*L1(v1@K_ij[22]).T@C, z, -1*L1(v1@K_ij[23]).T@C],
                         [L1(v1@K_ij[31])@M, L2(v2@K_ij[31])@C, L1(v1@K_ij[32])@M, L2(v2@K_ij[32])@C, L1(v1@K_ij[33])@M, L2(v2@K_ij[33])@C],
                                             [z, -1*L1(v1@K_ij[31]).T@C, z, -1*L1(v1@K_ij[32]).T@C, z, -1*L1(v1@K_ij[32]).T@C]])
     return JacNL_elem1 - JacNL_elem2
 
    def JacNL_gl(self,V):
     JacNL_global = np.zeros((self.n_nodes*12, self.n_nodes*12))
     for el in self.elems.values():
         i = el[0]
         V_el = V[(i-1)*12:(i+2)*12]
         JacNL_global[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += self.JacNL_elem(V_el)
     return JacNL_global
 
    def res_jac_legacy(self,V):
        interior=slice(6,-6)
        jacres_int = self.K_lin_gl + self.JacNL_gl(V)
        jacres = np.eye(self.n_nodes*12)
        jacres[interior,interior] = jacres_int[interior,interior]
        return jacres
    
    def res_jac(self, V):
        interior=slice(6,-6)
        Vc = V.reshape((-1)) 
        H = self.K_nl_gl_.reshape(self.n_nodes*12, self.n_nodes*12, self.n_nodes*12)
        jacres_int = self.K_lin_gl + np.einsum('ijk,j->ik', H, Vc) + np.einsum('ijk,k->ij', H, Vc)        
        jacres = np.eye(self.n_nodes*12)
        jacres[interior,interior] = jacres_int[interior,interior]
        return jacres
    
    def dynamic_solver(self,F,tspan,V0=None,NLinclude=1):
        interior = slice(6,-6)
        V0 = self.V0 if V0 is None else V0
        H = self.K_nl_gl_.reshape(self.n_nodes*12, self.n_nodes*12, self.n_nodes*12)
        if not callable(F):
            F0 = F
            F = lambda t: F0*(1 - np.exp(-t*50))
        def dVdt(t,V_sol):
            print(t, V_sol.max())
            V_sol = V_sol.reshape((-1,1))
            NL = self.K_nl_gl_ @ np.kron(V_sol, V_sol).reshape((-1,1))
            res_int = F(t) - self.K_lin_gl@V_sol - NL*NLinclude
            ddt = self.M_gl_inv @ res_int
            out = np.zeros_like(V_sol)
            out[interior] = ddt[interior]
            return out.reshape((-1,))
        def jac(t,V):
            Vc = V.reshape((-1)) 
            jacres_int = self.K_lin_gl + np.einsum('ijk,j->ik', H, Vc) + np.einsum('ijk,k->ij', H, Vc)        
            jacres = np.eye(self.n_nodes*12)
            jacres[interior,interior] = jacres_int[interior,interior]
            return -self.M_gl_inv @ jacres
        start = time.time()
        sol = solve_ivp(dVdt, tspan, V0.reshape(-1), method='BDF', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, jac=jac)
        FOMtime = time.time() - start
        return sol.y, sol.t, FOMtime
        
    
    
    
        
class BeamNL_POD:
    def __init__(self, beamNL, l, r = None, center=0, interior_lims = (6,6)):
        self.interior = slice(interior_lims[0], -interior_lims[1])
        self.interior_lims = interior_lims
        self.n_nodes = beamNL.n_nodes
        self.n_joins = beamNL.n_joins
        self.l = l
        self.r = r if not r is None else l
        self.M_gl = beamNL.M_gl
        self.K_lin_gl = beamNL.K_lin_gl
        self.n_nodes_per_elem = beamNL.nodes_per_elem
        self._L1, self._L2, self.M, self.C, self.h = beamNL._L1, beamNL._L2, beamNL.M, beamNL.C, beamNL.h
        self.eta_grid = beamNL.eta_grid
        self.post = beamNL.post
        self.V0 = beamNL.V0
        self.center = center
        self.elems = beamNL.elems
        self.history = []
        self.GBC, self.GBC_orth, self.gBC = beamNL.GBC, beamNL.GBC_orth, beamNL.gBC
        self.nodes_per_elem = beamNL.nodes_per_elem
        self.index_to_elem = beamNL.index_to_elem
        self.JacNL_gl = beamNL.JacNL_gl
        self.M_gl, self.K_lin_gl = beamNL.M_gl, beamNL.K_lin_gl
        self.K_nl_gl = beamNL.K_nl_gl
        self.K_nl_gl_ = beamNL.K_nl_gl_
        self.K5 = beamNL.K5
        self.NL_kronecker_build = beamNL.NL_kronecker_build
        self.phiphiphi = beamNL.phiphiphi
        
    def K_nl_gl_kron_r(self,PHI,interior):
        """
        VERIFIED
        """
        fNL = self.NL_kronecker_build()
        K_el = (self.h/210)*np.kron(self.phiphiphi, fNL) 
        l = K_el.shape
        n_notjoins = self.n_nodes - self.n_joins
        K_nl_global = np.zeros((self.n_nodes*12, 12*12*(5*self.n_joins + 3*n_notjoins)))
        end = None
        end_row = None
        for el in self.elems.values():
            start = end - 12*12 if not end is None else 0
            end = start + l[1]
            start_row = end_row - 12 if not end_row is None else 0
            end_row = start_row + l[0]
            #print(i, start, end, start_row, end_row)
            K_nl_global[start_row:end_row,start:end] += K_el
        K_nl_gl_r_outerdim = K_nl_global @ self.K5 @ np.kron(PHI,PHI)
        return PHI.T @ K_nl_gl_r_outerdim[interior,:]
    
    def res_jac_POD(self,V):
        #interior=slice(6,-6)
        Vc = V.reshape((-1)) 
        H = self.K_nl_gl_r_.reshape(self.l, self.l, self.l)
        jacres_int = self.K_lin_gl_r + np.einsum('ijk,j->ik', H, Vc) + np.einsum('ijk,k->ij', H, Vc)        
        #jacres = np.eye(self.n_nodes*12)
        #jacres[interior,interior] = jacres_int[interior,interior]
        return jacres_int
    
    #POD OFFLINE
    def _POD_offline(self, X):
        n_dim, n_samples = X.shape
        V_mean = X.mean(axis=1, keepdims=True)
        V_mean_mat = np.hstack(tuple([V_mean for _ in range(n_samples)]))
        X_cen = X - self.center*V_mean_mat
        POD_basis, sigmas, _ = np.linalg.svd(X_cen, full_matrices=False)
        PHI = POD_basis[:,:self.l]
        self.V_mean = V_mean
        return PHI, sigmas
    
    def POD_offline(self, X_samples):
        #POD OFFLINE
        interior = self.interior
        PHI, self.sigmas = self._POD_offline(X_samples)       
        self.PHI = PHI
        #self.M_gl_r = PHI.T @ self.M_gl @ PHI
        #self.K_lin_gl_r = PHI.T @ self.K_lin_gl @ PHI
        #self.V0_r = PHI.T @ self.V0    
        #self.K_nl_gl_r_ = PHI.T @ self.K_nl_gl_ @ np.kron(PHI,PHI)
        self.M_gl_r = PHI.T @ self.M_gl[interior,interior] @ PHI
        self.K_lin_gl_r = PHI.T @ self.K_lin_gl[interior,interior] @ PHI
        self.V0_r = PHI.T @ self.V0[interior]    
        self.K_nl_gl_r_ = self.K_nl_gl_kron_r(PHI,interior)
    
    def static_POD_legacy(self, V_sol_r, F_r, NLinclude):
        #print('Legacy mode, using old definition of K_nl_gl without the Kronecker product')
        interior, BC_dims, PHI = self.interior, sum(self.interior_lims), self.PHI
        interior_dims, l = PHI.shape
        V_sol_r = V_sol_r.reshape((-1,1))
        V_sol_recon = np.zeros((interior_dims + BC_dims,1))
        V_sol_recon[self.interior] = PHI @ V_sol_r
        NL = self.K_nl_gl(V_sol_recon) @ V_sol_recon
        NL_r = PHI.T @ NL[interior]
        L_r = self.K_lin_gl_r @ V_sol_r
        res = L_r - F_r.reshape((-1,1)) + NL_r*NLinclude
        return res.reshape((-1,))
    
    def static_POD(self, V_sol_r, F_r, NLinclude):
        NL_r = self.K_nl_gl_r_ @ np.kron(V_sol_r,V_sol_r)
        L_r = self.K_lin_gl_r @ V_sol_r
        res = L_r + NL_r*NLinclude - F_r
        return res.reshape((-1,))
    
    def static_diagnostic(self, V_sol, F):
        print('Diagnostic mode, using the FOM solver to find solution and using that for diagnosis of POD tools')
        interior = slice(6,-6)
        V_sol = V_sol.reshape((-1,1))
        NL = self.K_nl_gl_ @ np.kron(V_sol, V_sol).reshape((-1,1))
        
        NL_r2 = self.PHI.T @ NL[interior]   
        V_sol_r = self.PHI.T @ V_sol[interior]
        NL_r1 = self.K_nl_gl_r_ @ np.kron(V_sol_r,V_sol_r)
        
        print('NL_r, trad = ', np.round(np.max(NL_r2), 10))
        print('NL_r diff = ',np.max(np.abs(NL_r2 - NL_r1)))

        res_int = self.K_lin_gl@V_sol - F.reshape((-1,1))  + NL
        res = self.GBC@V_sol - self.gBC
        res[interior] = res_int[interior]
        return res.reshape((-1,))
    
    
        
    def static_solver_POD(self, F, full_output=True, V0=None, anal_jac=True, legacy = False, NLinclude = 1, diagnostic = False, timer=False):
        interior = self.interior
        V_POD = np.zeros_like(self.V0)
        F_r = self.PHI.T @ F[interior]
        V0_r = self.V0_r if V0 is None else self.PHI.T @ V0[interior]
        if legacy:
            static = self.static_POD_legacy
        else:
            static = self.static_POD
        
        if anal_jac:
            fprime = self.res_jac_POD
        else:
            fprime=None
        if diagnostic:
            V_FOM = fsolve(lambda V: self.static_diagnostic(V, F.reshape(-1)), self.V0.reshape(-1), fprime=None ,full_output=True)
            return V_FOM[0], V_FOM[-1]
        start = time.time()
        V_POD_r = fsolve(lambda V_r: static(V_r, F_r.reshape(-1), NLinclude), V0_r.reshape(-1), fprime=fprime ,full_output=full_output)
        elasped = time.time() - start
        V_POD[interior] = self.PHI @ V_POD_r[0].reshape((-1,1)) #+ self.V_mean*self.center
        out = [V_POD]
        if full_output:
            out.append(V_POD_r[-1])
        if timer:
            out.append(elasped)
        return out
    
    def dynamic_solver_POD(self,F,tspan,V0=None,NLinclude=1):
        interior = self.interior
        V0_r = self.V0_r if V0 is None else self.PHI.T @ V0[interior]
        F0_r = self.PHI.T @ F[interior]
        M_gl_r_inv = inv(self.M_gl_r)
        if not callable(F):
            F0_r = self.PHI.T @ F[interior]
            def F_r(t):
                if t < 0.01:
                    return F0_r*(1 - np.exp(-t*50))
                else:
                    return 0*F0_r
        else:
            print('NOT IMPLEMENTED YET')
            raise
        def dV_rdt(t,V_r):
            return -1*M_gl_r_inv @ self.static_POD(V_r, F_r(t).reshape(-1), NLinclude)
        def jac(t,V):
            return -M_gl_r_inv @ self.res_jac_POD(V)
        start = time.time()
        sol = solve_ivp(dV_rdt, tspan, V0_r.reshape(-1), method='BDF', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, jac=jac)
        PODtime = time.time() - start
        soly_r = sol.y
        soly = np.zeros((self.n_nodes*12,soly_r.shape[1]))
        soly[interior,:] = self.PHI @ soly_r
        return soly, sol.t, PODtime
        
    def _POD_DEIM_offline(self, X, NL,):
        r = self.l if self.r is None else self.r
        center=self.center
        n_dim, n_samples = X.shape
        V_mean = X.mean(axis=1, keepdims=True)
        V_mean_mat = np.hstack(tuple([V_mean for _ in range(n_samples)]))
        X_cen = X - center*V_mean_mat
        POD_basis, sigmas, _ = np.linalg.svd(X_cen)
        PHI = POD_basis[:,:self.l]
        XI, _, _ = np.linalg.svd(NL, full_matrices=False)
        XI_r = XI[:,0].reshape(n_dim,1)
        z = np.zeros((n_dim,1))
        P = np.copy(z)
        index_1 = np.argmax(np.abs(XI_r))
        P[index_1] = 1
        for i in range(1,r):
            XI_rp1 = XI[:,i].reshape(n_dim,1)
            c = np.linalg.solve(P.T@XI_r, P.T@XI_rp1)
            res = XI_rp1 - XI_r@c 
            index_i = np.argmax(np.abs(res))
            P_new = np.copy(z)
            P_new[index_i] = 1
            P = np.concatenate((P, P_new), axis = 1)
            XI_r = np.concatenate((XI_r, XI_rp1), axis = 1)
        N_approx = XI_r @ np.linalg.inv(P.T @ XI_r)
        N_approx_l = PHI.T @ N_approx    
        PT_PHI = P.T @ PHI    
        return PHI, N_approx_l, PT_PHI, P, N_approx
    
    def POD_DEIM_offline(self, X_samples, NL_samples):
        interior = self.interior
        PHI, self.N_approx_l, self.PT_PHI, self.P, self.N_approx = self._POD_DEIM_offline(X_samples, NL_samples)
        self.PHI = PHI
        self.V_mean = np.zeros_like(self.V0)
        self.M_gl_r = PHI.T @ self.M_gl[interior,interior] @ PHI
        self.K_lin_gl_r = PHI.T @ self.K_lin_gl[interior,interior] @ PHI
        self.V0_r = PHI.T @ self.V0[interior]       
        self.DEIM_ps = np.where(self.P.T == 1)[1]
        finder = self.kron_finder()
        self.relevant_xkronx = []
        self.DEIM_idx = {}
        for j in self.DEIM_ps:
            res = finder[j]
            self.relevant_xkronx.append(res)
            for entry in  res.split(':'):
                i,j = entry.split(',')
                i = int(i.strip('[] ')) - 1
                j = int(j.strip('[] ')) - 1
                self.DEIM_idx[entry.strip()] = i*12 + j
        self.PHI_DEIM = PHI[[v for v in self.DEIM_idx.values()],:]
        self.PTK_nl_gl_ = self.P.T @ self.K_nl_gl_kron_DEIM(interior)
            
    def kron_finder(self):
        out = []
        self.finder2 = {}
        k = 0
        for i in range((self.n_nodes-1)*12): #ONLY INNER NODES
            node_i = i // 12
            inner_i = i % 12
            for j in range((self.n_nodes-1)*12):
                node_j = j // 12
                inner_j = j % 12
                out.append(f'[{node_i + 1},{inner_i + 1}] : [{node_j + 1},{inner_j + 1}]')
                self.finder2[f'[{node_i + 1},{inner_i + 1}] : [{node_j + 1},{inner_j + 1}]'] = k
                k+=1
                
        return out
    
    def K_nl_gl_kron_DEIM(self,interior):
        fNL = self.NL_kronecker_build()
        K_el = (self.h/210)*np.kron(self.phiphiphi, fNL) 
        l = K_el.shape
        n_notjoins = self.n_nodes - self.n_joins
        K_nl_global = np.zeros((self.n_nodes*12, 12*12*(5*self.n_joins + 3*n_notjoins)))
        end = None
        end_row = None
        for el in self.elems.values():
            start = end - 12*12 if not end is None else 0
            end = start + l[1]
            start_row = end_row - 12 if not end_row is None else 0
            end_row = start_row + l[0]
            #print(i, start, end, start_row, end_row)
            K_nl_global[start_row:end_row,start:end] += K_el
        
        K_idx = []
        for entry in self.relevant_xkronx:
            a,b = entry.split(':')
            a,b = a.strip(),b.strip()
            node_i,inner_i = a.split(',')
            node_i,inner_i = int(node_i.strip('[] ')) - 1, int(inner_i.strip('[] ')) - 1

            node_j,inner_j = b.split(',')
            node_j, inner_j = int(node_j.strip('[] ')) - 1, int(inner_j.strip('[] ')) - 1
            
            key = f'[{node_i + 1},{inner_i + 1}] : [{node_j + 1},{inner_j + 1}]'
            K_idx.append(self.finder2[key])
        K_nl_gl_r_outerdim = K_nl_global @ self.K5
        
        return K_nl_gl_r_outerdim[interior,K_idx]        
    
    def kronV_DEIM(self,V_r):
        V_DEIM = self.PHI_DEIM @ V_r
        out = np.zeros((self.r,1))
        for i in range(self.r):
            x,y = self.relevant_xkronx[i].split(':')
            x = x.strip()
            y = y.strip()
            out[i] = self.DEIM_idx[x] * self.DEIM_idx[y]
        return out

    def NL_DEIM(self, V_r):
        return self.N_approx_l @ self.PTK_nl_gl_ @ self.kronV_DEIM(V_r)

    def static_DEIM(self, V_r, F_r, NLinclude):
        V_r = V_r.reshape((-1,1))
        NL_r = self.NL_DEIM(V_r)
        L_r = self.K_lin_gl_r @ V_r
        res = L_r + NL_r*NLinclude - F_r
        return res.reshape((-1,))
    
    
    def static_solver_DEIM(self, F, full_output=True, V0=None, anal_jac=False, NLinclude = 1, diagnostic = False, timer=False):
        interior = self.interior
        V_DEIM = np.zeros_like(self.V0)
        F_r = self.PHI.T @ F[interior]
        V0_r = self.V0_r if V0 is None else self.PHI.T @ V0[interior]
        static = self.static_DEIM
        if anal_jac:
            raise('NOT IMPLEMENTED...')
            #fprime = self.res_jac_POD
        else:
            fprime=None
        start = time.time()
        V_DEIM_r = fsolve(lambda V_r: static(V_r, F_r.reshape((-1,1)), NLinclude), V0_r.reshape(-1), fprime=fprime ,full_output=full_output)
        elasped = time.time() - start
        V_DEIM[interior] = self.PHI @ V_DEIM_r[0].reshape((-1,1)) #+ self.V_mean*self.center
        out = [V_DEIM]
        if full_output:
            out.append(V_DEIM_r[-1])
        if timer:
            out.append(elasped)
        return out
 
#IMPORTING SAMPLES

#NL_samples2 = pd.read_csv("NL_samples_during_sol.csv", header=None).to_numpy()


    
def plot_block_sparsity(A, block_size=144):
    import scipy.sparse as sp
    
    A_coo = A.tocoo()
    n_rows, n_cols = A.shape
    n_blocks_r = int(np.ceil(n_rows / block_size))
    n_blocks_c = int(np.ceil(n_cols / block_size))

    # Map each nonzero to its block
    block_rows = A_coo.row // block_size
    block_cols = A_coo.col // block_size

    # Build block map by just marking occupied blocks
    block_map = np.zeros((n_blocks_r, n_blocks_c))
    block_map[block_rows, block_cols] = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(block_map, ax=ax, cmap=["white", "steelblue"],
                cbar=False, linewidths=0.2, linecolor='gray', square=True)
    ax.set_title(f"Visualization of matrix $K_4$ (in {block_size}×{block_size} blocks)")
    plt.tight_layout()
    plt.show()
    return block_map
