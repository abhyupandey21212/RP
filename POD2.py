# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:30:36 2025

@author: abhyu
"""
import numpy as np

def RK4solver(dudt, u0, dt, nt, nx, BC = None):
    U = np.zeros((nx, nt))
    U[:,0] = u0
    for n in range(nt-1):
        u = U[:,n].copy()
        k1 = dudt(u)
        k2 = dudt(u + 0.5*dt*k1)
        k3 = dudt(u + 0.5*dt*k2)
        k4 = dudt(u + dt*k3)
        u_new = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        if BC:
            u_new[0], u_new[-1] = BC   # enforce BCs
        U[:, n+1] = u_new
    return U

def POD_DEIM_offline(U, NL, L, l, dx, r=None):
    if r is None:
        r = l
    interior = slice(1, -1)
    U_interior = U[interior,:]
    m,n = U_interior.shape
    NL_interior = NL[interior,:]
    L_interior = L[interior, interior]
    print(U_interior.shape)
    Phi, _, _ = np.linalg.svd(U_interior, full_matrices=False)
    POD_basis = Phi[:,:l]
    
    """
    NL = NL[interior,:]
    XI,S_NL,WT = np.linalg.svd(NL,full_matrices=0)
    nmax = np.argmax(np.abs(XI[:,0]))
    XI_m = XI[:,0].reshape(m_interior,1)
    z = np.zeros((m_interior,1))
    P = np.copy(z)
    P[nmax] = 1    
    for jj in range(1,r):
        c=np.linalg.solve(P.T@XI_m, P.T@XI[:,jj].reshape(m_interior,1))
        res = XI[:,jj].reshape(m_interior,1) - XI_m @ c
        nmax = np.argmax(np.abs(res))
        XI_m=np.concatenate((XI_m,XI[:,jj].reshape(m_interior,1)),axis=1)
        P = np.concatenate((P,z),axis=1)
        P[nmax,jj] = 1
    XI_r = XI[:,:r]
    """
    
    XI, _, _ = np.linalg.svd(NL_interior, full_matrices=False)
    XI_r = XI[:,0].reshape(m,1)
    z = np.zeros((m,1))
    P = np.copy(z)
    index_1 = np.argmax(np.abs(XI_r))
    P[index_1] = 1
    for i in range(1,r):
        XI_rp1 = XI[:,i].reshape(m,1)
        c = np.linalg.solve(P.T@XI_r, P.T@XI_rp1)
        res = XI_rp1 - XI_r@c 
        index_i = np.argmax(np.abs(res))
        P_new = np.copy(z)
        P_new[index_i] = 1
        P = np.concatenate((P, P_new), axis = 1)
        XI_r = np.concatenate((XI_r, XI_rp1), axis = 1)
    print('look here')
    print(P.shape, POD_basis.shape)
    N_approx = XI_r @ np.linalg.inv(P.T @ XI_r)
    N_approx_l = POD_basis.T @ N_approx
    
    DX = np.zeros((m,m))
    DX[0,0] = -1
    DX[0,1] = 1
    DX[-1,-1] = 1
    DX[-1,-2] = -1
    for i in range(1,m-1):
        DX[i,i-1] = -1
        DX[i,i+1] = 1
    DX = DX/(2*dx)
    
    P_POD_basis = P.T @ POD_basis
    
    P_DX_POD_basis = P.T @ (DX @ POD_basis)
    print(POD_basis.shape, L_interior.shape)
    L_l = POD_basis.T @ (L_interior @ POD_basis)
    u0_l = POD_basis.T @ U_interior[:,0]
    
    return POD_basis, L_l, u0_l, N_approx_l, P_POD_basis, P_DX_POD_basis

def POD_DEIM_online(POD_basis, L_l, u0_l, N_approx_l, P_POD_basis, P_DX_POD_basis, n, m, dt):
    interior = slice(1, -1)
    l = POD_basis.shape[1]
    def dudt_ROM(u_l):
        u_DEIM = P_POD_basis @ u_l
        dudx_DEIM = P_DX_POD_basis @ u_l
        PTN = -1 * u_DEIM * dudx_DEIM
        NL_l = N_approx_l @ PTN
        return L_l @ u_l + NL_l
    U_l = RK4solver(dudt_ROM, u0_l, dt, n, l)
    U_rec_int = POD_basis @ U_l
    U_rec = np.zeros((m,n))
    U_rec[interior,:] = U_rec_int
    return U_rec, U_l

def POD_online(POD_basis, L_l, u0_l, n, m, dt, dx):
    interior = slice(1, -1)
    l = POD_basis.shape[1]
    m_int=m-2
    DX = np.zeros((m_int,m_int))
    DX[0,0] = -1
    DX[0,1] = 1
    DX[-1,-1] = 1
    DX[-1,-2] = -1
    for i in range(1,m_int-1):
        DX[i,i-1] = -1
        DX[i,i+1] = 1
    DX = DX/(2*dx)
    
    def dudt_POD(u_l):
        u = POD_basis @ u_l
        dudx = DX  @ u
        NL = -1 * u * dudx
        NL_l = POD_basis.T @ NL
        return L_l @ u_l + NL_l
    U_l = RK4solver(dudt_POD, u0_l, dt, n, l)
    U_rec_int = POD_basis @ U_l
    U_rec = np.zeros((m,n))
    U_rec[interior,:] = U_rec_int
    return U_rec, U_l
    