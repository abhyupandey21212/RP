# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:09:07 2025

@author: abhyu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FOM import *


nu = 0.01

def POD_offline(U, l, L):
    interior = slice(1, -1)
    n = len(T_span)
    m = len(X_span)
    m = m - 2
    weights = [dt for _ in T_span]
    weights[0] = dt/2
    weights[-1] = dt/2
    D = np.diag(weights) #trapezoidal weights for D
    W = np.identity(m) #Not sure what to make the weights of the inner product
    U_interior = U[interior, :]   # shape (nx_interior, nt)
    A = U_interior@D@U_interior.T@W
    eig, basis = np.linalg.eigh(A) 
    index = np.argsort(eig)[::-1]
    POD_basis = basis[:,index[:l]]
    u0 = U[:,0]
    u0_l = POD_basis.T @ u0[interior]
    L_interior = L[interior, interior]
    L_l = POD_basis.T @ L_interior @ POD_basis
    return POD_basis, u0_l, L_l

def defunctPOD_offline(U, l):

    # Compute SVD
    Uu, s, Vt = np.linalg.svd(U, full_matrices=False)

    # Take first l spatial modes
    POD_basis = Uu[:, :l]

    return POD_basis

def DEIM_offline(POD_basis, NL, r, nu, m, dx):
    interior = slice(1, -1)
    m_interior = m-2
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
    
    P_NL = POD_basis.T @ (XI_r @ np.linalg.inv(P.T @ XI_r)) #Projection matrix fornnonlinearity
    P_POD_basis = P.T @ POD_basis # interpolation of Psi
    
    dudx_matrix = np.zeros((m_interior,m_interior))
    for i in range(1, m_interior-1):
        dudx_matrix[i,i+1] = 1
        dudx_matrix[i,i-1] = -1
    P_dx_POD_basis = P.T @ dudx_matrix @ POD_basis #Projection of dudx
    
    return P, P_NL, P_POD_basis, P_dx_POD_basis

def POD_online(u0, POD_basis):
    l = POD_basis.shape[1]
    def burger_dudt_rom_full(u_l):
        u = POD_basis.dot(u_l)
        dudt = burger_dudt(u)
        dudt_l = POD_basis.T.dot(dudt)
        return dudt_l
        
    u0_l = POD_basis.T.dot(u0)
    U_l = RK4solver(dudt=burger_dudt_rom_full, u0=u0_l, dt=dt, nt=nt, nx=l)
    U_recon = POD_basis@U_l
    return U_recon

def POD_DEIM_online(u0, POD_basis, P_NL, P_POD_basis, P_dx_POD_basis, L_l, u0_l, dt = dt, nt = n):
    """
    POD-DEIM reduced order model integration.
    u0: initial condition
    POD_basis: POD basis for state
    V_deim, P: DEIM basis and projection matrix
    NL_func: function computing nonlinear term (takes u)
    """
    interior = slice(1, -1)
    l = POD_basis.shape[1]

    def burger_dudt_rom(u_l, L_l, P_NL,P_POD_basis,P_dx_POD_basis):
        u_DEIM = P_POD_basis@u_l
        dudx_DEIM = P_dx_POD_basis@u_l
        N_r = -1*u_DEIM*dudx_DEIM
        dudt_l = L_l @ u_l + P_NL @ N_r
        return dudt_l

    U_l = np.zeros((l, nt))
    U_l[:, 0] = u0_l
    U_l = RK4solver(dudt=lambda u_l: burger_dudt_rom(u_l, L_l, P_NL, P_POD_basis, P_dx_POD_basis), u0=u0_l, dt=dt, nt=nt, nx=l)
    U_rec_interior = POD_basis @ U_l
    U_rec = np.zeros((nx, nt))
    U_rec[interior, :] = U_rec_interior
    U_rec[0, :] = 0
    U_rec[-1, :] = 0
    return U_rec
def POD_test_online(u0_l, Phi, L_l, dt, nt):
    interior = slice(1, -1)
    l = len(u0_l)
    def dudt_rom(u_l): return L_l @ u_l
    U_l = RK4solver(dudt_rom, u0_l, dt, nt, nx=l)
    U_rec_interior = Phi @ U_l
    U_rec = np.zeros((nx, nt))
    U_rec[interior, :] = U_rec_interior
    U_rec[0, :] = 0
    U_rec[-1, :] = 0
    return U_rec
# %%
if __name__ == '__main__':
    l = 20
    r = 50
    POD_basis = POD_offline(U, l)
    P, XI_l = DEIM_offline(NL, r)
    U_recon = POD_DEIM_online(u0, POD_basis, XI_l, P)
    #U_recon = POD_online(u0, POD_basis)
    
    
    # --- animation ---
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)
    
    ax.set_xlim(-X, X)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("1D viscous Burgers' equation")
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def animate(i):
        line1.set_data(x, U[:, i])
        line2.set_data(x, U_recon[:, i])
    
        ax.set_title(f"t = {t[i]:.3f}")
        return line1, line2
    
    ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                                  interval=2, blit=True)
    
    plt.show()
    
