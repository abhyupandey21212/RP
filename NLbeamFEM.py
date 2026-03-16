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
from dimentions import *



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

def define_FEM_matricies(n_nodes):
    E = np.block([[cross_mat(k0), np.zeros_like(cross_mat(k0))],
                  [cross_mat(e1), cross_mat(k0)]])
    ET = E.T
    
    #Element matrix
    def M_el(M, C, h):
        z = np.zeros_like(M)
        return (h/15)*np.block([[4*M, z, 2*M, z, -1*M, z],
                                [z, 4*C, z, 2*C, z, -1*C],
                                [2*M, z, 16*M, z, 2*M, z],
                                [z, 2*C, z, 16*C, z, 2*C],
                                [-1*M, z, 2*M, z, 4*M, z],
                                [z, -1*C, z, 2*C, z, 4*C]])
    
    def K_lin_el(E, ET, h):
        z = np.zeros_like(E)
        I = np.eye(6)
        K_lin_el_1 = (h/15)*np.block([[z, 4*E, z, 2*E, z, -1*E],
                                      [-4*ET, z, -2*ET, z, 1*ET, z],
                                      [z, 2*E, z, 16*E, z, 2*E],
                                      [-2*ET, z, -16*ET, z, 2*ET, z],
                                      [z, -1*E, z, 2*E, z, 4*E],
                                      [1*ET, z, -2*ET, z, -4*ET, z]])
        
        K_lin_el_2 = (1/6)*np.block([[z, -3*I, z, 4*I, z, -1*I],
                                      [-3*I, z, 4*I, z, -1*I, z],
                                      [z, -4*I, z, z, z, 4*I],
                                      [-4*I, z, z, z, 4*I, z],
                                      [z, 1*I, z, -4*I, z, 3*I],
                                      [1*I, z, -4*I, z, 3*I, z]])
        
        #[2*ET, z, 16*ET, z, 2*ET, z], #My derivation
        #[2*ET, z, 16*ET, z, -2*ET, z], #IDENTICAL TO Artola

        return -K_lin_el_1 - K_lin_el_2
    
    def K_nl_el(L1, L2, M, C, V):
        v1 = np.array([[V[:6], V[12:18], V[24:30]]]).reshape((-1,3)) #v11, v12, v13
        v2 = np.array([[V[6:12], V[18:24], V[30:36]]]).reshape((-1,3)) #v21, v22, v23
        z = np.zeros_like(M)
        
        ints = {'111': 39, '112': 20, '113': -3, '122': 16, '123': -8, '133': -3, '222': 192, '223': 16, '233': 20, '333': 39}
        K_ij = {}
        for i in range(1, 4):
            for j in range(1, 4):
                Kij = np.array([ints[''.join(sorted(f'{i}{j}{k}'))] for k in [1, 2, 3]])
                K_ij[int(f'{i}{j}')] = Kij
    
        return (h/210)*np.block([[L1(v1@K_ij[11])@M, L2(v2@K_ij[11])@C, L1(v1@K_ij[12])@M, L2(v2@K_ij[12])@C, L1(v1@K_ij[13])@M, L2(v2@K_ij[13])@C],
                            [z, -1*L1(v1@K_ij[11]).T@C, z, -1*L1(v1@K_ij[12]).T@C, z, -1*L1(v1@K_ij[13]).T@C],
                            [L1(v1@K_ij[21])@M, L2(v2@K_ij[21])@C, L1(v1@K_ij[22])@M, L2(v2@K_ij[22])@C, L1(v1@K_ij[23])@M, L2(v2@K_ij[23])@C],
                            [z, -1*L1(v1@K_ij[21]).T@C, z, -1*L1(v1@K_ij[22]).T@C, z, -1*L1(v1@K_ij[23]).T@C],
                            [L1(v1@K_ij[31])@M, L2(v2@K_ij[31])@C, L1(v1@K_ij[32])@M, L2(v2@K_ij[32])@C, L1(v1@K_ij[33])@M, L2(v2@K_ij[33])@C],
                                                [z, -1*L1(v1@K_ij[31]).T@C, z, -1*L1(v1@K_ij[32]).T@C, z, -1*L1(v1@K_ij[32]).T@C]])
    
    #Global matrix
    nodes_per_elem = 3
    nodes = {i+1: eta for i, eta in enumerate(eta_grid)}
    no_elems = (n_nodes - 1)//(nodes_per_elem - 1)
    if (n_nodes - 1) % (nodes_per_elem - 1) != 0:
        raise ValueError
    
    elems = {i+1: [j+1 for j in range(2*i, 2*i + 3)] for i in range(no_elems)}
    
    M_gl = np.zeros((n_nodes*12, n_nodes*12))
    K_lin_gl = np.zeros((n_nodes*12, n_nodes*12))
    for el in elems.values():
        i = el[0]
        M_gl[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += M_el(M, C, h)
        K_lin_gl[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += K_lin_el(E, ET, h)
                
    
    def K_nl_gl(V):
        K_nl_global = np.zeros((n_nodes*12, n_nodes*12))
        for el in elems.values():
            i = el[0]
            V_el = V[(i-1)*12:(i+2)*12]
            K_nl_global[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += K_nl_el(L1, L2, M, C, V_el)
        return K_nl_global
    
    def NL_DEIM(V_r, PHI, DEIM_elems, n_nodes_per_elem = 3):
        #Assumeing contigunous elements
        n_DEIM_elems = len(DEIM_elems)
        n_nodes = (n_DEIM_elems - 1)*(n_nodes_per_elem - 1) + n_nodes_per_elem
        
        first_node = (min(min(DEIM_elems.values())) - 1)*12
        final_node = (max(max(DEIM_elems.values())) - 0)*12
        
        PHI_DEIM = PHI[:n_nodes * 12 - 6,:]#Missing BC nodes
        V_DEIM = np.zeros((n_nodes*12,1))
        V_DEIM[6:,:] = PHI_DEIM @ V_r #Missing BC nodes

        K_nl_DEIM = np.zeros((n_nodes*12, n_nodes*12))
        for el in DEIM_elems.values():
            i = el[0]
            V_el = V_DEIM[(i-1)*12:(i+2)*12]
            K_nl_DEIM[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += K_nl_el(L1, L2, M, C, V_el)
        return K_nl_DEIM @ V_DEIM, n_nodes
    
    #Loads
    f_tip_ext = np.array([0,0,1])
    m_tip_ext = np.array([0,0,0])
    
    F_gl = np.block([np.zeros((1,(n_nodes-1)*12)), f_tip_ext,m_tip_ext,np.zeros((1,6))]).T

    V0 = np.zeros((n_nodes*12, 1))
    
    G0 = np.block([[np.eye(6), np.zeros((6,6))], [np.zeros((6,6)), np.zeros((6,6))]])
    G0_orth = np.block([[np.zeros((6,6)), np.zeros((6,6))], [np.zeros((6,6)), np.eye(6)]])

    #sL will be free hence
    GL = G0_orth
    GL_orth = G0
    g0 = gL = np.zeros((12, 1))
    gBC = np.vstack((g0, np.zeros(((n_nodes - 2)*12,1)), gL))

    GBC = scipy.linalg.block_diag(G0, np.zeros(((n_nodes - 2)*12,(n_nodes - 2)*12)), GL)
    GBC_orth = scipy.linalg.block_diag(G0_orth, np.eye((n_nodes - 2)*12), GL_orth)

    
    return M_gl, K_lin_gl, K_nl_gl, F_gl, V0, eta_grid, h, GBC, gBC, GBC_orth, NL_DEIM, elems

def K_nl_gl2(V):
    #Full state including BC nodes
    K_nl_global = np.zeros((n_nodes*12, n_nodes*12))
    for el in elems.values():
        i = el[0]
        V_el = V[(i-1)*12:(i+2)*12]
        K_nl_global[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += K_nl_el(L1, L2, M, C, V_el)
    return K_nl_global

def static(V_sol, M, K_lin, K_nl, F, G, g):
    interior = slice(6,-6)
    V_sol = V_sol.reshape((-1,1))
    res_int = K_lin@V_sol + K_nl(V_sol)@V_sol - F 
    res_BC = G@V_sol - g
    res = res_BC
    res[interior] = res_int[interior]
    #Since Vn has already been multiplied by G_ortho, we have set the BC nodes to zero, so their values will be zero when multiplying by eveyrhing, so we can just ADD the BC conditons to thsi equations as they act on indep coordinates in the vector
    return res.reshape((-1,))

def static_POD(V_sol_r, M_r, K_lin_r, K_nl, F_r, PHI, interior_lims = (6,6)):
    interior = slice(interior_lims[0], -interior_lims[1])
    BC_dims = sum(interior_lims)
    interior_dims, l = PHI.shape
    V_sol_r = V_sol_r.reshape((-1,1))
    V_sol_recon = np.zeros((interior_dims + BC_dims,1))
    V_sol_recon[interior] = PHI @ V_sol_r
    NL = K_nl(V_sol_recon) @ V_sol_recon
    res = K_lin_r @ V_sol_r + PHI.T @ NL[interior] - F_r 
    return res.reshape((-1,))

def static_DEIM(V_sol_r, M_r, K_lin_r, NL_DEIM, F_r, PHI, P, N_approx_l, DEIM_elems, interior_lims = (6,6)):
    #interior = slice(interior_lims[0], -interior_lims[1])
    #BC_dims = sum(interior_lims)
    interior_dims, l = PHI.shape
    V_sol_r = V_sol_r.reshape((-1,1))
    #V_sol_recon = np.zeros((interior_dims + BC_dims,1))
    #V_sol_recon[interior] = PHI @ V_sol_r
    NL_r, n_DEIM_nodes = NL_DEIM(V_sol_r, PHI, DEIM_elems)
    print(NL_r.max())
    NL_l = N_approx_l @ P.T[:,:n_DEIM_nodes*12] @ NL_r
    res = K_lin_r @ V_sol_r + NL_l - F_r 
    return res.reshape((-1,))

def dynamic(t, V, Minv, K_lin, K_nl, F, G, gdot):
    print(t)
    interior = slice(6,-6)
    V = V.reshape((-1,1))
    dVdt = Vn = fsolve(lambda Vn: static(Vn, V, M_gl, K_lin_gl, K_nl_gl, t*P*F_gl, GBC, gBC), V,full_output=False)
    dVdt += G@(gdot)
    print(dVdt.max())
    return dVdt.reshape((-1,))

def post_intrinsic(Vn, C):
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


if __name__ == '__main__':
    #IC and BC
    V0 = np.zeros((n_nodes*12, 1))
    
    #s0 will be clamped hence
    


# %%

#POD OFFLINE
def POD_offline(X, l, center=0):
    n_dim, n_samples = X.shape
    V_mean = X.mean(axis=1, keepdims=True)
    V_mean_mat = np.hstack(tuple([V_mean for _ in range(n_samples)]))
    X_cen = X - center*V_mean_mat
    POD_basis, sigmas, _ = np.linalg.svd(X_cen)
    PHI = POD_basis[:,:l]
    return PHI,V_mean, sigmas

def POD_DEIM_offline(U, NL, l, r=None):
    if r is None:
        r = l
    #interior = slice(1, -1)
    #U_interior = U[interior,:]
    m,n = U.shape
    #NL_interior = NL[interior,:]
    Phi, _, _ = np.linalg.svd(U, full_matrices=False)
    POD_basis = Phi[:,:l]    
    XI, _, _ = np.linalg.svd(NL, full_matrices=False)
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
    N_approx = XI_r @ np.linalg.inv(P.T @ XI_r)
    N_approx_l = POD_basis.T @ N_approx    
    PT_POD_basis = P.T @ POD_basis    
    return POD_basis, N_approx_l, PT_POD_basis, P

#PHI, NL_l, P_PHI, P = POD_DEIM_offline(V_FOM2, NL, 3, 5)
# %%
def animate():
    V_POD = V_rec
    V_FOM = V_FOM2
    
    velz_FOM = V_FOM[2::12,:]
    velz_POD = V_POD[2::12,:]
    velz_mean = V_mean_mat[2::12,:]
    m, nt = velz_FOM.shape
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    line1, = ax.plot(eta_grid, velz_FOM[:,0], c='b', label='FOM')
    line2, = ax.plot(eta_grid, velz_POD[:,0], c='r', label='ROM')
    line3, = ax.plot(eta_grid, velz_mean[:,0], c='g', label='mean')
    
    ax.set_xlim(eta_grid.min(), eta_grid.max())
    ax.set_ylim(-0.3, 0.1)
    
    ax.legend()
    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 't', 0, nt-1, valinit=0, valstep=1)
    
    def update(i):
        line1.set_ydata(velz_FOM[:,int(i)])
        line2.set_ydata(velz_POD[:,int(i)])
        line3.set_ydata(velz_mean[:,int(i)])
    
        return line1, line2, line3,
    
    slider.on_changed(update)
    
    # Animation
    def animate(i):
        slider.set_val(i)
        return line1, line2, line3,
    
    ani = FuncAnimation(fig, animate, frames=nt, interval=50)
    plt.show()
    
    
