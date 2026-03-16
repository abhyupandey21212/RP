# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 11:39:46 2026

@author: abhyu
"""
import numpy as np
inv = np.linalg.inv
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy


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
    
    

#Dimention
span, chord, height = l, b, h = [16, 1, 0.25] #meters, span, chord, height
mass_per_len = dm = 0.75 #kg/m
mom_of_iner = 0.1 #kgm at 50% chord, which is where both the COM and eta axis are

l_bending_rigidity = EIy = 20000 #Nm2 lengthwise EI_yy
torsional_rigidity = GJ = 10000 #Nm2 elastic axis EI_xx
c_bending_rigidity = EIz = 4000000 #Nm2 chordwise EI_zz
EA = 1e9
GAy = 1e9
GAz = 1e9
#Defining the geo and the material
e1 = np.array([[1, 0, 0]]).T
k0 = np.array([[0, 0, 0]]).T #inital curvature is zero, initially straight beam
E = np.block([[cross_mat(k0), np.zeros_like(cross_mat(k0))],
              [cross_mat(e1), cross_mat(k0)]])
ET = E.T

#Moments per span
Ixx = b*h*(b*b + h*h)/(12*l) #kgm2 / m
Iyy = l*h*(l*l + h*h)/(12*l) #kgm2 / m
Izz = b*l*(b*b + l*l)/(12*l) #kgm2 / m
mass_mom_per_span = np.diag([Ixx, Iyy, Izz])

#Discritizaton
n_nodes = 7
dt = 0.001
eta_grid = np.linspace(0, span, n_nodes)
h = eta_grid[1] - eta_grid[0]

time_grid = np.arange(0, 0.1+dt, dt)


#Mass matrix (in continous) per span
M = np.block([[dm*np.eye(3), np.zeros((3,3))],
              [np.zeros((3,3)), mass_mom_per_span]])


#Stiffness matrix (in continous) per span
Cinv = np.diag([EA, GAy, GAz, GJ, EIy, EIz])
C = inv(Cinv)

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
    K_lin_el_1 = (h/15)*np.block([[z, -4*E, z, -2*E, z, 1*E],
                                  [4*ET, z, 2*ET, z, -1*ET, z],
                                  [z, -2*E, z, -16*E, z, -2*E],
                                  [2*ET, z, 16*ET, z, 2*ET, z], #My derivation
                                  #[2*ET, z, 16*ET, z, -2*ET, z], #IDENTICAL TO Artola
                                  [z, 1*E, z, -2*E, z, -4*E],
                                  [-1*ET, z, 2*ET, z, 4*ET, z]])
    K_lin_el_2 = (-1/6)*np.block([[z, -3*I, z, 4*I, z, -1*I],
                                  [-3*I, z, 4*I, z, -1*I, z],
                                  [z, -4*I, z, z, z, 4*I],
                                  [-4*I, z, z, z, 4*I, z],
                                  [z, 1*I, z, -4*I, z, 3*I],
                                  [1*I, z, -4*I, z, 3*I, z]])
    return K_lin_el_1 + K_lin_el_2

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

def K_nl_gl2(V_interior):
    #Removes BC DOFs
    V = np.zeros((84,1))
    V[interior] = V_interior
    K_nl_global = np.zeros((n_nodes*12, n_nodes*12))
    for el in elems.values():
        i = el[0]
        V_el = V[(i-1)*12:(i+2)*12]
        K_nl_global[(i-1)*12:(i+2)*12 ,(i-1)*12:(i+2)*12] += K_nl_el(L1, L2, M, C, V_el)
    return K_nl_global[interior, interior]


#Loads
f_tip_ext = np.array([0,0,-50])
m_tip_ext = np.array([0,0,0])
F_gl = np.block([np.zeros((1,(n_nodes-1)*12)), f_tip_ext,m_tip_ext,np.zeros((1,6))]).T

def F_gl_t(t):
    if t < 0.03:
        return F_gl# * np.sin(5*t)
    else:
        return F_gl*0
#IC and BC
V0 = np.zeros((n_nodes*12, 1))

#s0 will be clamped hence
G0 = np.block([[np.eye(6), np.zeros((6,6))], [np.zeros((6,6)), np.zeros((6,6))]])
G0_orth = np.block([[np.zeros((6,6)), np.zeros((6,6))], [np.zeros((6,6)), np.eye(6)]])

#sL will be free hence
GL = G0_orth
GL_orth = G0
g0 = gL = np.zeros((12, 1))
gBC = np.zeros_like(V0)
gBC[:12] = g0
gBC[-12:] = gL

GBC = scipy.linalg.block_diag(G0, np.eye(V0.shape[0] - G0.shape[0] - GL.shape[0]), GL)
GBC_orth = scipy.linalg.block_diag(G0_orth, np.eye(V0.shape[0] - G0.shape[0] - GL.shape[0]), GL_orth)

def implicit_ODE(M_gl, K_gl, F_gl_half, V, Vn, G0, GL, g0, gL):
    #The idea is to use Newton-Ralphson to solve for Vn by finding the root of this equation for each time step
    V = V.reshape((-1, 1))
    Vn = Vn.reshape((-1, 1))
    Vhalf = (V + Vn)/2
    
    residual = M_gl@(Vn - V)/dt + K_gl@Vhalf  + K_nl_gl(Vhalf)@Vhalf - F_gl_half

    #Impose BC on s0 and sL by imposing G0v0 = g0, GLvL = gL as part of Newton-Raphson
    VnBC = GBC @ Vn - gBC
    residual[:6]    = VnBC[:6]
    residual[-6:]   = VnBC[-6:]
    return  residual.reshape((-1,))

def implicit_ODE2(M_gl, K_gl, F_gl_half, V, Vn):
    #The idea is to use Newton-Ralphson to solve for Vn by finding the root of this equation for each time step
    #This time we encoperate the BC into the matricies
    V = V.reshape((-1, 1))
    Vn = Vn.reshape((-1, 1))
    Vhalf = (V + Vn)/2    
    residual_i = M_gl@(Vn - V)/dt + K_gl@Vhalf  + K_nl_gl2(Vhalf)@Vhalf - F_gl_half
    return  residual_i.reshape((-1,))

def ODE_NL_sampler(V, Vn):
    #The idea is to use Newton-Ralphson to solve for Vn by finding the root of this equation for each time step
    #This time we encoperate the BC into the matricies
    V = V.reshape((-1, 1))
    Vn = Vn.reshape((-1, 1))
    Vhalf = (V + Vn)/2    
    return K_nl_gl(Vhalf)@Vhalf

def implicit_ODE_lin_prime(M_gl, K_gl, Vn):
    pass #SHOULLD PROBABLY FIND TO SPEED UP CALC


# %%
#SOLUTION

V_mat = np.zeros((V0.shape[0], len(time_grid)))
NL = np.zeros_like(V_mat)

V_mat[:,0] = V0.reshape((-1,))
for i, t in enumerate(time_grid[1:]):
    print(t)
    #initazlie with current V
    V = V_mat[:,-1]
    #Impose BC by premultiplying G0_orth and GL_orth to first and last nodes
    V = GBC_orth @ V
    Vn = fsolve(lambda Vn: implicit_ODE(M_gl, K_lin_gl, F_gl_t(t), V, Vn, G0, GL, g0, gL), V)
    NLn = ODE_NL_sampler(V, Vn)
    #add Vn to the list of outputs
    V_mat[:,i+1] = Vn
    NL[:,i+1] = NLn.reshape((-1,))
V_FOM = V_mat
velz = V_mat[2::12,:]

    
# %%

# %%
#SOLUTION 2
interior=slice(6,-6)

V_mat = np.zeros((V0[interior].shape[0], len(time_grid)))
V_mat[:,0] = V0[interior].reshape((-1,))
for i, t in enumerate(time_grid[1:]):
    print(t)
    #initazlie with current V
    V = V_mat[:,-1]
    Vn = fsolve(lambda Vn: implicit_ODE2(M_gl[interior,interior], K_lin_gl[interior,interior], F_gl_t(t)[interior], V, Vn), V)
    #add Vn to the list of outputs
    V_mat[:,i+1] = Vn   
    
V_FOM2 = np.zeros((V0.shape[0], len(time_grid)))
V_FOM2[interior,:] = V_mat
velz = V_FOM2[2::12,:]

    
# %%
#PLOTTING
#NOTE, these are nodal values, for proper visualization they have to be plotted with the shape function integrals
    
velz = V_FOM[2::12,:]#FOM2[2::12,:]

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

m, nt = velz.shape

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(eta_grid, velz[:,0], c='b')
ax.set_xlim(eta_grid.min(), eta_grid.max())
ax.set_ylim(velz.min(), velz.max())

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 't', 0, nt-1, valinit=0, valstep=1)

def update(i):
    line.set_ydata(velz[:,int(i)])
    return line,

slider.on_changed(update)

# Animation
def animate(i):
    slider.set_val(i)
    return line,

ani = FuncAnimation(fig, animate, frames=nt, interval=50)

plt.show()
    
# %%

#POD OFFLINE
def POD_offline(X, l):
    V_mean = X.mean(axis=1, keepdims=True)
    V_mean_mat = np.hstack(tuple([V_mean for _ in range(len(time_grid))]))
    X_cen = X - V_mean_mat
    POD_basis, sigmas, _ = np.linalg.svd(X_cen)
    PHI = POD_basis[:,:l]
    PHI[:6,:] = 0
    PHI[-6:,:] = 0
    return PHI
PHI = POD_offline(V_FOM2, 3)
M_gl_r = PHI.T @ M_gl @ PHI
K_lin_gl_r = PHI.T @ K_lin_gl @ PHI
V0_r = PHI.T @ V0

# %%
#POD ONLINE
def implicit_ODE_POD(M_gl_r, K_gl_r, F_gl_half_r, V_r, Vn_r, PHI):
    V_r = V_r.reshape((-1, 1))
    Vn_r = Vn_r.reshape((-1, 1))
    Vhalf_r = (V_r + Vn_r)/2
    Vhalf = PHI @ Vhalf_r
    residual_interior = M_gl_r@(Vn_r - V_r)/dt + K_gl_r@Vhalf_r  + PHI.T @ (K_nl_gl(Vhalf) @ Vhalf) - F_gl_half_r
    #residual = # some combo of residual_interior and something to enforce the BC
    return  residual_interior.reshape((-1,))


V_matPOD = np.zeros((V0_r.shape[0], len(time_grid)))
V_rec = np.zeros_like(V_FOM2)
V_matPOD[:,0] = V0_r.reshape((-1,))
V_rec[:,0] = V_FOM2[:,0]
for i, t in enumerate(time_grid[1:]):
    print(t)
    #initazlie with current V
    V_r = V_matPOD[:,-1]
    F_gl_r = PHI.T @ F_gl_t(t)

    Vn_r = fsolve(lambda Vn_r: implicit_ODE_POD(M_gl_r, K_lin_gl_r, F_gl_r, V_r, Vn_r, PHI), V_r)
    #add Vn to the list of outputs
    V_matPOD[:,i+1] = Vn_r.reshape((-1,))

V_rec = PHI @ V_matPOD #+ V_mean_mat
velz_POD = V_rec[2::12,:]

#Notes and next steps:
#Super weird that I dont need to add the mean back after subtracting it before, dont really understand why that is
#Next I need to make a better snapshot, maybe a freqeuncy sweep so that we capture all the dynamics of the beam in the POD basis
#Then we can apply DEIM, and then compare with modal decomp
#Then AERO model for forcing - NEED TO READ UP FOR THIS


#Notes from meeting:
#    check if physically correct with small deflectionsna nd linear beam thoery
#    Try random noise as the input for POD data capture
#    Check FEM model by doing modal analysis and see if those freq line up with that in the paper from which I took the wing data
#    Figure out postprocessing for getting the positons
# %%

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
    P_POD_basis = P.T @ POD_basis    
    return POD_basis, N_approx_l, P_POD_basis, P

PHI, NL_l, P_PHI, P = POD_DEIM_offline(V_FOM2, NL, 3, 5)
# %%
PHIT_XI_inv_PT_XI = NL_l


def implicit_ODE_PODDEIM(M_gl_r, K_gl_r, F_gl_half_r, V_r, Vn_r, PHI):
    V_r = V_r.reshape((-1, 1))
    Vn_r = Vn_r.reshape((-1, 1))
    Vhalf_r = (V_r + Vn_r)/2
    Vhalf_DEIM = P_PHI @ Vhalf_r
    residual_interior = M_gl_r@(Vn_r - V_r)/dt + K_gl_r@Vhalf_r  + NL_l.T @ (K_nl_gl_DEIM(Vhalf_DEIM) @ Vhalf_DEIM) - F_gl_half_r
    #residual = # some combo of residual_interior and something to enforce the BC
    return  residual_interior.reshape((-1,))


V_matPOD = np.zeros((V0_r.shape[0], len(time_grid)))
V_rec = np.zeros_like(V_FOM2)
V_matPOD[:,0] = V0_r.reshape((-1,))
V_rec[:,0] = V_FOM2[:,0]
for i, t in enumerate(time_grid[1:]):
    print(t)
    #initazlie with current V
    V_r = V_matPOD[:,-1]
    F_gl_r = PHI.T @ F_gl_t(t)

    Vn_r = fsolve(lambda Vn_r: implicit_ODE_POD(M_gl_r, K_lin_gl_r, F_gl_r, V_r, Vn_r, PHI), V_r)
    #add Vn to the list of outputs
    V_matPOD[:,i+1] = Vn_r.reshape((-1,))

V_rec = PHI @ V_matPOD #+ V_mean_mat
velz_POD = V_rec[2::12,:]
# %%
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

# %%
from scipy.integrate import solve_ivp

def dVdt_POD(t, V_r, a, L, NL, PHI_NL, PHI, interior = slice(6,-6)):
    # a = PHI.T @ inv(M) @ F_ext
    # L = PHI.T @ inv(M) @ K_lin @ PHI
    # PHI_NL = PHI.T @ inv(M)
    print(t)
    V_r = V_r.reshape((-1, 1))
    V_rec = np.zeros((84,1))
    V_rec[interior] = PHI @ V_r
    dudt = a - L@V_r #- PHI_NL @ NL(V_rec)[interior,interior] @ PHI @ V_r
    return  dudt.reshape((-1,))

a = PHI.T @ inv(M_gl[interior,interior]) @ F_gl[interior]
L = PHI.T @ inv(M_gl[interior,interior]) @ K_lin_gl[interior,interior] @ PHI
PHI_NL = PHI.T @ inv(M_gl[interior,interior])

sol = solve_ivp(lambda t, V_r: dVdt_POD(t, V_r, a, L, K_nl_gl, PHI_NL, PHI),
                t_span=(time_grid[0], time_grid[-1]),
                #t_eval=time_grid, 
                y0=V0_r.reshape((-1,)), 
                method='RK45')
