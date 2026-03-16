# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:44:58 2025

@author: abhyu
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Parameters
# ----------------------------
T = 10
X = 5
dx, dt = 0.1, 0.01
T_span = np.arange(0, T+dt, dt)
X_span = np.arange(-X, X+dx, dx)
nx, nt = len(X_span), len(T_span)

interior = slice(1, -1)
nx_interior = nx - 2
nu = 0.01

# ----------------------------
# Linear operator L for 1D diffusion
# ----------------------------
def burger_L(nu, dx, nx):
    L = np.zeros((nx, nx))
    for i in range(1, nx-1):
        L[i, i]   = -2
        L[i, i-1] = 1
        L[i, i+1] = 1
    return L * nu / dx**2

# ----------------------------
# Initial condition
# ----------------------------
def N_wave_IC(X_span):
    u = np.exp(-0.5*(X_span-1)**2) - np.exp(-0.5*(X_span+1)**2)
    u[0], u[-1] = 0, 0
    return u

# ----------------------------
# RK4 solver (general)
# ----------------------------
def RK4solver(dudt, u0, dt, nt, nx):
    U = np.zeros((nx, nt))
    U[:,0] = u0
    for n in range(nt-1):
        u = U[:,n].copy()
        k1 = dudt(u)
        k2 = dudt(u + 0.5*dt*k1)
        k3 = dudt(u + 0.5*dt*k2)
        k4 = dudt(u + dt*k3)
        u_new = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        #u_new[0], u_new[-1] = 0,0   # enforce BCs
        U[:, n+1] = u_new
    return U

# ----------------------------
# Compute FOM snapshots
# ----------------------------
u0 = N_wave_IC(X_span)
L = burger_L(nu, dx, nx)
U = RK4solver(lambda u: L@u, u0, dt, nt, nx)
U_interior = U[interior, :]   # shape (nx_interior, nt)

# ----------------------------
# POD offline via SVD
# ----------------------------
def POD_offline(U, l):
    Uu, s, Vt = np.linalg.svd(U, full_matrices=False)
    Phi = Uu[:, :l]   # spatial modes
    return Phi, s

l = 50
Phi, s = POD_offline(U_interior, l)

# ----------------------------
# Reduced L operator
# ----------------------------
L_interior = L[interior, interior]
L_l = Phi.T @ L_interior @ Phi
u0_l = Phi.T @ u0[interior]

# ----------------------------
# POD online / ROM solve
# ----------------------------
def POD_online_linear(u0_l, Phi, L_l, dt, nt):
    l = len(u0_l)
    def dudt_rom(u_l): return L_l @ u_l
    U_l = RK4solver(dudt_rom, u0_l, dt, nt, nx=l)
    U_rec = Phi @ U_l
    return U_rec

U_rec_interior = POD_online_linear(u0_l, Phi, L_l, dt, nt)
U_rec = np.zeros((nx, nt))
U_rec[interior, :] = U_rec_interior
U_rec[0, :] = 0
U_rec[-1, :] = 0
# ----------------------------
# Compare FOM vs ROM
# ----------------------------
rel_error = np.linalg.norm(U - U_rec) / np.linalg.norm(U)
print("Relative error between FOM and ROM:", rel_error)

# ----------------------------
# Plot a snapshot
# ----------------------------
plt.plot(X_span, U[:,100], label="FOM")
plt.plot(X_span, U_rec[:,100], "--", label="ROM")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()


# %%
x = X_span
t = T_span
# --- animation ---
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, label = 'FOM')
#line2, = ax.plot([], [], lw=2, label = 'POD', ls = ':')
line3, = ax.plot([], [], lw=2, label = 'DEIM', ls = '--')

ax.set_xlim(-X, X)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("1D viscous Burgers' equation")
ax.legend()

def init():
    line1.set_data([], [])
    #line2.set_data([], [])
    line3.set_data([], [])
    return line1, line3#, line2

def animate(i):
    line1.set_data(x, U[:, i])
    #line2.set_data(x, U_recon_POD[:, i])
    line3.set_data(x, U_rec[:, i])

    ax.set_title(f"t = {t[i]:.3f}")
    return line1, line3#, line2

ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                              interval=0.2, blit=True)

plt.show()