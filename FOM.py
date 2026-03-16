# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:58:01 2025

@author: abhyu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

T = 2
X = 5
dx, dt = 0.01, 0.001
T_span = np.arange(0, T+dt, dt)
X_span = np.arange(-X, X+dx, dx)
nx, nt = len(X_span), len(T_span)

interior = slice(1, -1)
nx_interior = nx - 2
nu = 0.01
n = nt
m = nx
BC = [0, 0]

def burger_L(nu, dx, m):
    A = np.zeros((m, m))
    for i in range(1, m - 1):
        A[i, i] = -2
        A[i, i-1] = 1
        A[i, i+1] = 1
    return A*nu/(dx*dx)

def burger_NL_indexwise(u_i, u_im1, u_ip1, dx):
    dudx = (u_ip1 - u_im1)/(2*dx)
    return -1*u_i*dudx

def burger_dudt(u, L = burger_L(nu, dx, m), nu = nu, dx = dx, m = m):
    NL = np.zeros((m,))
    for i in range(1, m-1):
        NL[i] = burger_NL_indexwise(u[i], u[i-1], u[i+1], dx)
    return L@u + NL

def N_wave_IC(X_span):
    u = np.array([np.exp(-0.5*(x - 1)**2) - np.exp(-0.5*(x + 1)**2) for x in X_span])
    u[0], u[-1] = 0,0
    return u

def RK4solver(dudt, u0, dt, nt = nt, nx = nx):
    U = np.zeros((nx, nt))
    U[:,0] = u0
    for n in range(nt - 1):
        u = U[:,n].copy()
        k1 = dudt(u)
        k2 = dudt(u + 0.5*dt*k1)
        k3 = dudt(u + 0.5*dt*k2)
        k4 = dudt(u + dt*k3)
        u = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        #u[0], u[-1] = 0,0
        U[:, n+1] = u
    return U

# %%
def calculate_FOM():
    u0 = N_wave_IC(X_span)
    U = RK4solver(burger_dudt, u0, dt, nt, nx) #Solution snapshots
    NL = np.zeros_like(U) #Nonlinear function snapshots
    for t in range(n):
        u = U[:,t]
        for i in range(1, m-1):
            NL[i,t] = burger_NL_indexwise(u[i], u[i-1], u[i+1], dx)
    return u0, U, NL


# %%

"""

plt.figure()
plt.title('OG')
for i in range(len(t)):
    plt.plot(x, U[:, i])
plt.show()

# --- animation ---
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2)

ax.set_xlim(-X, X)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("1D viscous Burgers' equation")

def init():
    line1.set_data([], [])
    return line1,

def animate(i):
    line1.set_data(x, U[:, i])
    ax.set_title(f"t = {t[i]:.3f}")
    return line1,

ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                              interval=20, blit=True)

plt.show()
"""
        
    
    
    
    