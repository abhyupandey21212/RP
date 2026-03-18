# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 10:18:28 2026

@author: abhyu
"""

import numpy as np
inv = np.linalg.inv
rk = np.linalg.matrix_rank
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

import matplotlib.pyplot as plt
import scipy

#Dimention
from dimentions import *


#Element matrix
def K_el(EA, EIy, EIz, GJ, L):
    return np.array([
                     [EA/L, 0, 0, 0, 0, 0, -EA/L, 0, 0, 0, 0, 0],
                     [0, 12*EIz/L**3, 0, 0, 0, 6*EIz/L**2, 0, -12*EIz/L**3, 0, 0, 0, 6*EIz/L**2],
                     [0, 0, 12*EIy/L**3, 0, -6*EIy/L**2, 0, 0, 0, -12*EIy/L**3, 0, -6*EIy/L**2, 0],
                     [0, 0, 0, GJ/L, 0, 0, 0, 0, 0, -GJ/L, 0, 0],
                     [0, 0, -6*EIy/L**2, 0, 4*EIy/L, 0, 0, 0, 6*EIy/L**2, 0, 2*EIy/L, 0],
                     [0, 6*EIz/L**2, 0, 0, 0, 4*EIz/L, 0, -6*EIz/L**2, 0, 0, 0, 2*EIz/L],
                     [-EA/L, 0, 0, 0, 0, 0, EA/L, 0, 0, 0, 0, 0],
                     [0, -12*EIz/L**3, 0, 0, 0, -6*EIz/L**2, 0, 12*EIz/L**3, 0, 0, 0, -6*EIz/L**2],
                     [0, 0, -12*EIy/L**3, 0, 6*EIy/L**2, 0, 0, 0, 12*EIy/L**3, 0, 6*EIy/L**2, 0],
                     [0, 0, 0, -GJ/L, 0, 0, 0, 0, 0, GJ/L, 0, 0],
                     [0, 0, -6*EIy/L**2, 0, 2*EIy/L, 0, 0, 0, 6*EIy/L**2, 0, 4*EIy/L, 0],
                     [0, 6*EIz/L**2, 0, 0, 0, 2*EIz/L, 0, -6*EIz/L**2, 0, 0, 0, 4*EIz/L]
                     ])

def M_el(h):
    return h*np.block([[M, np.zeros_like(M)],
                       [np.zeros_like(M), M]])

#Global matrix
nodes_per_elem = 2
nodes = {i+1: eta for i, eta in enumerate(eta_grid)}
no_elems = (n_nodes - 1)//(nodes_per_elem - 1)
if (n_nodes - 1) % (nodes_per_elem - 1) != 0:
    raise ValueError

elems = {i+1: [j+1 for j in range(i, i+2)] for i in range(no_elems)}

K_gl = np.zeros((n_nodes*6, n_nodes*6))
M_gl = np.zeros((n_nodes*6, n_nodes*6))

for el in elems.values():
    i = el[0]
    K_gl[(i-1)*6:(i+1)*6 ,(i-1)*6:(i+1)*6] += K_el(EA, EIy, EIz, GJ, h)
    M_gl[(i-1)*6:(i+1)*6 ,(i-1)*6:(i+1)*6] += M_el(h)

M_gl_inv = inv(M_gl)
#Loads
f_tip_ext = np.array([[0,0,1]]).T
m_tip_ext = np.array([[0,0,0]]).T
F_gl = np.zeros((n_nodes*6, 1))
F_gl[-6:,:] = np.vstack((f_tip_ext, m_tip_ext))

F_gl_euler = F_gl


A = np.block([[np.zeros_like(K_gl), np.identity(n_nodes*6)],
               [-M_gl_inv @ K_gl, -0.0 * M_gl_inv]])



Bu0 = np.vstack((0*F_gl, 0*F_gl))
# %%
#Dynamic solver
def Bu_t(t, x):
    pos = x[:n_nodes*6]
    theta = pos[-2]
    out = F_gl*np.cos(theta)
    return np.vstack((0*F_gl, M_gl_inv @ out)) # 

def F_gl_t(t, x):
    pos = x[:n_nodes*6]
    theta = pos[-2]
    out = F_gl*np.cos(theta)
    return out

def dXdt(A, Bu, X):
    X = X.reshape((-1,1))
    ddt = A@X + Bu
    #BC
    ddt[:6] = 0
    ddt[n_nodes*6:n_nodes*6 + 6] = 0
    print(ddt.max())
    return ddt.reshape((-1,))

def static_ODE(K, F, X):
    X = X.reshape((-1,1))
    res = np.zeros_like(X)
    pos = X[:n_nodes*6]
    res[:n_nodes*6] = K @ pos - F
    #BC
    res[:6] = X[:6]
    #res[n_nodes*6:n_nodes*6 + 6] = X[n_nodes*6:n_nodes*6 + 6]
    return res.reshape((-1,))


def dyn_sol(X0, P):
    sol = solve_ivp(lambda t,x: dXdt(A, P*Bu_t(t,x), x), t_span = (0, time_grid[-1]), t_eval = time_grid, y0 = X0.reshape((-1,)))
    
    X_FOM = sol.y[:n_nodes*6,:] #Positions
    V_FOM = sol.y[n_nodes*6:,:] #Velocities
    return X_FOM, V_FOM

#Static solver
P = -5

def posz_EULER(P):
    return [P*x*x*(3*span - x)/(6*EIy) for x in eta_grid]


def modal():
    beta = np.array([0.5968, 1.4942, 2.5002, 3.4999])*(np.pi / span)
    w_theory = beta**2 * np.sqrt(EIy / dm)

    lam, Phi = eigh(K_gl[6:,6:], M_gl[6:,6:])
    #lam = lam[lam > 1e-6]
    w = np.sqrt(lam[:4])
    print(w_theory)
    print(w)
    
def stat_sol(P, X0 = None):
    if X0 is None:
        x0 = np.zeros((n_nodes*6, 1))
        xdot0 = np.zeros((n_nodes*6, 1))
        X0 = np.vstack((x0, xdot0))
    X = fsolve(lambda x: static_ODE(K_gl, P*F_gl_euler, x), X0)
    X_FOM = X[:n_nodes*6] #Positions
    V_FOM = X[n_nodes*6:] #Velocities
    return X_FOM, V_FOM


# %%
if __name__ == '__main__':
    animate = True
    
    if not animate:
        X_FOM, V_FOM = stat_sol(X0, P)
        posz = X_FOM[2::6]
        plt.figure()
        plt.plot(eta_grid, posz, label = 'FEM')
        plt.plot(eta_grid, posz_theory, label = 'Euler analytical')
        plt.legend()
        plt.show
    
    
    if animate:
        X_FOM, V_FOM = dyn_sol(X0, P)
        posz = X_FOM[2::6,:]
        velz = V_FOM[2::6,:]
        
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
        
        ani = FuncAnimation(fig, animate, frames=nt, interval=2)
        
        plt.show()