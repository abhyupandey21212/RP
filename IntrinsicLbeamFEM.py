# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 21:19:32 2026

@author: abhyu
"""

import NLbeamFEM
import numpy as np
from scipy.integrate import solve_ivp

#Discritizaton
from dimentions import *


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



M_gl, K_lin_gl, K_nl_gl, F_gl, V0, eta_grid, h = NLbeamFEM.define_FEM_matricies(n_nodes)

def F_gl_t(t):
    if t < 0.03:
        return F_gl# * np.sin(5*t)
    else:
        return F_gl*0
#SOLUTION

V_mat = np.zeros((V0.shape[0], len(time_grid)))
V_mat[:,0] = V0.reshape((-1,))
for i, t in enumerate(time_grid[1:3]):
    print(t)
    #initazlie with current V
    V = V_mat[:,-1]
    #Impose BC by premultiplying G0_orth and GL_orth to first and last nodes
    V = GBC_orth @ V
    Vn = fsolve(lambda Vn: implicit_ODE(M_gl, K_lin_gl, K_nl_gl, F_gl_t(t), V, Vn, GBC, gBC), V)
    #add Vn to the list of outputs
    V_mat[:,i+1] = Vn
V_FOM = V_mat
velz = V_mat[2::12,:]

# %%
V0alt = V0
V0[2::12,:] = np.array([[6.37845187e-25, 8.81518959e-02, 1.17481532e+00]]).T
V_mat = np.zeros((V0.shape[0], len(time_grid)))
M_gl_inv = inv(M_gl)
sol = solve_ivp(lambda t,x: dVdt_LIN(M_gl_inv, K_lin_gl, K_nl_gl, 0*F_gl_t(t), x, t), t_span = (0, 0.1), t_eval = time_grid, y0 = V0.reshape((-1,)))

V_FOM = sol.y
velz = V_FOM[2::12,:]#FOM2[2::12,:]

#The reason it doesnt return to equilib when starting from not equilib is NOT coz I REMOVED THE PRETWIST TERM, thats for when the equilib SHOULD be twisted

    
# %%
#PLOTTING
#NOTE, these are nodal values, for proper visualization they have to be plotted with the shape function integrals
from LbeamFEM import velz as velzLin
    
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


velz_FOM = V_FOM[2::12,:]
velz_POD = velzLin
m, nt = velz_FOM.shape

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

line1, = ax.plot(eta_grid, velz_FOM[:,0], c='b', label='Intrinsic')
line2, = ax.plot(eta_grid, velz_POD[:,0], c='r', label='Euler')
#line3, = ax.plot(eta_grid, velz_mean[:,0], c='g', label='mean')

ax.set_xlim(eta_grid.min(), eta_grid.max())
ax.set_ylim(-3, 3)

ax.legend()
# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 't', 0, nt-1, valinit=0, valstep=1)

def update(i):
    line1.set_ydata(velz_FOM[:,int(i)])
    line2.set_ydata(velz_POD[:,int(i)])
    #line3.set_ydata(velz_mean[:,int(i)])

    return line1, line2, #line3,

slider.on_changed(update)

# Animation
def animate(i):
    slider.set_val(i)
    return line1, line2, #line3,

ani = FuncAnimation(fig, animate, frames=nt, interval=50)
plt.show()