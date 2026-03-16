# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 21:19:32 2026

@author: abhyu
"""

import NLbeamFEM 
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
inv = np.linalg.inv
from dimentions import *
import matplotlib.pyplot as plt
import pandas as pd

M_gl, K_lin_gl, K_nl_gl, F_gl, V0, eta_grid, h, GBC, gBC, GBC_orth, NL_DEIM, elems = NLbeamFEM.define_FEM_matricies(n_nodes)
M_gl_inv = inv(M_gl)
# %%
#SOLUTION

V0 = np.zeros((n_nodes*12,1))
def stat_sol(F, V0 = V0, dt = 0.1, full_output = False, false_time_march = False):
    #initazlie with current V
    #Impose BC by premultiplying G0_orth and GL_orth to first and last nodes
    V0 = GBC_orth @ V0
    if not false_time_march:
        V_sol = fsolve(lambda V: NLbeamFEM.static(V, M_gl, K_lin_gl, K_nl_gl, F, GBC, gBC), V0,full_output=full_output)
    else:
        V = V0
        for t in np.arange(0, 1+dt, dt):
            print(t)
            Vn = fsolve(lambda Vn: NLbeamFEM.static(Vn, V, M_gl, K_lin_gl, K_nl_gl, t*P*F_gl, GBC, gBC), V,full_output=False)
            V = Vn
        V_sol = Vn
    if full_output:
        return V_sol[0], V_sol[-1]
    else:
        return V_sol
P = -5#*np.random.uniform(0, 1)
#print(P)
Vn, mesg = stat_sol(P*F_gl, dt=0.5, full_output=True)
velz = Vn[2::12]


# %%
#Post processing
X = NLbeamFEM.post_intrinsic(Vn, C)
posz = X[2::6]

# %%
#Sampling
def sample(P_max = 5, n_samples = 500, NL_sam = False, full_random = False, resample=False):
    try:
        V_samples = pd.read_csv("V_samples.csv", header=None).to_numpy()
        F_samples = pd.read_csv("F_samples.csv", header=None).to_numpy()
        if NL_sam:
            NL_samples = pd.read_csv("NL_samples.csv", header=None).to_numpy()
        if V_samples.shape[-1] < n_samples or resample:
            print('Mismatching sample size, resampling')
            raise()
        else:
            print('Found the sample')
            if NL_sam:
                return V_samples[:,:n_samples], F_samples[:,:n_samples], NL_samples[:,:n_samples]
            else:
                return V_samples[:,:n_samples], F_samples[:,:n_samples]
    except:
        fy_ext = np.array([[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
        fz_ext = np.array([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
        Fy_gl = np.vstack([fy_ext for _ in range(n_nodes)])
        Fz_gl = np.vstack([fz_ext for _ in range(n_nodes)])
        
        Fdic = {'y': Fy_gl, 'z': Fz_gl}
        
        F_samples = np.zeros((n_nodes*12, n_samples))
        V_samples = np.zeros((n_nodes*12, n_samples))
        NL_samples = np.zeros((n_nodes*12, n_samples))
        i = 0
        while i < n_samples:
            print(f'Sample {i}')
            if full_random:
                ax = np.random.choice(list(Fdic.keys()))
                print(f'Random force along {ax}-axis')
                F = Fdic[ax]
                F *= np.random.uniform(-P_max, P_max, F.size).reshape((-1,1))
            else:
                F = F_gl*P_max*np.random.uniform(-1, 1)
            sol_i = stat_sol(F,full_output=True)
            if not sol_i[-1] == 'The solution converged.':
                print(f'Sample {i} did not converge, trying again...')
                continue
            print(f'Sample {i} converged.')

            V = sol_i[0]
            V_samples[:,i] = V
            NL_samples[:,i] = K_nl_gl(V) @ V
            F_samples[:,i] = F.reshape((-1,))
            i += 1
        dfV = pd.DataFrame(V_samples)
        dfV.to_csv("V_samples.csv", index=False, header=False)
        dfF = pd.DataFrame(F_samples)
        dfF.to_csv("F_samples.csv", index=False, header=False)
        dfNL = pd.DataFrame(NL_samples)
        dfNL.to_csv("NL_samples.csv", index=False, header=False)
    if NL_sam:
        return V_samples, F_samples, NL_samples
    return V_samples, F_samples
V_samples, F_samples, NL_samples = sample(n_samples=500, NL_sam=True, full_random=True, resample=False)
X_samples = NLbeamFEM.post_intrinsic(V_samples, C)
# %% 
#POD
import LbeamFEM
import time
POD = True
DEIM = True

if POD == True:
    #settings
    l = 4 #less dims means the model does not capture the input F, so gives a zero solutions (close to), more dims means the problem simply does NOT converge, I found 4 to be the perfect compromise
    center = 0 #Centering the data seems to make things worse not better
    interior = slice(6, -6) #POD only on the non-constrainted DOFs

    P = 5*np.random.uniform(-1,1) #So far I only sampled within \pm 5N so I am only testing that
    F = P*F_gl
    #Euler
    X_eulerFEM, _ = LbeamFEM.stat_sol(P)
    posz_eulerFEM = X_eulerFEM[2::6]
    posz_theory = LbeamFEM.posz_EULER(P)
    #FOM
    start = time.time()
    V_FOM = stat_sol(F)
    FOM_time = time.time() - start
    X_FOM = NLbeamFEM.post_intrinsic(V_FOM, C)
    posz_FOM = X_FOM[2::6]
    
    #POD OFFLINE
    PHI, mean, sig = NLbeamFEM.POD_offline(V_samples[interior,:], l, center=center)
    V_mean = np.zeros_like(V0)
    V_mean[interior] = mean
    F_r = PHI.T @ F[interior]
    M_gl_r = PHI.T @ M_gl[interior,interior] @ PHI
    K_lin_gl_r = PHI.T @ K_lin_gl[interior,interior] @ PHI
    V0_r = PHI.T @ V0[interior]    
    V_POD = np.zeros_like(V_FOM)
    
    #POD ONLINE
    start = time.time()
    V_POD_r = fsolve(lambda V_r: NLbeamFEM.static_POD(V_r, M_gl_r, K_lin_gl_r, K_nl_gl, F_r, PHI), V0_r)
    POD_time = time.time() - start
    V_POD[interior] = PHI @ V_POD_r
    V_POD += center*V_mean.reshape((-1,))
    X_POD = NLbeamFEM.post_intrinsic(V_POD, C)
    posz_POD = X_POD[2::6]
    
    if not DEIM:
        print(f'FOM: {FOM_time:.6f} \n POD: {POD_time:.6f}')
        
        plt.figure()
        plt.plot(eta_grid, posz_POD, label=f'POD, {POD_time:.4f}s')
        plt.plot(eta_grid, posz_FOM, label=f'FOM, {FOM_time:.4f}s', ls = 'dotted')
        plt.plot(eta_grid, posz_eulerFEM, label='Euler, FEM')
        plt.plot(eta_grid, posz_theory, label='EUler, theory')
        plt.title(f'Deflection under tip load of {P:.3f} N')
        plt.xlabel('Span, m')
        plt.ylabel('Displacement, m')
        plt.legend()
        plt.show()  


# %%
#POD DEIM
#Guiding princ: we approximate the nonlinear function N as 
# N ~ Xi @ inv(P.T @ Xi) @ P.T @ N 
#Where P.T @ N contains only r coords of the N vector, so we can speed up calculations by only calculating those coords. P.T has rows of e_i (indexing vectoring)
#This is easy if the N function is elementwise, but in our case the N vector depends on neighbouring coords (nodes) as well, so we dont have as big of a speed up. So the way I have implemented it, which is a relatively simple but NOT the smallest possible implementation is to keep all the nodes of whichever element e_i happens to fall within. Eg if i = 10, then that falls within the first element (which contains the first 36 dims), so we keeps the whole of the first elem and the correspoding nl matrix. and so on...

if DEIM and POD:
    l = 4
    r = 10
    #POD DEIM OFFLINE
    PHI, NL_l, P_PHI, P_l = NLbeamFEM.POD_DEIM_offline(V_samples[interior,:], NL_samples[interior,:], l=l, r=r)
    
    DEIM_elems = {} #the elements whose nodes need to be kept in the reduced dimention K_nl_gl function
    for p in np.where(P_l == 1)[0]:
        for el in elems:
            if p//12 + 1 in elems[el]:
                DEIM_elems[el] = elems[el]
                
    F_r = PHI.T @ F[interior]
    M_gl_r = PHI.T @ M_gl[interior,interior] @ PHI
    K_lin_gl_r = PHI.T @ K_lin_gl[interior,interior] @ PHI
    V0_r = PHI.T @ V0[interior]   
    V_DEIM = np.zeros_like(V_FOM)
    
    #POD DEIM ONLINE
    start = time.time()
    out = fsolve(lambda V_r: NLbeamFEM.static_DEIM(V_r, M_gl_r, K_lin_gl_r, NL_DEIM, 100*F_r, PHI, P_l, NL_l, DEIM_elems), V0_r, full_output = True)
    V_DEIM_r = out[0]
    DEIM_time = time.time() - start
    V_DEIM[interior] = PHI @ V_DEIM_r
    X_DEIM = NLbeamFEM.post_intrinsic(V_DEIM, C)
    posz_DEIM = X_DEIM[2::6]
    
    #Plottings
    #print(f'FOM: {FOM_time:.6f} \n POD: {POD_time:.6f}')
    plt.figure()
    plt.plot(eta_grid, posz_DEIM, label=f'DEIM, {DEIM_time:.4f}s')
    plt.plot(eta_grid, posz_POD, label=f'POD, {POD_time:.4f}s', ls = 'dashed')
    plt.plot(eta_grid, posz_FOM, label=f'FOM, {FOM_time:.4f}s', ls = 'dotted')
    plt.plot(eta_grid, posz_eulerFEM, label='Euler, FEM')
    plt.plot(eta_grid, posz_theory, label='EUler, theory')
    plt.title(f'Deflection under tip load of {P:.3f} N')
    plt.xlabel('Span, m')
    plt.ylabel('Displacement, m')
    plt.legend()
    plt.show() 
    
    
    #IMPLEMENTATION NEEDS new NL func that can take just a few nodes and their surronding nodes and provide the NL func at those points (not the global NL func), we need a psuedo pointwise NL func (pseudo coz it will need a certain number of neighbours), the nodes at which NL needs to be evalutaed is the 1s of P
    #V_DEIM = np.zeros_like(V_FOM)
    
# %%
#PLOTTING
animate = False
if animate: 
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
if animate == False:
    X_eulerFEM, _ = LbeamFEM.stat_sol(P)
    posz_eulerFEM = X_eulerFEM[2::6]
    posz_theory = LbeamFEM.posz_EULER(P)

    plt.figure()
    plt.plot(eta_grid, posz, label='NL Intrinsic FEM')
    plt.plot(eta_grid, posz_eulerFEM, label='Euler FEM')
    plt.plot(eta_grid, posz_theory, label='Euler analytical', ls='dotted')
    plt.title(f'Deflection under tip load of {P} N')
    plt.xlabel('Span, m')
    plt.ylabel('Displacement, m')
    plt.legend()
    plt.show()    