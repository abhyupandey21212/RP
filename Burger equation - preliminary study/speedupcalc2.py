# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:24:16 2025

@author: abhyu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import FOM
import POD2 as POD
import time
from FOM import T, X, dx, dt, nx, nt, m, n, nu, BC, X_span, T_span
import os

redo = True
FOM_res = "FOM_res.npz"
if os.path.isfile(FOM_res) and not redo:
    data = np.load(FOM_res)
    u0, U, NL, FOM_time = data["u0"], data["U"], data["NL"], float(data["FOM_time"])
    print('Found data')

else:
    FOM_time = time.time()
    u0, U, NL = FOM.calculate_FOM()
    FOM_time = time.time() - FOM_time
    np.savez(FOM_res, u0=u0, U=U, NL=NL, FOM_time=FOM_time)

print(f'FOM took {FOM_time}')


# %%
def error(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(a)

sizes = [2, 5, 10, 25, 50, 100, 200, 300, 400, 500]#, 20, 30, 50]
POD_errors = []
DEIM_errors = []
POD_times = []
DEIM_times = []
Usets_DEIM = []
Usets_POD = []
for l in sizes:
    print(f'With POD of size {l}')
    r=l
    L = FOM.burger_L(nu, dx, m)
    offline_time = time.time()
    POD_basis, L_l, u0_l, N_approx_l, P_POD_basis, P_DX_POD_basis = POD.POD_DEIM_offline(U, NL, L, l, dx)
    print(f'Offline computation took {time.time() - offline_time} s')
    
    t0 = time.time()
    U_recon_POD, U_l = POD.POD_online(POD_basis, L_l, u0_l, n, m, dt, dx)
    Usets_POD.append(U_recon_POD)
    POD_times.append(time.time() - t0)
    POD_errors.append(error(U, U_recon_POD))
    print(f'POD took {POD_errors[-1]}')
    t0 = time.time()
    U_recon_DEIM, U_l = POD.POD_DEIM_online(POD_basis, L_l, u0_l, N_approx_l, P_POD_basis, P_DX_POD_basis, n, m, dt)
    Usets_DEIM.append(U_recon_DEIM)
    DEIM_times.append(time.time() - t0)
    DEIM_errors.append(error(U, U_recon_DEIM))
    print(f'DEIM took {DEIM_errors[-1]}')

# %%

rel_sizes = np.array(sizes)/m
plt.figure()
plt.title('Time')
plt.xlabel('Relative ROM size')
plt.ylabel('Relative computation time')
plt.plot(rel_sizes, np.array(POD_times)/FOM_time, label = 'POD')
plt.plot(rel_sizes, np.array(DEIM_times)/FOM_time, label = 'POD-DEIM')
plt.hlines(FOM_time/FOM_time, xmin=0, xmax=rel_sizes[-1], label = 'FOM', colors= 'grey', ls = '-')
plt.legend()
plt.show()

plt.figure()
plt.title('Error')
plt.xlabel('Relative ROM size')
plt.ylabel('Relative error')
plt.semilogy(np.array(sizes)/m, POD_errors, label = 'POD')
plt.semilogy(np.array(sizes)/m, DEIM_errors, label = 'POD-DEIM')
plt.legend()
plt.show()
# %%




plt.figure()
plt.title('Error')
for i, size in enumerate(sizes):
    DEIM_error_history = [error(u_FOM, u_DEIM) for u_DEIM, u_FOM in zip(U.T, Usets_DEIM[i].T)]
    POD_error_history = [error(u_FOM, u_POD) for u_POD, u_FOM in zip(U.T, Usets_POD[i].T)]
    plt.plot(T_span, POD_error_history, label = f'POD, {size}',ls=':')
    plt.plot(T_span, DEIM_error_history, label = f'DEIM, {size}')
plt.legend()
plt.show()
# %%
U_recon_DEIM = Usets_DEIM[3]
U_recon_POD = Usets_POD[3]

x = X_span
t = T_span
# --- animation ---
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, label = 'FOM')
line2, = ax.plot([], [], lw=2, label = 'POD', ls = ':')
line3, = ax.plot([], [], lw=2, label = 'POD-DEIM', ls = '--')

ax.set_xlim(-X, X)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("1D viscous Burgers' equation")
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line3, line2

def animate(i):
    line1.set_data(x, U[:, i])
    line2.set_data(x, U_recon_POD[:, i])
    line3.set_data(x, U_recon_DEIM[:, i])

    ax.set_title(f"t = {t[i]:.3f}")
    return line1, line3, line2

ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                              interval=0.2, blit=True)

plt.show()
# %%
plt.figure()
for i in range(nt):
    i = int(i)
    if not i%100 == 0:
        continue
    #print(i)
    plt.plot(x, U[:,i], color = 'b')
    #plt.plot(x, U_recon_DEIM[:,i], color = 'b')
    #plt.plot(x, U_recon_POD[:,i], color = 'g')
plt.title('Convection-diffusion of N-wave by Bateman-Burger equation')
plt.xlabel('Spatial domain, x')
plt.ylabel('Wave amplitude')
plt.show()


