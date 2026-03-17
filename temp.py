from IntrinsicBeamNL import *
import dimentions as dim
import numpy as np
import pandas as pd
#POD PARAM
l = 98
#DEIM PARAM
r = 150
V_samples = pd.read_csv(f"Data/NONLn_nodes={dim.n_nodes}/V_samplesNONL.csv", header=None).to_numpy()  
NL_samples = pd.read_csv(f"Data/n_nodes={dim.n_nodes}/NL_samples{dim.n_nodes}.csv", header=None).to_numpy()
interior = slice(6,-6)

#DEFINIING BEAM OBJECTS and training
beamNL = BeamNL(n_nodes=dim.n_nodes, span=dim.span, chord=dim.chord, dm=dim.dm, r_g=dim.r_g, I_per_span=dim.mass_mom_per_span, Cinv=dim.Cinv)
beamPOD = BeamNL_POD(beamNL, l=l,r=r)  
beamPOD.center = 1
beamPOD.POD_offline(V_samples[interior,:])
beamPOD.POD_DEIM_offline(V_samples[interior,:], NL_samples[interior,:])
F_tip = beamNL.force_templates()[1]
P_max = 20
P = np.random.uniform(-P_max,P_max)
P = np.round(P,2)
F = F_tip*P
#l_list = [50,60,70,80,90,100,110,120,130,140,150]
#l_list_fine = [90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110]
l_list = [2,3,4,5,6,7,8,9,10]
NLinc = 0


test = Tester(10)
res_dic = {}
key = {'speedup':0,'percentCV':1,'X error':2}
test_points = 100

for l in l_list:
    print(l)
    beamPOD = BeamNL_POD(beamNL, l=l)  
    beamPOD.POD_offline(V_samples[interior,:])
    res = test.test_POD(beamNL, beamPOD, F_tip, test_points=test_points,full_output=True, NLinclude=0)
    res_dic[l] = res.copy()
   
"""

print('------FOM-------')
start = time.time()
V_FOM,mes = beamNL.static_solver(F, anal_jac=True, legacy=False, full_output=True, NLinclude = NLinc)
#V_FOM = np.zeros_like(beamNL.V0)
#V_FOM[interior] = V_FOM_int.reshape((-1,1))
print(mes)

FOM_time = time.time() - start
X_FOM = beamNL.post(V_FOM)
print('------POD-------')
V_POD,mes,POD_time = beamPOD.static_solver_POD(F, anal_jac=True, full_output=True, V0=None, legacy=False, NLinclude = NLinc, diagnostic=False, timer=True)
print(mes)
X_POD = beamPOD.post(V_POD)
#print('------DEIM-------')
#V_DEIM,mes,POD_time = beamPOD.static_solver_DEIM(F, full_output=True, V0=None, NLinclude = NLinc, timer=True)
#print(mes)
#X_DEIM = beamPOD.post(V_DEIM)

"""
# %%
plotting = False
if plotting:
    x = beamNL.eta_grid
    plt.figure()
    plt.plot(x, X_FOM[2::6], label=f'FOM, t = {np.round(FOM_time,6)}s')
    plt.plot(x, X_POD[2::6], label=f'POD, l = {l}, t = {np.round(POD_time,6)}s', ls='dashed')
    #plt.plot(x, X_DEIM[2::6], label='DEIM')
    plt.legend()
    plt.title(f'Deflection of beam under tip force of {P} N')
    plt.xlabel('Span, m')
    plt.ylabel('Deflection, m')



""" DYN SOLVER IS NOT YET IMPLEMENTED
def dynamic(t, V, Minv, K_lin, K_nl, F, G, gdot):
print(t)
interior = slice(6,-6)
V = V.reshape((-1,1))
dVdt = Vn = fsolve(lambda Vn: static(Vn, V, M_gl, K_lin_gl, K_nl_gl, t*P*F_gl, GBC, gBC), V,full_output=False)
dVdt += G@(gdot)
print(dVdt.max())
return dVdt.reshape((-1,))





# %%



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


"""