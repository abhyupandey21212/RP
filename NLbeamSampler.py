# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 13:51:56 2026

@author: abhyu
"""

#Sampling
from IntrinsicBeamNL import BeamNL, BeamNL_POD
import dimentions as dim
import numpy as np
import pandas as pd
from LbeamFEM import posz_EULER
#import threading
#from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor



#Sampling
def sample(sampler, P_max = 200, n_samples = 500, full_random = False, resample=False, n_nodes=dim.n_nodes, name=''):
    name = str(name)
    try:
        V_samples = pd.read_csv("V_samples"+name+".csv", header=None).to_numpy()
        F_samples = pd.read_csv("F_samples"+name+".csv", header=None).to_numpy()
        NL_samples = pd.read_csv("NL_samples"+name+".csv", header=None).to_numpy()
        NL_samples_during_sol = pd.read_csv('NL_samples_during_sol'+name+'.csv', header=None).to_numpy()
        if resample:
            raise()
        else:
            print('Found the sample')
            return V_samples, F_samples, NL_samples, NL_samples_during_sol

    except:
        Fdic = {'y': Fy_gl, 'z': Fz_gl}
        
        F_samples = np.zeros((n_nodes*12, n_samples))
        V_samples = np.zeros((n_nodes*12, n_samples))
        NL_samples = np.zeros((n_nodes*12, n_samples))
        NL_samples_during_sol = []
        i = 0
        while i < n_samples:
            if full_random:
                ax = np.random.choice(list(Fdic.keys()))
                #print(f'Random force along {ax}-axis')
                F = Fdic[ax]
                F *= np.random.uniform(-P_max, P_max, F.size).reshape((-1,1))
            else:
                P = np.random.uniform(-P_max, P_max)
                F = Fztip_gl*P
                #print(f'Sample {i}, tip force {np.round(P, 1)}')
            sol_i = sampler.static_solver(F,full_output=True, anal_jac=True, legacy=False)
            if not sol_i[-1] == 'The solution converged.':
                #print(f'Sample {i} did not converge, trying again...')
                continue
            #print(f'Sample {i} converged.')

            V = sol_i[0]
            V_samples[:,i] = V
            NL_samples[:,i] = sampler.K_nl_gl(V) @ V
            F_samples[:,i] = F.reshape((-1,))
            if len(NL_samples_during_sol) <= 10000:
                NL_samples_during_sol = NL_samples_during_sol + sampler.NL_samples.copy()
            i += 1
        NL_samples_during_sol = np.array(NL_samples_during_sol)
        NL_samples_during_sol = NL_samples_during_sol.reshape((sampler.n_nodes*12,-1))
        dfV = pd.DataFrame(V_samples)
        dfV.to_csv("V_samples"+name+".csv", index=False, header=False)
        dfF = pd.DataFrame(F_samples)
        dfF.to_csv("F_samples"+name+".csv", index=False, header=False)
        dfNL = pd.DataFrame(NL_samples)
        dfNL.to_csv("NL_samples"+name+".csv", index=False, header=False)
        dfNL = pd.DataFrame(NL_samples_during_sol)
        dfNL.to_csv("NL_samples_during_sol"+name+".csv", index=False, header=False)
    return V_samples, F_samples, NL_samples, NL_samples_during_sol

def NL_sample(n_samples, sampler, n_nodes, interior):   
    try:
        NL_samples = pd.read_csv("NL_samples2.csv", header=None).to_numpy()
        if NL_samples.shape[1] != n_samples:
            raise
        print('Found samples...')
    except:
        NL_samples = NL_samples = np.zeros((n_nodes*12, n_samples))
        print('Collecting samples...')
        for i in range(n_samples):
            #print('Sample ', i+1)
            #V = np.zeros((n_nodes*12,1))
            #V[interior,:] = np.random.uniform(-50, 50, n_nodes*12 - 12).reshape((-1,1))
            #V_rand = np.random.multivariate_normal(V_mean, cov=3*np.diag(V_stdev))
            V_rand = np.random.uniform(V_mean - 3*V_stdev, V_mean + 3*V_stdev)

            V = V_rand
            NL = sampler.K_nl_gl(V) @ V
            NL_samples[:,i] = NL.reshape((-1,))
        dfNL = pd.DataFrame(NL_samples)
        dfNL.to_csv("NL_samples2.csv", index=False, header=False)
    return NL_samples

def NL_sample_from_POD_rand(VPOD_samples, n_samples, sampler, n_nodes, interior, use_stored = True):   
    try:
        NL_samples = pd.read_csv("NL_samples_POD.csv", header=None).to_numpy()
        if not use_stored:
            raise
        print('Found samples...')
    except:
        PHI = sampler.PHI
        V_samplesT = PHI @ VPOD_samples
        V_samples = np.zeros((12*n_nodes,VPOD_samples.shape[1]))
        V_samples[interior,:] = V_samplesT
        V_mean = np.mean(V_samples, axis=-1)
        V_stdev = np.std(V_samples, axis=-1)
        NL_samples = np.zeros((n_nodes*12, n_samples))
        interior = sampler.interior
        print('Collecting samples...')
        for i in range(n_samples):
            #print('Sample ', i+1)
            #V = np.zeros((n_nodes*12,1))
            #V[interior,:] = np.random.uniform(-50, 50, n_nodes*12 - 12).reshape((-1,1))
            #V_rand = np.random.multivariate_normal(V_mean, cov=3*np.diag(V_stdev))
            V_rand = np.random.uniform(V_mean - 3*V_stdev, V_mean + 3*V_stdev)
            V = V_rand
            NL = sampler.K_nl_gl(V) @ V
            NL_samples[:,i] = NL.reshape((-1,))
        dfNL = pd.DataFrame(NL_samples)
        dfNL.to_csv("NL_samples_POD.csv", index=False, header=False)
    return NL_samples

def NL_sample_from_POD_no_rand(VPOD_samples, sampler, n_nodes, interior, use_stored = True):  
     try:
         NL_samples = pd.read_csv("NL_samples_POD_no_random.csv", header=None).to_numpy()
         if not use_stored:
            raise
         print('Found samples...')
         
     except:
         PHI = sampler.PHI
         n_samples = VPOD_samples.shape[1]

         NL_samples = []
         interior = sampler.interior
         print('Collecting samples...')
         for i in range(n_samples):
             #print('Sample ', i+1)
             V = np.zeros((n_nodes*12,1))
             V_r = VPOD_samples[:,i].reshape((-1,1))
             if V_r.max() < 1e-10:
                 #print('too small')
                 continue
             V[interior,:] = PHI @ V_r
             NL = sampler.K_nl_gl(V) @ V
             NL_samples.append(NL.reshape((-1,)))
         NL_samples = np.array(NL_samples).T
         dfNL = pd.DataFrame(NL_samples)
         dfNL.to_csv("NL_samples_POD_no_random.csv", index=False, header=False)
     return NL_samples

def VPOD_sample(n_samples, sampler, n_nodes, interior, l, P_max = 5, use_stored = True):   
    """This time we generate random tip force, solve it with POD and store ALL intermediate V_r that fsolve tries, we then calculate NL of all of those V_r (by firsting finding V_recon with PHI) and use those as our samples"""
    try:
        VPOD_samples = pd.read_csv("VPOD_samples.csv", header=None).to_numpy()
        if not use_stored:
            raise
        print('Found samples...')
    except:
        VPOD_samples = np.zeros((n_nodes*12, n_samples))
        print('Collecting samples...')
        i=0
        while i  < n_samples:
            print(i)
            F = Fztip_gl*P_max*np.random.uniform(-1, 1)
            sol_POD = sampler.static_sampler_POD(F,full_output=True)
            if not sol_POD[-1] == 'The solution converged.':
                print(f'Sample {i} did not converge, trying again...')
                continue
            i += 1
        VPOD_samples = np.array(sampler.history)
        #print(VPOD_samples.shape)
            
        dfNL = pd.DataFrame(VPOD_samples.reshape((-1,l)))
        dfNL.to_csv("VPOD_samples.csv", index=False, header=False)
    return VPOD_samples.reshape((-1,l)).T

def NL_DEIM_tester(n_samples, approx, sampler, Fz_gl, Fy_gl, Fztip_gl, Fytip_gl, P_max = 5, full_random=False):  
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
            F = Fztip_gl*P_max*np.random.uniform(-1, 1)
        sol_FOM = sampler.static_solver(F,full_output=True)
        if not sol_FOM[-1] == 'The solution converged.':
            print(f'Sample {i} did not converge, trying again...')
            continue
        sol_POD = approx.static_solver_POD(F,full_output=True)
        if not sol_POD[-1] == 'The solution converged.':
            print(f'Sample {i} did not converge, trying again...')
            continue
        i += 1
    
        V = sol_POD[0].reshape((-1,1))
        NL_FOM = sampler.K_nl_gl(V) @ V
        PHI = approx.PHI
        V_r = PHI.T @ V[interior,:]
        NL_DEIM = approx.NL_DEIM2(V_r)
        err = np.linalg.norm(NL_FOM[interior] - NL_DEIM) / np.linalg.norm(NL_FOM[interior])
        print(f'Extrapolation error : {100 * np.round(err, 4)}%')

interior = slice(6,-6)
samplerFOM = BeamNL(dim.n_nodes, dim.span, dim.chord, dim.dm, dim.r_g, dim.mass_mom_per_span, dim.Cinv)
Fytip_gl, Fztip_gl, Fy_gl, Fz_gl = samplerFOM.force_templates()
#V_samples, F_samples, NL_samples, NL_samples_during_sol = sample(samplerFOM, n_samples=4000,resample=True, full_random=False)
def sample_wrapper(name):
    return sample(samplerFOM, n_samples=1000,resample=True, full_random=False, name=name)

def stitcher(name, n):
    samples = []
    for i in range(n):
        file = name + str(i+1) + '.csv'
        temp=pd.read_csv(file, header=None).to_numpy()
        samples.append(temp)
        print(temp.shape)
    out = np.hstack(tuple(samples))
    dfNL = pd.DataFrame(out)
    dfNL.to_csv(name+".csv", index=False, header=False)
    return out

def mesh_CV_worker(pars):
    n, n_samples, P = pars
    beamNL = BeamNL(n_nodes=n, span=dim.span, chord=dim.chord, dm=dim.dm,
                    r_g=dim.r_g, I_per_span=dim.mass_mom_per_span, Cinv=dim.Cinv)
    F_tip = beamNL.force_templates()[1]
    s = []
    attempt = 0
    while len(s) < n_samples:
        if attempt >= len(P):
            break
        F = F_tip * P[attempt]
        attempt += 1
        Vsol, mes = beamNL.static_solver(F, anal_jac=True, legacy=False,
                                          full_output=True, NLinclude=1)
        if mes == 'The solution converged.':
            X = beamNL.post(Vsol)
            s.append(X.max())
    return np.array(s)

def mesh_convergence_FOM(meshes=[5,7,9,11,13,15], n_samples=1000, P_max=200, no_threads=10):
    P = np.random.uniform(-P_max, P_max, n_samples)
    pars = [[n, n_samples, P] for n in meshes]
    with ProcessPoolExecutor(max_workers=no_threads) as executor:
        results = list(executor.map(mesh_CV_worker, pars))
    samples = {m: r for m, r in zip(meshes, results)}
    return samples

def mesh_convergence_post(samples_input):
    samples = samples_input.copy()
    m = max(samples.keys())
    ans = samples.pop(m)
    errors = {s:0 for s in samples.keys()}
    errors[m] = 0
    for s in samples:
        r = []
        for i,a in enumerate(ans):
            r.append((samples[s][i]/a) - 1)
        errors[s] = np.mean(r)
    plt.figure()
    plt.plot(errors.keys(), errors.values())
    plt.title('Relative error in tip deflection')
    plt.xlabel('Number of nodes in the mesh')
    plt.ylabel('Relative error')
    #plt.ylim((0,0.5))
    return errors


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    no_threads = 10
    #F_samples = stitcher('F_samples', no_threads)
    #NL_samples_fsolve = stitcher('NL_samples_during_sol', no_threads)
    samples = mesh_convergence_FOM(meshes=[5,7,9,11,13], n_samples = 100)
    errors2 = mesh_convergence_post(samples)
    
#samlpe = False
#if sample:
#    with ProcessPoolExecutor(max_workers=no_threads) as executor:
#        for i in range(4):
#            executor.submit(sample_wrapper, f'{i+1}')
#        results = list(executor.map(sample_wrapper, range(1,no_threads+1)))
#    NL_samples = stitcher('NL_samples', no_threads)    
#    V_samples = stitcher('V_samples', no_threads)


""" 
Fytip_gl, Fztip_gl, Fy_gl, Fz_gl = samplerFOM.force_templates()
V_mean = np.mean(V_samples, axis=-1)
V_stdev = np.std(V_samples, axis=-1)

#NL_samples2 = NL_sample(10000, sampler, n_nodes, interior)
l = 4
samplerPOD = BeamNL_POD(samplerFOM, l=l)   
samplerPOD.POD_offline(V_samples[interior,:])
VPOD_samples = VPOD_sample(1000, samplerPOD, dim.n_nodes, interior, l=4)
#NL_samples_POD = NL_sample_from_POD(VPOD_samples, 10000, samplerPOD, dim.n_nodes, interior)
NL_samples_POD = NL_sample_from_POD_no_rand(VPOD_samples, samplerPOD, dim.n_nodes, interior, use_stored=False)
approx = BeamNL_POD(samplerFOM, l=4, r=20)    
approx.POD_DEIM_offline(V_samples[interior,:], NL_samples_POD[interior,:])
#print(np.where(tester.P.T==1))

#NL_DEIM_tester(100, approx, sampler, Fz_gl, Fy_gl, Fztip_gl, Fytip_gl)
"""

        
