# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 13:51:56 2026

@author: abhyu
"""

from IntrinsicBeamNL import BeamNL, BeamNL_POD
import dimentions as dim
import numpy as np
import pandas as pd
#from LbeamFEM import posz_EULER
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import time



#Sampling
def sample(sampler, P_max = 200, n_samples = 500, full_random = False, resample=False, n_nodes=dim.n_nodes, name='',NLinclude=1):
    name = str(name)
    Fytip_gl, Fztip_gl, Fy_gl, Fz_gl = sampler.force_templates()
    try:
        V_samples = pd.read_csv("V_samples"+name+".csv", header=None).to_numpy()
        F_samples = pd.read_csv("F_samples"+name+".csv", header=None).to_numpy()
        NL_samples = pd.read_csv("NL_samples"+name+".csv", header=None).to_numpy()
        if resample:
            raise()
        else:
            print('Found the sample')
            return V_samples, F_samples, NL_samples

    except:
        Fdic = {'y': Fy_gl, 'z': Fz_gl}
        
        F_samples = np.zeros((n_nodes*12, n_samples))
        V_samples = np.zeros((n_nodes*12, n_samples))
        NL_samples = np.zeros((n_nodes*12, n_samples))
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
            sol_i = sampler.static_solver(F,full_output=True, anal_jac=True, legacy=False, NLinclude=NLinclude)
            if not sol_i[-1] == 'The solution converged.':
                print(f'Sample {i} did not converge, trying again...')
                continue
            #print(f'Sample {i} converged.')

            V = sol_i[0]
            V_samples[:,i] = V
            NL_samples[:,i] = sampler.K_nl_gl(V) @ V
            F_samples[:,i] = F.reshape((-1,))
           
            i += 1
        dfV = pd.DataFrame(V_samples)
        dfV.to_csv("V_samples"+name+".csv", index=False, header=False)
        dfF = pd.DataFrame(F_samples)
        dfF.to_csv("F_samples"+name+".csv", index=False, header=False)
        dfNL = pd.DataFrame(NL_samples)
        dfNL.to_csv("NL_samples"+name+".csv", index=False, header=False)
    return V_samples, F_samples, NL_samples


#V_samples, F_samples, NL_samples, NL_samples_during_sol = sample(samplerFOM, n_samples=4000,resample=True, full_random=False)
def sample_wrapper(pars):
    sampler,P_max,n_samples,full_random,resample,name,NLinclude = pars
    return sample(sampler=sampler, n_samples=n_samples,full_random=full_random,resample=resample,name=name, P_max=P_max,NLinclude=NLinclude)

def stitcher(name, n):
    samples = []
    del_dir = []
    for i in range(n):
        file = name + str(i+1) + '.csv'
        temp=pd.read_csv(file, header=None).to_numpy()
        samples.append(temp)
        print(temp.shape)
        del_dir.append(file)
    out = np.hstack(tuple(samples))
    dfNL = pd.DataFrame(out)
    dfNL.to_csv(name+".csv", index=False, header=False)
    for file in del_dir:
        os.remove(file)
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

class Tester:
    def __init__(self, n_threads):
        self.n_threads = n_threads
        
    def FOM_timer(self, beamNL,F_shape,P_max=200,test_points=1000):
        times = []
        i=0
        while i < test_points:
            #print(f'-----{i}-----')
            P = np.random.uniform(-P_max,P_max)
            F = F_shape*P
            #print('------FOM-------')
            start = time.time()
            V_FOM,mes = beamNL.static_solver(F)
            times.append(time.time() - start)
            if mes == 'The solution converged.':
                i+=1
        return np.mean(times)
                
    def jac_tester_worker(self,pars):
        n_nodes, n_samples, F_idx, P_max = pars
        beamNL = BeamNL(n_nodes=n_nodes, span=dim.span, chord=dim.chord, dm=dim.dm,r_g=dim.r_g, I_per_span=dim.mass_mom_per_span, Cinv=dim.Cinv)
        F_shape = beamNL.force_templates()[F_idx]
        fd_time = 0
        jac_time = 0
        for i in range(n_samples):
            P = np.random.uniform(-P_max, P_max)
            F = F_shape * P

            start = time.time()
            V = beamNL.static_solver(F, anal_jac=False)
            fd_time += time.time() - start

            start = time.time()
            V = beamNL.static_solver(F, anal_jac=True)
            jac_time += time.time() - start
        return n_nodes, fd_time/n_samples, jac_time/n_samples
    
    def jac_tester_worker_POD(self,pars):
        interior=slice(6,-6)
        n_nodes, l, n_samples, F_idx, P_max = pars
        beamNL = BeamNL(n_nodes=n_nodes, span=dim.span, chord=dim.chord, dm=dim.dm,r_g=dim.r_g, I_per_span=dim.mass_mom_per_span, Cinv=dim.Cinv)
        F_shape = beamNL.force_templates()[F_idx]
        beamPOD = beamPOD = BeamNL_POD(beamNL, l=l) 
        V_samples = pd.read_csv(f"Data/n_nodes={n_nodes}/V_samples.csv", header=None).to_numpy()  
        beamPOD.POD_offline(V_samples[interior,:])


        fd_time = 0
        jac_time = 0
        for i in range(n_samples):
            P = np.random.uniform(-P_max, P_max)
            F = F_shape * P

            start = time.time()
            V = beamPOD.static_solver_POD(F, anal_jac=False)
            fd_time += time.time() - start

            start = time.time()
            V = beamPOD.static_solver_POD(F, anal_jac=True)
            jac_time += time.time() - start
        return n_nodes, fd_time/n_samples, jac_time/n_samples
    
    def jac_tester(self, n_nodes_list, F_idx, P_max=200, n_samples=100, multithreading=True, POD=False):
        n_threads = self.n_threads
        worker = self.jac_tester_worker
        pars_list = [[n_nodes,n_samples,F_idx,P_max] for n_nodes in n_nodes_list]

        if POD:
            worker = self.jac_tester_worker_POD
            pars_list = [[n_nodes,l,n_samples,F_idx,P_max] for n_nodes,l in n_nodes_list]
        times = [{},{},{}]
        if multithreading:
            with ProcessPoolExecutor(max_workers=n_threads) as executor:
                results = list(executor.map(worker, pars_list))
        else:
            results = []
            for pars in pars_list:
                results.append(self.jac_tester_worker(pars))
    
        for n_nodes, fd_time, jac_time in results:
            times[1][n_nodes] = fd_time
            times[2][n_nodes] = jac_time
            speedup = fd_time/jac_time
            times[0][n_nodes] = speedup
        return times
    
    
    
    def rel_error(self, true, approx):
        return np.linalg.norm(true - approx) / np.linalg.norm(true)
    
    def test_POD(self, beamNL, beamPOD, F_shape, test_points=100, P_max = 150, redoifnoCV = True, full_output = False, NLinclude=1):
        FOM_times = []
        POD_times = []
        POD_timesCV = []

        POD_Xerrors = []
        POD_XerrorsCV = []

        POD_Verrors = []
        i=0
        j = 0
        while i < test_points:
            P = np.random.uniform(-P_max,P_max)
            F = F_shape*P
            start = time.time()
            V_FOM,mes = beamNL.static_solver(F,NLinclude=NLinclude)
            FOM_times.append(time.time() - start)
            X_FOM = beamNL.post(V_FOM)
            
            V_POD, mes1, POD_time = beamPOD.static_solver_POD(F,timer=True,NLinclude=NLinclude)
            POD_times.append(POD_time)
            X_POD = beamPOD.post(V_POD)
            Xerr = self.rel_error(X_FOM, X_POD)
            POD_Verrors.append(self.rel_error(V_FOM, V_POD))
            POD_Xerrors.append(Xerr)
            ifmes1 = mes1 == 'The solution converged.' 
            if ifmes1:
                i+=1
                POD_timesCV.append(POD_time)
                POD_XerrorsCV.append(Xerr)
            j+=1
            if j > 1.5*test_points:
                break

        eV = np.mean(POD_Verrors)
        eX = np.mean(POD_Xerrors)
        eXCV = np.mean(POD_XerrorsCV)
        tPOD  = np.mean(POD_times)
        tFOM = np.mean(FOM_times)
        tPODCV = np.mean(POD_timesCV)
        CV = i/test_points
        speedup = tFOM/tPOD
        out = [speedup, CV,eX]
        if full_output:
            out = out + [eV,tPOD,tFOM]
        return out

    def _test_POD_worker(self, l):
        """Picklable wrapper for multiprocessing."""
        #print(l)
        interior = slice(6,-6)
        self.beamPOD = BeamNL_POD(self.beamNL, l=l)    
        self.beamPOD.POD_offline(self.V_samples[interior,:])
        return self.test_POD(self.beamNL, self.beamPOD, self.F_shape,           test_points=self.test_points, P_max=self.P_max,         redoifnoCV=self.redoifnoCV)
    
    def batch_test_POD(self, beamNL, V_samples, l_list,
                       F_shape, test_points=100, P_max=200, redoifnoCV=False, plotting=False):
        self.beamNL = beamNL
        self.V_samples = V_samples
        self.F_shape = F_shape
        self.test_points = test_points
        self.P_max = P_max
        self.redoifnoCV = redoifnoCV
        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            results = list(executor.map(
                self._test_POD_worker,
                l_list))
        self.results = np.array(results)
        return results
    


if __name__ == '__main__':
    """
    interior = slice(6,-6)
    test = Tester(10)
    beamNL = BeamNL(n_nodes=dim.n_nodes, span=dim.span, chord=dim.chord, dm=dim.dm, r_g=dim.r_g, I_per_span=dim.mass_mom_per_span, Cinv=dim.Cinv)
    F_idx = 1
    res = test.jac_tester([(9,58),(15,98)], 1,POD=True)
    """
    samplerFOM = BeamNL(dim.n_nodes, dim.span, dim.chord, dim.dm, dim.r_g, dim.mass_mom_per_span, dim.Cinv)
    Fytip_gl, Fztip_gl, Fy_gl, Fz_gl = samplerFOM.force_templates()
    no_threads = 10

    pars = [[samplerFOM,20,500,False,True,f'{i+1}',1] for i in range(no_threads)]
    
    do_sample = True
    if do_sample:
        with ProcessPoolExecutor(max_workers=no_threads) as executor:
            results = list(executor.map(sample_wrapper, pars))
    F_samples = stitcher('F_samples', no_threads)    
    NL_samples = stitcher('NL_samples', no_threads)    
    V_samples = stitcher('V_samples', no_threads)
    

        
