import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd
from joblib import Parallel, delayed
import scipy.optimize as opt


home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
from model_linear import *




def run_4_scalar(succs, alphas_t, max_steps, success):
    beta_t = succs / (B/(R.sum()*max_steps))
    sols = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)(beta_t, alphas_t) for itera in range(sample_size))
    gammas = []
    for sol in sols:
        gammas += sol[1]
    rate = np.mean(gammas)
    return abs(rate-success)



def run_ppi_parallel(beta_t, alphas_t):
    betas_t = np.ones(N)*beta_t
    outputs = run_ppi(T, A=A, alpha=alphas_t, phi=phi, tau=tau, R=R, 
            B=B_n, beta=beta_t, betas=betas_t, get_gammas=True, max_steps=max_steps)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI[:,-1], gammas)




def run_single(T, A, alphas, phi, tau, R, B, beta, betas, max_steps):
    outputs = run_ppi(T, A=A, alpha=alphas, phi=phi, tau=tau, R=R, 
            B=B, beta=beta, betas=betas, max_steps=max_steps)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H = outputs
    return tsI[:,-1]



def run_many_for_node(alpha, node, est_alphas, sampleSize, T, A, R, phi, tau,
         B, bs, betas, beta, max_steps):
    alphas = est_alphas.copy()
    alphas[node] = alpha
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_single)(T, A, alphas, phi, tau, R, B, beta, betas, max_steps) for itera in range(sample_size))
    errors = np.abs(np.mean(final_values, axis=0)[node] - T[node])
    return errors




def estimate_linear(T, A, R, phi, tau, 
             B=None, bs=None, betas=None, beta=None,
             parallel_processes=4, sample_size=1000, alphas=None,
             nodes0=None, max_steps=None):
    
    est_alphas = alphas.copy()
    if nodes0 is None:
        nodes = np.arange(len(R))
    else:
        nodes = nodes0.copy()
    np.random.shuffle(nodes)
    for node in nodes:
        sol = opt.minimize_scalar(run_many_for_node, args=(node, est_alphas, sample_size, T, A, R, phi, tau,
                                                  B, bs, betas, beta, max_steps), bounds=[0, 10.], method='Bounded')
        est_alphas[node] = sol.x
    
    final_alphas = est_alphas.copy()
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_single)(T, A, final_alphas, phi, tau, R, B, beta, betas, max_steps) for itera in range(sample_size))
    error = np.abs(np.mean(final_values, axis=0) - T).mean()
    
    return final_alphas, error




subfolder = 'data/est/'

 
# Dataset
df = pd.read_csv(home+"data/final_sample.csv")
colYears = [col for col in df.columns if col.isnumeric()]
num_years = len(colYears)
scalar = 100
min_value = 1e-2


tol_error_beta = .0075
tol_error_alpha = .25

sample_size = 100
parallel_processes = 20
max_steps = 50

countries = df.countryCode.unique()
np.random.shuffle(countries)
# countries = ['GNB']
for country in countries:

    done = [name for name in os.listdir(home+subfolder) if '.csv' in name]
    
    if country+'.csv' in done:
        continue
    else:
        file = open(home+subfolder+country+'.csv', 'w')
        file.close()
    
    
    dft = df[df.countryCode == country]
    A = np.loadtxt(home+'/data/nets/A_'+country+'.csv', dtype=float, delimiter=",")  
    phi = dft.gov_cc.mean()
    tau = dft.gov_rl.mean()
    series = dft[colYears].values
    N = len(dft)
    
    # Build variables (gaps)
    R = (dft.instrumental.values == 1).astype(int)
    T = series[:,-1] - series[:,0]
    T *= scalar
    T[T<min_value] = (np.max(series[:,1::], axis=1) - series[:,0])[T<min_value]*scalar
    T[T<min_value] = min_value
    B = dft.budget.values[0]*num_years
    
    
    # Global expenditure returns (homogeneous because we don't know expenditure amond indicators)
    sc = series[:, 1::]-series[:, 0:-1] # get changes in indicators
    scr = sc # isolate instrumentals
    success_emp = np.sum(scr>0)/(scr.shape[0]*scr.shape[1]) # compute rate of success pooling data

    # Initial factors
    est_alphas = np.ones(N)*.25
    
    
    B_n = B/max_steps
    
    
    error_beta = 10
    error_alpha = 10
    counter = 0
    
    best_error_beta = 10
    best_error_alpha = 10
    
    best_alphas = est_alphas.copy()
    best_beta = np.ones(N)
    


    while (error_alpha > tol_error_alpha or error_beta > tol_error_beta) and (counter < 10):
        
        print(country, 'finding alpha and beta...',)
        
        counter += 1
                        
        # second optimizaion of beta
        sol = opt.minimize_scalar(run_4_scalar, args=(est_alphas, max_steps, success_emp), bounds=[0, 3], method='Bounded')
        best_succs = sol.x
        est_beta = best_succs / (B/(R.sum()*max_steps))
        error_beta = sol.fun
        
        est_alphas, error_alpha = estimate_linear(T, A=A, R=R, phi=phi, tau=tau, alphas=est_alphas,
                                          B=B_n, beta=est_beta, betas=np.ones(N)*est_beta,
                                          sample_size=sample_size, 
                                          parallel_processes=parallel_processes, max_steps=max_steps)
        

        
        # evaluate errors of final estimates
        evals = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)(est_beta, est_alphas) for itera in range(sample_size*10))
        eval_indis, eval_gammas = [], []
        for evalu in evals:
            eval_indis.append(evalu[0])
            eval_gammas += evalu[1]
        error_alpha = np.abs(np.mean(eval_indis, axis=0)-T).mean()
        error_beta = abs(np.mean(eval_gammas) - success_emp)
        
        if error_alpha < best_error_alpha:
            best_alphas = est_alphas
            best_beta = est_beta
            best_error_beta = error_beta
            best_error_alpha = error_alpha
        
        print('beta error:', error_beta, )
        print('alpha error:', error_alpha, )
        print('counter:', counter)
        print()
    
    
    dfc = pd.DataFrame([[a, best_beta, max_steps, num_years, best_error_alpha, best_error_beta, scalar, min_value] for a in best_alphas], 
                       columns=['alphas', 'beta', 'steps', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value'])
    dfc.to_csv(home+subfolder+country+'.csv', index=False)
    









