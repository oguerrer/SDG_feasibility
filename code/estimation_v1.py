import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd
from joblib import Parallel, delayed
import scipy.optimize as opt
import time

import cvxpy as cp



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


def run_many_for_all_nodes(alphas, sampleSize, T, A, R, phi, tau,
         B, bs, betas, beta, max_steps):
    
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_single)(T, A,
                            alphas, phi, tau, R, B, beta, betas, max_steps) for itera in range(sample_size))
    
    error = np.abs(np.mean(final_values, axis=0) - T).mean()
    return error




def estimate_linear(T, A, R, phi, tau, 
             B=None, bs=None, betas=None, beta=None,
             parallel_processes=2, sample_size=2, alphas=None,
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



def estimate_linear_all_nodes(T, A, R, phi, tau, 
             B=None, bs=None, betas=None, beta=None,
             parallel_processes=4, sample_size=1000, alphas=None,
             max_steps=None):
    
    est_alphas = alphas.copy()
    
    bounds_all = [(0,10.) for i in range(0,len(est_alphas))]

    sol = opt.minimize(run_many_for_all_nodes, args=(sample_size, T, A, R, phi, tau,
                               B, bs, betas, beta, max_steps),x0 = alphas,
    bounds=bounds_all, method='COBYLA')
    
    est_alphas = sol.x
    
    final_alphas = est_alphas.copy()
    
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)\
    (delayed(run_single)(T, A, final_alphas, phi, tau, R, B, beta, betas, max_steps)\
     for itera in range(sample_size))
    
    error = np.abs(np.mean(final_values, axis=0) - T).mean()
    
    return final_alphas, error

def lambda_approx_func(lambdas,AA,z):
    
    temp = np.dot(lambdas,AA)
    temp = 2*np.multiply(temp,lambdas)
    
    diff = np.linalg.norm(temp-z)
    
    return diff


def approximate_markov_chain(alphas,T,A,B,beta,tau):
    
    # Get number of indicators
    n = len(T)
    
    # Get q values
    q = T / np.amax(T)
    
    # Get average allocation vector
    P = (B/np.sum(q))*q
    
    # Get average benefit vector
    F = alphas / np.sum(alphas)
    
    # Get contribution vector
    supportvec = np.ones(n)*0.5+F/4
    C = np.multiply(P,supportvec)
    
    # Compute lambdas
    lambdas = np.random.uniform(size=n)
    z = np.multiply(beta,C)+np.mean(C)
    
    bounds_all = [(0,1) for i in range(0,len(lambdas))]
    
    # Solve the problem
    lambda_sol = opt.minimize(lambda_approx_func, args=(A,z),x0 = lambdas,
    bounds=bounds_all, method='COBYLA')
    
    # Get optimal lambdas
    lambdas = lambda_sol.x
    
    # Approximate final indicator values
    I = tau * np.dot(alphas,np.diag(lambdas))
    
    return I





def run_many_approximate_mc(alphas,T,A,B,beta,tau,sample_size):
    
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(approximate_markov_chain)\
    (alphas,T,A,B,beta,tau) for itera in range(sample_size))
    
    error = np.abs(np.mean(final_values, axis=0) - T).mean()
    return error






def estimate_approximation_all_nodes(T, A, R, phi, tau, 
             B=None, bs=None, betas=None, beta=None,
             parallel_processes=4, sample_size=1000, alphas=None,
             max_steps=None,min_value=1e-2):
    
    est_alphas = alphas.copy()
    
    bounds_all = [(0,10.) for i in range(0,len(est_alphas))]
    
    # Watch out for max_steps - tau change!!
    sol = opt.minimize(run_many_approximate_mc,
                       args=(T,A,B,beta,max_steps,sample_size),
                       x0 = alphas,
    bounds=bounds_all, method='COBYLA')
    
    est_alphas = sol.x
    
    final_alphas = est_alphas.copy()
    final_alphas[final_alphas<min_value] = min_value
    
    
    final_values = Parallel(n_jobs=parallel_processes, verbose=0)\
    (delayed(run_single)(T, A, final_alphas, phi, tau, R, B, beta, betas, max_steps)\
     for itera in range(sample_size))
    
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

sample_size = 2
parallel_processes = 2
max_steps = 20

methods = ['original', 'multivariate', 'approximation']
method = 'multivariate' #'original' vs 'multivariate' vs 'approximation'

countries = df.countryCode.unique()
np.random.shuffle(countries)

countries = ['GNB']

# Initialize dictionary to store the results
method_comparison_results = {}

for method in methods:


    for country in countries:
        
        starttime = time.time()
        
        done = [name for name in os.listdir(home+subfolder) if '.csv' in name]
        
        if country+'.csv' in done:
            continue
        else:
            pass
            #file = open(home+subfolder+country+'.csv', 'w')
            #file.close()
        
        
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
            
            
            if method == 'original':
                est_alphas, error_alpha = estimate_linear(T, A=A, R=R, phi=phi, tau=tau, alphas=est_alphas,
                                                  B=B_n, beta=est_beta, betas=np.ones(N)*est_beta,
                                                  sample_size=sample_size, 
                                                  parallel_processes=parallel_processes, max_steps=max_steps)
            
            elif method == 'multivariate':
                
                est_alphas, error_alpha = estimate_linear_all_nodes(T, A=A, R=R, phi=phi, tau=tau, alphas=est_alphas,
                                                  B=B_n, beta=est_beta, betas=np.ones(N)*est_beta,
                                                  sample_size=sample_size, 
                                                  parallel_processes=parallel_processes, max_steps=max_steps)
            
            
            elif method == 'approximation':
                
                est_alphas, error_alpha = estimate_approximation_all_nodes(T, A=A, R=R,
                                                phi=phi, tau=tau, alphas=est_alphas,
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
        #dfc.to_csv(home+subfolder+country+'.csv', index=False)
        
        
        endtime = time.time()
        time_elapsed = round(endtime-starttime,2)
        print("Time elapsed for one country: {} seconds".format(time_elapsed))
        
    # Save results
    method_comparison_results[method] = {'beta error': error_beta,
                                         'alpha error': error_alpha,
                                         'time_elapsed':time_elapsed}
    
    



# Convert and Export method comparison results
method_comparison_results = pd.DataFrame(method_comparison_results)


method_comparison_results.to_csv('../method_comparison_results/method_comparison_results.csv')
    
    
    
# Experiment with Markov-chain approach


n = 10
lambdas = np.random.uniform(size=n)

bounds_all = [(0,1) for i in range(0,len(lambdas))]

AA = np.random.rand(n,n)
z = np.random.uniform(size=n)



# Solve the problem
sol = opt.minimize(lambda_approx_func, args=(AA,z),x0 = lambdas,
    bounds=bounds_all, method='COBYLA')

# Print optimal lambdas
print(sol.x)

# Get residual value
print(sol.fun)


# TRY new functions
run_many_approximate_mc(best_alphas,T,A,B,best_beta,tau,sample_size)



approximate_markov_chain(best_alphas,T,A,B,best_beta,tau)


#### Try particle method

# General parameters
sampleSize = 2
B=None
bs=None
betas=None
beta=None
parallel_processes=2
max_steps=10

# K iterations
K = 20

# Generate candidates
candidate_alphas = [np.random.uniform(0,10,N) for i in range(K)]

# Define current best
x = candidate_alphas[0]

# Initialize CVX parameters
P = np.random.normal(size = (N,N))
q = np.random.normal(size = N)
r = np.random.normal(size = None)

y = []

# Evaluate function
for alpha in candidate_alphas:

    y.append(run_many_for_all_nodes(alphas, sampleSize, T, A, R, phi, tau,
             B, bs, betas, beta, max_steps))
    
    
# Define objective function
def particle_obj(candidate_alphas,x,y,P,q,r):
    
    summa = 0
    for i in range(0,len(candidate_alphas)):
        diff = (candidate_alphas[i]-x)
        summa += (diff.T@P@diff + \
        q.T@diff + r - y[i])**2
        
    return summa


# Try objective function
particle_obj(candidate_alphas,x,y,P,q,r)






