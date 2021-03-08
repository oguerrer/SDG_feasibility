import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed


home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
import ppi






def run_ppi_parallel(I0, A, alpha, cc, rl, R, B, beta, betas, max_steps, scalar):
    outputs = ppi.run_ppi(I0, A=A, alpha=alpha, cc=cc, rl=rl, R=R, 
            B=B, beta=beta, betas=betas, get_gammas=True, max_steps=max_steps, scalar=scalar)
    tsI, tsC, tsF, tsP, tsD, tsS, times, H, gammas = outputs
    return (tsI[:,-1], gammas)



def fobj2(I0, A, alpha, cc, rl, R, B, beta, betas, max_steps, scalar, sample_size, G, success_emp):
    sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, A, alpha, cc, rl, R, B, beta, betas, max_steps, scalar) for itera in range(sample_size)))
    FIs = []
    gammas = []
    for sol in sols:
        FIs.append(sol[0])
        gammas += sol[1]

    mean_indis = np.mean(FIs, axis=0)
    error_alpha = G - mean_indis
    mean_gamma = np.mean(gammas)
    error_beta = success_emp - mean_gamma

    return error_alpha.tolist()+[error_beta]



subfolder = 'data/parameters/'

 
# Dataset
df = pd.read_csv(home+"data/dataset_final.csv")
colYears = [col for col in df.columns if col.isnumeric()]
num_years = len(colYears)
scalar = 100
min_value = 1e-2

parallel_processes = 5
max_steps = 50
parallel_countries = 10

countries = df.countryCode.unique()
np.random.shuffle(countries)


def calibrate(country):

    dft = df[df.countryCode == country]
    A = np.loadtxt(home+'/data/networks/A_'+country+'.csv', dtype=float, delimiter=",")  
    series = dft[colYears].values
    N = len(dft)
    
    # Build variables (gaps)
    R = (dft.instrumental.values == 1).astype(int)
    G = series[:,-1] - series[:,0]
    G *= scalar
    G[G<min_value] = (np.max(series[:,1::], axis=1) - series[:,0])[G<min_value]*scalar
    G[G<min_value] = min_value
    G += series[:,0]*scalar
    I0 = series[:,0]*scalar
    B = dft.budget_pc.values[0]*num_years
    
    # Global expenditure returns (homogeneous because we don't know expenditure amond indicators)
    sc = series[:, 1::]-series[:, 0:-1] # get changes in indicators
    scr = sc # isolate instrumentals
    success_emp = np.sum(scr>0)/(scr.shape[0]*scr.shape[1]) # compute rate of success pooling data

    # Initial factors
    params = np.ones(N+1)*.5
    
    B_n = B/max_steps
    
    cc = dft.cc.values[0]
    rl = dft.cc.values[0]

    mean_abs_error = 100
    sample_size = 10
    counter = 0
    while mean_abs_error > .05:
        
        counter += 1
        alphas_t = params[0:-1]
        beta_t = params[-1]
        betas_t = np.ones(N)*beta_t
        
        errors = np.array(fobj2(I0, A, alphas_t, cc, rl, R, B_n, beta_t, betas_t, max_steps, scalar, sample_size, G, success_emp))
        normed_errors = errors/np.array((G-I0).tolist()+[success_emp])
        abs_errors = np.abs(errors)
        abs_normed_errrors = np.abs(normed_errors)
        
        mean_abs_error = np.mean(abs_errors)
        
        params[errors<0] *= np.clip(1-abs_normed_errrors[errors<0], .5, 1)
        params[errors>0] *= np.clip(1+abs_normed_errrors[errors>0], 1, 1.5)
        
        sample_size += 100
        
        print( country, mean_abs_error, sample_size, counter )

    
    print('copmuting final estimate...')
    print()
    sample_size = 10000
    alphas_est = params[0:-1]
    beta_est = params[-1]
    betas_est = np.ones(N)*beta_est
    errors_est = np.array(fobj2(I0, A, alphas_est, cc, rl, R, B_n, beta_est, betas_est, max_steps, scalar, sample_size, G, success_emp))
    errors_alpha = errors_est[0:-1]
    error_beta = errors_est[-1]
    
    GoF_alpha = 1 - np.abs(errors_alpha)/(G-I0)
    GoF_beta = 1 - np.abs(error_beta)/success_emp
    
    dfc = pd.DataFrame([[alphas_est[i], beta_est, max_steps, num_years, errors_alpha[i]/scalar, error_beta, scalar, min_value, GoF_alpha[i], GoF_beta] \
                        if i==0 else [alphas_est[i], np.nan, np.nan, np.nan, errors_alpha[i]/scalar, np.nan, np.nan, np.nan, GoF_alpha[i], np.nan] \
                       for i in range(N)], 
                       columns=['alphas', 'beta', 'steps', 'years', 'error_alpha', 'error_beta', 'scalar', 'min_value', 'GoF_alpha', 'GoF_beta'])
    dfc.to_csv(home+subfolder+country+'.csv', index=False)
    


for country in countries:
    done = [name for name in os.listdir(home+subfolder) if '.csv' in name]
    if country+'.csv' in done:
        continue
    else:
        file = open(home+subfolder+country+'.csv', 'w')
        file.close()
        calibrate(country)





