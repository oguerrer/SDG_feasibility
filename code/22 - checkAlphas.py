import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd
from joblib import Parallel, delayed
import scipy.optimize as opt
import seaborn as sns


home =  os.getcwd()[:-4]

os.chdir(home+'/code/')
from model_linear import *

path = '/Users/tequilamambo/Dropbox/Apps/ShareLaTeX/accelerators_ppi/figs/'


df = pd.read_csv(home+"data/final_sample.csv")

current_palette = sns.color_palette('muted', 7)
colors = dict(zip(sorted(df.group.unique()), current_palette.as_hex()))


all_gaps = {}



colYears = [col for col in df.columns if col.isnumeric()][16::]
num_years = len(colYears)
scalar = 100
min_value = 1e-2


countries = df.countryCode.unique()
for country in countries:
    
    dft = df[df.countryCode == country]
    A = np.loadtxt(home+'/data/nets_5/A_'+country+'.csv', dtype=float, delimiter=",")  
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
    
    subfolder = 'data/est_5/'
    adf = pd.read_csv(home+subfolder+country+".csv")
    alphas = adf.alphas.values
    
    all_gaps[country] = {21:0, 10:0, 5:alphas}
    
    

colYears = [col for col in df.columns if col.isnumeric()][11::]
num_years = len(colYears)
scalar = 100
min_value = 1e-2


countries = df.countryCode.unique()
for country in countries:

    dft = df[df.countryCode == country]
    A = np.loadtxt(home+'/data/nets_10/A_'+country+'.csv', dtype=float, delimiter=",")  
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
    
    subfolder = 'data/est_10/'
    adf = pd.read_csv(home+subfolder+country+".csv")
    alphas = adf.alphas.values
        
    all_gaps[country][10] = alphas
    
    
    

colYears = [col for col in df.columns if col.isnumeric()][0::]
num_years = len(colYears)
scalar = 100
min_value = 1e-2


countries = df.countryCode.unique()
for country in countries:

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
    
    subfolder = 'data/est/'
    adf = pd.read_csv(home+subfolder+country+".csv")
    alphas = adf.alphas.values
        
    all_gaps[country][21] = alphas
    
    
    


regs = dict(zip(df.countryCode, df.group))




plt.figure(figsize=(6,3.5))
plt.plot(-10,-10, 'ok', mfc='none', markersize=5, label='21 years')
plt.plot(-10,-10, '.k',  label='10 years')
plt.plot(-10,-10, '_k', label='5 years')
i = 0
for country, gaps in all_gaps.items():
    
    plt.plot( i, all_gaps[country][21].mean(), 'o', mec=colors[regs[country]], mfc='none', markersize=5)
    plt.plot( i, all_gaps[country][10].mean(), '.', color=colors[regs[country]] )
    plt.plot( i, all_gaps[country][5].mean(), '_', color=colors[regs[country]] )
    i+=1

plt.xlim(-2,len(all_gaps)+1)
plt.ylim(0, 1)
plt.xticks([])
plt.yticks([])
plt.xlabel('country', fontsize=14)
plt.ylabel('structural factors', fontsize=14)
plt.legend(fontsize=12, ncol=3)
plt.tight_layout()
plt.savefig(path+'alphas_raw.pdf')
plt.show()








plt.figure(figsize=(6,3.5))
plt.plot(-10,-10, 'ok', mfc='none', markersize=5, label='21 years')
plt.plot(-10,-10, '.k',  label='10 years')
plt.plot(-10,-10, '_k', label='5 years')
i = 0
for country, gaps in all_gaps.items():
    
    plt.plot( i, all_gaps[country][21].mean()/21, 'o', mec=colors[regs[country]], mfc='none', markersize=5)
    plt.plot( i, all_gaps[country][10].mean()/10, '.', color=colors[regs[country]] )
    plt.plot( i, all_gaps[country][5].mean()/5, '_', color=colors[regs[country]] )
    i+=1

plt.xlim(-2,len(all_gaps)+1)
plt.ylim(0, .1)
plt.xticks([])
plt.yticks([])
plt.xlabel('country', fontsize=14)
plt.ylabel('normalized alpha', fontsize=14)
plt.legend(fontsize=12, ncol=3)
plt.tight_layout()
plt.savefig(path+'alphas_norm.pdf')
plt.show()
































































    
    