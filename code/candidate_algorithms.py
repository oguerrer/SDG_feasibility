#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:49:25 2021

@author: MacBook
"""

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


# Define objective function for particle method
def particle_obj(candidate_alphas,x,y,P,q,r):
    
    summa = 0
    for i in range(0,len(candidate_alphas)):
        diff = (candidate_alphas[i]-x)
        summa += (diff.T@P@diff + \
        q.T@diff + r - y[i])**2
        
    return summa

