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