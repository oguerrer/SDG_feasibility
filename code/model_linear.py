# -*- coding: utf-8 -*-
"""Policy Priority Inference for Sustainable Development

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Pyhton 3.7
Acknowledgments: This product was developed through the sponsorship of the
    United Nations Development Programme (bureau for Latin America) 
    and with the support of the National Laboratory for Public Policies (Mexico City), 
    the Centro de Investigación y Docencia Económica (CIDE, Mexico City), 
    and The Alan Turing Institute (London).

This file contains all the necessary functions to reproduce the analysis presented
in the methodological and technical reports. The accompanying data can be 
obtained from the public repository: https://github.com/oguerrer/PPI4SD. 
There are two functions in this script:
    
    run_ppi : the main function that simulates the policymaking process and
    generates synthetic development-indicator data.
    get_targets : a support function to transform a collection of series 
    where one or more targets are less or equals to the initial value of the series.

Further information can be found in each function's code.


Example
-------
To run PPI in a Python script, just add the following line:

    tsI, tsC, tsF, tsP, tsD, tsS, ticks, H = run_ppi(I0, T)
    
This will simulate the policymaking process for initial values I0 and targets T.
This example assumes no network of spillovers. All other arguments can be
passed as explained in the function run_ppi.


Rquired external libraries
--------------------------
- Numpy


"""

# import necessary libraries
from __future__ import division, print_function
import numpy as np
import warnings
warnings.simplefilter("ignore")


def run_ppi(T, A=None, alpha=.1, phi=.5, tau=.5, R=None, 
            gov_func=None, P0=None, H0=None, PF=None, pf=1, RD=None, 
            B=None, bs=None, betas=None, beta=None, node=None, 
            time=None, get_gammas=False, force_gammas=False, 
            max_steps=100, converge=False, max_theo=None):
    """Function to run one simulation of the Policy Priority Inference model.

    Parameters
    ----------
        I0: numpy array 
            Initial values of the development indicators.
        T: numpy array 
            Target values for development indicators. These values represent 
            the government's goals or aspirations. For a retrospective analysis, 
            it is usually assumed that the targets correspond to the final values 
            of the series. They should be higher than I0 or the model will not 
            converge.
        A:  2D numpy array
            The adjacency matrix of the spillover network of development 
            indicators. If not given, the model assumes a zero-matrix, so there 
            are no spillovers.
        alpha: float, optional
            A vector of growth factors in (0,1).
        phi: float, optional
            Scalar in [0,1] or numpy array (a vector) with values in [0,1] that 
            represent the quality of the government's monitoring mechanisms.
        tau: float, optional 
            Scalar in [0,1] or numpy array (a vector) with values in [0,1] that 
            represent the quality of the rule of law.
        R: numpy array, optional
            Binary vector indicating which nodes are instrumental (value 1) and 
            which are not (value 0). If not provided, it is assumed that all
            nodes are instrumental (a vector of ones).
        gov_func: python function, optional
            A custom function that that returns the policy priority of the government
        P0: numpy array, optional
            An array with the initial allocation profile.
        H0: numpy array, optional
            The initial vector of historical inefficiencies
        PF: numpy array, optional
            An exogenous vector of policy priorities.
        pf: float, optional
            The probability with which the exogenous priorities are followed
            each period. It must be in [0,1].
        RD: numpy array, optional
            A vector indicating the level of fiscal rigidity of each instrumental
            indicator. RD should be provided together with PF, and each element 
            of RD should be at most as big as the corresponding element in PF.
        B: float, optional
            A factor for the overall size of the total budget.
            It should only be used in counterfactual analysis where the baseline
            estimation has a B=1, so if B=0.9, it means that the overall budget
            for transformative resources shrinked 10% with respect to the
            baseline estimation.
        b: numpy array, optional
            A vector with expenditure factors.
        tolerance: float, optional
            The precision to consider that an indicator has reached its goal.
            Unless you understand very well how PPI works, this should not be
            changed.
        
    Returns
    -------
        tsI: 2D numpy array
            Matrix with the time series of the simulated indicators. Each column 
            corresponds to a simulation step.
        tsC: 2D numpy array
            Matrix with the time series of the simulated contributions. Each column 
            corresponds to a simulation step.
        tsF: 2D numpy array 
            Matrix with the time series of the simulated agents' benefits. Each column 
            corresponds to a simulation step.
        tsP: 2D numpy array 
            Matrix with the time series of the simulated resource allocations. 
            Each column corresponds to a simulation step.
        tsD: 2D numpy array 
            Matrix with the time series of the simulated inefficiencies. Each column 
            corresponds to a simulation step.
        tsS: 2D numpy array 
            Matrix with the time series of the simulated spillovers. Each column 
            corresponds to a simulation step.
        ticks: numpy array
            A vector with the simulation step in which each indicator reached its target.
        H: numpy array
            A vector with historical inefficiencies,
    """
       
    assert np.sum(T<=0) == 0, 'Target values must be positive numbers'
    assert np.sum(np.isnan(T)) == 0, 'Target values must be valid numbers'
    assert type(max_steps) is int, 'The number of iterations must be an integer value'
    if max_theo is not None:
        assert len(max_theo) == len(T), 'The number of maximum theoretical values needs to be the same as indicators.'
    
    
    N = len(T) # number of indicators
    
    # transform indicators of instrumental variables into boolean types
    if R is None:
        R = np.ones(N).astype(bool)
    else:
        R[R!=1] = 0
        R = R.astype(bool)
    Rnum = np.sum(R)
    
    # if no network is provided, create a zero-matrix
    if A is None:
        A = np.zeros((N,N))
    else:
        A = A.copy()
        np.fill_diagonal(A, 0)
    
    if B is None:
        B = 1.
    
    n = np.sum(R) # number of instrumental nodes
    
    tsI = [] # stores time series of indicators
    tsC = [] # stores time series of contributions
    tsF = [] # stores time series of benefits
    tsP = [] # stores time series of allocations
    tsD = [] # stores time series of corruption
    tsX = [] # stores time series of actions
    tsS = [] # stores time series of spillovers
    tsb = [] # stores time series of exogenous budget
    
    qs = np.ones(n) # propensities to allocate resources (initially homogeneous)
    F = np.random.rand(n) # vector of benefits
    Ft = np.random.rand(n) # vectors of lagged benefits
    I = np.zeros(N) # vector of indicators
    It = np.zeros(N) # vector of lagged indicators
    X = np.random.rand(n)-.5 # vector of actions
    Xt = np.random.rand(n)-.5 # vector of lagged actions
    H = np.ones(n) # vector of historical inefficiencies
    HC = np.ones(n)
    signt = np.sign(np.random.rand(n)-.5) # vector of previous signs for directed learning
    changeFt = np.random.rand(n)-.5 # vector of changes in benefits
    b = np.ones(Rnum)
    P = B*T[R]/np.sum(T[R]) # vector of allocations (initially homogeneous)
    P0 = T[R]/T[R].max()
    C = np.random.rand(n)*P # vector of contributions

    step = 1 # iteration counter
    ticks = np.ones(N)*np.nan # simulation period in which each indicator reaches its target
    
    # in case the user provides initial allocation or historical inefficiencies
    if P0 is not None:
        P = P0/P0.sum()
    if H0 is not None:
        H = H0
            
    if bs is None:
        bs = np.ones(Rnum)
    
    if betas is None:
        betas = np.ones(N)
    betas = betas.copy()
    betas[~R] = 0
    
    if beta is None:
        beta = 1.
    
    if PF is not None:
        PF = PF.copy()
        PF[PF==0] = 1e-12
        P = B*PF/PF.sum()
    
    all_gammas = []
    gammas = np.ones(N)
    
    finish = False # a flag to halt the simulation (activates when all indicators reach their targets)
    while not finish: # iterate until the flag indicates otherwise
        
        step += 1 # increase counter (used to indicate period of convergence, so starting value is 2)
        tsI.append(I.copy()) # store this period's indicators
        tsP.append(P.copy()) # store this period's allocations
        tsb.append(b.copy()) # store this period's exogenous allocation

        deltaIAbs = I-It # change of all indicators
        deltaIIns = deltaIAbs[R] # change of instrumental indicators
        deltaBin = (T>It).astype(int)
        
        # relative change of instrumental indicators
        if np.sum(deltaIIns) == 0:
            deltaIIns = np.zeros(len(deltaIIns))
        else:
            deltaIIns = deltaIIns/np.sum(np.abs(deltaIIns))
        

        ### DETERMINE CONTRIBUTIONS ###
        
        changeF = F - Ft # change in benefits
        changeX = X - Xt # change in actions
        sign = np.sign(changeF*changeX) # sign for the direction of the next action
        changeF[changeF==0] = changeFt[changeF==0] # if the benefit did not change, keep the last change
        sign[sign==0] = signt[sign==0] # if the sign is undefined, keep the last one
        Xt = X.copy() # update lagged actions
        X = X + sign*np.abs(changeF) # determine current action
        assert np.sum(np.isnan(X)) == 0, 'X has invalid values!'
        C = P/(1 + np.exp(-X)) # map action into contribution
        assert np.sum(np.isnan(C)) == 0, 'C has invalid values!'
        signt = sign.copy() # update previous signs
        changeFt = changeF.copy() # update previous changes in benefits
        
        tsC.append(C.copy()) # store this period's contributions
        tsD.append((P-C).copy()) # store this period's inefficiencies
        tsF.append(F.copy()) # store this period's benefits
        tsX.append(X.copy()) # store this period's actions
        
                
        
        ### DETERMINE BENEFITS ###
        
        trial = (np.random.rand(n) < phi * P/P.max() * (P-C)/P) # monitoring outcomes
        theta = trial.astype(float) # indicator function of uncovering inefficiencies
        H[theta==1] += (P[theta==1] - C[theta==1])/P[theta==1]
        HC[theta==1] += 1
        newF = deltaIIns*C/P + (1-theta*tau)*(P-C)/P # compute benefits
        Ft = F.copy() # update lagged benefits
        F = newF # update benefits
        assert np.sum(np.isnan(F)) == 0, 'F has invalid values!'
        
        
        ### DETERMINE INDICATORS ###
        
        deltaM = np.array([deltaBin,]*len(deltaBin)).T # reshape deltaIAbs into a matrix
        S = np.sum(deltaM*A, axis=0) # compute spillovers
        assert np.sum(np.isnan(S)) == 0, 'S has invalid values!'
        tsS.append(S) # save spillovers
        cnorm = np.zeros(N) # initialize a zero-vector to store the normalized contributions
        cnorm[R] = C # compute contributions only for instrumental nodes
        gammas = ( beta*C.mean() + betas*cnorm )/( 1 + np.exp(-S)) # compute probability of succesful growth
        assert np.sum(np.isnan(gammas)) == 0, 'gammas has invalid values!'
        
        if force_gammas:
            succsess = np.ones(N).astype(int)
        else:      
            succsess = (np.random.rand(N) < gammas).astype(int) # determine if there is succesful growrth
        newI = I + alpha * succsess # compute new indicators
        if max_theo is not None:
            newI[newI > max_theo] = max_theo[newI > max_theo]
        It = I.copy() # update lagged indicators
        I =  newI.copy() # update indicators
        
        if get_gammas:
            all_gammas += gammas[R].tolist()
        
        
                
        ### DETERMINE ALLOCATIONS ###
        
        if PF is None:
            P0 += np.random.rand(n)*H/HC
            assert np.sum(np.isnan(P0)) == 0, 'P0 has invalid values!'
            assert np.sum(P0==0) == 0, 'P0 has a zero value!'
            q = P0/P0.sum()
            assert np.sum(np.isnan(q)) == 0, 'q has invalid values!'
            qs_hat = q**bs
            P = B*qs_hat/qs_hat.sum()
            assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
            assert np.sum(P==0) == 0, 'P has zero values!'
                    
        else:
            
            P = B*PF/PF.sum()
            assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
            assert np.sum(P==0) == 0, 'P has zero values!'


        # update convergence ticks
        converged = I >= T
        ticks[(converged) & (np.isnan(ticks))] = step

        
        # check if all indicators have converged
        if converge:
            if np.sum(converged)==N:
                finish = True
        else:
            if step>max_steps:
                finish = True
            
            
    if get_gammas:
        return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsD).T, np.array(tsS).T, ticks, H, all_gammas
        
    else:
        return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsD).T, np.array(tsS).T, ticks, H



    




def get_targets(series, tol=1e-2):
    """Transforms a collection of series where one or more targets are less or
    equals to the initial value of the series.

    Parameters
    ----------
        series: numpy 2D array 
            A matrix containing the time series of each indicator. Each row
            corresponds to a series and each column to a period.
        
    Returns
    -------
        I0: numpy array
            The initial values of each series.
        T: numpy array 
            The transformed targets (final values) of each series.
    """
    
    gaps = series[:,-1]-series[:,0]
    I0 = series[:,0]
    if np.sum(gaps<0) > 0:
        T = series[:,-1] + np.abs(np.min(gaps)) + tol
    elif np.min(gaps) < tol:
        T = series[:,-1] + tol - np.min(gaps)
    else:
        T = series[:,-1]
    
    return I0, T





