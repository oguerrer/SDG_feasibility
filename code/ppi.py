# -*- coding: utf-8 -*-
"""Policy Priority Inference for Sustainable Development

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Pyhton 3.7
Acknowledgments: This product was developed through the sponsorship of the
    United Nations Development Programme (bureau for Latin America) 
    and with the support of the National Laboratory for Public Policies (Mexico City), 
    the Centro de Investigación y Docencia Económica (CIDE, Mexico City), 
    and The Alan Turing Institute (London).


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


def run_ppi(I0, G=None, A=None, alpha=.1, cc=.5, rl=.5, R=None, PF=None, B=None, 
            bs=None, betas=None, beta=None, get_gammas=False, force_gammas=False, 
            max_steps=50, max_theo=None, scalar=1.):
    """Function to run one simulation of the Policy Priority Inference model.

    Parameters
    ----------
        I0: numpy array 
            Initial values of the development indicators.
        G: numpy array 
            Development goals.
        A:  2D numpy array
            The adjacency matrix of the spillover network of development 
            indicators. If not given, the model assumes a zero-matrix, so there 
            are no spillovers.
        
    Returns
    -------
        tsI: 2D numpy array
            Matrix with the time series of the simulated indicators. Each column 
            corresponds to a simulation step.
        tsC: 2D numpy array
            Matrix with the time series of the simulated contributions. Each column 
            corresponds to a simulation step.
    """
    
    N = len(I0) # number of indicators
    
    ## Check data integrity
    if G is not None:
        assert np.sum(G<=0) == 0, 'Goals must be positive numbers'
        assert np.sum(G<=I0) == 0, 'Goals must be larger than innitial values'
        assert np.sum(np.isnan(G)) == 0, 'Goals must be valid numbers'
    assert np.sum(np.isnan(I0)) == 0, 'Initial values must be valid numbers'
    assert type(max_steps) is int, 'The number of iterations must be an integer value'
    if max_theo is not None:
        assert len(max_theo) == N, 'The number of maximum theoretical values needs to be the same as indicators.'
    
    
    
    if type(cc) is int:
        assert cc < N, 'Index of control of corruption is out of range'
    if type(rl) is int:
        assert rl < N, 'Index of rule of law is out of range'
    
    # transform indicators of instrumental variables into boolean types
    if R is None:
        R = np.ones(N).astype(bool)
    else:
        R[R!=1] = 0
        R = R.astype(bool)
        assert np.sum(R) > 0, 'At least one instrumental indicator is needed'
        
    # if no network is provided, create a zero-matrix
    if A is None:
        A = np.zeros((N,N))
    else:
        assert np.sum(np.isnan(A)) == 0, 'The spillover network contains invalid values'
        A = A.copy()
        np.fill_diagonal(A, 0)
    
    # If no budget provided, set to 1
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
    
    F = np.random.rand(n) # vector of benefits
    Ft = np.random.rand(n) # vectors of lagged benefits
    I = I0.copy() # vector of indicators
    It = np.random.rand(N)*I # vector of lagged indicators
    X = np.random.rand(n)-.5 # vector of actions
    Xt = np.random.rand(n)-.5 # vector of lagged actions
    H = np.ones(n) # vector of historical inefficiencies
    HC = np.ones(n)
    signt = np.sign(np.random.rand(n)-.5) # vector of previous signs for directed learning
    changeFt = np.random.rand(n)-.5 # vector of changes in benefits
    P = np.random.rand(n)
    if G is not None:
        P = G[R] - I0[R]
    P /= P.sum()
    P *= B # vector of allocations (initially random)
    P0 = P.copy()
    C = np.random.rand(n)*P # vector of contributions

    step = 1 # iteration counter (useful to track when indicator reach their goals)
    ticks = np.ones(N)*np.nan # simulation period in which each indicator reaches its target
    
    if bs is None:
        bs = np.ones(n)
    
    if betas is None:
        betas = np.ones(N)
    betas = betas.copy()
    betas[~R] = 0
    
    if beta is None:
        beta = 1.
    
    if PF is not None:
        PF = PF.copy()
        P = B*PF/PF.sum()
    
    all_gammas = []
    gammas = np.ones(N)
    
    for step in range(max_steps): # iterate until the flag indicates otherwise
        
        step += 1 # increase counter (used to indicate period of convergence, so starting value is 2)
        tsI.append(I.copy()) # store this period's indicators
        tsP.append(P.copy()) # store this period's allocations

        deltaIAbs = I-It # change of all indicators
        deltaIIns = deltaIAbs[R] # change of instrumental indicators
        deltaBin = (I>It).astype(int)
        
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
        assert np.sum(P < C)==0, 'C larger than P!'
        tsF.append(F.copy()) # store this period's benefits
        tsX.append(X.copy()) # store this period's actions
        
                
        
        ### DETERMINE BENEFITS ###
        
        if type(cc) is int or type(cc) is np.int64:
            trial = (np.random.rand(n) < (I[cc]/scalar) * P/P.max() * (P-C)/P) # monitoring outcomes
        else:
            trial = (np.random.rand(n) < cc * P/P.max() * (P-C)/P)
        theta = trial.astype(float) # indicator function of uncovering inefficiencies
        H[theta==1] += (P[theta==1] - C[theta==1])/P[theta==1]
        HC[theta==1] += 1
        if type(rl) is int or type(rl) is np.int64:
            newF = deltaIIns*C/P + (1-theta*(I[rl]/scalar))*(P-C)/P # compute benefits
        else:
            newF = deltaIIns*C/P + (1-theta*rl)*(P-C)/P
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
            success = np.ones(N).astype(int)
        else:      
            success = (np.random.rand(N) < gammas).astype(int) # determine if there is succesful growrth
        newI = I + alpha * success # compute new indicators
        assert np.sum(newI < 0) == 0, 'indicators cannot be negative!'
        
        # if theoretical maximums are provided, make sure the indicators do not surpass them
        if max_theo is not None:
            with_bound = ~np.isnan(max_theo)
            newI[with_bound][newI[with_bound] > max_theo[with_bound]] = max_theo[with_bound][newI[with_bound] > max_theo[with_bound]]
            
        # if governance parameters are endogenous, make sure they are not larger than 1
        if (type(cc) is int or type(cc) is np.int64) and newI[cc] > scalar:
            newI[cc] = scalar
        
        if (type(rl) is int or type(rl) is np.int64) and newI[rl] > scalar:
            newI[rl] = scalar
            
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
                    
        else:
            q = PF/PF.sum()
            qs_hat = q**bs
            P = B*qs_hat/qs_hat.sum()
            
        # assert np.sum(qs_hat>1)==0, 'Propensities larger than 1!'
        assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
        assert np.sum(P==0) == 0, 'P has zero values!'


        # update convergence ticks
        if G is not None:
            converged = I >= G
            ticks[(converged) & (np.isnan(ticks))] = step


            
    if get_gammas:
        return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsD).T, np.array(tsS).T, ticks, H, all_gammas
        
    else:
        return np.array(tsI).T, np.array(tsC).T, np.array(tsF).T, np.array(tsP).T, np.array(tsD).T, np.array(tsS).T, ticks, H



    



