# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:42:25 2024

Authors: Francisco J. Aragón-Artacho, Pedro Pérez-Aros, David Torregrosa-Belén

Code associated with the paper:

F.J. Aragón-Artacho, P. Pérez-Aros, D. Torregrosa-Belén: 
The Boosted Double-proximal Subgradient Algorithm for nonconvex optimization.
(https://arxiv.org/abs/2306.17144)

#####################################################
"Section: Avoiding Non-Optimal Critical Points"

For generating Table 2
#####################################################

"""

from numpy.linalg import norm
from numpy.random import seed, random
import numpy as np

seed(4)


n = 2 #dimension: change and execute to get the rows in the table

"Auxiliary functions:"

def phi(x):
    g = norm(x,2)**2
    f =   -sum(abs(x)) - sum(np.log(2+np.exp(2*x)) )
    return g + f

def varphi(x):
    return phi(x)
    
vphi = np.vectorize(phi)


def grad_log_e(x):
    return 2*np.exp(2*x)/(2+np.exp(2*x))


def subl1_neg(x):
    p = -np.sign(x)
    p[p==0] = -1 
    return p
        
        
"Codes for algorithms:"

def inertial_proximal_linearized(x0,igam,rho,prec):
    
    output = {}
    
    
    kk= 0
    xkold = x0.copy()
    xprev = xkold.copy()
    xk = x0.copy()
    
    while kk==0 or norm(xkold-xk)/n > prec:
        
        xprev = xkold.copy()
        xkold = xk.copy()
        
        subh = grad_log_e(xk) + np.sign(xk)
        temp1 = xk + subh  + igam/rho*(xk-xprev) 
        xkn = rho*temp1/(rho+2)
        xk = xkn.copy()
        kk+=1
        
    

    output['xk'] = xk
    output['iter'] = kk
    output['f'] = phi(xk)
    return output


def BDSA(x0,gamma,prec,alph=.5,barlam0=2,barlamk=2,N=2):
    output = {}
    
    kk = 0
    xkold = x0.copy()
    xk = x0.copy()
    
    while kk == 0 or norm(xkold-xk)/n>prec:
        xkold = xk.copy()

        subh = -grad_log_e(xk) + subl1_neg(xk)
        temp1 = xk-gamma*subh
        hatxk = temp1/(1+2*gamma)
        dxk = hatxk - xk
        r = 0 
        lamk = barlamk
        if (dxk==0).all():
            xkn = hatxk.copy()
        else: 
            while varphi(hatxk+lamk*dxk) > varphi(hatxk) - 0.1*lamk**2*norm(dxk)**2:
                r+=1 
                if r < N:
                    lamk = (alph**r)*lamk
                else:
                    lamk = 0 
                    break 
            if r==0:
                barlamk = 2*barlamk 
            else: barlamk = max((alph**r)*lamk,barlam0)
            xkn = hatxk + lamk*dxk
        xk = xkn.copy()
        kk += 1 

    
    output['xk'] = xk
    output['iter'] = kk
    output['f'] = phi(xk)
    
    return output

"Main experiments:"


sol = 1.389525545*np.ones(n)
phisol = phi(sol)


rep = 10000 # number of problems
gam_BDSA = 1

rho = 1
igam = 0.999*rho/2

prec = 1e-6

BDSA_global = 0 
iDCA_global = 0

BDSA_wins = 0
iDCA_wins = 0
ties = 0

for ll in range(rep):
    x0 = 6*random(n)-2.5
    
    
    outputBDSA = BDSA(x0,gam_BDSA,prec,alph=.5,barlam0=2,barlamk=2,N=2)

    fBDSA = outputBDSA['f']
    
    outputiDCA = inertial_proximal_linearized(x0,igam,rho,prec)


    fiDCA = outputiDCA['f']
    
    if abs(fBDSA - phisol) < 1e-3:
        BDSA_global += 1
    
    if abs(fiDCA - phisol) < 1e-3:
        iDCA_global += 1
    
    if fBDSA > fiDCA and abs(fBDSA-fiDCA) > 1e-3:
        iDCA_wins += 1
    elif fBDSA < fiDCA and abs(fBDSA-fiDCA) > 1e-3:
        BDSA_wins += 1
    else:
        ties += 1
   
print('Results:')
print('n=',n)
print('BDSA conveged to global:', BDSA_global)
print('iDCA converged to global:',iDCA_global)
print('BDSA lower objective value',BDSA_wins)
print('iDSA lowe objective value',iDCA_wins)
print('Both methods tie:',ties)








