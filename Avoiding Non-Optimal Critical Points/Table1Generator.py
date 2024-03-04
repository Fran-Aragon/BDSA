# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:38:30 2023


Authors: Francisco J. Aragón-Artacho, Pedro Pérez-Aros, David Torregrosa-Belén

Code associated with the paper:

F.J. Aragón-Artacho, P. Pérez-Aros, D. Torregrosa-Belén: 
The Boosted Double-proximal Subgradient Algorithm for nonconvex optimization.
(https://arxiv.org/abs/2306.17144)

#####################################################
"Section: Avoiding Non-Optimal Critical Points"

For generating  the data in table 1

Change n and p in lines 94-95 to get the different rows

#####################################################

"""

from numpy.linalg import norm
from numpy.random import random, seed
import numpy as np
#import pylab as plt



seed(4)


cte = 1  #constant multiplying ||x||^2
prec=1e-6 # precision stopping criterion
prec0=1e-6 # precision for objective function value 

"Auxiliary functions"

def varphi(x):
    sumat = 0
    for ii in range(1,p+1):
        sumat += np.sum(abs(x-ii)) + np.sum(abs(x+ii))
    return cte*norm(x)**2 -np.sum(abs(x))-sumat -np.sum(abs(x-p-1))

def varphiplot(x,y):
    sumat = 0
    for ii in range(1,p+1):
        sumat += abs(x-ii) + abs(x+ii) + abs(y-ii) + abs(y+ii)
    return cte*( x**2+y**2)-abs(x)-abs(y)-sumat - abs(x-p-1)- abs(y-p-1)

def minsign(x):
    s = np.ones(len(x))
    s[x>0] = -1
    return s


def subh(x): #sub. of -h
    sumat = 0
    for ii in range(1,p+1):
        sumat += minsign(x-ii) + minsign(x+ii)
    return minsign(x) + sumat +minsign(x-p-1)

def subh0(x):  #sub  of h (taking 0 )
    sumat = 0
    for ii in range(1,p+1):
        sumat += np.sign(x-ii) + np.sign(x+ii)
    return np.sign(x) + sumat +np.sign(x-p-1)
        

    
def Phi(x,y):
    if (-1>y).any() or (y > 1).any():
        return np.inf
    else:
        sumat = 0
        for ii in range(1,p+1):
            sumat += tauvect[2*ii-1]*nones.T@y[2*ii-1,:] +  tauvect[2*ii]*nones.T@y[2*ii,:]
        return cte*norm(x)**2   +  sumat + (p+1)*nones.T@y[2*p+1,:] - x.T@np.sum(y,axis=0)

def proxl1(x,tau,mu): #calcula  prox_mu de la conjugada de |x-tau|_l1
    return np.clip(x-mu*tau,-1,1)
    

def prox_minus_l1(x,gamma):
    return  x + gamma*np.sign(x)


def subh_minus_l1(x): #sub. of -h
    sumat = 0
    for ii in range(1,p+1):
        sumat += minsign(x-ii) + minsign(x+ii)
    return sumat +minsign(x-p-1)


"Experiment"
################ BEGINNING EXPERIMENT ##############################

# Change n and p (q) for obtaining the different rows 
# in the table
n = 20 #dimensión espacio
p = 3



rep  = 10000 #repeticiones
gamma = 1
mu = 1
prec = 1e-6
precO = 1e-9
tauvect = np.zeros(2*p+2)
ltauvect = len(tauvect)

for ii in range(1,p+1):
    tauvect[2*ii-1] = ii
    tauvect[2*ii] = -ii
tauvect[2*p+1] = p+1

nones = np.ones(n)
sol = -(p+1)*np.ones(n)  #Cambiar si se cambia p
phisol = varphi(sol)



exitosDP = 0
exitosB = 0
exitosDCA = 0
exitosBDCA = 0
exitosPrimal = 0
exitosBPrimal = 0
exitosiDCA = 0
exitosDSA = 0 
exitosBDSA = 0

BDPgana = 0
DPempate = 0
DPgana = 0

DCAgana = 0
DCAempate = 0
BDCAgana = 0

for rr in range(rep):
    x0 = 2*(p+2)*random(n)-(p+2)
    y0 = 2*random(n)-1
    
    "Double Proximal Gradient Algorithm"
    
    xk = x0.copy() #initial point for the double-Prox
    yk = np.tile(y0,(ltauvect,1))
    
  
    xkold = xk.copy()
    ykold = yk.copy()
    temp1 = xk + gamma*np.sum(yk,0) #argument proxg
    xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
    for ii in range(ltauvect):
        yki = yk[ii,:]
        taui = tauvect[ii]
        tempi = yki + mu*xkn  #argument prxh*
        ykni = proxl1(tempi,taui,mu)
        yk[ii,:] = ykni
    #update
    xk = xkn.copy()
    while norm(xkold -xk) +  np.sum(norm(ykold-yk,axis=1)) > n*prec:
        xkold = xk.copy()
        ykold = yk.copy()
        temp1 = xk + gamma*np.sum(yk,0) #argument proxg
        xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
        for ii in range(ltauvect):
            yki = yk[ii,:]
            taui = tauvect[ii]
            tempi = yki + mu*xkn  #argument prxh*
            ykni = proxl1(tempi,taui,mu)
            yk[ii,:] = ykni
        #update
        xk = xkn.copy()
        
    
    if norm(varphi(xk)-phisol) < precO:
        exitosDP +=1 
    
    DPphixk = varphi(xk)
    
    
    " Boosted Double Prox"

    alph = 0.5
    barlam0= 2
    barlamk = 2
    N = 2
    
    xk = x0.copy() #initial point for the double-Prox
    yk = np.tile(y0,(ltauvect,1))
    
    
    xkold = xk.copy()
    ykold = yk.copy()
    temp1 = xk + gamma*np.sum(yk,0) #argument proxg
    xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
    ykn = yk.copy()
    for ii in range(ltauvect):
        yki = yk[ii,:]
        taui = tauvect[ii]
        tempi = yki + mu*xkn  #argument prxh*
        ykni = proxl1(tempi,taui,mu)
        ykn[ii,:] = ykni
    dxk = xkn-xk
    dyk = ykn-yk
    if (dxk==0).all() and (dyk==0).all():
        xk = xkn.copy()
        yk = ykn.copy()
    else:
        r = 0 
        lamk = barlamk
        while Phi(xkn+lamk*dxk,ykn+lamk*dyk) >= Phi(xkn,ykn) - 0.1*lamk**2*(norm(dxk)**2 +np.sum(norm(dyk,axis=1)**2)):
            r += 1
            if r < N:
                lamk = (alph**r)*barlamk
            else:
                lamk = 0
                break
        if r == 0:
            barlamk= 2*barlamk
        else: barlamk = max(barlam0,(alph**r)*barlamk)
        xk = xkn + lamk*dxk
        yk = ykn + lamk*dyk 
    while norm(xkold-xk) + np.sum(norm(ykold-yk,axis=1)) > n*prec:
        xkold = xk.copy()
        ykold = yk.copy()
        temp1 = xk + gamma*np.sum(yk,0) #argument proxg
        xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
        dxk=xkn-xk
        ykn = yk.copy()
        for ii in range(ltauvect):
            yki = yk[ii,:]
            taui = tauvect[ii]
            tempi = yki + mu*xkn  #argument prxh*
            ykni = proxl1(tempi,taui,mu)
            ykn[ii,:] = ykni
        dyk = ykn-yk
        if (dxk==0).all() and (dyk==0).all():
            xk = xkn.copy()
            yk = ykn.copy()
        else:
            r = 0 
            lamk = barlamk
            while Phi(xkn+lamk*dxk,ykn+lamk*dyk) >= Phi(xkn,ykn) - 0.1*lamk**2*(norm(dxk)**2 +np.sum(norm(dyk,axis=1)**2)):
                r += 1
                if r < N:
                    lamk = (alph**r)*barlamk
                else:
                    lamk = 0
                    break
            if r == 0:
                barlamk= 2*barlamk
            else: barlamk = max(barlam0,(alph**r)*barlamk)
            xk = xkn + lamk*dxk
            yk = ykn + lamk*dyk
    
    if norm(varphi(xk)-phisol) < precO:
        exitosB +=1 
            
    BDPphixk = varphi(xk)
  
    if BDPphixk - DPphixk < -prec0:
        BDPgana += 1
    elif abs(BDPphixk - DPphixk)<=prec0:
        DPempate +=1
    else:
        DPgana += 1  


    " DSA (no linesearch) "
    
    xk = x0.copy()
    gam_min_l1 = 0.49
    
    xkold  =  xk.copy()
    temp1 = xk - gam_min_l1*(2*xk + subh_minus_l1(xk)) 
    xkn = prox_minus_l1(temp1, gam_min_l1)
    xk = xkn.copy()
            
    while norm(xkold-xk)/n > prec:
        xkold = xk.copy() 
        temp1 = xk- gam_min_l1*(2*xk + subh_minus_l1(xk)) 
        xkn = prox_minus_l1(temp1, gam_min_l1)
        xk = xkn.copy() 
        
    if norm(varphi(xk)-phisol) < precO:
        exitosDSA += 1    
    
    
    " BDSA (with linesearch)"
    
    xk = x0.copy()
    gam_min_l1 = 0.49
    
    alph = 0.5
    barlam0 = 2
    barlamk = barlam0
    N = 2
    
    xkold  =  xk.copy()
    temp1 = xk - gam_min_l1*(2*xk + subh_minus_l1(xk)) 
    hatxk = prox_minus_l1(temp1, gam_min_l1)
    dxk=hatxk-xk
    r = 0
    lamk = barlamk
    if (dxk==0).all():
        xkn = hatxk.copy()
    else:
        while varphi(hatxk+lamk*dxk) > varphi(hatxk) -0.1*lamk**2*norm(dxk)**2:
            r += 1
            if r < N:
                lamk =  (alph**r)*lamk
            else:
                lamk = 0
                break 
        if r == 0:
                barlamk = 2*barlamk
        else: barlamk = max((alph**r)*lamk,barlam0)
        xkn = hatxk +lamk*dxk                                 
    xk = xkn.copy()
    
    while norm(xkold-xk)/n > prec:
        xkold = xk.copy() 
        temp1 = xk- gam_min_l1*(2*xk + subh_minus_l1(xk)) 
        hatxk = prox_minus_l1(temp1, gam_min_l1)
        # Boosted step
        dxk=hatxk-xk
        r = 0
        lamk = barlamk
        if (dxk==0).all():
            xkn = hatxk.copy()
        else:
            while varphi(hatxk+lamk*dxk) > varphi(hatxk) -0.1*lamk**2*norm(dxk)**2:
                r += 1
                if r < N:
                    lamk =  (alph**r)*lamk
                else:
                    lamk = 0
                    break 
            if r == 0:
                    barlamk = 2*barlamk
            else: barlamk = max((alph**r)*lamk,barlam0)
            xkn = hatxk +lamk*dxk                                 
        xk = xkn.copy()
        
    if norm(varphi(xk)-phisol) < precO:
        exitosBDSA += 1
       
 
        
    
    "Proximal DC algorithm"
         
        
       
    xk = x0.copy()
      
    xkold = xk.copy()
    temp1 = xk - gamma*subh(xk) #argument proxg
    xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
    xk = xkn.copy()
    while norm(xkold-xk)/n > prec:
        xkold = xk.copy()
        temp1 = xk - gamma*subh(xk) #argument proxg
        xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
        xk = xkn.copy()
        
    if norm(varphi(xk)-phisol) < precO:
        exitosPrimal += 1
        
    "Boosted Proximal DC algorithm"
    
    
    alph = 0.5
    barlam0 = 2
    barlamk = barlam0
    N = 2
       
    xk = x0.copy()
    
    xkold = xk.copy()
    temp1 = xk - gamma*subh(xk)#argument proxg
    hatxk =  temp1/(1+2*cte*gamma)   #evaluation prog x
    dxk=hatxk-xk
    r = 0
    lamk = barlamk
    if (dxk==0).all():
        xkn = hatxk.copy()
    else:
        while varphi(hatxk+lamk*dxk) > varphi(hatxk) -0.1*lamk**2*norm(dxk)**2:
            r += 1
            if r < N:
                lamk =  (alph**r)*lamk
            else:
                lamk = 0
                break 
        if r == 0:
                barlamk = 2*barlamk
        else: barlamk = max((alph**r)*lamk,barlam0)
        xkn = hatxk +lamk*dxk                                 
    xk = xkn.copy()
    
    while norm(xkold-xk)/n > prec:
        xkold = xk.copy()
        temp1 = xk - gamma*subh(xk)#argument proxg
        hatxk =  temp1/(1+2*cte*gamma)   #evaluation prog x
        dxk=hatxk-xk
        r = 0
        lamk = barlamk
        if (dxk==0).all():
            xkn = hatxk.copy()
        else:
            while varphi(hatxk+lamk*dxk) > varphi(hatxk) -0.1*lamk**2*norm(dxk)**2:
                r += 1
                if r < N:
                    lamk =  (alph**r)*lamk
                else:
                    lamk = 0
                    break 
            if r == 0:
                    barlamk = 2*barlamk
            else: barlamk = max((alph**r)*lamk,barlam0)
            xkn = hatxk +lamk*dxk                             
        xk = xkn.copy()
        
    if norm(varphi(xk)-phisol) < precO:
        exitosBPrimal += 1
        
        
    " Intertial DCA "
    
    # Decomposition for iDCA:
    # varphi(x) = f1(x) - f2(x)
    # f1(x) = ||x||^2 + rho/2 ||x||^2
    # f2(x) = 1/2||x||^2 + ||x||_1 + sum_{j=1}^q ( ... ) + ||x-(q+1)e||1
    
    xk = x0.copy()
    rho = 1
    igam = .999*rho/2
    
    xkold = xk.copy()
    xprev = xkold.copy()
    temp1 = xk - subh(xk) + igam/rho*(xk-xprev) #argument prox
    xkn = rho*temp1/(rho+2)  # evaluation proc
    xk = xkn.copy()
    while  norm(xkold-xk)/n > prec:
        xprev = xkold.copy()
        xkold = xk.copy()
        temp1 = xk - subh(xk) + igam/rho*(xk-xprev) #argument prox
        xkn = rho*temp1/(rho+2)  # evaluation proc
        xk = xkn.copy()
        
    if norm(varphi(xk)-phisol) < prec0:
        exitosiDCA += 1
        
    
    if rr%50==0:
        print('Repeticiones realizadas:', rr)
        
print("Número de veces que cada método alcanza el mínimo")
print("Exitos Double Prox", exitosDP)
print("Exitos Boosted Double Prox", exitosB)
print("Exitos DCA", exitosDCA)
print("Exitos BDCA", exitosBDCA)
print("Exitos Primal", exitosPrimal)
print("Exitos Primal Boosted", exitosBPrimal)
print("Exitos iDCA", exitosiDCA)
print("Exitos DSA", exitosDSA)
print("Exitos BDSA", exitosBDSA)
print("Comparativa Double-Prox")
print("# veces Boosted DP alcanza mejor valor que DP:", BDPgana)
print("# número veces alcanzan mismo valor:", DPempate)
print("# veces DP alcanza mejor valor que el boosted:", DPgana)
print("ComparativaDCA")
print("# veces BDCA alcanza mejor valor que DCA:", BDCAgana)
print("# número veces alcanzan mismo valor:", DCAempate)
print("# veces DCA alcanza mejor valor que el boosted:", DCAgana)




