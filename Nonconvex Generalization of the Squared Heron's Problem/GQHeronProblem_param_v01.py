# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:06:29 2023

@author: david

Generalized quadratic Heron Problem

We test the nonconvex splitting for a generalized Heron problem
consisting in finding a point x in a cc set C0 which minimizes the sum of 
the quadratic distances of the image of x by some quadratic transformation
to some cc sets C1,...,Cr

Start of the experiment line 205
"""

from numpy import array, concatenate, where, argmin, maximum, zeros, tile, repeat, newaxis, append, arange
from numpy.linalg import norm, eig
import math
import numpy.linalg as LA
from numpy.random import random, seed
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import time



seed(4) #4


"Auxiliary functions"


def PC0(x):
    if norm(x) > rC0:
        return rC0*x/norm(x)
    else:
        return x


def PCr(x,c):
    ub = c + np.sqrt(2)/2
    lb = c - np.sqrt(2)/2 
    return np.clip(x,lb,ub)




"Psi(x)"
def Qx(x):
    tempQx = Q@x
    return x.T @ tempQx.T

"Objective function is the following:"

def varphi(x):
    if (x==PC0(x)).all:
        fQx = Qx(x)
        dist = 0
        for ii in range(r):
            cii = cCr[ii,:]
            dist += norm(fQx-PCr(fQx,cii))**2/2
        return dist
    else:
        return np.inf
    

def Phi(x,y):
    if (x==PC0(x)).all:
        for ii in range(r):
            yii = y[ii,:]
            cii = cCr[ii,:]
            if norm(yii-PCr(yii,cii)) > 0:
                return np.inf
        fQx = Qx(x)
        ny2 = norm(y,axis=1)**2
        return r*norm(fQx)**2/2 + sum(ny2)/2  - fQx @ y.sum(0)
    else:
        return np.inf



" DSA (no linesearch)"

def NCsplitting(x,gamma,mu,kappa,Lips_Qx,beta=0,tol=1e-6):
    " x - initial starting point"
    "the dual variables start all as the 0 vector"
    xk = x.copy()
    yk = np.zeros([r,m])
    "first update"
    gradQx = 2*(Q @ xk).T   
    gam  = gamma*1/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
    xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk)) #update en xk
    
    wk = (1-beta)*xkn + beta*xk
    "update yk"
    ykold =yk.copy()
    for ii in range(r):
        tempyii = yk[ii,:]
        cii = cCr[ii,:]
        yk[ii,:] = PCr((tempyii+mu*Qx(wk))/(1+mu),cii)
    "here starts the loop"
    k = 1
    while abs(Phi(xkn,yk)-Phi(xk,ykold)) > tol:  # and k <10:
        xk = xkn.copy()
        gradQx = 2*(Q @ xk).T 
        gam  = gamma/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
        "update  xk"
        xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk))
       
        wk = (1-beta)*xkn + beta*xk
        "update  yk"
        ykold = yk.copy()
        for ii in range(r):
            tempyii = yk[ii,:]
            cii = cCr[ii,:]
            yk[ii,:] = PCr((tempyii+mu*Qx(wk))/(1+mu),cii)
        k += 1
    return xkn, yk, k



" BDSA (with lineasearch):"

def NCsplitting_B(x,gamma,mu,alph,kappa,Lips_Qx,barlam0 = 2,Ny=2,tol=1e-6):
    " x - initial starting point"
    "the dual variables start all as the 0 vector"
    barlamk = barlam0
    xk = x.copy()
    yk = cCr.copy()
    gradQx = 2*(Q @ xk).T # no need to do tranpose since we have to do it again later inside PC0
    gam  = gamma*1/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
     
  
    "first update"

    xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk)) #update en xk
 
    
    ykn =yk.copy()
    for ii in range(r):
        tempyii = yk[ii,:]
        cii = cCr[ii,:]
        ykn[ii,:] = PCr((tempyii+mu*Qx(xkn))/(1+mu),cii)
    "Linear Search"
    dxk=xkn-xk
    dyk=ykn-yk
    exp_alph = 0
    lamk = barlamk
    while Phi(xkn+lamk*dxk,ykn+lamk*dyk) > Phi(xkn,ykn) -0.1*lamk**2*(norm(dxk)**2 + np.sum(norm(dyk,axis=1)**2)) :
        exp_alph += 1
        if exp_alph < Ny:
            lamk =  (alph**exp_alph)*barlamk
        else:
            lamk = 0
            break
    if exp_alph == 0:
            barlamk = 2*barlamk
    else: barlamk= max(barlam0,(alph**exp_alph)*barlamk)
    xkn = xkn  + lamk*dxk
    ykn = ykn + lamk*dyk
    "here starts the loop"
    k = 1
    while abs(Phi(xkn,ykn)-Phi(xk,yk)) > tol:  # and k <10:
        xk = xkn.copy()
        yk = ykn.copy()
        gradQx = 2*(Q @ xk).T 
        gam  = gamma/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
        
        "update xk"
        xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk))

        "update en yk"
        ykn= yk.copy()
        for ii in range(r):
            tempyii = yk[ii,:]
            cii = cCr[ii,:]
            ykn[ii,:] = PCr((tempyii+mu*Qx(xkn))/(1+mu),cii)
        k += 1
        "Linear Search"
        dxk=xkn-xk
        dyk=ykn-yk
        exp_alph = 0
        lamk = barlamk
        while Phi(xkn+lamk*dxk,ykn+lamk*dyk) > Phi(xkn,ykn) -0.1*lamk**2*(norm(dxk)**2 +np.sum(norm(dyk,axis=1)**2)) :
            exp_alph += 1
            if exp_alph < Ny:
                lamk =  (alph**exp_alph)*barlamk
            else:
                lamk = 0
                break
        if exp_alph == 0:
                barlamk = 2*barlamk
        else: barlamk= max(barlam0,(alph**exp_alph)*barlamk)
        xkn = xkn  + lamk*dxk
        ykn = ykn + lamk*dyk

    return xkn, ykn, k




"Start of the experiment"
############ Experiment  ###############

use_saved_data = True #True: for data in the paper, False: new data

repP = 5  #number of problems
repS = 5  #number of starting point for each problem

n = 3#20  # dimension of the primal space
m = 4  # math.floor(1.5*n) #23 # dimension of the dual space
r = 3 # number of soft constraints

betas = [0]
gammas = [0.1,0.3,0.5,0.7,0.9,0.99] 
mus = [0.5,1,5]

Asol = zeros([repP,repS,len(betas),len(mus),len(gammas)])
Aiter = zeros([repP,repS,len(betas),len(mus),len(gammas)])
Atime = zeros([repP,repS,len(betas),len(mus),len(gammas)])

if use_saved_data == False:
    for rPP in range(repP):
        " Generating data "
        
        rC0 = 5 #radius of the ball
        
        # center of the hypercubes
        normcCr = 3*random(r)+7
        cCr = 2*random([r,m])-1
        ncCr = norm(cCr, axis = 1 )
        cCr = normcCr[:, None] * cCr / ncCr[:, None] #estos son los centros de los hipercubos
        
        #Operator Psi:
        Q = zeros([m,n,n])
        singvaluesQ = zeros(m)
    
    
        for mm in range(m):
            tempD = 2*random(n)-1
            tempQ = np.diag(tempD) #dividimos por radio espectral, radio bola y raíz de m
            Q[mm,:,:] =  tempQ
            singvaluesQ[mm] = norm(tempQ,2) #guardamos el radio spectral de cada Q_mm
        
        QM = zeros([m*n,n])         #for writting Q in matrix form
        for mm in range(m):
            QM[mm*n:(mm+1)*n,:] = Q[mm]
        normQ =  norm(QM,2) #
        maxu = 2*normQ*rC0;
        Lips_Qx =   2*normQ #Lipschitz constant of  grad of Psi(x) = Qx(x)
        sumrhoi = np.sum(singvaluesQ**2)
        kappa = 6*r*rC0**2*normQ*np.linalg.norm(np.array([np.linalg.norm(Q[i],ord=2) for i in range(m)]))/2
        
        
        for rSS in range(repS):
            "Generate random starting point"
            x0 = 10*random(n)
            
            for iibeta in range(len(betas)):
                beta = betas[iibeta]
                for iimu in range(len(mus)):
                    mu = mus[iimu]
                    for iigamma in range(len(gammas)):
                        gamma = gammas[iigamma]
                        if beta == 0:
                            starttime = time.time()
                            sol,_,it = NCsplitting(x0,gamma,mu,kappa,Lips_Qx,tol=1e-6)
                            stoptime = time.time()
                            print('done!!')
    
                            
                        Asol[rPP,rSS,iibeta,iimu,iigamma] = varphi(sol)
                        Aiter[rPP,rSS,iibeta,iimu,iigamma] = it
                        Atime[rPP,rSS,iibeta,iimu,iigamma] = stoptime-starttime
                            
                    
    "To  save data"     
    #np.savez('Exp_Param_Splitting_20230626', Asol, Aiter, Atime)     

elif use_saved_data == True:
    # # "Load data" 
    npzfile = np.load('Exp_Param_Splitting_20230626.npz',allow_pickle = True )
    Asol = npzfile['arr_0']
    Aiter = npzfile['arr_1']
    Atime = npzfile['arr_2']
    


averagesol0 = np.sum(Asol,axis=1)/repS
averagesol  = np.sum(averagesol0,axis=0)/repP

averageiter0 = np.sum(Aiter,axis=1)/repS
averageiter = np.sum(averageiter0,axis=0)/repP

averagetime0 = np.sum(Aiter,axis=1)/repS
averagetime = np.sum(averagetime0,axis=0)/repP













        
    
            
