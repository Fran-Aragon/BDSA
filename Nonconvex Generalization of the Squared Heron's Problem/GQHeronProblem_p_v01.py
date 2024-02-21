# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:06:29 2023

@author: david

Generalized quadratic Heron Problem

We test the nonconvex splitting for a generalized Heron problem
consisting in finding a point x in a cc set C0 which minimizes the sum of 
the quadratic distances of the image of x by some quadratic transformation
to some cc sets C1,...,Cr

Experiment starts line 205
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
#import pylab as plt




seed(4) #4



def PC0(x):
    if norm(x) > rC0:
        return rC0*x/norm(x)
    else:
        return x

def PCr(x,c):
    ub = c + np.sqrt(2)/2
    lb = c - np.sqrt(2)/2 
    return np.clip(x,lb,ub)


"función para calcular Psi(x)"
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
    
"Formulación primal-dual de la función objetivo"
"i.e. r*1/2||Psi(x)||**2) + iC0(x) + sum(iCi(yi) + ||yi||**2/2)-sum(Psi(x),y_i)"
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



" DSA (no linesearch):"

def NCsplitting(x,gamma,mu,kappa,Lips_Qx,beta=0,tol=1e-6):
    " x - initial starting point"
    "the dual variables start all as the 0 vector"
    xk = x.copy()
    yk = np.zeros([r,m])
    "first update"
    gradQx = 2*(Q @ xk).T   
    gam  = gamma*1/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1))) 
    xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk)) 

    wk = (1-beta)*xkn + beta*xk
    "update  yk"
    ykold =yk.copy()
    for ii in range(r):
        tempyii = yk[ii,:]
        cii = cCr[ii,:]
        yk[ii,:] = PCr((tempyii+mu*Qx(wk))/(1+mu),cii)
    "here starts the loop"
    k = 1
    while abs(Phi(xkn,yk)-Phi(xk,ykold)) > tol: 
        xk = xkn.copy()
        gradQx = 2*(Q @ xk).T 
        gam  = gamma/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
        "update en xk"
        xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk))

        wk = (1-beta)*xkn + beta*xk
        "update en yk"
        ykold = yk.copy()
        for ii in range(r):
            tempyii = yk[ii,:]
            cii = cCr[ii,:]
            yk[ii,:] = PCr((tempyii+mu*Qx(wk))/(1+mu),cii)
        k += 1
    return xkn, yk, k



" BDSA (linesearch):"

def NCsplitting_B(x,gamma,mu,alph,kappa,Lips_Qx,barlam0 = 2,Ny=2,tol=1e-6):
    " x - initial starting point"
    "the dual variables start all as the 0 vector"
    barlamk = barlam0
    xk = x.copy()
    yk = cCr.copy()
    gradQx = 2*(Q @ xk).T 
    gam  = gamma*1/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))
     
  
    "first update"

    xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk)) #update en xk

    "update  yk"
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

    while abs(Phi(xkn,ykn)-Phi(xk,yk)) > tol:  
        xk = xkn.copy()
        yk = ykn.copy()
        gradQx = 2*(Q @ xk).T 
        gam  = gamma/(2*kappa +  Lips_Qx*sum(norm(yk,axis=1)))

        "update  xk"
        xkn = PC0( xk + gam*gradQx @ yk.sum(0) - r*gam*gradQx@Qx(xk))

        "update  yk"
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





"START EXPERIMENT"
##################### EXPERIMENT  ###################################


use_saved_data = False # True for using data paper, false for generate new

repP = 10
repS = 1

n = 20 
m = math.floor(0.8*n)
sizesr = [3, 5, 7, 10, 15, 20]

E3_Ssol = zeros([len(sizesr),repP,repS])
E3_Siter = zeros([len(sizesr),repP,repS])
E3_Stime = zeros([len(sizesr),repP,repS])

E3_Bsol = zeros([len(sizesr),repP,repS])
E3_Biter = zeros([len(sizesr),repP,repS])
E3_Btime = zeros([len(sizesr),repP,repS])

"Parameters"
gamma = 0.99
mu = 0.5

"Parameters boosted"

alph = 0.5
delta = 2

if use_saved_data == False:
    for NN in range(len(sizesr)):
        r = sizesr[NN]
        m = math.floor(0.8*n)
        for rPP in range(repP):
            
            " Generating data "
            
            rC0 = 5 #radius of the ball
            
            # center of the hypercubes
            normcCr = 3*random(r)+7
            cCr = 2*random([r,m])-1
            ncCr = norm(cCr, axis = 1 )
            cCr = normcCr[:, None] * cCr / ncCr[:, None] 
            
            #Operator Psi:
            Q = zeros([m,n,n])
            singvaluesQ = zeros(m)
    
            for mm in range(m):
                tempD = 2*random(n)-1
                tempQ = np.diag(tempD) 
                Q[mm,:,:] =  tempQ
                singvaluesQ[mm] = norm(tempQ,2) 
            
            QM = zeros([m*n,n])         
            for mm in range(m):
                QM[mm*n:(mm+1)*n,:] = Q[mm]
            normQ =  norm(QM,2) #
            maxu = 2*normQ*rC0
            Lips_Qx = 2*normQ*rC0  
            kappa = 6*r*rC0**2*normQ*np.linalg.norm(np.array([np.linalg.norm(Q[i],ord=2) for i in range(m)]))/2
            
            for rSS in range(repS):
                "Generate random starting point"
                x0 = 10*random(n)
                
                starttime = time.time()
                sol,_,it = NCsplitting(x0,gamma,mu,kappa,Lips_Qx,tol=1e-6)
                stoptime = time.time()
                
                E3_Ssol[NN,rPP,rSS] = varphi(sol)
                E3_Siter[NN,rPP,rSS] = it
                E3_Stime[NN,rPP,rSS] = starttime-stoptime
                
                print('Split done!!')
                        
                
                start1 = time.time()
                sol,_,it = NCsplitting_B(x0,gamma,mu,alph,kappa,Lips_Qx,barlam0=2,Ny=2,tol=1e-6)
                stop1 = time.time()
                
                E3_Bsol[NN,rPP,rSS] = varphi(sol)
                E3_Biter[NN,rPP,rSS] = it
                E3_Btime[NN,rPP,rSS] = start1-stop1
                
                print('Boosted done!!')
                
                "To  save data"     
                #np.savez('Exp_HP_SplitVSBoosted_r_v01', E3_Ssol, E3_Siter, E3_Stime, E3_Bsol, E3_Biter, E3_Btime) 
     
elif use_saved_data==True:
    npzfile = np.load('Exp_HP_SplitVSBoosted_r_v01.npz',allow_pickle = True )
    E3_Ssol = npzfile['arr_0']
    E3_Siter = npzfile['arr_1']
    E3_Stime = npzfile['arr_2']
    E3_Bsol= npzfile['arr_3']
    E3_Biter = npzfile['arr_4']
    E3_Btime = npzfile['arr_5']




averagesol0_S = np.sum(E3_Ssol,axis=2)/repS
averagesol_S  = np.sum(averagesol0_S,axis=1)/repP


averagesol0_B = np.sum(E3_Bsol,axis=2)/repS
averagesol_B  = np.sum(averagesol0_B,axis=1)/repP

difsol0 = averagesol0_S/ averagesol0_B
difsol= averagesol_S / averagesol_B

averageiter0_S = np.sum(E3_Siter,axis=2)/repS
averageiter_S  = np.sum(averageiter0_S,axis=1)/repP


averageiter0_B = np.sum(E3_Biter,axis=2)/repS
averageiter_B  = np.sum(averageiter0_B,axis=1)/repP

difiter0 = averageiter0_S/ averageiter0_B
difiter = averageiter_S / averageiter_B


averagetime0_S = np.sum(E3_Stime,axis=2)/repS
averagetime_S  = np.sum(averagetime0_S,axis=1)/repP

averagetime0_B = np.sum(E3_Btime,axis=2)/repS
averagetime_B  = np.sum(averagetime0_B,axis=1)/repP

diftime0 = averagetime0_S / averagetime0_B
diftime = averagetime_S / averagetime_B


difitertotal = np.sum(difiter)/len(sizesr)
diftimetotal = np.sum(diftime) / len(sizesr)


" Plot de Iteraciones "

plt.clf()
ax = plt.gca()



for NN in range(len(sizesr)):
    ax.plot(NN*np.ones(repP),difiter0[NN,:],'o', markersize = 8,fillstyle = 'none', color = 'cornflowerblue')
    


ax.plot(difiter[0:len(sizesr)],'o', markersize = 8, color = 'k')
ax.hlines(difitertotal,-0.5,len(sizesr)-0.5,linestyle = 'dashed', color = 'mediumblue')

ax.set_xlim([-0.5,len(sizesr)-.5])
ax.set_ylim([4,5.75])
ax.set_xlabel('p')
ax.set_ylabel('Iterations ratio DSA/BDSA')

ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(['3', '5', '7', '10', '15','20'])
plt.title('Varying p')

plt.savefig('HP_SplitVSBoosted_r_Iter.pdf',bbox_inches='tight',dpi=400)
plt.show()    


plt.clf()
ax = plt.gca()



for NN in range(len(sizesr)):
    ax.plot(NN*np.ones(repP),diftime0[NN,:],'o', markersize = 8, fillstyle = 'none', color = 'lightcoral')
    

ax.plot(diftime[0:len(sizesr)],'o',color = 'k',markersize=8)
ax.hlines(diftimetotal,-0.5,len(sizesr)-.5,color='crimson',linestyles='dashed')
ax.set_xlim([-0.5,len(sizesr)-0.5])
ax.set_ylim([1.4, 4])
ax.set_xlabel('p')
ax.set_ylabel('Time ratio DSA/BDSA')
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels([ '3', '5', '7', '10', '15','20'])
plt.title('Varying p')

plt.savefig('HP_SplitVSBoosted_r_Time.pdf',bbox_inches='tight',dpi=400)
plt.show()            
















        
    
            
