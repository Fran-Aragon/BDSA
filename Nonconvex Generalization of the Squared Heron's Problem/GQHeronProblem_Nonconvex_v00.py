# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:06:29 2023

@author: david

Generalized quadratic Heron Problem NONCOVEX SETS

We test the nonconvex splitting for a generalized Heron problem
consisting in finding a point x in a cc set C0 which minimizes the sum of 
the quadratic distances of the image of x by some quadratic transformation
to some cc sets C1,...,Cr

The experiment starst in line 208
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


" Auxiliary functions "



def PC0(x):
    if norm(x) > rC0:
        return rC0*x/norm(x)
    else:
        return x

    

def PCr(x,cc):
    p = x
    dist = np.inf
    for ii in range(len(cc)):
        c = cc[ii]
        ub = c + np.sqrt(2)/2
        lb = c - np.sqrt(2)/2 
        tempp = np.clip(x,lb,ub)
        tempdist = norm(x-p)
        if tempdist < dist:
            dist = tempdist
            p = tempp   
    return p


"Objective function is the following:"

def varphi(x):
    if (x==PC0(x)).all:
        return 1/2*norm(L@x-PCr(L@x,cc))
    else:
        return np.inf
    


" Phi = varphi"

def Phi(x):
    return varphi(x)


"DSA (no linesearch):"

def NCsplitting(x,gamma,kappa,tol=1e-6):   #Lips_Qx,
    " x - initial starting point"
    gam  = gamma*1/(2*kappa )
    xk = x.copy()
    #yk = np.zeros([r,m])
    "first update"
    Lxk = L@xk
    subdf= L.T@(Lxk-PCr(Lxk, cc))  
    xkn = PC0( xk  - gam*subdf) #update en xk
    "here starts the loop"
    k = 1
    while abs(varphi(xkn)-varphi(xk)) > tol:  # and k <10:
        xk = xkn.copy()
        Lxk = L@xk
        subdf= L.T@(Lxk-PCr(Lxk, cc))   
        "update xk"
        xkn = PC0( xk - gam*subdf)
        k += 1
    return xkn, k

"BDSA (with linesearch):"

def NCsplitting2(x,gamma,alph,kappa,Ny=2,tol=1e-6):
    " x - initial starting point"
    barlam0=2
    barlamk = barlam0
    xk = x.copy()
    gam  = gamma*1/(2*kappa) # 
     
  
    "first update"

    Lxk = L@xk
    subdf= L.T@(Lxk-PCr(Lxk, cc))   
    xkn = PC0( xk - gam*subdf) #update en xk
    "Linear Search"
    dxk=xkn-xk
    exp_alph = 0
    lamk = barlamk
    while varphi(xkn+lamk*dxk) > Phi(xkn) - 0.1*lamk**2*norm(dxk)**2  :
        exp_alph += 1
        if exp_alph < Ny:
            lamk = (alph**exp_alph)*barlamk
        else:
            lamk = 0
            break
    if exp_alph == 0:
            barlamk = 2*barlamk
    else:barlamk= max((alph**exp_alph)*barlamk,barlam0)
    xkn = xkn  + lamk*dxk
    k = 1
    while abs(varphi(xkn)-varphi(xk)) > tol:  # and k <10:
        xk = xkn.copy()
        "update  xk"
        Lxk = L@xk
        subdf= L.T@(Lxk-PCr(Lxk, cc))   
        k += 1
        "Linear Search"
        dxk=xkn-xk
        exp_alph = 0
        lamk = barlamk
        while varphi(xkn+lamk*dxk) > varphi(xkn) -0.1*lamk**2*norm(dxk)**2 :
            exp_alph += 1
            if exp_alph < Ny:
                lamk =   (alph**exp_alph)*barlamk
            else:
                lamk = 0
                break
        if exp_alph == 0:
                barlamk = 2*barlamk
        else: barlamk= max((alph**exp_alph)*barlamk,barlam0)
        xkn = xkn  + lamk*dxk

    return xkn,  k



  
"The experiment starts here"
##################### EXPERIMENT  ###################################


use_saved_data = True
repP = 10
repS = 1

sizesn = [ 50, 100, 200, 300, 500]

r = 5

E3_Ssol = zeros([len(sizesn),repP,repS])
E3_Siter = zeros([len(sizesn),repP,repS])
E3_Stime = zeros([len(sizesn),repP,repS])

E3_Bsol = zeros([len(sizesn),repP,repS])
E3_Biter = zeros([len(sizesn),repP,repS])
E3_Btime = zeros([len(sizesn),repP,repS])

"Parameters"
gamma = 0.99

"Parameters boosted"

alph = 0.5
delta = 2

if use_saved_data == False:
    for NN in range(len(sizesn)):
        n = sizesn[NN]
        m = math.floor(1.2*n)
        for rPP in range(repP):
            
            " Generating data "
            
            rC0 = 5 #radius of the ball
            
            # center of the hypercubes
            normcCr = 3*random(r)+7
            cCr = 2*random([r,m])-1
            ncCr = norm(cCr, axis = 1 )
            cc = normcCr[:, None] * cCr / ncCr[:, None] #estos son los centros de los hipercubos
            
            #Operator Psi:
            L = random([m,n])

            kappa = norm(L,2)**2/2
            
            for rSS in range(repS):
                "Generate random starting point"
                x0 = 10*random(n)
                
                starttime = time.time()
                sol,it = NCsplitting(x0,gamma,kappa,tol=1e-6)
                stoptime = time.time()
                
                E3_Ssol[NN,rPP,rSS] = varphi(sol)
                E3_Siter[NN,rPP,rSS] = it
                E3_Stime[NN,rPP,rSS] = starttime-stoptime
                
                print('Split done!!')
                        
                
                start1 = time.time()
                sol,it = NCsplitting2(x0,gamma,alph,kappa,Ny=2,tol=1e-6)  
                stop1 = time.time()
                
                E3_Bsol[NN,rPP,rSS] = varphi(sol)
                E3_Biter[NN,rPP,rSS] = it
                E3_Btime[NN,rPP,rSS] = start1-stop1
                
                print('Boosted done!!')
                
                "To  save data"     
                #np.savez('Exp_HP_SplitVSBoosted_NCS_v00', E3_Ssol, E3_Siter, E3_Stime, E3_Bsol, E3_Biter, E3_Btime) 
     
elif use_saved_data == True:
    npzfile = np.load('Exp_HP_SplitVSBoosted_NCS_v00.npz',allow_pickle = True )
    E3_Ssol = npzfile['arr_0']
    E3_Siter = npzfile['arr_1']
    E3_Stime = npzfile['arr_2']
    E3_Bsol = npzfile['arr_3']
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

difitertotal = np.sum(difiter)/(len(sizesn)-1)
diftimetotal = np.sum(diftime) /(len(sizesn)-1)



" Plot de Iteraciones "


" Plot de Iteraciones "

sizesn = [ 50, 100, 200, 300, 500]

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

difitertotal = np.sum(difiter)/(len(sizesn))
diftimetotal = np.sum(diftime) /(len(sizesn))



plt.clf()
ax = plt.gca()





for NN in range(len(sizesn)):
    ax.plot(NN*np.ones(repP),difiter0[NN,:],'o', markersize = 8, fillstyle = 'none', color = 'cornflowerblue')


ax.plot(difiter,'o',color = 'k',markersize=8)

ax.hlines(difitertotal,-.5,len(sizesn)-0.5,colors='mediumblue',linestyles='dashed')
ax.set_xlim([-0.5,len(sizesn)-0.5])
ax.set_ylim([1.9,7])
ax.set_xlabel('n')
ax.set_ylabel('Iterations ratio DSA/ BDSA')
ax.set_xticks([0,1,2,3,4])
plt.title('GHP with nonconvex sets')
ax.set_xticklabels(['50', '100','200','300','500'])

plt.savefig('HP_SplitVSBoosted_Iter_NC.pdf',bbox_inches='tight',dpi=400)
plt.show()    


plt.clf()
ax = plt.gca()



for NN in range(len(sizesn)):
    ax.plot(NN*np.ones(repP),diftime0[NN,:],'o',markersize = 8, fillstyle = 'none', color = 'lightcoral')


ax.plot(diftime,'o',color = 'k',markersize=8)

ax.hlines(diftimetotal,-0.5,len(sizesn)-0.5,colors='crimson',linestyles='dashed')
ax.set_xlim([-0.5,len(sizesn)-0.5])
ax.set_xlabel('n')
ax.set_ylabel('Time ratio DSA/BDSA')
ax.set_xticks([0,1,2,3,4])
plt.title('GHP with nonconvex sets')
ax.set_xticklabels(['50', '100','200','300','500'])

plt.savefig('HP_SplitVSBoosted_Time_NC.pdf',bbox_inches='tight',dpi=400)
plt.show()   



        
    
            
