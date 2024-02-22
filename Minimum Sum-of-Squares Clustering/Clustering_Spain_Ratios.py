# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:51:11 2018

@author: david


"""

from numpy import array, concatenate, where, argmin, maximum, zeros, tile, repeat, newaxis, append, arange
from numpy.linalg import norm
from numpy.random import random, seed
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import time
#import pylab as plt

seed(11)

saved = True # True for plotting the data used in the paper


A=np.array(pd.read_excel(io='Spain_peninsula_baleares_500.xlsx'))#.as_matrix()
[m,n]=A.shape
nC=[3,5,10,15,20,30,40,50]
eps = 1e-13
prec = 1e-4
R = 10 #número de repeticiones por cada numero de centro


"Setting the constraints"




def HPequation(xs,ys,pint):        #vertices cara, punto interior
    a = np.array([ys[1]-ys[0],-xs[1]+xs[0]])
    b = a@np.array([xs[0],ys[0]])
    if a@pint > b:
        return [-a,-b]
    else:
        return [a,b]
    
"Set 1"
pint = np.array([-8,43])
plt.plot(pint[0],pint[1],'y*',markersize=12)

plt.plot([-9,-7.8],[43,43.75],'y-',linewidth = 3)
S1_l1 = HPequation([-9,-7.8],[43,43.75],pint)

plt.plot([-8.5,-9],[42.1,43],'y-',linewidth = 3)
S1_l2 = HPequation([-8.5,-9],[42.1,43],pint)

plt.plot([-6.3,-8.5],[42.5,42.1],'y-',linewidth = 3)
S1_l3 = HPequation([-6.3,-8.5],[42.5,42.1],pint)

plt.plot([-7.4,-6.3],[43.6,42.5],'y-',linewidth = 3)
S1_l4 = HPequation([-7.4,-6.3],[43.6,42.5],pint)

plt.plot([-7.4,-7.8],[43.6,43.75],'y-',linewidth = 3)
S1_l5 = HPequation([-7.4,-7.8],[43.6,43.75],pint)

S1_H1 = np.array([S1_l1[0],S1_l2[0],S1_l3[0],S1_l4[0],S1_l5[0]])
S1_H2 = np.array([S1_l1[1],S1_l2[1],S1_l3[1],S1_l4[1],S1_l5[1]])
V1 = np.array([[-9,43],[-7.8,43.75],[-8.5,42.1],[-6.3,42.5],[-7.4,43.6]])

"Set 2"

pint = np.array([-4,43])
plt.plot(pint[0],pint[1],'y*',markersize=12)

S2_l1 = HPequation([-7.4,-6.3],[43.6,42.5],pint)


plt.plot([-6.3,-5.2],[42.5,42.5],'y-',linewidth = 3)
S2_l2 = HPequation([-6.3,-5.2],[42.5,42.5],pint)

plt.plot([-5.2,-1.8],[42.5,43.2],'y-',linewidth = 3)
S2_l3 = HPequation([-5.2,-1.8],[42.5,43.2],pint)

plt.plot([-1.8,-7.4],[43.2,43.6],'y-',linewidth = 3)
S2_l4 = HPequation([-1.8,-7.4],[43.2,43.6],pint)

S2_H1 = np.array([S2_l1[0],S2_l2[0],S2_l3[0],S2_l4[0]])
S2_H2 = np.array([S2_l1[1],S2_l2[1],S2_l3[1],S2_l4[1]])
V2 = np.array([[-7.4,43.6],[-6.3,42.5],[-5.2,42.5],[-1.8,43.2]])

"Set 3"

pint = np.array([-6,40])
plt.plot(pint[0],pint[1],'y*',markersize=12)

S3_l1 = HPequation([-6.3,-5.2],[42.5,42.5],pint)

plt.plot([-7.5,-6.3],[37.2,42.5],'y-',linewidth = 3)
S3_l2 = HPequation([-7.5,-6.3],[37.2,42.5],pint)

plt.plot([-7.5,-5.2],[37.2,37.2],'y-',linewidth = 3)
S3_l3 = HPequation([-7.5,-5.2],[37.2,37.2],pint)


plt.plot([-5.2,-5.2],[42.5,37.2],'y-',linewidth = 3)
S3_l4 = HPequation([-5.2,-5.2],[42.5,37.2],pint)

S3_H1 = np.array([S3_l1[0],S3_l2[0],S3_l3[0],S3_l4[0]])
S3_H2 = np.array([S3_l1[1],S3_l2[1],S3_l3[1],S3_l4[1]])
V3 = np.array([[-6.3,42.5],[-5.2,42.5],[-7.5,37.2],[-6.3,42.5],[-5.2,37.2]])

"Set 4"

pint = np.array([-4,42])
plt.plot(pint[0],pint[1],'y*',markersize=12)

S4_l1 = HPequation([-5.2,-1.8],[42.5,43.2],pint)

S4_l2 = HPequation([-5.2,-5.2],[41,42.5],pint)

plt.plot([-5.2,-3],[41,41],'y-',linewidth = 3)
S4_l3 = HPequation([-5.2,-3],[41,41],pint)

plt.plot([-3,-1.8],[41,43.2],'y-',linewidth = 3)
S4_l4 = HPequation([-5.2,-1.8],[41,43.2],pint)

S4_H1 = np.array([S4_l1[0],S4_l2[0],S4_l3[0],S4_l4[0]])
S4_H2 = np.array([S4_l1[1],S4_l2[1],S4_l3[1],S4_l4[1]])
V4 = np.array([[-5.2,42.5],[-1.8,43.2],[-5.2,41],[-3,41]])

"Set 8"

pint = np.array([-1,42])
plt.plot(pint[0],pint[1],'y*',markersize=12)


S8_l1 = HPequation([-5.2,-1.8],[41,43.2],pint)

plt.plot([1,-3],[41.2,41],'y-',linewidth = 3)
S8_l2 = HPequation([1,-3],[41.2,41],pint)


plt.plot([1,3.2],[41.2,41.8],'y-',linewidth = 3)
S8_l3 = HPequation([1,3.2],[41.2,41.8],pint)


plt.plot([3.2,3.2],[41.8,42.3],'y-',linewidth = 3)
S8_l4 = HPequation([3.2,3.2],[41.8,42.3],pint)


plt.plot([3.2,-1.8],[42.3,43.2], 'y-',linewidth = 3)
S8_l5 = HPequation([3.2,-1.8],[42.3,43.2],pint)

S8_H1 = np.array([S8_l1[0],S8_l2[0],S8_l3[0],S8_l4[0],S8_l5[0]])
S8_H2 = np.array([S8_l1[1],S8_l2[1],S8_l3[1],S8_l4[1],S8_l5[1]])
V8 = np.array([[-5.2,41],[-1.8,43.2],[1,41.2],[3.2,41.8],[3.2,42.3]])

"Set 7"

pint = np.array([-1,40])
plt.plot(pint[0],pint[1],'y*',markersize=12)

S7_l1 = HPequation([1,-3],[41.2,41],pint)

plt.plot([-3,-3],[41,39.4],'y-',linewidth = 3)
S7_l2 = HPequation([-3,-3],[41,39.4],pint)

plt.plot([-0.5,-3],[39.1,39.4],'y-',linewidth = 3)
S7_l3 = HPequation([-0.5,-3],[39.1,39.4],pint)


plt.plot([-0.5,1],[39.1,41.2],'y-',linewidth = 3)
S7_l4 = HPequation([-0.5,1],[39.1,41.2],pint)

S7_H1 = np.array([S7_l1[0],S7_l2[0],S7_l3[0],S7_l4[0]])
S7_H2 = np.array([S7_l1[1],S7_l2[1],S7_l3[1],S7_l4[1]])
V7 = np.array([[1,41.2],[-3,41],[-3,39.4],[-0.5,39.1]])

"Set 6"

pint = np.array([-2,38.5])
plt.plot(pint[0],pint[1],'y*',markersize=12)

S6_l1 = HPequation([-0.5,-3],[39.1,39.4],pint)

plt.plot([-0.5,0],[39.1,38.9],'y-',linewidth = 3)
S6_l2 = HPequation([-0.5,-0],[39.1,38.9],pint)


plt.plot([-2.6,0],[36.8,38.9],'y-',linewidth = 3)
S6_l3 = HPequation([-2.6,-0],[36.8,38.9],pint)

plt.plot([-2.6,-4.5],[36.8,36.7],'y-',linewidth = 3)
S6_l4 = HPequation([-2.6,-4.5],[36.8,36.7],pint)

plt.plot([-4.5,-5.2],[36.7,37.2],'y-',linewidth = 3)
S6_l5 = HPequation([-4.5,-5.2],[36.7,37.2],pint)

plt.plot([-5.2,-5.2],[39.4,37.2],'y-',linewidth = 3)
S6_l6 = HPequation([-5.2,-5.2],[39.4,37.2],pint)

plt.plot([-5.2,-3],[39.4,39.4],'y-',linewidth = 3)
S6_l7 = HPequation([-5.2,-3],[39.4,39.4],pint)

S6_H1 = np.array([S6_l1[0],S6_l2[0],S6_l3[0],S6_l4[0],S6_l5[0],S6_l6[0],S6_l7[0]])
S6_H2 = np.array([S6_l1[1],S6_l2[1],S6_l3[1],S6_l4[1],S6_l5[1],S6_l6[1],S6_l7[1]])
V6 = np.array([[-0.5,39.1],[0,38.9],[-2.6,36.8],[-4.5,36.7],[-5.2,37.2],[-5.2,39.4],[-3,39.4]])


"Set 5"

pint = np.array([-5.6,36.9])
plt.plot(pint[0],pint[1],'y*',markersize=12)

plt.plot([-4.5,-5.2],[36.7,37.2],'y-',linewidth = 3)
S5_l1 = HPequation([-4.5,-5.2],[36.7,37.2],pint)

plt.plot([-5.2,-6],[37.2,37.2],'y-',linewidth = 3)
S5_l2 = HPequation([-5.2,-6],[37.2,37.2],pint)

plt.plot([-6,-6.3],[37.2,36.5],'y-',linewidth = 3)
S5_l3 = HPequation([-6,-6.3],[37.2,36.5],pint)

plt.plot([-4.5,-4.9],[36.7,36.5],'y-',linewidth = 3)
S5_l4 = HPequation([-4.5,-4.9],[36.7,36.5],pint)

plt.plot([-4.9,-6.3],[36.5,36.5],'y-',linewidth = 3)
S5_l5 = HPequation([-4.9,-6.3],[36.5,36.5],pint)

S5_H1 = np.array([S5_l1[0],S5_l2[0],S5_l3[0],S5_l4[0],S5_l5[0]])
S5_H2 = np.array([S5_l1[1],S5_l2[1],S5_l3[1],S5_l4[1],S5_l5[1]])
V5 = np.array([[-4.5,36.7],[-5.2,37.2],[-6,37.2],[-6.3,36.5],[-4.9,36.5]])

"Set 9"

pint = np.array([3,39.7])
plt.plot(pint[0],pint[1],'y*',markersize=12)

plt.plot([2.35,3],[39.6,40],'y-',linewidth = 3)
S9_l1 = HPequation([2.35,3],[39.6,40],pint)

plt.plot([3,3.55],[40,39.75],'y-',linewidth = 3)
S9_l2 = HPequation([3,3.55],[40,39.75],pint)

plt.plot([2.35,3.05],[39.6,39.3],'y-',linewidth = 3)
S9_l3 = HPequation([2.35,3.05],[39.6,39.3],pint)

plt.plot([3.55,3.05],[39.75,39.3],'y-',linewidth = 3)
S9_l4 = HPequation([3.55,3.05],[39.75,39.3],pint)


S9_H1 = np.array([S9_l1[0],S9_l2[0],S9_l3[0],S9_l4[0]])
S9_H2 = np.array([S9_l1[1],S9_l2[1],S9_l3[1],S9_l4[1]])
V9 = np.array([[2.35,39.6],[3,40],[3.55,39.75],[3.05,39.5]])


a=A[0]


# 
H1 = [S1_H1,S2_H1,S3_H1,S4_H1,S5_H1,S6_H1,S7_H1,S8_H1,S9_H1]
H2 = [S1_H2,S2_H2,S3_H2,S4_H2,S5_H2,S6_H2,S7_H2,S8_H2,S9_H2]
V =  [V1,V2,V3,V4,V5,V6,V7,V8,V9]

def proxg(X,h1,h2,V):
    if iC(X,h1,h2) == 0:
        return X
    else: 
        [k, m] = X.shape
        p = np.zeros([k,m])
        nS = len(h1)
        for kk in range(k):
            x = X[kk,:]
            Feasible = False
            diffs= np.zeros([nS,m])
            dists = np.zeros(nS)
            for ii in range(nS):
                Si_h1 = h1[ii]
                Si_h2 = h2[ii]
                nH = len(Si_h1)   #number hyperplanes of the set
                di = np.zeros(nH)
                for jj in range(nH):
                    di[jj] = Si_h2[jj]-Si_h1[jj]@x
                if all(di>= 0-eps):
                    Feasible = True
                    p[kk,:] = x    
                else:
                    diffs_i = np.zeros([nH,m])
                    dists_i = np.zeros(nH)
                    for jj in range(nH):
                        diffs_i[jj] = di[jj]*Si_h1[jj]/norm(Si_h1[jj])**2
                        dists_i[jj] = norm(diffs_i[jj])
                        Pi = x+diffs_i[jj]
                        if iCk(Pi,Si_h1,Si_h2) != 0:
                            Vi = V[ii]
                            distV = np.zeros(len(Vi))
                            for iii in range(len(Vi)):
                                distV[iii] = norm(x-Vi[iii])
                            lll = list(distV).index(distV.min())
                            dists_i[jj] = norm(x-Vi[lll])
                            diffs_i[jj] = Vi[lll] - x                 
                    l = list(dists_i).index(dists_i.min())
                    diffs[ii] = diffs_i[l]
                    dists[ii] = dists_i[l]
            if Feasible == False:
                ll = list(dists).index(dists.min())
                p[kk,:] = x + diffs[ll]
    return p
          
def iC(X,h1,h2):
    k = X.shape[0]
    nS = len(h1)   #number sets
    Feasible = True
    for kk in range(k):
        x = X[kk,:]
        Feasiblek = False
        for ii in range(nS):
            Si_h1 = h1[ii]
            Si_h2 = h2[ii]
            nH = len(Si_h1)   #number hyperplanes of the set
            di= np.zeros(nH)
            for jj in range(nH):
                di[jj] =Si_h2[jj]-Si_h1[jj]@x
            if all(di>=0-eps):
                Feasiblek = True
        if Feasiblek==False:
                Feasible = False           
    if Feasible:
        return 0
    else:
        return  np.inf
    
def iCk(x,h1,h2):   #number sets
    Feasible = False    
    nH = len(h1)   #number hyperplanes of the set
    di= np.zeros(nH)
    for jj in range(nH):
        di[jj] =h2[jj]-h1[jj]@x
    if all(di>=0-eps):
        Feasible = True
    if Feasible:
        return 0
    else:
        return  np.inf
                
def f(x,A):
    return  ((norm(repeat(x[newaxis,:,:],A.shape[0],0)-A[:,newaxis,:],axis=2)**2).min(axis=1)).sum()/A.shape[0]


def FC(x,A):
    return iC(x,H1,H2) + f(x,A)


def gf1(x,a0):   #A0 is defined below
    return 2*(x-a0)

def gf2(x,A,lam=0.5):
    gf=zeros(x.shape)
    m=A.shape[0]
    for i in range(m):
        a=A[i]
        i_min=norm(a-x,axis=1).argmin()
        gf_add=2*(x-a)
        gf_add[i_min,:]=zeros(n)
        gf+=gf_add
    return gf/m



def compute_r(c,A):
    cc=zeros([c.shape[0],A.shape[0]])
    for k in range(c.shape[0]):
        cc[k]=norm(c[k]-A,axis=1)
    ind=list()
    for k in range(c.shape[0]):
        ind.append(where(argmin(cc,axis=0)==k)[0])
    r=zeros(c.shape[0])
    for k in range(c.shape[0]):
        r[k]=(cc[k][ind[k]]).max()
    return r

Fvalues_S = zeros([len(nC),R])
Fvalues_BS = zeros([len(nC),R])
Fvalues_E = zeros([len(nC),R])

times_S = zeros([len(nC),R])
times_BS = zeros([len(nC),R])
times_E = zeros([len(nC),R])

It_S = zeros([len(nC),R])
It_BS= zeros([len(nC),R])
It_E = zeros([len(nC),R])

contadorfails = 0
EwinsBDSA = 0 
EwinsGPPA = 0

if saved == False:
    
    for nc in range(len(nC)):
        c = nC[nc]
        a0=tile(sum(A)/m,[c,1])
        z=zeros(n)#array([0.0,0.0])   
        
        
        cmap=plt.cm.get_cmap('hsv')
        clineas=[1,1,1]
        
        x=zeros([c,n])
        v=A.max(axis=0)-A.min(axis=0)
        for k in range(n):
            x[:,k]=random(c)*v[k]+A.min(axis=0)[k]
        
        
        a=A[0]
        
        for rr in range(R):
            x=zeros([c,n])
            v=A.max(axis=0)-A.min(axis=0)
            for k in range(n):
                x[:,k]=random(c)*v[k]+A.min(axis=0)[k]
            #print(0,f(x,A))
            x0=x.copy()
            
            
            "Aquí comienza el  experimento  para el splitting"
            
            
        
            
            "Computing the solution with the Splitting and more iterations"
            
            Lf = 2
            kappa = Lf/2
            gamma = 0.9*(1/(2*kappa)) #0.9*(1/(2*Lf))
            
            iterations=25
            
            "Splitting"
            
            xS = x0.copy()
            
            "Boosted Splitting"
            
            xBS = x0.copy()
            
            " Extrapolated Proximal subgradient"
            xE = x0.copy()
            
            
            
            N = 2 #número de intentos a probar
            rho = .5
            barlam0 = 2
            barlamk = barlam0
            k = 0
            stopcrit = 100
            
            "Boosted Split"
            start_BS = time.time()
            while k<2 or stopcrit >= prec:
                vB = gf1(xBS,a0) -  gf2(xBS,A) 
                yBSn = proxg(xBS-gamma*vB, H1,H2,V)
                dB = yBSn-xBS
                lamk = barlamk
                exp_rho = 0
                while FC(yBSn+lamk*dB,A) > FC(yBSn,A) -0.1*lamk**2*norm(dB)**2:
                    exp_rho += 1
                    if exp_rho < N:
                        lamk = (rho**exp_rho)*barlamk
                    else:
                        lamk = 0
                xBSn = yBSn + lamk*dB  
                if exp_rho==0 :
                    barlamk = 2*lamk
                else:
                    barlamk = max((rho**exp_rho)*barlamk,barlam0)
            
    
                fxB = f(xBS,A)  
                xBS = xBSn
                fxBn = f(xBSn,A)
                k += 1
                if k>=2:
                    stopcrit = norm(fxB-fxBn)/norm(fxBn)
            stop_BS = time.time()
            
            FBoosted = fxBn
            
            It_BS[nc,rr] = k
            
            print('Boosted done')
            
        
            
            
            "Normal splitting"
            k = 0
            stopcrit = 100
            start_S = time.time()
            fxSn = f(xS,A)
            while k<2 or stopcrit >= prec:
                "Split"
                v = gf1(xS,a0) - gf2(xS,A) 
                xSn = proxg(xS-gamma*v, H1, H2,V)
                plt.plot(xSn[:,0],xSn[:,1],'k.',markersize=6)
                
                fxS = f(xS,A)
                xS = xSn
                fxSn = f(xSn,A)
                k += 1
                if k>=2:
                    stopcrit = norm(fxS-fxSn)/norm(fxSn)
                    
            stop_S = time.time()
            It_S[nc,rr] = k
    
            
            
            print('Splitting done')
    
            
            
            " Extrapolated GPPA"
    
    
            xE = x0.copy()
            xE0 = xE.copy()
            # Parameters extrapolation
            edel = 5*10**(-25)
            elamb = 0.1
            emub  = 1
            ekapn = 1
            ekap1 = ekapn
    
            k  = 0
            stopcrit = 100
            start_E = time.time()
            fxEn = f(xE,A)
            
            while k<2 or stopcrit >= prec:
                
                # Parameter update:
                taun = 1/(2*edel + 2*(2*elamb + 1) + 2*emub)
                emun = emub*taun
                elamn = elamb*(ekap1-1)/ekapn
                
                # Auxi vectors:
                un = xE + elamn*(xE - xE0)
                v = gf1(un,a0) - gf2(xE,A)
                xEn = proxg(xE + emun*(xE-xE0) - gamma*v, H1,H2,V )
                            
                k += 1
                # Variable update:
                
                xE0 = xE.copy()
                xE = xEn.copy()
                
                
                if k % 50:
                    # reset kappa
                    ekap1 = 1
                    ekapn = 1
                else:
                    ekap1 = ekapn
                    ekapn = (1+ np.sqrt(1+4*ekapn**2))/2
                
                fxE = f(xE0,A)
                fxEn = f(xEn,A)
                
                if  k>=2: 
                    stopcrit = norm(fxE-fxEn)/norm(fxEn)
                    
            stop_E = time.time()
            It_E[nc,rr] = k
            
            print('Extrapolated done')
            
            
                
            Fvalues_S[nc,rr] = fxSn
            Fvalues_BS[nc,rr] = fxBn
            Fvalues_E[nc,rr] = fxEn
            
            
            
            times_S[nc,rr] = stop_S - start_S
            times_BS[nc,rr] = stop_BS - start_BS
            times_E[nc,rr] = stop_E - start_E
            
            if fxSn > fxBn:
                contadorfails += 1 
            
            if fxE > fxBn:
                EwinsBDSA  +=1
                
            if fxE > fxSn:
                EwinsGPPA += 1
     
                    

    #np.savez('Exp_Clustering_Ratios_240221', contadorfails, Fvalues_S, Fvalues_BS, Fvalues_E, times_S, times_BS,times_E, It_BS, It_S, It_E)


elif saved == True:
    npzfile = np.load('Exp_Clustering_Ratios_240221.npz',allow_pickle = True )
    times_S = npzfile['arr_4']
    times_BS = npzfile['arr_5']
    times_E = npzfile['arr_6']
    It_BS = npzfile['arr_7']
    It_S = npzfile['arr_8']
    It_E = npzfile['arr_9']
    Fvalues_S = npzfile['arr_1']
    Fvalues_BS = npzfile['arr_2']
    Fvalues_E = npzfile['arr_3']




ratioiter = It_S / It_BS
ratiotime = times_S / times_BS
ratioiterE = It_E / It_BS
ratiotimeE = times_E / times_BS

ratioiter_clusters = np.sum(It_S,1) / np.sum(It_BS,1)
ratiotime_clusters = np.sum(times_S,1) / np.sum(times_BS,1)

ratioiterE_clusters = np.sum(It_E,1) / np.sum(It_BS,1)
ratiotimeE_clusters = np.sum(times_E,1) / np.sum(times_BS,1)

ratioiter_total = np.sum(np.sum(It_S))/np.sum(np.sum(It_BS))
ratiotime_total = np.sum(np.sum(times_S))/np.sum(np.sum(times_BS))

ratioiterE_total = np.sum(np.sum(It_E))/np.sum(np.sum(It_BS))
ratiotimeE_total = np.sum(np.sum(times_E))/np.sum(np.sum(times_BS))



plt.clf()
ax = plt.gca()
ax.plot(ratioiter_clusters,'o', markersize = 8, color = 'C0',zorder=4,markeredgecolor='k')
ax.plot(ratioiterE_clusters,'X',markersize = 9, color = 'C1',zorder=4,markeredgecolor='k')

ax.hlines(ratioiter_total,-0.5,len(nC)-0.5,color = 'C0', linestyle = 'dashed' )
ax.hlines(ratioiterE_total,-0.5,len(nC)-0.5,color = 'C1',linestyle='dotted')
for i in range(len(nC)):
    ax.plot(i*np.ones(len(ratioiter[i,:])),ratioiter[i,:], 'o', markersize = 8, fillstyle = 'none', color = 'C0')
    ax.plot(i*np.ones(len(ratioiterE[i,:])),ratioiterE[i,:],'x',markersize = 8, fillstyle = 'none', color = 'C1')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Iterations ratio')
ax.set_xticks([0,1,2,3,4,5,6,7])
plt.xlim([-0.5,len(nC)-.5])

ax.set_xticklabels(['3','5','10','15','20','30','40','50'])

#Legend:
legendS = Line2D([0],[0],label = 'GPPA/BDSA',marker = 'o', markersize = '7', 
                 color = 'C0', linestyle = 'dashed',markeredgecolor='k' )
legendE = Line2D([0],[0],label = 'ePSA/BDSA', marker = 'X', markersize = '8',
                 color = 'C1', linestyle = 'dotted',markeredgecolor='k')




handles, labels = ax.get_legend_handles_labels()

handles.extend([legendS,legendE])


plt.legend(handles=handles,loc = 'best')


plt.savefig('Cluster_Iterations.pdf',bbox_inches='tight',dpi=400)
plt.show()    

plt.clf()

ax = plt.gca()
ax.plot(ratiotime_clusters,'o', markersize = 8,markeredgecolor='k', color = 'C0',zorder=4)
ax.plot(ratiotimeE_clusters,'X',markersize = 9,markeredgecolor='k', color = 'C1',zorder=4)



ax.hlines(ratiotime_total,-.5,len(nC)-.5,color = 'C0', linestyle = 'dashed' )
ax.hlines(ratiotimeE_total,-0.5,len(nC)-0.5,color = 'C1',linestyle='dotted')

for i in range(len(nC)):
 
    
    ax.plot(i*np.ones(len(ratiotime[i,:])),ratiotime[i,:], 'o', markersize = 8, fillstyle = 'none', color = 'C0')
    ax.plot(i*np.ones(len(ratiotimeE[i,:])),ratiotimeE[i,:],'x',markersize = 8, fillstyle = 'none', color = 'C1')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Time ratio')
plt.xlim([-0.5,len(nC)-.5])

ax.set_xticks([0,1,2,3,4,5,6,7])
ax.set_xticklabels(['3','5','10','15','20','30','40','50'])

#Legend:
legendS = Line2D([0],[0],label = 'GPPA/BDSA',marker = 'o', markersize = '7', 
                 color = 'C0', linestyle = 'dashed',markeredgecolor='k' )
legendE = Line2D([0],[0],label = 'ePSA/BDSA', marker = 'X', markersize = '8',
                 color = 'C1', linestyle = 'dotted',markeredgecolor='k')




handles, labels = ax.get_legend_handles_labels()

handles.extend([legendS,legendE])


plt.legend(handles=handles,loc = 'best')

plt.savefig('Cluster_time.pdf',bbox_inches='tight',dpi=400)
plt.show()






    
