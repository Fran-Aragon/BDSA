# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:51:11 2018

@author: david

" This file generates Figure 6:"

"Line 30: defining the constraints"
"Line 306:  algorithm codes  and auxiliary functions"
"Line 466: running the experiment"
"Line 789: plots "
"""

from numpy import array, concatenate, where, argmin, maximum, zeros, tile, repeat, newaxis, append, arange
from numpy.linalg import norm
from numpy.random import random, seed
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
#import pylab as plt

seed(5)

A=np.array(pd.read_excel(io='Spain_peninsula_baleares_500.xlsx'))#.as_matrix()
[m,n]=A.shape
c=5
it=7
it_lim=50
eps = 1e-13


"Setting the constraints"

plt.figure(figsize=(12,6))
plt.plot(A[:,0],A[:,1],'s',color=r'#695FFF',markersize=4,markeredgecolor='k',markeredgewidth=0.3) ##8eb9ff
plt.plot([])
ax = plt.gca()



def HPequation(xs,ys,pint):        #vertices cara, punto interior
    a = np.array([ys[1]-ys[0],-xs[1]+xs[0]])
    b = a@np.array([xs[0],ys[0]])
    if a@pint > b:
        return [-a,-b]
    else:
        return [a,b]
    
"Set 1"
pint = np.array([-8,43])
plt.plot(pint[0],pint[1],'y*',markersize=20)

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

pint = np.array([-5.8,43])
plt.plot(pint[0],pint[1],'y*',markersize=20)

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
plt.plot(pint[0],pint[1],'y*',markersize=20)

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
plt.plot(pint[0],pint[1],'y*',markersize=20)

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

pint = np.array([-.2,42])
plt.plot(pint[0],pint[1],'y*',markersize=20)


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

pint = np.array([-1.8,40])
plt.plot(pint[0],pint[1],'y*',markersize=20)

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

pint = np.array([-3.3,38.3])
plt.plot(pint[0],pint[1],'y*',markersize=20)

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
plt.plot(pint[0],pint[1],'y*',markersize=20)

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
plt.plot(pint[0],pint[1],'y*',markersize=20)

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

ax.axis([A.min(axis=0)[0]-0.3,A.max(axis=0)[0]+0.8,A.min(axis=0)[1]-0.6,A.max(axis=0)[1]+1.2])
a=A[0]
plt.plot(a[0],a[1],'s',color=r'#695FFF',markersize=4,label='$a^i,\, i=1,\ldots,'+str(len(A))+'$',markeredgecolor='k',markeredgewidth=0.3)

ax.set_aspect('equal')
plt.legend(loc='best',ncol=2)
plt.savefig('Clustering_Spain_1.pdf',bbox_inches='tight',dpi=800)
plt.show()


# 
H1 = [S1_H1,S2_H1,S3_H1,S4_H1,S5_H1,S6_H1,S7_H1,S8_H1,S9_H1]
H2 = [S1_H2,S2_H2,S3_H2,S4_H2,S5_H2,S6_H2,S7_H2,S8_H2,S9_H2]
V =  [V1,V2,V3,V4,V5,V6,V7,V8,V9]





" Algorithm codes and auxiliary functions"


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
            #print("i",ii)
            for jj in range(nH):
                #print("j",jj)
                di[jj] =Si_h2[jj]-Si_h1[jj]@x
            #print(di)
            if all(di>=0-eps):
                #print(di)
                #print("Feasible")
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



    

a0=tile(sum(A)/m,[c,1])
z=zeros(n)#array([0.0,0.0])   



cmap=plt.cm.get_cmap('hsv')
clineas=[1,1,1]

x=zeros([c,n])
v=A.max(axis=0)-A.min(axis=0)
for k in range(n):
    x[:,k]=random(c)*v[k]+A.min(axis=0)[k]
print(0,f(x,A))

x0=x.copy()
a=A[0]





"Start of the main experiment"


#Initializing the new plot
cmap=plt.cm.get_cmap('hsv')
clineas=[1,1,1]

plt.figure(figsize=(12,6))
plt.plot(A[:,0],A[:,1],'s',color=r'#695FFF',markersize=4,markeredgecolor='k',markeredgewidth=0.3) ##8eb9ff
plt.plot(a[0],a[1],'s',color=r'#695FFF',markersize=4,label='$a^i,\, i=1,\ldots,'+str(len(A))+'$',markeredgecolor='k',markeredgewidth=0.3)
plt.plot(x0[:,0],x0[:,1],'k.',markerfacecolor='k',markersize=12,markeredgewidth=0.7,zorder=4)
ax = plt.gca()
#ax.axis('off')
ax.axis([A.min(axis=0)[0]-0.3,A.max(axis=0)[0]+0.8,A.min(axis=0)[1]-0.6,A.max(axis=0)[1]+1.2])
ax.set_aspect('equal')

"Fill"
ALPHA=0.4
orden=3
col=r'#7cd929'#'C1'
plt.fill([-7.8,-9,-8.5,-6.3,-7.4],[43.75,43,42.1,42.5,43.6],col,alpha=ALPHA,zorder=orden)
plt.fill([-6.3,-5.2,-1.8,-7.4],[42.5,42.5,43.2,43.6],col,alpha=ALPHA,zorder=orden)
plt.fill([-5.2,-6.3,-7.5,-5.2],[42.5,42.5,37.2,37.2],col,alpha=ALPHA,zorder=orden)
plt.fill([-1.8,-5.2,-5.2,-3],[43.2,42.5,41,41],col,alpha=ALPHA,zorder=orden)
plt.fill([-3,1,3.2,3.2,-1.8],[41,41.2,41.8,42.3,43.2],col,alpha=ALPHA,zorder=orden)
plt.fill([-3,-3,-0.5,1],[41,39.4,39.1,41.2],col,alpha=ALPHA,zorder=orden)
plt.fill([-0.5,0,-2.6,-4.5,-5.2,-5.2,-3],[39.1,38.9,36.8,36.7,37.2,39.4,39.4],col,alpha=ALPHA,zorder=orden)
plt.fill([-4.5,-5.2,-6,-6.3,-4.9],[36.7,37.2,37.2,36.5,36.5],col,alpha=ALPHA,zorder=orden)
plt.fill([2.35,3,3.55,3.05],[39.6,40,39.75,39.3],col,alpha=ALPHA,zorder=orden)


"Computing the solution with the Splitting and more iterations"

Lf = 2
kappa = Lf/2
gamma = 0.9*(1/(2*kappa)) #0.9*(1/(2*Lf))

iterations= 1000

val_S = zeros(iterations+1)
val_BS = zeros(iterations+1)
lambdas = np.ones(iterations+1)
val_E = zeros(iterations+1)





eps_stop=1e-4

"Splitting"

xS = x0.copy()
val_S[0] = 5#FC(xS,A)

"Boosted Splitting"

xBS = x0.copy()
val_BS[0] = 5#FC(xBS,A)


barlamk0 = 2 #tamaño de paso dir de descenso
barlamk = barlamk0
N = 2 #número de intentos a probar
rho = .5

kF = 0
    
"Boosted Split"
    
for k in range(iterations):
    xBold = xBS.copy()
    vB = gf1(xBS,a0) -  gf2(xBS,A) 
    yBSn = proxg(xBS-gamma*vB, H1,H2,V)
    dB = yBSn-xBS
    lamk = barlamk
    r = 0
    while FC(yBSn+lamk*dB,A) > FC(yBSn,A)-0.1*lamk**2*norm(dB)**2:
        r += 1
        if r < N:
            lamk = (rho**r)*lamk
        else:
            lamk = 0
    xBSn = yBSn + lamk*dB  
    lambdas[k] = lamk
    if r==0:
        barlamk=2*barlamk
    else:
        barlamk = max(barlamk0,(rho**r)*lamk) #t00
    
    plt.plot(xBSn[:,0],xBSn[:,1],'k+',markeredgewidth=0.7,markersize=6,zorder=4)
    for j in range(c):
        plt.plot([xBSn[j,0],xBS[j,0]],[xBSn[j,1],xBS[j,1]],'k',linewidth=2,zorder=4)
    plt.plot([xBSn[0],xBS[0]],[xBSn[1],xBS[1]],'k',linewidth=2,zorder=4)
    plt.plot([xBSn[2],xBS[2]],[xBSn[3],xBS[3]],'k',linewidth=2,zorder=4)
    
    
    ax = plt.gca()
    ax.axis([A.min(axis=0)[0]-0.3,A.max(axis=0)[0]+0.8,A.min(axis=0)[1]-0.6,A.max(axis=0)[1]+1.2])
    ax.set_aspect('equal')
    if kF ==0:
        plt.plot(xBSn[0],xBSn[1],'k+-',markeredgewidth=.7,markersize=5,zorder=4)
    else:
        
        plt.plot(xBSn[:,0],xBSn[:,1],'k*',markersize=20)

        plt.plot(xBSn[:,0],xBSn[:,1],'k*',markersize=16)
        plt.plot(xBSn[:,0],xBSn[:,1],'k*',markersize=14,label='critical point')

    handles, labels = ax.get_legend_handles_labels()
    
    

    lineG = Line2D([0], [0], color= 'r', linestyle='-', marker='.', label ='GPPA')


    lineB = Line2D([0], [0], color='k',  linestyle='-',marker='x',label='BDSA')

    
    handles[:0] = [lineG, lineB] 

        
    
    plt.legend(handles=handles, loc='best',ncol=2)
    

 
    if FC(xBSn,A) > FC(xBS,A):
        print("Error:", k)
    if abs(FC(xBS,A)-FC(xBSn,A))/FC(xBSn,A)<eps_stop:
        itBDSA = k+1
        print('Iterations BDSA', k)
        break
        
    xBS = xBSn.copy()
    val_BS[k+1] = FC(xBS,A)
    
    print(k+1, FC(xS,A), FC(xBS,A), barlamk, r, lamk, norm(gf1(xS,a0)-gf2(xS,A)) , norm(gf1(xBS,a0)-gf2(xBS,A)))
    


cGPPA='r'

for k in range(iterations):
    
    xSold = xS.copy()
    "Split"
    v = gf1(xS,a0) - gf2(xS,A) 
    xSn = proxg(xS-gamma*v, H1, H2,V)
    plt.plot(xSn[:,0],xSn[:,1],'.',color=cGPPA,markersize=3,zorder=6)
    
    for j in range(c):
        plt.plot([xSn[j,0],xS[j,0]],[xSn[j,1],xS[j,1]],'-',color=cGPPA,linewidth=1,zorder=6)
 
    xS = xSn
    val_S[k+1] = FC(xS,A)
    
    if norm(FC(xSold,A)-FC(xS,A))/FC(xS,A)<eps_stop:
        itGPPA = k +1
        print('Iterations GPPA', k)
        break

  
" Extrapolated GPPA"

cEGPPA = 'orange'
mEGPPA = 'x'
val_E[0] = 5
xE = x0.copy()
xE0 = xE.copy()
# Parameters extrapolation
edel = 5e-25
elamb = 0.1
emub  = 1
ekapn = 1
ekap1 = ekapn


for k in range(iterations):
    
    
    # Parameter update:
    taun = 1/(2*edel + 2*(2*elamb + 1) + 2*emub)
    emun = emub*taun
    elamn = elamb*(ekap1-1)/ekapn
    
    # Auxi vectors:
    un = xE + elamn*(xE - xE0)
    v = gf1(un,a0) - gf2(xE,A)
    xEn = proxg(xE + emun*(xE-xE0) - gamma*v, H1,H2,V )
    plt.plot(xEn[:,0],xEn[:,1],marker=mEGPPA,markeredgewidth=.5,linestyle='none',color = cEGPPA,markersize = 5,markerfacecolor='none',zorder = 4)
    
    # Variable update:
    if (k+1)%50:
        # reset kappa
        ekap1 = 1
        ekapn = 1
    else:
        ekap1 = ekapn
        ekapn = (1+ np.sqrt(1+4*ekapn**2))/2
    
    for j in range(c):
        plt.plot([xEn[j,0],xE[j,0]],[xEn[j,1],xE[j,1]],'-',markeredgewidth=.5,color=cEGPPA,linewidth=1,zorder=4)

    
    xE0 = xE.copy()
    xE = xEn.copy()
    
    val_E[k+1] = FC(xEn,A)
    
    if norm(FC(xE0,A)-FC(xE,A))/FC(xE,A)<eps_stop:
        itePSA = k+1
        print('Iterations ePSA:', k)
        break

print('Objective values:')
print('BDSA:', FC(xBS,A))
print('GPPA:', FC(xS,A))
print('ePSA:', FC(xE,A))

pp = 8*27
xlambdas = np.linspace(0,iterations,pp)
ylambdas = np.zeros(len(xlambdas))
for ll in range(iterations):
    sep = int(pp/iterations)
    ylambdas[ll*sep] = lambdas[ll]

ax = plt.gca()
ax.axis([A.min(axis=0)[0]-0.3,A.max(axis=0)[0]+0.8,A.min(axis=0)[1]-0.6,A.max(axis=0)[1]+1.2])
ax.set_aspect('equal')


plt.plot(xSn[:,0],xSn[:,1],'C2*',markersize=16,markerfacecolor=cGPPA,zorder=4)

plt.plot(xBSn[:,0],xBSn[:,1],'C2*',markersize=16,markerfacecolor='k',zorder=4)
plt.plot(xBSn[:,0],xBSn[:,1],'C2*',markersize=16,label='critical points',markerfacecolor='none',zorder=0)
plt.plot(xE[:,0],xE[:,1],'C2*',markersize = 16, markerfacecolor=cEGPPA,zorder=4)


plt.plot(xSn[0],xSn[1],'.-',color=cGPPA,markersize=6,label='GPPA')
plt.plot(xBSn[0],xBSn[1],'k+-',markersize=6,label='BDSA')
plt.plot(xE[0],xEn[1],'x-',color = cEGPPA,markersize=5,label= 'ePSA')

handles, labels = ax.get_legend_handles_labels()


handles0 =[handles[2], handles[3], handles[4], handles[0], handles[1]]


plt.legend(handles = handles0, loc='best',ncol=2)

plt.savefig('Figures\Clustering_Spain_Fill_Worst.jpg',bbox_inches='tight',dpi=300)


"plot decreasing and stepsize"
fig, ax1 =  plt.subplots(figsize=(6,6))
ax2 = ax1.twinx()
ax1.axis([0,iterations,val_BS[iterations]-0.5,val_S[1]+0.5])
ax1.set_xlabel('$k$')
ax1.set_ylabel('$f(X^k)$')
ax2.set_ylabel('$\\lambda_{k}$')
ax1.plot(val_S[:itGPPA],color=cGPPA, marker='.',markersize=5,label = 'GPPA',markevery=4)
ax1.plot(val_BS[:itBDSA], 'k', marker='+',label = 'BDSA',markevery=4)
ax1.plot(val_E[:itePSA],'-', color='C1',marker = mEGPPA, markersize = 5,label = 'ePSA', markevery=4)
ax1.plot(0,0,'k',linestyle='dotted',label='$\\lambda_{k}$')
ax2.plot(range(itBDSA),lambdas[:itBDSA],'k',linestyle='dotted',label='$\\lambda_{k}$') #'$\\alpha^{r}\lambda_k$'
ax1.legend(loc='best')
ax1.set_ylim(0,ax1.get_ylim()[1])
ax2.set_ylim(0,ax2.get_ylim()[1])
plt.xlim([0,itGPPA])
plt.savefig('Figures\Clustering_decrease_Worst.pdf',bbox_inches='tight',dpi=800)


    
