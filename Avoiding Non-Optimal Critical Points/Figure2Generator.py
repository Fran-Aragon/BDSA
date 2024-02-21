# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:38:30 2023

Last modified Wed Jan 10

@author: David Torregrosa-Belén
"""

"Section: Avoiding Non-Optimal Critical Points"

" For generating fig 2 "

from numpy import array, concatenate, where, argmin, maximum, zeros, tile, repeat, newaxis, append, arange
from numpy.linalg import norm, eig
import math
import numpy.linalg as LA
from numpy.random import random, seed
from matplotlib import pyplot as plt
import numpy as np



seed(4)

n = 2 #  space dimension
p = 3 # ||x||^2 -|x| - sum_i=1^p (|x-i|+|x+i|) - |x-p-1|
cte = 1  #contant multiplying ||x||^2
nones = np.ones(n)
prec=1e-6
prec0=1e-6


"Auxiliary functions "
tauvect = np.zeros(2*p+2)
ltauvect = len(tauvect)

for ii in range(1,p+1):
    tauvect[2*ii-1] = ii
    tauvect[2*ii] = -ii
tauvect[2*p+1] = p+1


def varphi(x):
    sumat = 0
    for ii in range(1,p+1):
        sumat += np.sum(abs(x-ii)) + np.sum(abs(x+ii))
    return cte*norm(x)**2 -np.sum(abs(x))-sumat -np.sum(abs(x-p-1))

phisol = varphi(-4*np.ones(n))

# def varphi(x):
#     return cte*norm(x)**2 - np.sum(abs(x))-np.sum(abs(1-x))

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


" Initialization"
    
x0 = np.array([1.5, -0.5]) #initial points
y0= np.array([0,0.])   



xmin,xmax=-5,3
ymin,ymax=-5,2

"Parameters"
gamma = 1
mu = 1
    
"Double OProx"

it = 1000

DCA = np.zeros([it+1,2])
BDCA = np.zeros([it+1,2])
DPGA = np.zeros([it+1,2])
BDPGA = np.zeros([it+1,2])
OneProx = np.zeros([it+1,2]) #PPA
BOneProx = np.zeros([it+1,2]) #BPPA
iDCA = np.zeros([it+1,2])
BDSA = np.zeros([it+1,2])
BDSA_ls = np.zeros([it+1,2])



xk = x0.copy() #initial point for the double-Prox
yk = np.tile(y0,(ltauvect,1))



DCA[0,:] = x0
BDCA[0,:] = x0
DPGA[0,:] = x0
BDPGA[0,:] = x0
OneProx[0,:] = x0
BOneProx[0,:] = x0
iDCA[0,:] = x0
BDSA[0,:] = x0
BDSA_ls[0,:] = x0

"Double-Prox Gradient algorithm"

kk = 1
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
DPGA[kk,:] = xk
while norm(xkold -xk) + np.sum(norm(ykold-yk,axis=1)) > n*prec:
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
    kk += 1

    DPGA[kk,:] = xk

DPGA=DPGA[0:kk,:]

    
" Boosted Double-Prox Gradient Algorithm"

alph = 0.5
barlam0= 2
barlamk = 2
N = 2

xk = x0.copy() #initial point for the double-Prox
yk = np.tile(y0,(ltauvect,1))


kk=1
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
BDPGA[kk,:] = xk
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
        kk+=1
    
    BDPGA[kk,:] = xk
    
BDPGA = BDPGA[0:kk,:] 
        


"Proximal DC algorithm"
     
    
   
xk = x0.copy()
  
kk=1
xkold = xk.copy()
temp1 = xk - gamma*subh(xk) #argument proxg
xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
xk = xkn.copy()
OneProx[kk,:] = xk
while norm(xkold-xk)/n > prec:
    xkold = xk.copy()
    temp1 = xk - gamma*subh(xk) #argument proxg
    xkn =  temp1/(1+2*cte*gamma)   #evaluation prog x
    xk = xkn.copy()
    kk+=1
    
    OneProx[kk,:] = xk
    
OneProx=OneProx[0:kk,:]  

"Boosted Proximal DC algorihm"

alph = 0.5
barlam0 = 2
barlamk = barlam0
N = 2
   

   
xk = x0.copy()

kk=1
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
BOneProx[kk,:] = xk
# breakpoint()
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
    kk+=1
    
    BOneProx[kk,:] = xk


BOneProx = BOneProx[0:kk,:]

" Intertial DCA "

# Decomposition for iDCA:
# varphi(x) = f1(x) - f2(x)
# f1(x) = ||x||^2 + rho/2 ||x||^2
# f2(x) = 1/2||x||^2 + ||x||_1 + sum_{j=1}^q ( ... ) + ||x-(q+1)e||1

xk = x0.copy()
rho = 1
igam = .999*rho/2

kk = 1
xkold = xk.copy()
xprev = xkold.copy()
temp1 = xk + subh0(xk) + igam/rho*(xk-xprev) #argument prox
xkn = rho*temp1/(rho+2)  # evaluation proc
xk = xkn.copy()
iDCA[kk,:] = xk
while  norm(xkold-xk)/n > prec:
    xprev = xkold.copy()
    xkold = xk.copy()
    temp1 = xk + subh0(xk) + igam/rho*(xk-xprev) #argument prox
    xkn = rho*temp1/(rho+2)  # evaluation proc
    xk = xkn.copy()
    kk+=1
    
    iDCA[kk,:] = xk
    
iDCA = iDCA[0:kk,:]

" BDSA no linesearch "

xk = x0.copy()
gam_min_l1 = 0.9

kk= 1 
xkold  =  xk.copy()
temp1 = xk - gam_min_l1*(2*xk + subh_minus_l1(xk)) 
xkn = prox_minus_l1(temp1, gam_min_l1)
xk = xkn.copy()
BDSA[kk,:] = xk

while norm(xkold-xk)/n > prec:
    xkold = xk.copy() 
    temp1 = xk- gam_min_l1*(2*xk + subh_minus_l1(xk)) 
    xkn = prox_minus_l1(temp1, gam_min_l1)
    xk = xkn.copy() 
    kk += 1
    
    BDSA[kk,:] = xk

BDSA = BDSA[0:kk,:]


" BDSA linesearch"

xk = x0.copy()
gam_min_l1 = 0.9

alph = 0.5
barlam0 = 2
barlamk = barlam0
#eta = -1/gamma
N = 2

kk= 1
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
BDSA_ls[kk,:] = xk

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
    kk += 1
    
    BDSA_ls[kk,:] = xk

BDSA_ls = BDSA_ls[0:kk,:]

"PLOT"    

xpoints = np.linspace(xmin,xmax,600)

# List of points in y axis
ypoints     = np.linspace(ymin,ymax,600)

X, Y = np.meshgrid(xpoints,ypoints)
Z = varphiplot(X,Y)

plt.figure()
plt.contour(X, Y, Z, 100, cmap='rainbow',linewidths=.8)
plt.axis([xmin,xmax,ymin,ymax])
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()



linea=1.2
mlinea=.8
plt.plot(BDSA[:,0],BDSA[:,1],'s-', markersize = 3, color='C2',label='DSA',lw=linea,mew=mlinea)

plt.plot(OneProx[:,0],OneProx[:,1],'x-', markersize = 6, color='k',label='PDCA',lw=linea,mew=mlinea)
plt.plot(DPGA[:,0],DPGA[:,1],'.-',markersize = 6,color='C5',label='DGA',lw=linea,mew=mlinea)
plt.plot(iDCA[:,0],iDCA[:,1],'d-',markersize = 4,color = 'C0', label = 'iDCA',zorder=2,lw=linea,mew=mlinea)
plt.plot(BDSA_ls[:,0],BDSA_ls[:,1],'s--', markersize = 4, color=[0.3,0.3,0.3],markerfacecolor='None',label='BDSA',lw=linea,mew=mlinea)
plt.plot(BOneProx[:,0],BOneProx[:,1],'X--',markersize = 5,markerfacecolor='None', color='C1',label='BPDCA',zorder=3,lw=linea,mew=mlinea)

plt.plot(BDPGA[:,0],BDPGA[:,1],'.--', markersize = 6, color='C3',markerfacecolor='none',label='BDGA',lw=linea,mew=mlinea)

plt.legend(loc = 'upper left',ncol=2)


plt.savefig('Counterexample.pdf',bbox_inches='tight',dpi=400)
plt.show() 

