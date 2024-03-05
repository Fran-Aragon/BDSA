# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:00:25 2024

Authors: Francisco J. Aragón-Artacho, Pedro Pérez-Aros, David Torregrosa-Belén

Code associated with the paper:

F.J. Aragón-Artacho, P. Pérez-Aros, D. Torregrosa-Belén: 
The Boosted Double-proximal Subgradient Algorithm for nonconvex optimization.
(https://arxiv.org/abs/2306.17144)

#####################################################
"Section: Avoiding Non-Optimal Critical Points"

 For generating  the plots in Figure 3

#####################################################

"""



from numpy.linalg import norm
from numpy.random import seed
from matplotlib import pyplot as plt
import numpy as np

# seed(4)

n=2

" Auxiliary functions"

def phi(x1,x2):
    g = x1**2 + x2**2
    f =   - abs(x1)-abs(x2) - np.log(2+np.exp(2*x1)) - np.log(2+np.exp(2*x2))
    return g + f

def varphi(x):
    x1 = x[0]
    x2 = x[1]
    return phi(x1,x2)
    
vphi = np.vectorize(phi)


def grad_log_e(x):
    return 2*np.exp(2*x)/(2+np.exp(2*x))


def subl1_neg(x):
    p = -np.sign(x)
    p[p==0] = -1 
    return p

def sign0(x):
    p = np.sign(x)
    p[x==0] = 1
    return p
        
def prox_minus_l1(x,gamma):
    return  x + gamma*sign0(x)
        
        
" Algorithms:"

def inertial_proximal_linearized(x0,igam,rho,prec):
    
    output = {}
    
    iDCA = np.zeros([1000,2])
    iDCA[0,:] = x0
    
    kk= 1
    xkold = x0.copy()
    xprev = xkold.copy()
    xk = x0.copy()
    iDCA[kk,:] = xk
    while kk==1 or norm(xkold-xk)/n > prec:
        
        xprev = xkold.copy()
        xkold = xk.copy()
        
        subh = grad_log_e(xk) + np.sign(xk)
        temp1 = xk + subh  + igam/rho*(xk-xprev) 
        xkn = rho*temp1/(rho+2)
        xk = xkn.copy()
        kk+=1
        iDCA[kk,:] = xk
        
    
    iDCA = iDCA[0:kk,:]

    output['iDCA'] = iDCA
    output['iter'] = kk
    return output

def BDSA(x0,gamma,prec,alph=.5,barlam0=2,barlamk=2,N=2):
    output = {}
    
    BDSA = np.zeros([1000,2])
    BDSA[0,:] = x0
    
    kk = 1
    xkold = x0.copy()
    xk = x0.copy()
    BDSA[kk,:] = xk
    
    while kk == 1 or norm(xkold-xk)/n>prec:
        xkold = xk.copy()

        subh = 2*xk-grad_log_e(xk) 
        temp1 = xk-gamma*subh
        hatxk = prox_minus_l1(temp1, gamma)
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
        BDSA[kk,:] = xk 
    BDSA = BDSA[0:kk,:]

    
    output['BDSA'] = BDSA
    output['iter'] = kk
    
    return output

def BPDCA(x0,gamma,prec,alph=.5,barlam0=2,barlamk=2,N=2):
    output = {}
    
    BPDCA = np.zeros([1000,2])
    BPDCA[0,:] = x0
    
    kk = 1
    xkold = x0.copy()
    xk = x0.copy()
    BPDCA[kk,:] = xk
    
    while kk == 1 or norm(xkold-xk)/n>prec:
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
        BPDCA[kk,:] = xk 
    BPDCA = BPDCA[0:kk,:]
    
    output['BPDCA'] = BPDCA
    output['iter'] = kk
    
    return output


" Main experiments:"

prec = 1e-6
initial_points = np.array([[-.6,-1.3],[.5,-1.4]]) #[-.9,-2], 



" BDSA gamma=...."

gam_BDSA = 0.49

" Inertial DCA igam=..."

rho = 1
igam = 0.999*rho/2

" BPDCA gamma=...."

gam_BPDCA = 1


xmin,xmax=-2.5,3.5
ymin,ymax=-2.5,3.5


xpoints = np.linspace(xmin,xmax,100)

ypoints     = np.linspace(ymin,ymax,100)

X, Y = np.meshgrid(xpoints,ypoints)
Z = vphi(X,Y)

linea=1.2
mlinea=.8


    

for ll in range(2):
    
    # Contour plot

    plt.figure()
    plt.contour(X, Y, Z, 100, cmap='rainbow',linewidths=.8)
    plt.axis([xmin,xmax,ymin,ymax])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar()
    
    # Algorithms
    
    "BDSA"
   
    
    x0 = initial_points[ll]
    # x0=np.random.rand(2)*6-2.5
    
    outputBDSA = BDSA(x0,gam_BDSA,prec,alph=.5,barlam0=2,barlamk=2,N=2)
    

    BDSAsol = outputBDSA['BDSA']
    
    plt.plot(BDSAsol[1:3,0],BDSAsol[1:3,1],'--', color=[0.3,0.3,0.3],lw=linea,mew=mlinea)
    plt.plot(BDSAsol[2:,0],BDSAsol[2:,1],'s--', markersize = 4, color=[0.3,0.3,0.3],markerfacecolor='None',label='BDSA',lw=linea,mew=mlinea)
  

    "BPDCA"
    
    outputBPDCA = BPDCA(x0,gam_BPDCA,prec,alph=.5,barlam0=2,barlamk=2,N=2)
    
    BPDCAsol = outputBPDCA['BPDCA']
    
    plt.plot(BPDCAsol[1:3,0],BPDCAsol[1:3,1],'--',markersize = 5,markerfacecolor='None', color='C1',zorder=3,lw=linea,mew=mlinea)
    plt.plot(BPDCAsol[2:,0],BPDCAsol[2:,1],'X--',markersize = 5,markerfacecolor='None', color='C1',label='BPDCA',zorder=3,lw=linea,mew=mlinea)
    

    "iDCA"
    
    outputiDCA = inertial_proximal_linearized(x0,igam,rho,prec)


    iDCAsol = outputiDCA['iDCA']
    
    plt.plot(iDCAsol[1:3,0],iDCAsol[1:3,1],'-',markersize = 4,color = 'C0', zorder=2,lw=linea,mew=mlinea)
    plt.plot(iDCAsol[2:,0],iDCAsol[2:,1],'d-',markersize = 4,color = 'C0', label = 'iDCA',zorder=2,lw=linea,mew=mlinea)
  
    plt.plot(x0[0],x0[1],'ko',ms=4,zorder=4)

    plt.legend(loc = 'upper left', ncol=1)


    plt.savefig('Log_e_example_'+str(ll)+'.pdf',bbox_inches='tight',dpi=400)
    plt.show() 







