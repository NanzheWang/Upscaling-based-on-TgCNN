# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:11:49 2021

@author: WNZ
"""

# %%几个reshape和transpose是关键，matlab和python里的转置和重构貌似不一致

from scipy.sparse import spdiags
import numpy as np

def fun_add_bound(x):
    xx=np.hstack((x[:,0:1],x,x[:,-1:]))
    xxx=np.vstack((xx[0:1,:],xx,xx[-1:,:]))
    return xxx


def fun_P5_periodic(Kx,Ky,Kz,Nx,Ny,Nz,hx,hy,hz):

    K=np.ones((3,Nx+2,Ny+2,Nz))
    K[0:1,:,:,:]=fun_add_bound(Kx).reshape(1,Nx+2,Ny+2,Nz);
    K[1:2,:,:,:]=fun_add_bound(Ky).reshape(1,Nx+2,Ny+2,Nz);
    K[2:3,:,:,:]=fun_add_bound(Kz).reshape(1,Nx+2,Ny+2,Nz);
    
    #  Compute K
    KM = K
     
    Nx=Nx+2
    Ny=Ny+2
    N=Nx*Ny*Nz
    
    L = 1/KM
    tx = 2*hy*hz/hx
    TX = np.zeros((Nx+1,Ny,Nz))
    
    ty = 2*hx*hz/hy
    TY = np.zeros((Nx,Ny+1,Nz))
    
    tz = 2*hx*hy/hz
    TZ = np.zeros((Nx,Ny,Nz+1))
    
    TX[1:Nx,:,:] = tx/(L[0,0:Nx-1,:,:]+L[0,1:Nx,:,:])
    TY[:,1:Ny,:] = ty/(L[1,:,0:Ny-1,:]+L[1,:,1:Ny,:])
    TZ[:,:,1:Nz] = tz/(L[2,:,:,0:Nz-1]+L[2,:,:,1:Nz])
    
    # %% updated
    TX[1,:,:] = TX[1,:,:]*2
    TX[Nx-1,:,:] = TX[Nx-1,:,:]*2
    TY[:,1,:] = TY[:,1,:]*2
    TY[:,Ny-1,:] = TY[:,Ny-1,:]*2
    
    # % Assemble TPFA discretization matrix.
    x1 = TX[0:Nx,:,:].transpose().reshape(1,N)
    x2 = TX[1:,:,:].transpose().reshape(1,N)
    y1 = TY[:,0:Ny,:].transpose().reshape(1,N)
    y2 = TY[:,1:,:].transpose().reshape(1,N)
    z1 = TZ[:,:,0:Nz].transpose().reshape(1,N)
    z2 = TZ[:,:,1:].transpose().reshape(1,N)
    DiagVecs = np.vstack((-z2,-y2,-x2,x1+x2+y1+y2+z1+z2,-x1,-y1,-z1))
    DiagIndx = np.array([-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny])
    A = spdiags(DiagVecs,DiagIndx,N,N)

    b=np.zeros((N,1))
    
    A =A.toarray()
    # % change boundaries
    # % 4 corners useless
    for i in [0, Nx-1, Nx*(Ny-1), Nx*Ny-1]:
        A[i,:]=0
        A[i,i]=1
        b[i,0]=0

    
    # % 4 sides
    # % top bottom same p
    for i in range(1,Nx-1):
        A[i,:]=0
        A[i,i]=1
        A[i,i+Nx*(Ny-1)]=-1
    
    # % top bottom same v
    for i in range(1,Nx-1):
        A[i+Nx*(Ny-1),:]=0
        A[i+Nx*(Ny-1),i]=-TY[i,1]
        A[i+Nx*(Ny-1),i+Nx]=TY[i,1]
        A[i+Nx*(Ny-1),i+Nx*(Ny-2)]=TY[i,Ny-1]
        A[i+Nx*(Ny-1),i+Nx*(Ny-1)]=-TY[i,Ny-1]

    # % left right dp=1
    for i in range(1,Ny-1):
        A[i*Nx,:]=0
        A[i*Nx,i*Nx]=1
        A[i*Nx,i*Nx+Nx-1]=-1
        b[i*Nx,0]=1
    
    # % left right same v
    for i in range(1,Ny-1):
        A[i*Nx+Nx-1,:]=0
        A[i*Nx+Nx-1,i*Nx]=-TX[1,i]
        A[i*Nx+Nx-1,i*Nx+1]=TX[1,i]
        A[i*Nx+Nx-1,i*Nx+Nx-2]=TX[Nx-1,i]
        A[i*Nx+Nx-1,i*Nx+Nx-1]=-TX[Nx-1,i]
 
    
   
    
    # % solve sigular problem
    i=Nx+1
    A[i,:]=0
    A[i,i]=1
    b[i,0]=0.95 # % any real value is ok
    
    # % Solve linear system and extract interface fluxes.
    x = np.linalg.inv(A).dot(b)
    P = x.reshape(Nx,Ny,Nz).transpose(1,0,2)
    
    Vx = np.zeros((Nx+1,Ny,Nz))
    Vy = np.zeros((Nx,Ny+1,Nz))
    Vz = np.zeros((Nx,Ny,Nz+1))
    Vx[1:Nx,:,:] = (P[0:Nx-1,:,:]-P[1:Nx,:,:])*TX[1:Nx,:,:]
    Vy[:,1:Ny,:] = (P[:,0:Ny-1,:]-P[:,1:Ny,:])*TY[:,1:Ny,:]
    Vz[:,:,1:Nz] = (P[:,:,0:Nz-1]-P[:,:,1:Nz])*TZ[:,:,1:Nz]
    
    return P[:,:,0],Vx[:,:,0],Vy[:,:,0],Vz[:,:,0]