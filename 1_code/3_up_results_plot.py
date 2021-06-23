# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:36:28 2021

@author: WNZ
"""



import torch
from torch.nn import Linear,Tanh,Sequential,ReLU,Softplus
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as func
import random
import scipy.io as scio

torch.manual_seed(100) 
torch.set_printoptions(precision=8, threshold=None, edgeitems=None, linewidth=None, profile=None)


from pyDOE import lhs
import math
import numpy as np

import matplotlib.pyplot as plt



import matplotlib.gridspec as gridspec


import time

import re
import os
import os.path


import sys


plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率

#########################################################################
#设备设置
device = torch.device('cuda:0')


##################################
#参数设置
L_x= 100    #区域长度
L_y= 100
L_z= 1

nx=100
ny=100
nz=1

nx_up=10
ny_up=10
nz_up=1

nux=int(nx/nx_up)
nuy=int(ny/ny_up)

dx=L_x/nx
dy=L_y/ny
dz=L_z/nz

x=np.arange(1,nx+1,1)
x=x*dx
y=np.arange(1,ny+1,1)
y=y*dy

#渗透率场设置

domain=L_x*L_y
weight=0.9
#渗透率场设置
mean_logk=0
var=1.0
eta=0.2*L_x   #相关长度

n_logk=300
#####################################################################################################
#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()
#################################################################################
#切换工作目录
path = "\\2_data\\"

# 查看当前工作目录
now = os.getcwd()
print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
print ("目录修改成功 %s" % new_path)

#data load
k_true_set = scio.loadmat('k_true_n=%d.mat'%(n_logk))
p_true_set = scio.loadmat('p_true_set_n=%d.mat'%(n_logk))
vx_true_set = scio.loadmat('vx_true_set_n=%d.mat'%(n_logk))
vy_true_set = scio.loadmat('vy_true_set_n=%d.mat'%(n_logk))

p_up_true_set = scio.loadmat('p_up_true_set_n=%d.mat'%(n_logk))
vx_up_true_set = scio.loadmat('vx_up_true_set_n=%d.mat'%(n_logk))
vy_up_true_set = scio.loadmat('vy_up_true_set_n=%d.mat'%(n_logk))

kx_up_set=scio.loadmat('kx_num_n=%d.mat'%(n_logk))
p_up_num_set = scio.loadmat('p_up_num_set_n=%d.mat'%(n_logk))
vx_up_num_set = scio.loadmat('vx_up_num_set_n=%d.mat'%(n_logk))
vy_up_num_set = scio.loadmat('vy_up_num_set_n=%d.mat'%(n_logk))

kx_pre_up_set=scio.loadmat('kx_pre_n=%d.mat'%(n_logk))
p_up_pre_set = scio.loadmat('p_up_pre_set_n=%d.mat'%(n_logk))
vx_up_pre_set = scio.loadmat('vx_up_pre_set_n=%d.mat'%(n_logk))
vy_up_pre_set = scio.loadmat('vy_up_pre_set_n=%d.mat'%(n_logk))


k_true_set_0=np.log(k_true_set['k_true_set'])
p_true_set_0=p_true_set['p_true_set']
vx_true_set_0=vx_true_set['vx_true_set']
vy_true_set_0=vy_true_set['vy_true_set']

p_up_true_set_0=p_up_true_set['p_up_true_set']
vx_up_true_set_0=vx_up_true_set['vx_up_true_set']
vy_up_true_set_0=vy_up_true_set['vy_up_true_set']

kx_up_set_0=kx_up_set['kx_up_set']
p_up_num_set_0=p_up_num_set['p_up_num_set']
vx_up_num_set_0=vx_up_num_set['vx_up_num_set']
vy_up_num_set_0=vy_up_num_set['vy_up_num_set']

kx_pre_up_set_0=kx_pre_up_set['kx_pre_up_set']
p_up_pre_set_0=p_up_pre_set['p_up_pre_set']
vx_up_pre_set_0=vx_up_pre_set['vx_up_pre_set']
vy_up_pre_set_0=vy_up_pre_set['vy_up_pre_set']



"""=================模型测试=================="""

p_l2_set=np.zeros((n_logk))
p_r2_set=np.zeros((n_logk))
vx_l2_set=np.zeros((n_logk))
vx_r2_set=np.zeros((n_logk))
vy_l2_set=np.zeros((n_logk))
vy_r2_set=np.zeros((n_logk))


for ik in range(n_logk):

    ##############################################
    #  calculate error
            
    print('Realization %d' % (ik))
    
    error_l2 = np.linalg.norm(p_up_true_set_0[ik].flatten()-p_up_pre_set_0[ik].flatten(),2)/np.linalg.norm(p_up_true_set_0[ik].flatten(),2)
    print('p Error L2: %e' % (error_l2))
    p_l2_set[ik]=error_l2    
    R2=1-np.sum((p_up_true_set_0[ik].flatten()-p_up_pre_set_0[ik].flatten())**2)/np.sum((p_up_true_set_0[ik].flatten()-p_up_true_set_0[ik].flatten().mean())**2)
    print('p coefficient of determination  R2: %e' % (R2))
    p_r2_set[ik]=R2

    error_l2 = np.linalg.norm(vx_up_true_set_0[ik].flatten()-vx_up_pre_set_0[ik].flatten(),2)/np.linalg.norm(vx_up_true_set_0[ik].flatten(),2)
    print('vx Error L2: %e' % (error_l2))
    vx_l2_set[ik]=error_l2    
    R2=1-np.sum((vx_up_true_set_0[ik].flatten()-vx_up_pre_set_0[ik].flatten())**2)/np.sum((vx_up_true_set_0[ik].flatten()-vx_up_true_set_0[ik].flatten().mean())**2)
    print('vx coefficient of determination  R2: %e' % (R2))
    vx_r2_set[ik]=R2

    error_l2 = np.linalg.norm(vy_up_true_set_0[ik].flatten()-vy_up_pre_set_0[ik].flatten(),2)/np.linalg.norm(vy_up_true_set_0[ik].flatten(),2)
    print('vy Error L2: %e' % (error_l2))
    vy_l2_set[ik]=error_l2    
    R2=1-np.sum((vy_up_true_set_0[ik].flatten()-vy_up_pre_set_0[ik].flatten())**2)/np.sum((vy_up_true_set_0[ik].flatten()-vy_up_true_set_0[ik].flatten().mean())**2)
    print('vy coefficient of determination  R2: %e' % (R2))
    vy_r2_set[ik]=R2
    

p_l2_mean=np.mean(p_l2_set)
p_l2_var=np.var(p_l2_set)
p_r2_mean=np.mean(p_r2_set)
p_r2_var=np.var(p_r2_set)

vx_l2_mean=np.mean(vx_l2_set)
vx_l2_var=np.var(vx_l2_set)
vx_r2_mean=np.mean(vx_r2_set)
vx_r2_var=np.var(vx_r2_set)

vy_l2_mean=np.mean(vy_l2_set)
vy_l2_var=np.var(vy_l2_set)
vy_r2_mean=np.mean(vy_r2_set)
vy_r2_var=np.var(vy_r2_set)

print('p L2 mean:')
print(p_l2_mean)
print('p L2 var:')
print(p_l2_var)

print('p R2 mean:')
print(p_r2_mean)
print('p R2 var:')
print(p_r2_var)

print('vx L2 mean:')
print(vx_l2_mean)
print('vx L2 var:')
print(vx_l2_var)

print('vx R2 mean:')
print(vx_r2_mean)
print('vx R2 var:')
print(vx_r2_var)


print('vy L2 mean:')
print(vy_l2_mean)
print('vy L2 var:')
print(vy_l2_var)

print('vy R2 mean:')
print(vy_r2_mean)
print('vy R2 var:')
print(vy_r2_var)

##########################################################
x_up=np.arange(1,nx_up+1,1)
y_up=np.arange(1,ny_up+1,1)

######################################################################
############################# Plotting_1 #############################
######################################################################   



num_bins = 15
plt.figure(figsize=(5,3))
plt.hist(p_r2_set, num_bins,fill=False, hatch='///',edgecolor="red")
plt.xlabel('$R^2$ score',fontsize=15)
# plt.xlim(0.95,1)
# # plt.legend()
# # plt.xticks(l2_x_ticks)

num_bins = 15
plt.figure(figsize=(5,3))
plt.hist(vx_r2_set, num_bins,fill=False, hatch='///',edgecolor="red")
plt.xlabel('$R^2$ score',fontsize=15)
# plt.xlim(0.95,1)
# # plt.legend()
# # plt.xticks(l2_x_ticks)

num_bins = 15
plt.figure(figsize=(5,3))
plt.hist(vy_r2_set, num_bins,fill=False, hatch='///',edgecolor="red")
plt.xlabel('$R^2$ score',fontsize=15)
# plt.xlim(0.95,1)
# # plt.legend()
# # plt.xticks(l2_x_ticks)



# num_bins2 = 15
# plt.figure(figsize=(6,4))
# plt.hist(TgNN_R2_set, num_bins2)
# plt.title(r'$Histogram\ \ of\ \  R^2\ \ score$')
#plt.title(r'$Histogram\ \ of\ \  R^2\ \ score\ \ \left(TgNN\right)$')
# plt.xlim(0.9,1)




######################################################################
############################# Plotting_2 #############################
######################################################################   
#统计结果展示    
if n_logk>1:
    n_plot=2
else:
    n_plot=1
for sam1 in range(n_plot):
# for sam1 in plot_n:
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    plt.contourf(k_true_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                              extent=[x.min(), x.max(), y.min(),y.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    cbar=plt.colorbar()
    cbar.set_clim( -3, 2 )
    
# plt.figure(figsize=(18,3))
# plt.subplot(141)
# mm=plt.contourf(k_true_set_0[sam1,0:25,-25:],levels=np.linspace(-3,2,11), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                               extent=[x[-25:].min(), x[-25:].max(), y[-25:].min(),y[-25:].max()],vmin=-3,vmax=2)
# plt.xlabel('$x$',fontsize=15)
# plt.ylabel('$y$',fontsize=15)
# plt.title("ln$K(x,y)$",fontsize=15)
# cbar=plt.colorbar(mm)
# cbar.set_clim( -3, 2 )

    
    plt.subplot(142)
    plt.contourf(p_true_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.contourf(vx_true_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.contourf(vy_true_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()

#########################################
#numerical 
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    # plt.contourf(np.log(kx_up_set_0[sam1]),10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
    #                           extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.imshow(np.log(kx_up_set_0[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                              extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()],vmin=-3,vmax=2)
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    plt.colorbar()
    
    
    plt.subplot(142)
    plt.contourf(p_up_num_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.contourf(vx_up_num_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.contourf(vy_up_num_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()
#########################################
#deep learning
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    plt.contourf(np.log(kx_pre_up_set_0[sam1]),10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                              extent=[x.min(), x.max(), y.min(),y.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    plt.colorbar()
    
    
    plt.subplot(142)
    plt.contourf(p_up_pre_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.contourf(vx_up_pre_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.contourf(vy_up_pre_set_0[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()


x_id=5
y_id=5
cor_ticks3=np.arange(-0.2,0.3,0.1)
id_plot=[0]
for sam1 in id_plot:

    fig=plt.figure(figsize=(13.8,13))
    plt.subplot(331)
    plt.title("Upscaled h",fontsize=13)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_true_set_0[:,x_id,y_id],p_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')       
    plt.xlabel('True',fontsize=13)
    plt.ylabel('Numerical',fontsize=13)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)

    
    plt.subplot(332)
    plt.title("Upscaled vx",fontsize=13)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_true_set_0[:,x_id,y_id],vx_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')        
    plt.xlabel('True',fontsize=13)
    plt.ylabel('Numerical',fontsize=13)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)


    
    plt.subplot(333)
    plt.title("Upscaled vy",fontsize=13)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_true_set_0[:,x_id,y_id],vy_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')     
    plt.xlabel('True',fontsize=13)
    plt.ylabel('Numerical',fontsize=13)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)     
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)

 
    plt.subplot(334)
    # plt.title("Upscaled h",fontsize=13)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_true_set_0[:,x_id,y_id],p_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')      
    plt.xlabel('True',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
 
    
    plt.subplot(335)
    # plt.title("Upscaled vx",fontsize=13)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_true_set_0[:,x_id,y_id],vx_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')       
    plt.xlabel('True',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    fig.subplots_adjust(right=0.9)

   
    
    plt.subplot(336)
    # plt.title("Upscaled vy",fontsize=13)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_true_set_0[:,x_id,y_id],vy_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')        
    plt.xlabel('True',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)     
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)


    cor_ticks=np.arange(400,1150,200)
    plt.subplot(337)
    # plt.title("Upscaled h",fontsize=13)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_num_set_0[:,x_id,y_id],p_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')       
    plt.xlabel('Numerical',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
 
    
    plt.subplot(338)
    # plt.title("Upscaled vx",fontsize=13)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_num_set_0[:,x_id,y_id],vx_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')     
    plt.xlabel('Numerical',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    
    
    plt.subplot(339)
    # plt.title("Upscaled vy",fontsize=13)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_num_set_0[:,x_id,y_id],vy_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')   
    plt.xlabel('Numerical',fontsize=13)
    plt.ylabel('TgCNN',fontsize=13)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)    
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    
  




x_id=5
y_id=5
cor_ticks3=np.arange(-0.2,0.3,0.1)
id_plot=[0]
for sam1 in id_plot:

    fig=plt.figure(figsize=(13.8,13))
    plt.subplot(331)
    plt.title("Upscaled h",fontsize=15)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_true_set_0[:,x_id,y_id],p_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')       
    plt.xlabel('True',fontsize=15)
    plt.ylabel('Numerical',fontsize=15)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)

    
    plt.subplot(332)
    plt.title("Upscaled vx",fontsize=15)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_true_set_0[:,x_id,y_id],vx_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')        
    plt.xlabel('True',fontsize=15)
    plt.ylabel('Numerical',fontsize=15)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)


    
    plt.subplot(333)
    plt.title("Upscaled vy",fontsize=15)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_true_set_0[:,x_id,y_id],vy_up_num_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='b',label='Point 1')     
    plt.xlabel('True',fontsize=15)
    plt.ylabel('Numerical',fontsize=15)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)     
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)

 
    plt.subplot(334)
    # plt.title("Upscaled h",fontsize=13)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_true_set_0[:,x_id,y_id],p_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')      
    plt.xlabel('True',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
 
    
    plt.subplot(335)
    # plt.title("Upscaled vx",fontsize=13)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_true_set_0[:,x_id,y_id],vx_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')       
    plt.xlabel('True',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    fig.subplots_adjust(right=0.9)

   
    
    plt.subplot(336)
    # plt.title("Upscaled vy",fontsize=13)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_true_set_0[:,x_id,y_id],vy_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='r',label='Point 1')        
    plt.xlabel('True',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)     
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)


    cor_ticks=np.arange(400,1150,200)
    plt.subplot(337)
    # plt.title("Upscaled h",fontsize=13)
    plt.plot([0,1],[0,1],'k-',linewidth=2)
    plt.scatter(p_up_num_set_0[:,x_id,y_id],p_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')       
    plt.xlabel('Numerical',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
 
    
    plt.subplot(338)
    # plt.title("Upscaled vx",fontsize=13)
    plt.plot([-0.2,2],[-0.2,2],'k-',linewidth=2)
    plt.scatter(vx_up_num_set_0[:,x_id,y_id],vx_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')     
    plt.xlabel('Numerical',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(-0.1,1)
    plt.ylim(-0.1,1)
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    
    
    plt.subplot(339)
    # plt.title("Upscaled vy",fontsize=13)
    plt.plot([-0.25,2],[-0.25,2],'k-',linewidth=2)
    plt.scatter(vy_up_num_set_0[:,x_id,y_id],vy_up_pre_set_0[:,x_id,y_id],marker='o',c='',s=20,edgecolors='lime',label='Point 1')   
    plt.xlabel('Numerical',fontsize=15)
    plt.ylabel('TgCNN',fontsize=15)
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xticks(cor_ticks3)
    plt.yticks(cor_ticks3)    
    # plt.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.3)
    
  







