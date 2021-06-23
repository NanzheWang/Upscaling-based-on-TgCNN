# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:40:18 2021

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
import operator

import sys

from KLE import eigen_value_solution
from KLE import sort_lamda
from KLE import eigen_func
from fun_P5_periodic import fun_P5_periodic
from fun_up import fun_up

from MyConvModel import conv_upscale,conv_upscale_32,conv_upscale_elu

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

#################################################
###Data Processing
#################################################
# bigx=100
# L_x=L_x/bigx
# L_y= L_y/bigx
# eta=eta/bigx
# domain=L_x*L_y
# x=x/bigx
# y=y/bigx
# Ss=Ss*bigx*bigx
# dx=dx/bigx
# dy=dy/bigx

#################################################
# #h的无量纲化
# h_boun1=(h_boun1_0-h_c0)/(h_boun1_0-h_c0)
# h_boun2=(h_boun2_0-h_c0)/(h_boun1_0-h_c0)
# h_c=(h_c0-h_c0)/(h_boun1_0-h_c0)
###########################################
###计算所需特征值个数
#########################################
n_test=50
lamda_x,w_x0,cumulate_lamda_x=eigen_value_solution(eta,L_x,var,n_test)
lamda_y,w_y0,cumulate_lamda_y=eigen_value_solution(eta,L_y,var,n_test)

############################################################
#二维特征值计算，混合，排序，截断
lamda_xy,w_x,w_y,n_eigen,cum_lamda=sort_lamda(lamda_x,w_x0,lamda_y,w_y0,domain,var,weight)


#########################################################
#根据weight获取所需计算特征值个数,并计算特征值以及特征函数值
#################################################
fn_x=[]
fn_y=[]


for i_x in range(nx):
    f_x=eigen_func(n_eigen,w_x,eta,L_x,x[i_x])
    fn_x.append([f_x,x[i_x]])
    
for i_y in range(ny):
    f_y=eigen_func(n_eigen,w_y,eta,L_y,y[i_y])
    fn_y.append([f_y,y[i_y]])

print('特征函数计算完成')




#########################################################
#生成随机数组，生成渗透率场实现
#################################################
seed_n=200
np.random.seed(seed_n)
n_logk=50  #渗透率场实现个数


kesi=np.zeros((n_logk,n_eigen))   #随机数数组
logk=np.zeros((n_logk,nx,ny))       #渗透率场数组

for i_logk in range(n_logk):
    kesi[i_logk,:]=np.random.randn(n_eigen)   #随机数数组
    
    #由随机数计算渗透率场
    for i_x in range(nx):
        for i_y in range(ny):
            logk[i_logk,i_y,i_x]=mean_logk+np.sum(np.sqrt(lamda_xy)*fn_x[i_x][0]*fn_y[i_y][0]*kesi[i_logk:i_logk+1].transpose())



#渗透率场对数转化
k=np.exp(logk)
plt.figure(figsize=(3,3))

mm=plt.imshow(logk[0],origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
            extent=[0, 10, 0,10])


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("$K(x,y)$")
# plt.xlim(0,10)
# plt.ylim(0,10)
plt.colorbar(mm,fraction=0.046, pad=0.04)

# sys.exit(0)

############################################
#分割函数
def patch(k,nx_up,ny_up):
    nk,nx,ny=np.shape(k)
    nux=int(nx/nx_up)
    nuy=int(ny/ny_up)
    n_patch=int(nk*nux*nuy)
    k_patch=np.zeros((n_patch,nx_up,ny_up))
    for m in range(nk):
        for i in range(nux):
            for j in range(nuy):
                k_patch[m*nux*nuy+i*nux+j]=k[m,i*nx_up:(i+1)*nx_up,j*ny_up:(j+1)*ny_up]
    return k_patch

logk_up=patch(logk,nx_up,ny_up)

n_up_logk=int(n_logk*nux*nuy)
logk_image=logk_up.reshape(n_logk,nux*nuy,1,nx_up,ny_up)

#####################################################################################################
#渗透率场写入程序并进行模拟

########################################################################
#数据空间定义
hh=np.zeros((n_up_logk,nx_up+2,ny_up+2))
vx=np.zeros((n_up_logk,nx_up+3,ny_up+2))
vy=np.zeros((n_up_logk,nx_up+2,ny_up+3))
vz=np.zeros((n_up_logk,nx_up+2,ny_up+2))

kx_up=np.zeros((n_logk,int(nx/nx_up),int(ny/ny_up)))
ky_up=np.zeros((n_logk,int(nx/nx_up),int(ny/ny_up)))
kxy_up=np.zeros((n_logk,int(nx/nx_up),int(ny/ny_up)))

start=time.time()
for ik in range(n_logk):
    print('Calculating Realization %d' % (ik))
    for i in range(int(nx/nx_up)):
        for j in range(int(ny/ny_up)):
            kx_up[ik,i,j],ky_up[ik,i,j],kxy_up[ik,i,j]=fun_up(np.exp(logk_up[ik*nux*nuy+i*nux+j]),1,1,\
                                                      dx,dy,dz)
end=time.time()
calculate_time = end  - start              
print('Calculate time: %.4f' % (calculate_time))

# sys.exit(0)
###################################################################################
##储存到二进制文件npy
# np.save('hh_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),hh)
# np.save('vx_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),vx)
# np.save('vy_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),vy)


#2导入数据
#读取二进制文件npy

# hh=np.load('hh_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))
# vx=np.load('vx_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))
# vy=np.load('vy_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))

# sys.exit(0)
#################################################################

# hh_image=hh.reshape(n_up_logk,1,nx_up+2,ny_up+2)



##############################################
#函数定义
def fun_add_bound_numpy(x):
    xx=np.hstack((x[:,0:1],x,x[:,-1:]))
    xxx=np.vstack((xx[0:1,:],xx,xx[-1:,:]))
    return xxx

def fun_add_bound(x):
    xx=torch.cat((x[:,:,:,0:1],x,x[:,:,:,-1:]),3)
    xxx=torch.cat((xx[:,:,0:1,:],xx,xx[:,:,-1:,:]),2)
    return xxx
def fun_add_bound2(x):
    xx=np.concatenate((x[:,:,:,0:1],x,x[:,:,:,-1:]),3)
    xxx=np.concatenate((xx[:,:,0:1,:],xx,xx[:,:,-1:,:]),2)
    return xxx
########################################################################

lamx=2*dy*dz/dx
lamy=2*dx*dz/dy
############################################################################

Net = conv_upscale_elu().to(device)

#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()
#切换工作目录
path = "\\3_model_parameter_results\\"

# 查看当前工作目录
now = os.getcwd()
#print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
#print ("目录修改成功 %s" % new_path)


Net.load_state_dict(torch.load('5_label_free_N_logk=1_epoch=1000_batchsize=100_N_col=5_seed=38_lam1=1_lam2=1_lam3=1_lam4=1_lam5=1_t=158.657.ckpt'))









"""=================模型测试=================="""
Net.eval()

########################################################################

logk_test_image = torch.from_numpy(logk_image)
logk_test_image = logk_test_image.type(torch.FloatTensor)

logk_test_image=logk_test_image.to(device)
error_l2_set=np.empty((n_logk))
R2_set=np.empty((n_logk))

hh_test_pred=np.zeros((n_up_logk,nx_up+2,ny_up+2))
vx_test_pred=np.zeros((nux*nuy,1,nx_up+3,ny_up+2))
vy_test_pred=np.zeros((nux*nuy,1,nx_up+2,ny_up+3))
        
kx_test_pred=np.zeros((n_logk,nux,nuy))
ky_test_pred=np.zeros((n_logk,nux,nuy))
kxy_test_pred=np.zeros((n_logk,nux,nuy))


start_time = time.time()


for ik in range(n_logk):

    h_test_pred=Net(logk_test_image[ik]).cpu().detach().numpy()
    # sys.exit(0)
    # h_test_pred=h_test_pred.reshape(nx_up+2,ny_up+2)
    h_test_pred[:,:,0,0]=0
    h_test_pred[:,:,0,-1]=0
    h_test_pred[:,:,-1,0]=0
    h_test_pred[:,:,-1,-1]=0
    
    kk=fun_add_bound2(np.exp(logk_image[ik]))
    l=1/kk
    # kk=fun_add_bound(torch.exp(logk_test_image[ik]))
    # l=torch.pow(kk,-1).cpu().detach().numpy()
        
    tx=np.zeros((nux*nuy,1,nx_up+3,ny_up+2))
    ty=np.zeros((nux*nuy,1,nx_up+2,ny_up+3))
        
    tx[:,:,1:-1,:]=lamx/(l[:,:,0:-1,:]+l[:,:,1:,:])
    ty[:,:,:,1:-1]=lamy/(l[:,:,:,0:-1]+l[:,:,:,1:])
    tx[:,:,1,:]=tx[:,:,1,:]*2
    tx[:,:,-2,:]=tx[:,:,-2,:]*2
    ty[:,:,:,1]=ty[:,:,:,1]*2
    ty[:,:,:,-2]=ty[:,:,:,-2]*2
    vx_test_pred[:,:,1:-1,:]=(h_test_pred[:,:,0:-1,:]-h_test_pred[:,:,1:,:])*tx[:,:,1:-1,:]
    vy_test_pred[:,:,:,1:-1]=(h_test_pred[:,:,:,0:-1]-h_test_pred[:,:,:,1:])*ty[:,:,:,1:-1]
    
    dp=np.mean(h_test_pred[:,:,0,1:-1]-h_test_pred[:,:,-1,1:-1],2 )
    kxx=(np.sum(vx_test_pred[:,:,1,1:-1],2)/dy/(dp/dx)).reshape(nux,nuy)
    kyx=(np.sum(vy_test_pred[:,:,1:-1,1],2)/dx/(dp/dx)).reshape(nux,nuy)
    
    # sys.exit(0)
    h_test_pred=Net(logk_test_image[ik].transpose(3,2)).cpu().detach().numpy()
    h_test_pred[:,:,0,0]=0
    h_test_pred[:,:,0,-1]=0
    h_test_pred[:,:,-1,0]=0
    h_test_pred[:,:,-1,-1]=0
    
    kk=fun_add_bound2(np.exp(logk_image[ik].transpose(0,1,3,2)))    
    l=1/kk
    # kk=fun_add_bound(torch.exp(logk_test_image[ik].transpose(3,2)))
    # l=torch.pow(kk,-1).cpu().detach().numpy()
        
    tx=np.zeros((nux*nuy,1,nx_up+3,ny_up+2))
    ty=np.zeros((nux*nuy,1,nx_up+2,ny_up+3))
    tx[:,:,1:-1,:]=lamx/(l[:,:,0:-1,:]+l[:,:,1:,:])
    ty[:,:,:,1:-1]=lamy/(l[:,:,:,0:-1]+l[:,:,:,1:])
    tx[:,:,1,:]=tx[:,:,1,:]*2
    tx[:,:,-2,:]=tx[:,:,-2,:]*2
    ty[:,:,:,1]=ty[:,:,:,1]*2
    ty[:,:,:,-2]=ty[:,:,:,-2]*2
    vx_test_pred[:,:,1:-1,:]=(h_test_pred[:,:,0:-1,:]-h_test_pred[:,:,1:,:])*tx[:,:,1:-1,:]
    vy_test_pred[:,:,:,1:-1]=(h_test_pred[:,:,:,0:-1]-h_test_pred[:,:,:,1:])*ty[:,:,:,1:-1]
    dp=np.mean(h_test_pred[:,:,0,1:-1]-h_test_pred[:,:,-1,1:-1],2 )
    kyy=(np.sum(vx_test_pred[:,:,1,1:-1],2)/dy/(dp/dx)).reshape(nux,nuy)
    kxy=(np.sum(vy_test_pred[:,:,1:-1,1],2)/dx/(dp/dx)).reshape(nux,nuy)
    
    kx_test_pred[ik]=kxx
    ky_test_pred[ik]=kyy
    kxy_test_pred[ik]=(kyx+kxy)/2

#    print('Predicting realization %d'%(i+1))

    ##############################################
    #  calculate error
            
    print('Realization %d' % (ik))
    
    error_l2 = np.linalg.norm(kx_up[ik].flatten()-kx_test_pred[ik].flatten(),2)/np.linalg.norm(kx_up[ik].flatten(),2)
    print('Error L2: %e' % (error_l2))
    error_l2_set[ik]=error_l2
    
    R2=1-np.sum((kx_up[ik].flatten()-kx_test_pred[ik].flatten())**2)/np.sum((kx_up[ik].flatten()-kx_up[ik].flatten().mean())**2)
    print('coefficient of determination  R2: %e' % (R2))
    R2_set[ik]=R2


elapsed = time.time() - start_time                
print('Prediction time: %.4f' % (elapsed))

calculate_time = end  - start              
print('Numerical Calculate time: %.4f' % (calculate_time))

L2_mean=np.mean(error_l2_set)
L2_var=np.var(error_l2_set)
R2_mean=np.mean(R2_set)
R2_var=np.var(R2_set)

print('L2 mean:')
print(L2_mean)
print('L2 var:')
print(L2_var)

print('R2 mean:')
print(R2_mean)
print('R2 var:')
print(R2_var)


# sys.exit(0)
##########################################################
####Post Process for data #####
####  Just once  ##########
# x=x*bigx
# y=y*bigx



#######################################################
#     PINN plot
#结果展示    

ind_1=2
ind_2=5
ind_3=8
x_up=np.arange(1,nx_up+1,1)
y_up=np.arange(1,ny_up+1,1)



######################################################################
############################# Plotting_1 #############################
######################################################################   


TgNN_R2_set=R2_set
TgNN_error_l2_set=error_l2_set



num_bins = 15

l2_x_ticks = np.arange(0,0.0016, 0.0003)

plt.figure(figsize=(6,4))
plt.hist(TgNN_error_l2_set, num_bins)
plt.title(r'$Histogram\ \ of\ \  relative\ \ L_2\ \ error$')
#plt.title(r'$Histogram\ \ of\ \  relative\ \ L_2\ \ error\ \ \left(TgNN\right)$')
#plt.xlim(0,0.0015)
# plt.xticks(l2_x_ticks)


num_bins2 = 15
plt.figure(figsize=(6,4))
plt.hist(TgNN_R2_set, num_bins2)
plt.title(r'$Histogram\ \ of\ \  R^2\ \ score$')
#plt.title(r'$Histogram\ \ of\ \  R^2\ \ score\ \ \left(TgNN\right)$')
# plt.xlim(0.9,1)




######################################################################
############################# Plotting_2 #############################
######################################################################   
#统计结果展示    
if n_logk>1:
    n_plot=6
else:
    n_plot=1
for sam1 in range(n_plot):
# for sam1 in plot_n:
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    plt.contourf(logk[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                              extent=[x.min(), x.max(), y.min(),y.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    plt.colorbar()
    
    
    plt.subplot(142)
    plt.contourf(np.log(kx_up[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.contourf(np.log(kx_test_pred[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    # cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.contourf(abs(np.log(kx_test_pred[sam1])-np.log(kx_up[sam1])), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()




# n_plot=6
# for sam1 in range(n_plot):
# # for sam1 in plot_n:
#     plt.figure(figsize=(18,3))
#     plt.subplot(141)
#     plt.imshow(logk_up[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                              extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
#     plt.xlabel('$x$',fontsize=15)
#     plt.ylabel('$y$',fontsize=15)
#     plt.title("ln$K(x,y)$",fontsize=15)
#     plt.colorbar()
    
    
#     plt.subplot(142)
#     plt.imshow(vx_test[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                 extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
#     plt.xlabel('$x$',fontsize=15)
#     #plt.ylabel('y')
#     plt.title('Reference',fontsize=15)
#     cbar=plt.colorbar()
#     cbar.set_ticks(np.linspace(200,202,5))
    
    
#     plt.subplot(143)
#     plt.imshow(vx_test_pred[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                 extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
#     plt.xlabel('$x$',fontsize=15)
#     #plt.ylabel('y')
#     plt.title('Prediction',fontsize=15)
#     cbar=plt.colorbar()
#     cbar.set_ticks(np.linspace(200,202,5))
    
    
#     plt.subplot(144)
#     plt.imshow(abs(vx_test_pred[sam1]-vx_test[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                 extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
#     plt.xlabel('$x$',fontsize=15)
#     #plt.ylabel('y')
#     plt.title('Error',fontsize=15)
#     plt.colorbar()




# #matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()
# k_m=matlab.double(kx_up[0].tolist())
# P,V=eng.fun_MPFA_no_struct(k_m,k_m,k_m,k_m,nx,ny,nz,1,0)
# eng.sum(k_m)
########################################################
#数据保存
#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()

print ("目录修改成功 %s" % retval)
path = "\\2_data\\"

# 查看当前工作目录
now = os.getcwd()
print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
print ("目录修改成功 %s" % new_path)


data1 = 'k_true_n=%d.mat'%(n_logk)
scio.savemat(data1, {'k_true_set':k})

data2 = 'kx_num_n=%d.mat'%(n_logk)
scio.savemat(data2, {'kx_up_set':kx_up})

data3 = 'ky_num_n=%d.mat'%(n_logk)
scio.savemat(data3, {'ky_up_set':ky_up})

data4 = 'kxy_num_n=%d.mat'%(n_logk)
scio.savemat(data4, {'kxy_up_set':kxy_up})

data5 = 'kx_pre_n=%d.mat'%(n_logk)
scio.savemat(data5, {'kx_pre_up_set':kx_test_pred})

data6 = 'ky_pre_n=%d.mat'%(n_logk)
scio.savemat(data6, {'ky_pre_up_set':ky_test_pred})

data7 = 'kxy_pre_n=%d.mat'%(n_logk)
scio.savemat(data7, {'kxy_pre_up_set':kxy_test_pred})




