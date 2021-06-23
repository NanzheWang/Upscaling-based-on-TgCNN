# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:21:38 2021

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


torch.manual_seed(100) 


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
seed_n=38
np.random.seed(seed_n)
n_logk=1  #渗透率场实现个数


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
    
# for i in range(-10,-1):
#     plt.figure(figsize=(3,3))
    
#     mm=plt.imshow(logk_up[i],origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#                 extent=[0, 10, 0,10])
    
    
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.title("$K(x,y)$")
#     # plt.xlim(0,10)
#     # plt.ylim(0,10)
#     plt.colorbar(mm,fraction=0.046, pad=0.04)

n_up_logk=int(n_logk*(nx/nx_up)*(ny/ny_up))
logk_image=logk_up.reshape(n_up_logk,1,nx_up,ny_up)

#####################################################################################################
#渗透率场写入程序并进行模拟

########################################################################
#数据空间定义
hh=np.zeros((n_up_logk,nx_up+2,ny_up+2))
vx=np.zeros((n_up_logk,nx_up+3,ny_up+2))
vy=np.zeros((n_up_logk,nx_up+2,ny_up+3))
vz=np.zeros((n_up_logk,nx_up+2,ny_up+2))
start=time.time()
for i in range(n_up_logk):
    hh[i],vx[i],vy[i],vz[i]=fun_P5_periodic(np.exp(logk_up[i]),np.exp(logk_up[i]),\
                          np.exp(logk_up[i]),nx_up,ny_up,nz_up,dx,dy,dz)
end=time.time()
calculate_time = end  - start              
print('Calculate time: %.4f' % (calculate_time))

# sys.exit(0)
###################################################################################
##储存到二进制文件npy
np.save('hh_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),hh)
np.save('vx_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),vx)
np.save('vy_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n),vy)


#2导入数据
#读取二进制文件npy

hh=np.load('hh_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))
vx_test=np.load('vx_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))
vy_test=np.load('vy_data_Nk=%d_nx_up=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk,nx_up,eta,weight,seed_n))

# sys.exit(0)
#################################################################
###################################################################
#虚拟实现生成
seed_v=50
np.random.seed(seed_v)
n_logk_v=5  #渗透率场实现个数


kesi_v=np.zeros((n_logk_v,n_eigen))   #随机数数组
logk_v=np.zeros((n_logk_v,nx,ny))       #渗透率场数组


for i_logk in range(n_logk_v):
    kesi_v[i_logk,:]=np.random.randn(n_eigen)   #随机数数组
    
    #由随机数计算渗透率场
    for i_x in range(nx):
        for i_y in range(ny):
            logk_v[i_logk,i_y,i_x]=mean_logk+np.sum(np.sqrt(lamda_xy)*fn_x[i_x][0]*fn_y[i_y][0]*kesi_v[i_logk:i_logk+1].transpose())


#渗透率场对数转化
k_v=np.exp(logk_v)

logk_up_v=patch(logk_v,nx_up,ny_up)
    
n_up_logk_v=int(n_logk_v*(nx/nx_up)*(ny/ny_up))
logk_image_v=logk_up_v.reshape(n_up_logk_v,1,nx_up,ny_up)



##############################################
#函数定义
def cut_boun(u,nb):
    u_cut=u[:,:,nb:-nb,nb:-nb]
    return u_cut

def diff_h(u,dx):
    u_l=u[:,:,:,0:-2]
    u_c=u[:,:,:,1:-1]
    u_r=u[:,:,:,2:]
    
    diff_u_left=(u_c-u_l)/dx
    diff_u_right=(u_r-u_c)/dx
    return diff_u_left,diff_u_right

def diff_v(u,dy):
    u_l=u[:,:,0:-2,:]
    u_c=u[:,:,1:-1,:]
    u_u=u[:,:,2:,:]
    
    diff_u_low=(u_c-u_l)/dx
    diff_u_up=(u_u-u_c)/dx
    return diff_u_low,diff_u_up

def harmonic_mean_h(k):
    k_l=k[:,:,:,0:-2]
    k_c=k[:,:,:,1:-1]
    k_r=k[:,:,:,2:]
    
    mean_k_left=2*k_c*k_l/(k_c+k_l)
    mean_k_right=2*k_c*k_r/(k_c+k_r)   
    
    return mean_k_left,mean_k_right

def harmonic_mean_v(k):
    k_l=k[:,:,0:-2,:]
    k_c=k[:,:,1:-1,:]
    k_u=k[:,:,2:,:]
    
    mean_k_low=2*k_c*k_l/(k_c+k_l)
    mean_k_up=2*k_c*k_u/(k_c+k_u)   
    
    return mean_k_low,mean_k_up

def fun_add_bound(x):
    xx=torch.cat((x[:,:,:,0:1],x,x[:,:,:,-1:]),3)
    xxx=torch.cat((xx[:,:,0:1,:],xx,xx[:,:,-1:,:]),2)
    return xxx

def fun_add_bound_numpy(x):
    xx=np.hstack((x[:,0:1],x,x[:,-1:]))
    xxx=np.vstack((xx[0:1,:],xx,xx[-1:,:]))
    return xxx


########################################################################
#########################################################################
#提取训练数据

logk_image_v_col=logk_image_v.copy()
np.random.shuffle(logk_image_v_col)
logk_train_v = torch.from_numpy(logk_image_v_col)
logk_train_v = logk_train_v.type(torch.FloatTensor)

n_col=len(logk_image_v_col)

############################################################################

Net = conv_upscale_elu().to(device)

"""=================训练神经网络=================="""
#训练数据分批处理
BATCH_SIZE = 100      # 批训练的数据个数
N_batch=math.ceil(n_col/BATCH_SIZE )
k_train_v_set=[]


for i_batch in range(int(N_batch)):
    k_train_v_set.append(logk_train_v[BATCH_SIZE*i_batch:BATCH_SIZE*(i_batch+1)])

# 定义神经网络优化器
LR=0.001

optimizer=torch.optim.Adam([
    {'params': Net.parameters()},   
# ],lr=LR,weight_decay=0.005)
],lr=LR)    

lr_list=[]

#定义loss数组
loss_set=[]
f1_set=[]
f2_set=[]  
f3_set=[]  
f4_set=[]  
f5_set=[] 
f6_set=[] 

lam1=1
lam2=1
lam3=1
lam4=1
lam5=1
lam6=1

lamx=2*dy*dz/dx
lamy=2*dx*dz/dy

num_epoch=1000

start_time = time.time()  
##########################################################################
##分批训练
for epoch in range(num_epoch):   # 训练所有!整套!数据 3 次
    for ite in range(N_batch):
        # batch_x=t_train_set[ite].to(device)
        # batch_y=h_train_set[ite].to(device)
        
        batch_x_v=k_train_v_set[ite].to(device)
        n_bat=len(batch_x_v)
        optimizer.zero_grad()
        # pred_f=Net(batch_x)
        pred_f_v=Net(batch_x_v)
        
        
        kk=fun_add_bound(torch.exp(batch_x_v))
        l=torch.pow(kk,-1)
        
        tx=torch.zeros((n_bat,1,nx_up+3,ny_up+2)).to(device)
        ty=torch.zeros((n_bat,1,nx_up+2,ny_up+3)).to(device)
        vx=torch.zeros((n_bat,1,nx_up+3,ny_up+2)).to(device)
        vy=torch.zeros((n_bat,1,nx_up+2,ny_up+3)).to(device)
        
        tx[:,:,1:-1,:]=lamx/(l[:,:,0:-1,:]+l[:,:,1:,:])
        ty[:,:,:,1:-1]=lamy/(l[:,:,:,0:-1]+l[:,:,:,1:])
        tx[:,:,1,:]=tx[:,:,1,:]*2
        tx[:,:,-2,:]=tx[:,:,-2,:]*2
        ty[:,:,:,1]=ty[:,:,:,1]*2
        ty[:,:,:,-2]=ty[:,:,:,-2]*2
        
        v_dp=pred_f_v[:,:,:,0]-pred_f_v[:,:,:,-1]
        h_dp=pred_f_v[:,:,0,:]-pred_f_v[:,:,-1,:]
           
        vx[:,:,1:-1,:]=(pred_f_v[:,:,0:-1,:]-pred_f_v[:,:,1:,:])*tx[:,:,1:-1,:]
        vxl=-vx[:,:,1:-2,1:-1]
        vxr=-vx[:,:,2:-1,1:-1]
        vy[:,:,:,1:-1]=(pred_f_v[:,:,:,0:-1]-pred_f_v[:,:,:,1:])*ty[:,:,:,1:-1]
        vyl=-vy[:,:,1:-1,1:-2]
        vyr=-vy[:,:,1:-1,2:-1]
        residual=(vxr-vxl)+(vyr-vyl)
        # f1=torch.pow((pred_f-batch_y),2).mean()
        f2=torch.pow((residual),2).mean()
        
        f3=torch.pow((v_dp-0),2).mean()+\
            torch.pow((h_dp-1),2).mean()
            
        f4=torch.pow((vx[:,:,1,:]-vx[:,:,-2,:]),2).mean()
        f5=torch.pow((vy[:,:,:,1]-vy[:,:,:,-2]),2).mean()
        f6=torch.pow((pred_f_v[:,:,1,1]-0.95),2).mean()
                
        loss=lam2*f2+lam3*f3+lam4*f4+lam5*f5+lam6*f6

        loss.backward()
        optimizer.step()        
        loss=loss.data
        # f1=f1.data
        f2=f2.data
        f3=f3.data
        f4=f4.data
        f5=f5.data
        f6=f6.data
        loss_set.append(loss)
        # f1_set.append(f1)
        f2_set.append(f2)
        f3_set.append(f3)
        f4_set.append(f4)
        f5_set.append(f5)
        f6_set.append(f6)
        print('Epoch: ', epoch, '| Step: ', ite, '|loss: ',loss)
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    if (epoch+1) % 100 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9



end_time = time.time() 

training_time = end_time  - start_time                
print('Training time: %.4f' % (training_time))

plt.figure()     
plt.plot(range(len(loss_set)),loss_set)
plt.xlabel('Iteration')
plt.ylabel('loss')


plt.figure()   
plt.plot(range(len(f1_set)),f1_set)
plt.xlabel('Iteration')
plt.ylabel('f1_loss')

plt.figure()     
plt.plot(range(len(f2_set)),f2_set)      
plt.xlabel('Iteration')
plt.ylabel('f2_loss')    

plt.figure()     
plt.plot(range(len(f3_set)),f3_set)      
plt.xlabel('Iteration')
plt.ylabel('f3_loss')    

plt.figure()     
plt.plot(range(len(f4_set)),f4_set)      
plt.xlabel('Iteration')
plt.ylabel('f4_loss')    

plt.figure()     
plt.plot(range(len(f5_set)),f5_set)      
plt.xlabel('Iteration')
plt.ylabel('f5_loss')  

plt.figure()     
plt.plot(range(len(f6_set)),f6_set)      
plt.xlabel('Iteration')
plt.ylabel('f6_loss')  

plt.figure()  
plt.plot(range(len(lr_list)),lr_list,color = 'r')  
plt.xlabel('Iteration')
plt.ylabel('Learning rate')    

################################################################################

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



torch.save(Net.state_dict(),'label_free_N_logk=%d_epoch=%d_batchsize=%d_N_col=%d_seed=%d_lam1=%d_lam2=%d_lam3=%d_lam4=%d_lam5=%d_t=%.3f.ckpt'%(n_logk,num_epoch,BATCH_SIZE,n_logk_v,seed_n,lam1,lam2,lam3,lam4,lam5,training_time))









"""=================模型测试=================="""
Net.eval()


##########################################################################
#Testing数据准备
#生成渗透率场

# seed=200
# n_logk_test=200 #渗透率场实现个数
# np.random.seed(seed)

# kesi_test=np.zeros((n_logk_test,n_eigen))   #随机数数组
# logk_test=np.zeros((n_logk_test,nx,ny))       #渗透率场数组

# for i_logk in range(n_logk_test):
#     kesi_test[i_logk,:]=np.random.randn(n_eigen)   #随机数数组
#     #由随机数计算渗透率场

#     for i_x in range(nx):
#         for i_y in range(ny):
#             logk_test[i_logk,i_y,i_x]=mean_logk+np.sum(np.sqrt(lamda_xy)*fn_x[i_x][0]*fn_y[i_y][0]*kesi_test[i_logk:i_logk+1].transpose())

# #渗透率场对数转化
# k_test=np.exp(logk_test)
# logk_test_image=logk_test.reshape(n_logk_test,1,nx,ny)



# ######################################################################################################
# ##渗透率场写入程序并进行模拟
# #########################################################################################################
# #
# #
# #修改工作目录
# path = "../"

# # 查看当前工作目录
# retval = os.getcwd()
# print ("当前工作目录为 %s" % retval)

# # 修改当前工作目录
# os.chdir( path )

# # 查看修改后的工作目录
# retval = os.getcwd()

# print ("目录修改成功 %s" % retval)


# # hh_test=forward_model(xp,yp,h_c0,k_test)   


# ########################################################################
# #################################################################################
# #切换工作目录
# path = "\\2data\\"

# # 查看当前工作目录
# now = os.getcwd()
# print ("当前工作目录为 %s" % now)

# # 修改当前工作目录
# os.chdir( now+path )

# # 查看修改后的工作目录
# new_path = os.getcwd()
# print ("目录修改成功 %s" % new_path)
# #
# #
# ###################################################################################
# ##储存到二进制文件npy
# # np.save('hh_test_data_N=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk_test,eta,weight,seed),hh_test)


# #2导入数据
# #读取二进制文件npy

# hh_test=np.load('hh_test_data_N=%d_eta=%d_weight=%f_seed=%d.npy'%(n_logk_test,eta,weight,seed))

########################################################################

logk_image = torch.from_numpy(logk_image)
logk_image = logk_image.type(torch.FloatTensor)

logk_image=logk_image.to(device)
error_l2_set=np.empty((n_up_logk))
R2_set=np.empty((n_up_logk))

hh_test_pred=np.zeros((n_up_logk,nx_up+2,ny_up+2))
vx_test_pred=np.zeros((n_up_logk,nx_up+3,ny_up+2))
vy_test_pred=np.zeros((n_up_logk,nx_up+2,ny_up+3))
hh_test=hh.copy()
start_time = time.time()


for i in range(n_up_logk):
    h_test_pred=Net(logk_image[i:i+1]).cpu().detach().numpy()
    # h_test_pred=h_test_pred*(h_boun1_0-h_c0)+h_c0
    h_test_pred=h_test_pred.reshape(nx_up+2,ny_up+2)
    h_test_pred[0,0]=0
    h_test_pred[0,-1]=0
    h_test_pred[-1,0]=0
    h_test_pred[-1,-1]=0
    
    kk=fun_add_bound_numpy(np.exp(logk_up[i]))
    l=1/kk
    tx=np.zeros((nx_up+3,ny_up+2))
    ty=np.zeros((nx_up+2,ny_up+3))
   
    
    tx[1:-1,:]=lamx/(l[0:-1,:]+l[1:,:])
    ty[:,1:-1]=lamy/(l[:,0:-1]+l[:,1:])
    tx[1,:]=tx[1,:]*2
    tx[-2,:]=tx[-2,:]*2
    ty[:,1]=ty[:,1]*2
    ty[:,-2]=ty[:,-2]*2
    
    vx_test_pred[i,1:-1,:]=(h_test_pred[0:-1,:]-h_test_pred[1:,:])*tx[1:-1,:]
    vy_test_pred[i,:,1:-1]=(h_test_pred[:,0:-1]-h_test_pred[:,1:])*ty[:,1:-1]
        
    hh_test_pred[i]=h_test_pred
   


#    print('Predicting realization %d'%(i+1))

    ##############################################
    #  calculate error
    
    print('Realization %d' % (i))
    
    error_l2 = np.linalg.norm(hh_test[i].flatten()-hh_test_pred[i].flatten(),2)/np.linalg.norm(hh_test[i].flatten(),2)
    print('Error L2: %e' % (error_l2))
    error_l2_set[i]=error_l2
    
    R2=1-np.sum((hh_test[i].flatten()-hh_test_pred[i].flatten())**2)/np.sum((hh_test[i].flatten()-hh_test[i].flatten().mean())**2)
    print('coefficient of determination  R2: %e' % (R2))
    R2_set[i]=R2


elapsed = time.time() - start_time                
print('Prediction time: %.4f' % (elapsed))


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



##########################################################
####Post Process for data #####
####  Just once  ##########
# x=x*bigx
# y=y*bigx



#######################################################
#     PINN plot
#结果展示    

n_logk_test_plot=10

ind_1=2
ind_2=5
ind_3=8
x_up=np.arange(1,nx_up+3,1)
y_up=np.arange(1,ny_up+3,1)

for i_sam in range(n_logk_test_plot):

    
    sam1=85+i_sam   
    
   
    
    ####### Row 0: h(t,x) ##################   
    
    
    plt.figure(figsize=(3,3))
    
    mm1=plt.imshow(hh_test[sam1], interpolation='nearest', cmap='rainbow', 
                  extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()], 
                  origin='lower')
    
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reference H',fontsize=10)
    plt.colorbar(mm1,fraction=0.046, pad=0.04)
    
    line = np.linspace(x_up.min(), x_up.max(), 2)[:,None]
    plt.plot(line, y_up[ind_1]*np.ones((2,1)), 'w-', linewidth = 1)
    plt.plot(line, y_up[ind_2]*np.ones((2,1)), 'w-', linewidth = 1)
    plt.plot(line, y_up[ind_3]*np.ones((2,1)), 'w-', linewidth = 1) 
    

    
    
    
    ####### Row 1: pred h(t,x) ##################  
    
    plt.figure(figsize=(3,3))
    
    mm2=plt.imshow(hh_test_pred[sam1,:,:], interpolation='nearest', cmap='rainbow', 
                  extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()], 
                  origin='lower')

    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Prediction H',fontsize=10)
    plt.colorbar(mm2,fraction=0.046, pad=0.04)
    
    line = np.linspace(x_up.min(), x_up.max(), 2)[:,None]
    plt.plot(line, y_up[ind_1]*np.ones((2,1)), 'w-', linewidth = 1)
    plt.plot(line, y_up[ind_2]*np.ones((2,1)), 'w-', linewidth = 1)
    plt.plot(line, y_up[ind_3]*np.ones((2,1)), 'w-', linewidth = 1) 
    
    
    
    
    
    ####### Row 2: u(t,x) slices ##################  
    plt.figure(figsize=(5,3.5))
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=-0.05, bottom=-0.5, left=0.1, right=0.9, wspace=0.6)
    
    y_ticks = np.arange(199,204, 1)


    
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_up,hh_test[sam1,:,ind_1], 'b-', linewidth = 2, label = 'Reference')       
    ax.plot(x_up,hh_test_pred[sam1,:,ind_1], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$H$')    
    ax.set_title('$y = 320$', fontsize = 10)
    ax.set_xlim([min(x_up)-0.1*max(x_up),max(x_up)+0.1*max(x_up)])
#    ax.set_ylim([hh.min()-0.1*hh.max(),hh.max()+0.1*hh.max()]) 
    # ax.set_ylim([hh_test.min()-0.003*hh_test.max(),hh_test.max()+0.003*hh_test.max()]) 
    # ax.set_yticks(y_ticks)
    
    
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_up,hh_test[sam1,:,ind_2], 'b-', linewidth = 2, label = 'Reference')       
    ax.plot(x_up,hh_test_pred[sam1,:,ind_2], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
#    ax.set_ylabel('$H$')
#    ax.axis('square')
    ax.set_xlim([min(x_up)-0.1*max(x_up),max(x_up)+0.1*max(x_up)])
#    ax.set_ylim([hh.min()-0.1*hh.max(),hh.max()+0.1*hh.max()]) 
    # ax.set_ylim([hh_test.min()-0.003*hh_test.max(),hh_test.max()+0.003*hh_test.max()]) 
    ax.set_title('$y = 620$', fontsize = 10)
    # ax.set_yticks(y_ticks)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x_up,hh_test[sam1,:,ind_3], 'b-', linewidth = 2, label = 'Reference')       
    ax.plot(x_up,hh_test_pred[sam1,:,ind_3], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
#    ax.set_ylabel('$H$')
#    ax.axis('square')
    ax.set_xlim([min(x_up)-0.1*max(x_up),max(x_up)+0.1*max(x_up)])
#    ax.set_ylim([hh.min()-0.1*hh.max(),hh.max()+0.1*hh.max()]) 
    # ax.set_ylim([hh_test.min()-0.003*hh_test.max(),hh_test.max()+0.003*hh_test.max()]) 
    ax.set_title('$y = 920$', fontsize = 10)
    # ax.set_yticks(y_ticks)





    ######################################################################
    ############################# Plotting_2 #############################
    ######################################################################   
#结果展示    

#选择观测时空点
#第一个点：时间t=5,位置x=200,y=200

obs_x1=5
obs_y1=5

obs_x2=9
obs_y2=9

obs_x3=2
obs_y3=2

real_h1=hh_test[:,obs_x1,obs_y1].flatten()
pred_h1=hh_test_pred[:,obs_x1,obs_y1].flatten()

real_h2=hh_test[:,obs_x2,obs_y2].flatten()
pred_h2=hh_test_pred[:,obs_x2,obs_y2].flatten()

real_h3=hh_test[:,obs_x3,obs_y3].flatten()
pred_h3=hh_test_pred[:,obs_x3,obs_y3].flatten()

plt.figure(figsize=(5,5))
plt.scatter(real_h1,pred_h1,marker='o',c='',edgecolors='b')
plt.xlabel('Prediction')
plt.ylabel('Exact')
plt.title("Results Comparison for TgNN at Point1")
plt.xlim(0,1)
plt.ylim(0,1)


#pred_h2=[]
#for i in range(n_logk_test):
#    pred_h2.append(hh_test_pred[i,obs_t2,obs_x2,obs_y2])


plt.figure(figsize=(5,5))
plt.scatter(real_h2,pred_h2,marker='o',c='',edgecolors='b')
plt.xlabel('Prediction')
plt.ylabel('Exact')
plt.title("Results Comparison for TgNN at Point2")
plt.xlim(0,1)
plt.ylim(0,1)


col_x_ticks = np.arange(199.7,202.3, 0.4)
plt.figure(figsize=(5,5))
plt.scatter(real_h1,pred_h1,marker='o',c='',edgecolors='b',label='Point1')
plt.scatter(real_h2,pred_h2,marker='o',c='',edgecolors='r',label='Point2')
plt.xlabel('Prediction')
plt.ylabel('Exact')
plt.title("Correlation between reference and prediction")
# plt.xlim(199.8,202)
# plt.ylim(199.8,202)
# plt.xticks(col_x_ticks)
# plt.yticks(col_x_ticks)



# col_x_ticks = np.arange(199.7,202.3, 0.4)
plt.figure(figsize=(5,5))
plt.plot([0,3],[0,3],'k-',linewidth=2)
plt.scatter(real_h1,pred_h1,marker='o',c='',edgecolors='b',label='Point 1')
plt.scatter(real_h2,pred_h2,marker='s',c='',edgecolors='r',label='Point 2')
plt.scatter(real_h3,pred_h3,marker='^',c='',edgecolors='c',label='Point 3')

plt.xlabel('Reference',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
# plt.title("Correlation between reference and prediction")
plt.xlim(0,1)
plt.ylim(0,1)
# plt.xticks(col_x_ticks,fontsize=12)
# plt.yticks(col_x_ticks,fontsize=12)
plt.legend(fontsize=12)

    ######################################################################
    ############################# Plotting_3 #############################
    ######################################################################   
#统计结果展示    





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



num_bins = 15
plt.figure(figsize=(5,3))
plt.hist(TgNN_error_l2_set, num_bins,fill=False, hatch='///',edgecolor="red")
plt.xlabel('Relative $L_2$ error',fontsize=15)
plt.xlim(0.,0.1)
# # plt.legend()
# # plt.xticks(l2_x_ticks)

num_bins = 15
plt.figure(figsize=(5,3))
plt.hist(TgNN_R2_set, num_bins,fill=False, hatch='///',edgecolor="red")
plt.xlabel('$R^2$ score',fontsize=15)
plt.xlim(0.95,1)
# # plt.legend()
# # plt.xticks(l2_x_ticks)







n_plot=6
for sam1 in range(n_plot):
# for sam1 in plot_n:
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    plt.imshow(logk_up[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                             extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    plt.colorbar()
    
    
    plt.subplot(142)
    plt.imshow(hh_test[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.imshow(hh_test_pred[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.imshow(abs(hh_test_pred[sam1]-hh_test[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()




n_plot=6
for sam1 in range(n_plot):
# for sam1 in plot_n:
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    plt.imshow(logk_up[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                             extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("ln$K(x,y)$",fontsize=15)
    plt.colorbar()
    
    
    plt.subplot(142)
    plt.imshow(vx_test[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Reference',fontsize=15)
    cbar=plt.colorbar()
    cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(143)
    plt.imshow(vx_test_pred[sam1], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar()
    cbar.set_ticks(np.linspace(200,202,5))
    
    
    plt.subplot(144)
    plt.imshow(abs(vx_test_pred[sam1]-vx_test[sam1]), origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    plt.colorbar()





x_up=np.arange(1,nx_up+1,1)
y_up=np.arange(1,ny_up+1,1)

#################################
#contourf
n_plot=6
for sam1 in range(n_plot):
# for sam1 in plot_n:
    plt.figure(figsize=(18,3))
    plt.subplot(141)
    mm=plt.contourf(logk_up[sam1],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                             extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.title("Pieced ln$K(x,y)$",fontsize=15)
    cbar=plt.colorbar(mm)
    cbar.set_clim( -2, 2 )
    
    
    plt.subplot(142)
    mm=plt.contourf(hh_test[sam1,1:11,1:11],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Numerical',fontsize=15)
    cbar=plt.colorbar(mm)
    cbar.set_ticks(np.linspace(0,1,5))
    
    
    plt.subplot(143)
    mm=plt.contourf(hh_test_pred[sam1,1:11,1:11],10, origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()])
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Prediction',fontsize=15)
    cbar=plt.colorbar(mm)
    cbar.set_ticks(np.linspace(0,1,5))
    
    
    plt.subplot(144)
    mm=plt.imshow(hh_test_pred[sam1,1:11,1:11]-hh_test[sam1,1:11,1:11], origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
                extent=[x_up.min(), x_up.max(), y_up.min(),y_up.max()],vmin=-0.1,vmax=0.1)
    plt.xlabel('$x$',fontsize=15)
    #plt.ylabel('y')
    plt.title('Error',fontsize=15)
    cbar=plt.colorbar(mm)
    cbar.set_ticks(np.linspace(-0.1,0.1,5))








