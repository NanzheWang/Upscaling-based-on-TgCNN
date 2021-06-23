# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:43:39 2020

@author: WNZ
"""


import torch
from torch.nn import Linear,Tanh,Sequential,ReLU,Softplus,ELU
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as func

#######################################################################
"""=================构建神经网络=================="""

class Swish(nn.Module):
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)
#设置保留小数位数
torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)


#%%定义一个神经网络
#######################################################################
# Convolutional neural network
class conv_upscale(nn.Module):
    def __init__(self, in_channel=1,out_channel=1,n_code=25):
        super(conv_upscale, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2)



            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            )

        self.fc = nn.Sequential(                
        nn.Linear(256*4 *4, n_code),
        # nn.ReLU(),
        nn.Linear(n_code, 256*4 *4),
        # nn.ReLU(),  
        )

        self.decode = nn.Sequential(
                
            nn.ConvTranspose2d( 256, 128, 3, 1, 0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 128, 64, 3, 1, 0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 64, 32, 3, 1, 0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),

            nn.ConvTranspose2d(32, 16, 3, 1, 0),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
    
     
            nn.ConvTranspose2d( 16, out_channel, 3, 1, 1),
            nn.Sigmoid())
#            nn.Tanh())
#            nn.LeakyReLU())

       
    def forward(self, x):
        code = self.encode(x)
        # code = code.view(code.size(0), -1)
        # code = self.fc(code)
        # code = code.view(-1, 256,4, 4)
        out = self.decode(code)
        return out
#######################################################################
#######################################################################
# Convolutional neural network
class conv_upscale_elu(nn.Module):
    def __init__(self, in_channel=1,out_channel=1,n_code=25):
        super(conv_upscale_elu, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2)



            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            )

        self.fc = nn.Sequential(                
        nn.Linear(256*4 *4, n_code),
        # nn.ReLU(),
        nn.Linear(n_code, 256*4 *4),
        # nn.ReLU(),  
        )

        self.decode = nn.Sequential(
                
            nn.ConvTranspose2d( 256, 128, 3, 1, 0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 128, 64, 3, 1, 0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 64, 32, 3, 1, 0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),

            nn.ConvTranspose2d(32, 16, 3, 1, 0),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
    
     
            nn.ConvTranspose2d( 16, out_channel, 3, 1, 1),
            # nn.Sigmoid())
#            nn.Tanh())
            # nn.LeakyReLU(0.2))
            # ELU())
            Swish())
            # )

       
    def forward(self, x):
        code = self.encode(x)
        # code = code.view(code.size(0), -1)
        # code = self.fc(code)
        # code = code.view(-1, 256,4, 4)
        out = self.decode(code)
        return out
#######################################################################

#######################################################################
# Convolutional neural network
class conv_upscale_32(nn.Module):
    def __init__(self, in_channel=1,out_channel=1,n_code=25):
        super(conv_upscale_32, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2)



            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            # nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),



            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            )

        self.fc = nn.Sequential(                
        nn.Linear(256*4 *4, n_code),
        # nn.ReLU(),
        nn.Linear(n_code, 256*4 *4),
        # nn.ReLU(),  
        )

        self.decode = nn.Sequential(
                
            nn.ConvTranspose2d( 512, 256, 3, 1, 0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 256, 128, 3, 1, 0),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
            
            nn.ConvTranspose2d( 128, 64, 3, 1, 0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),

            nn.ConvTranspose2d(64, 32, 3, 1, 0),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            Swish(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Softplus(beta=1, threshold=20),
    
     
            nn.ConvTranspose2d( 32, out_channel, 3, 1, 1),
            nn.Sigmoid())
#            nn.Tanh())
#            nn.LeakyReLU())

       
    def forward(self, x):
        code = self.encode(x)
        # code = code.view(code.size(0), -1)
        # code = self.fc(code)
        # code = code.view(-1, 256,4, 4)
        out = self.decode(code)
        return out
#######################################################################



















































