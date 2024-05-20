import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from attention import PAM_Module, CAM_Module
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish
from deform_conv import *

from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

def conv1x1x1(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)

def conv1x1x3(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)

def conv1x1x5(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,5), stride=stride,
                     padding=(0,0,2), groups=groups, bias=False)

def conv1x1x7(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,7), stride=stride,
                     padding=(0,0,3), groups=groups, bias=False)

def conv1x1x9(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,9), stride=stride,
                     padding=(0,0,4), groups=groups, bias=False)

def convg3(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)

def convg1(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
    
def convk3(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,1), stride=stride,
                     padding=(1,1,0), groups=groups, bias=False)
def convk1(in_planes,out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
def convnb(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, in_planes, kernel_size=(3,3,3), stride=stride,
                     padding=(1,1,1), groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1),padding=(0,0,0),stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,1), stride=stride,
                     padding=(1,1,0), groups=groups, bias=False)
def conv3x1(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)
def conv1x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)
def conv1x5(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,5), stride=stride,
                     padding=(0,0,2), groups=groups, bias=False)
def conv1x7(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,7), stride=stride,
                     padding=(0,0,3), groups=groups, bias=False)
def conv1x9(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,9), stride=stride,
                     padding=(0,0,4), groups=groups, bias=False)

class Pooling1(nn.Module):
    def __init__(self,in_planes):
        super(Pooling1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.conv3d = nn.Sequential(
             nn.Conv3d(96, in_planes, kernel_size=(1, 1, 1), stride=1, padding=0),
            #  nn.BatchNorm2d(24),
             nn.ReLU(inplace = True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,X):
        out1 = self.avg_pool(X)
        out2 = self.max_pool(X)
        out3 = torch.cat((out1, out2), dim = 1)
        out3 = self.avg_pool(out3)
        out5 = self.conv3d(out3)
        out = self.sigmoid(out5)
        # print(out.shape)
        return out


class Pooling2(nn.Module):
    def __init__(self,in_planes):
        super(Pooling2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.conv3d = nn.Sequential(
             nn.Conv3d(96, in_planes, kernel_size=(1, 1, 1), stride=1, padding=0),
            #  nn.BatchNorm2d(24),
             nn.ReLU(inplace = True),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):

        out1 = self.avg_pool(X)
        out2 = self.max_pool(X)
        out3 = torch.cat((out1, out2), dim = 1)
        out3 = self.avg_pool(out3)
        out5 = self.conv3d(out3)
        out = self.sigmoid(out5)

        return out
class Pooling3(nn.Module):
    def __init__(self,in_planes):
        super(Pooling3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.conv3d = nn.Sequential(
             nn.Conv3d(in_planes*2, in_planes, kernel_size=(1, 1, 1), stride=1, padding=0),
            #  nn.BatchNorm2d(24),
             nn.ReLU(inplace = True),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):

        out1 = self.avg_pool(X)
        out2 = self.max_pool(X)
        out3 = torch.cat((out1, out2), dim = 1)
        out3 = self.avg_pool(out3)
        out5 = self.conv3d(out3)
        out = self.sigmoid(out5)

        return out

        
class FECNet(nn.Module):
    def __init__(self, band, classes):
        super(FECNet, self).__init__()
        self.name = 'FECNet'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=48,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(48,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )

        self.conv111 = nn.Conv3d(in_channels=24, out_channels=12,
                                kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.convpa = nn.Conv3d(in_channels=1, out_channels=12,
                                kernel_size=(1, 1, 200), stride=(1, 1, 2))
        # Dense block
        self.batch_normpa = nn.Sequential(
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )
        
        #######################第一个空洞卷积块#################################################
        self.kongdong11 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong12 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        # self.kongdong12 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong12 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish() 
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong15 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        # self.kongdong15 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong15 = nn.Sequential(
                                    nn.BatchNorm3d(48,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        #######################第二个空洞卷积块#################################################
        self.kongdong21 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,3), dilation=2)
        self.batch_normkongdong21 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong22 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,3), dilation=2)
        # self.kongdong22 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong22 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong25 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,3), dilation=2)
        # self.kongdong25 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong25 = nn.Sequential(
                                    nn.BatchNorm3d(48,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        #######################第三个空洞卷积块#################################################
        self.kongdong31 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=5)
        self.batch_normkongdong31 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong32 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=5)
        # self.kongdong32 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong32 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )
        self.kongdong35 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=5)
        # self.kongdong35 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 1, 3), stride=(1, 1, 1),padding=(0,0,1), dilation=1)
        self.batch_normkongdong35 = nn.Sequential(
                                    nn.BatchNorm3d(48,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
                                    ReLU(inplace=True)
            # mish()
        )

        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.convp15 = nn.Conv3d(in_channels=48, out_channels=62,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化



        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(62,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(62, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(48)
        self.attention_spatial = PAM_Module(60)


        self.convp16 = nn.Conv3d(in_channels=12, out_channels=60,
                                kernel_size=(1, 1, 1), stride=(1, 1, 1)) # kernel size随数据变化


        self.pooling1 = Pooling1(48)
        self.pooling2 = Pooling2(48)
        self.pooling3 = Pooling3(48)

    def forward(self, X):

        Xout11 = self.conv11(X)
        Xnode1 = self.batch_norm11(Xout11)            
        Xout1 = self.kongdong11(Xnode1)
        Xout1 = self.batch_normkongdong11(Xout1)              
        Xout2 = self.kongdong12(Xout1)
        Xout2 = self.batch_normkongdong12(Xout2)           
        Xout3 = self.kongdong15(Xout2)
        Xnode2 = self.batch_normkongdong15(Xout3)           

        Xnode21 = self.pooling1(Xnode2)
        Xnode1 = Xnode1 + Xnode21

        Xout4 = self.kongdong21(Xnode2)
        Xout4 = self.batch_normkongdong21(Xout4 )
        Xout5 = self.kongdong22(Xout4 )
        Xout5 = self.batch_normkongdong22(Xout5)
        Xout6 = self.kongdong25(Xout5)
        Xnode3 = self.batch_normkongdong25(Xout6)

        Xnode22 = self.pooling2(Xnode3)
        Xnode1 = Xnode1 + Xnode22
        Xnode2 = Xnode2 + Xnode22

        Xout7 = self.kongdong31(Xnode3)
        Xout7 = self.batch_normkongdong31(Xout7)

        Xout8 = self.kongdong32(Xout7)
        Xout8 = self.batch_normkongdong32(Xout8)

        Xout9 = self.kongdong35(Xout8)
        Xnode4= self.batch_normkongdong35(Xout9)

        Xnode33 = self.pooling3(Xnode4)
        Xnode1 = Xnode1 + Xnode33
        Xnode2 = Xnode2 + Xnode33
        Xnode3 = Xnode3 + Xnode33

        Xout = self.batch_norm14(Xnode3)
        Xout = self.convp15(Xout)
        
        x1 = self.attention_spectral(Xout)
        x1 = torch.mul(x1, Xout)

        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        output = self.full_connection(x1)

        return output