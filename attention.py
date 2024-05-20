import numpy as np
import torch
import math
from torch import nn
from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

def conv1x1x1(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                     padding=(0,0,0), groups=groups, bias=False)

def conv1x1x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,3), stride=stride,
                     padding=(0,0,1), groups=groups, bias=False)

torch_ver = torch.__version__[:3]     #torch版本

__all__ = ['PAM_Module', 'CAM_Module']

class PAM_Module(Module):        #空间注意力模型
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):       #初始化层
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 通常 将 ResNet 最后的两个下采样层去除，使得到的特征图尺寸为原输入图像的 1/8
        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        #在 query_conv 中，输入 feature maps 为 B×C×W×H，输出为 B×C/8×W×H；
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        #在 key_conv 中，输入 feature maps 为 B×C×W×H，输出为 B×C/8×W×H；
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        #在 value_conv 中，输入 feature maps 为 B×C×W×H，输出为 B×C×W×H；
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)   #三维数据的最后一维
    def forward(self, x):     #前向传播与自动反向传播
        """
            inputs :
                x : input feature maps( B X C X H X W)  Batch_size × Channels × Width × Height
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        # print('x',x.shape)
        x = x.squeeze(-1)
        # print('x',x.shape)
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()    #从x中取出Batch_size × Channels × Width × Height
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        #proj_query 本质即加入了 reshape 的操作的卷积。首先对输入 feature map 进行 query_conv 卷积，
        #输出为 B×C/8×W×H；view 函数改变输出维度，就单张 feature map 而言，即将 W×H 大小拉直，变为 1×(W×H) 大小；
        #就 batch size 大小而言，输出就是 B×C/8×(W×H)；permute 函数则对第二维和第三维进行倒置，输出为 B×(W×H)×C/8。
        #proj_query 中的第 i 行表示第 i 个像素位置上所有通道的值
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        #proj_key 与 proj_query 相似，只是没有最后一步倒置，输出 shape 为 B×C/8×(W×H)
        energy = torch.bmm(proj_query, proj_key)
        #该步骤的意义是：energy 中第 (i, j) 位置的元素为输入特征图第 j 个元素对第 i 个元素的影响，从而实现全局上下文任意两个元素之间的依赖关系。
        attention = self.softmax(energy)
        #这一步将 energe 进行 softmax 归一化，注意是 对行的归一化。归一化后每行之和为1，对于 (i, j) 位置，可理解为第 j 位置对第 i 位置的权重，所有的 j 对 i 位置的权重之和为1，此时得到 attention_map。
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        #proj_value 和 proj_query 与 proj_key 一样，只是输入 shape 为 B×C×W×H，输出 shape 为 B×C×(W×H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        #在对 proj_value 与 attention_map 点乘前，先对 attention 转置。这是由于 attention 中每行的权重之和为1，
        #是原特征图第 j 个位置对第 i 个位置的权重，将其转置之后，每列之和为1；proj_value 的每一行与 attention 中的每一列点乘，
        #将权重施加于 proj_value 上，输出特征图 shape 为 B×C×(W×H) 。
        out = out.view(m_batchsize, C, height, width)   #目的是将多维的的数据如（none，36，2，2）平铺为一维如（none，144）。作用类似于keras中的Flatten函数
        out = (self.gamma*out + x).unsqueeze(-1)  #这一步是 对 attention 之后的 out 进行加权，x 是原始的特征图，将其叠加在原始特征图上。
        # 系数 gamma 是经过学习得到的，初始值为 0。输出即原始特征图，随着学习的深入，在原始特征图上增加了加权的 attention，得到特征图中 任意两个位置的全局依赖关系。
        # 这样做的好处是，gamma初始化为0.这使网络首先依赖局部特征（本地的，当前位置特征）的信息，然后再逐渐为非局部区域分配权重。这是一个由简单到复杂的过程。
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.inplanes = in_dim
        # self.conv=conv1x1x1(48, 24)
        self.conv=conv1x1x3(48, 24)
        self.batch_norm0 = nn.Sequential(
                                    # nn.BatchNorm3d(in_planes,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    ReLU()
                                    # mish()
        )



        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    # def forward(self,xl,xd):
    def forward(self,xl):
    # def forward(self,xl):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # x = torch.cat((xd, xl), dim=1)
        # # print('x',x.shape)
        # x1 = self.conv(x)
        # x2 = self.batch_norm0(x1)
        m_batchsize, C, height, width, channle = xl.size()
        #print(x.size())
        proj_query = xl.view(m_batchsize, C, -1)
        proj_key = xl.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # print('energy_new',energy_new.shape)
        attention = self.softmax(energy_new)
        # print('attention',attention.shape)
        proj_value = xl.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        # print('out',out.shape)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out',out.shape)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + xl  #C*H*W
        return out