import torch
import torch.nn as nn
from activation import *

class Bottleneck(nn.Module):
    def __init__(self, in_channels, flg=True, regular=None, inits=None):
        super(Bottleneck, self).__init__()
        self.flg = flg
        self.regular = regular
        self.inits = inits
        
        self.block_filters = [64, 64, 256]
        self.filter_sizes = [1, 3, 1]
        self.skips = [False, False, True]
        
        # 创建卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.shortcut_conv = None
        
        # 第一层卷积
        self.conv_layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.block_filters[1],
            kernel_size=self.filter_sizes[1],
            padding=self.filter_sizes[1]//2,
            bias=True
        ))
        self.bn_layers.append(nn.BatchNorm2d(self.block_filters[1], momentum=0.9))
        
        # 第二层卷积
        self.conv_layers.append(nn.Conv2d(
            in_channels=self.block_filters[1],
            out_channels=self.block_filters[2],
            kernel_size=self.filter_sizes[2],
            padding=self.filter_sizes[2]//2,
            bias=True
        ))
        self.bn_layers.append(nn.BatchNorm2d(self.block_filters[2], momentum=0.9))
        
        # 如果使用跳跃连接，创建shortcut卷积
        if self.skips[2]:
            self.shortcut_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.block_filters[2],
                kernel_size=self.filter_sizes[2],
                padding=self.filter_sizes[2]//2,
                bias=True
            )
    
    def forward(self, x):
        # 保存输入用于跳跃连接
        identity = x
        
        # 第一层卷积
        x = self.conv_layers[0](x)
        x = self.bn_layers[0](x)
        x = relu(x)
        
        # 第二层卷积
        x = self.conv_layers[1](x)
        x = self.bn_layers[1](x)
        
        # 如果使用跳跃连接
        if self.skips[2]:
            identity = self.shortcut_conv(identity)
            x = x + identity
        
        x = relu(x)
        
        return x

def bottleneck(x, flg, regular, inits, i):
    """
    为了保持与原代码接口一致，提供一个函数形式的接口
    """
    module = Bottleneck(x.size(1), flg, regular, inits)
    return module(x)