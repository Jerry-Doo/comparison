import torch
import torch.nn as nn
import numpy as np

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, 
                 trainable=True, activation=None, init_weights=None):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            dilation=dilation,
            bias=True
        )
        
        # 如果提供了初始化权重，则加载
        if init_weights is not None:
            if 'kernel' in init_weights:
                self.conv.weight.data = torch.from_numpy(np.array(init_weights['kernel']))
            if 'bias' in init_weights:
                self.conv.bias.data = torch.from_numpy(np.array(init_weights['bias']))
        
        self.activation = activation
        self.trainable = trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class TransposeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 trainable=True, activation=None, init_weights=None):
        super(TransposeConv2D, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=True
        )
        
        # 如果提供了初始化权重，则加载
        if init_weights is not None:
            if 'kernel' in init_weights:
                self.conv.weight.data = torch.from_numpy(np.array(init_weights['kernel']))
            if 'bias' in init_weights:
                self.conv.bias.data = torch.from_numpy(np.array(init_weights['bias']))
        
        self.activation = activation
        self.trainable = trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def conv(previous_pref=None, x=None, output_channel=None, filter_size=None, 
         dilation_rate=None, trainable=None, activation=None, conv_num=None):
    """
    为了保持与原代码接口一致，提供一个函数形式的接口
    """
    in_channels = x.size(1)
    module = Conv2D(
        in_channels=in_channels,
        out_channels=output_channel,
        kernel_size=filter_size,
        dilation=dilation_rate,
        trainable=trainable,
        activation=activation
    )
    return module(x)

def transpose_conv(pref=None, inits=None, current_input=None, output_channel=None, 
                  filter_size=None, strides=None, trainable=None, activation=None, conv_num=None):
    """
    为了保持与原代码接口一致，提供一个函数形式的接口
    """
    in_channels = current_input.size(1)
    module = TransposeConv2D(
        in_channels=in_channels,
        out_channels=output_channel,
        kernel_size=filter_size,
        stride=strides,
        trainable=trainable,
        activation=activation
    )
    return module(current_input)
