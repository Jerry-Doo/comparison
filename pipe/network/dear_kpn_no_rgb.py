import sys
sys.path.insert(0, '../module/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from module.activation import *

PI = 3.14159265358979323846
flg = False

class UNetSubnet(nn.Module):
    def __init__(self, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.n_filters = [64, 64, 128, 128, 128, 128, 256, 256]
        self.filter_sizes = [3] * 8
        self.pool_sizes = [1, 1, 2, 1, 2, 1, 2, 1]
        self.pool_strides = [1, 1, 2, 1, 2, 1, 2, 1]
        self.skips = [False, False, True, False, True, False, True, False]
        # Encoder convs
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = 2
        for i in range(len(self.n_filters)):
            self.enc_convs.append(
                nn.Conv2d(in_channels, self.n_filters[i], self.filter_sizes[i], padding=self.filter_sizes[i]//2)
            )
            if self.pool_sizes[i] != 1 or self.pool_strides[i] != 1:
                self.pools.append(nn.MaxPool2d(self.pool_sizes[i], self.pool_strides[i]))
            else:
                self.pools.append(None)
            in_channels = self.n_filters[i]
        # Decoder upconvs
        self.upconvs = nn.ModuleList()
        for i in range(len(self.n_filters)-2, 0, -1):
            ksize = self.filter_sizes[i]
            self.upconvs.append(
                nn.ConvTranspose2d(self.n_filters[i+1], self.n_filters[i], ksize, stride=self.pool_strides[i], padding=ksize//2, output_padding=self.pool_strides[i]-1 if self.pool_strides[i]>1 else 0)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
    def forward(self, x):
        conv = []
        pool = [x]
        current_input = x
        # Encoder
        for i in range(len(self.n_filters)):
            current_input = self.enc_convs[i](pool[-1])
            current_input = relu(current_input)
            conv.append(current_input)
            if self.pool_sizes[i] == 1 and self.pool_strides[i] == 1:
                pool.append(current_input)
            else:
                pooled = self.pools[i](current_input)
                pool.append(pooled)
            current_input = pool[-1]
        # Decoder
        upsamp = []
        current_input = pool[-1]
        up_idx = 0
        for i in range(len(self.n_filters)-2, 0, -1):
            current_input = self.upconvs[up_idx](current_input)
            current_input = relu(current_input)
            upsamp.append(current_input)
            if self.skips[i] == False and self.skips[i+1] == True:
                target_size = pool[i+1].shape[2:]  # 目标空间尺寸 (H, W)
                current_input = F.interpolate(current_input, size=target_size, mode='bilinear', align_corners=False)
                current_input = torch.cat([current_input, pool[i+1]], dim=1)
                C_in = current_input.shape[1]  # 通道数
                reduce_conv = nn.Conv2d(in_channels=C_in, out_channels=C_in // 2, kernel_size=1).to(current_input.device)
                current_input = reduce_conv(current_input)
            up_idx += 1
        features = upsamp[-1]
        return features

class DepthOutputSubnet(nn.Module):
    def __init__(self, in_channels, kernel_size=3, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

        out_channels = 1 + kernel_size ** 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        out = torch.sigmoid(self.conv(x))
        biases = out[:, 0:1, :, :]
        weights = out[:, 1:, :, :]
        return biases, weights


class DearKPNNoRGB(nn.Module):
    def __init__(self, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.kernel_size = 3
        self.unet = UNetSubnet(flg, regular, batch_size, deformable_range)
        self.depth_out = DepthOutputSubnet(in_channels=64, kernel_size=self.kernel_size,
                                  flg=flg, regular=regular,
                                  batch_size=batch_size, deformable_range=deformable_range)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
    def forward(self, x):
        # 后续的前向运算
        # 只实现HAMMER分支
        depth = x[:, 0:1, :, :]
        amplitude = x[:, 1:2, :, :]
        x1_input = torch.cat([depth, amplitude], dim=1)
        features = self.unet(x1_input)
        biases, weights = self.depth_out(features)
        weights = weights / (torch.sum(torch.abs(weights), dim=1, keepdim=True) + 1e-6)
        inputs = depth + biases
        column = im2col(inputs, kernel_size=self.kernel_size)
        #weights = weights.permute(0, 2, 3, 1)
        current_output = torch.sum(column * weights, dim=1, keepdim=True)
        depth_output = current_output
        return depth_output, current_output 