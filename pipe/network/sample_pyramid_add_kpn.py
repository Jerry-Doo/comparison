import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, './module/')
from dataset import *
from module.activation import *
from module.conv import conv
from module.dfus_block import dfus_block_add_output_conv
from module.utils import bilinear_warp, costvolumelayer

PI = 3.14159265358979323846
flg = False
dtype = torch.float32


class FeatureExtractorSubnet(nn.Module):
    def __init__(self, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.n_filters = [
        16, 16,
        32, 32,
        64, 64,
        96, 96,
        128, 128,
        192, 192,
    ]
        self.filter_sizes = [3] * len(self.n_filters)
        self.pool_sizes = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
        self.pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
        self.skips = [
        False, False,
        True, False,
        True, False,
        True, False,
        True, False,
        True, False,
    ]
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = 2
        for i in range(len(self.n_filters)):
            self.convs.append(
                nn.Conv2d(in_channels, self.n_filters[i], kernel_size=3, padding=1)
            )
            in_channels = self.n_filters[i]
            if self.pool_sizes[i] > 1:
                self.pools.append(
                    nn.MaxPool2d(kernel_size=self.pool_sizes[i], stride=self.pool_strides[i])
                )
            else:
                self.pools.append(None)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        features = []
        current_input = x
        for i in range(len(self.n_filters)):
            current_input = self.convs[i](current_input)
            current_input = F.relu(current_input)
            if self.pool_sizes[i] == 1 and self.pool_strides[i] == 1:
                if (i == len(self.n_filters) - 1) or (self.pool_sizes[i + 1] == 2 and self.pool_strides[i + 1] == 2):
                    features.append(current_input)
            else:
                current_input = self.pools[i](current_input)
        return features

class DepthResidualRegressionSubnet(nn.Module):
    def __init__(self, in_channels=None, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.n_filters = [128, 96, 64, 32, 16, 1]
        self.convs = nn.ModuleList()

        # 第一层使用 LazyConv2d，自动适配输入通道数
        self.convs.append(
            nn.LazyConv2d(self.n_filters[0], kernel_size=3, padding=1)
        )

        # 其余层使用标准 Conv2d
        for i in range(1, len(self.n_filters)):
            self.convs.append(
                nn.Conv2d(self.n_filters[i - 1], self.n_filters[i], kernel_size=3, padding=1)
            )

        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        current_input = x
        for i, conv in enumerate(self.convs):
            current_input = conv(current_input)
            if i != len(self.convs) - 1:
                current_input = F.relu(current_input)
        return current_input

class ResidualOutputSubnet(nn.Module):
    def __init__(self, in_channels, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        return self.conv(x)


class UNetSubnet(nn.Module):
    def __init__(self, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.n_filters = [16, 16, 32, 32, 64, 64, 128, 128]
        self.filter_sizes = [3] * len(self.n_filters)
        self.pool_sizes = [1, 1, 2, 1, 2, 1, 2, 1]
        self.pool_strides = [1, 1, 2, 1, 2, 1, 2, 1]
        self.skips = [False, False, True, False, True, False, True, False]
        self.encoder_convs = nn.ModuleList()
        in_channels = 1
        for i in range(len(self.n_filters)):
            self.encoder_convs.append(
                nn.Conv2d(in_channels, self.n_filters[i], kernel_size=3, padding=1)
            )
            in_channels = self.n_filters[i]
        # Decoder
        self.decoder_convs = nn.ModuleList()
        for i in range(len(self.n_filters) - 2, 0, -1):
            self.decoder_convs.append(
                nn.ConvTranspose2d(self.n_filters[i], self.n_filters[i-1], kernel_size=3, stride=self.pool_strides[i], padding=1, output_padding=self.pool_strides[i]-1)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        convs = []
        pools = [x]
        current_input = x
        for i in range(len(self.n_filters)):
            current_input = self.encoder_convs[i](current_input)
            current_input = F.relu(current_input)
            if self.pool_sizes[i] > 1:
                current_input = F.max_pool2d(current_input, kernel_size=self.pool_sizes[i], stride=self.pool_strides[i])
            convs.append(current_input)
            pools.append(current_input)
        current_input = pools[-1]
        for i, deconv in enumerate(self.decoder_convs):
            current_input = deconv(current_input)
            current_input = F.relu(current_input)
            if self.skips[i] and self.skips[i-1]:
                current_input = torch.cat([current_input, pools[i+1]], dim=1)
        return current_input


class DepthOutputSubnet(nn.Module):
    def __init__(self, in_channels, kernel_size, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class DearKPN(nn.Module):
    def __init__(self, kernel_size=3, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.unet = UNetSubnet(flg, regular, batch_size, deformable_range)
        self.depth_output = DepthOutputSubnet(16, kernel_size, flg, regular, batch_size, deformable_range)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        features = self.unet(x)
        weights = self.depth_output(features)
        weights = weights / (torch.sum(torch.abs(weights) + 1e-6, dim=1, keepdim=True))
        column = im2col(x, kernel_size=self.kernel_size)
        current_output = torch.sum(column * weights, dim=1, keepdim=True)
        return current_output


class SamplePyramidAddKPN(nn.Module):
    def __init__(self, batch_size, deformable_range, flg=None, regular=None):
        super().__init__()
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        self.depth_residual_weight = [0.32, 0.08, 0.02, 0.01, 0.005]
        self.feature_extractor = FeatureExtractorSubnet(flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        self.depth_residual_regressors = nn.ModuleList([
            DepthResidualRegressionSubnet(in_channels=None, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range) for _ in range(6)
        ])
        self.residual_output = ResidualOutputSubnet(6, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        self.dear_kpn = DearKPN(kernel_size=3, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular

    def forward(self, x):
        depth_residual = []
        depth_residual_input = []
        # 假设输入 x 形状为 (B, C, H, W)
        depth = x[:, 0:1, :, :]
        amplitude = x[:, 1:2, :, :]
        x1_input = torch.cat([depth, amplitude], dim=1)
        features = self.feature_extractor(x1_input)  
        h_max, w_max = x.shape[2], x.shape[3]
        for i in range(1, len(features) + 1):
            if i == 1:
                inputs = features[len(features) - i]
            else:
                feature_input = features[len(features) - i]
                h_max_low_scale, w_max_low_scale = feature_input.shape[2], feature_input.shape[3]
                depth_coarse_input = F.interpolate(depth_residual[-1], size=(h_max_low_scale, w_max_low_scale), mode='bicubic', align_corners=True)
                true_depth_coarse_input = F.interpolate(depth, size=(h_max_low_scale, w_max_low_scale), mode='bicubic', align_corners=True)
                inputs = torch.cat([feature_input, true_depth_coarse_input + depth_coarse_input], dim=1)
            current_depth_residual = self.depth_residual_regressors[i-1](inputs)
            depth_residual.append(current_depth_residual)
            current_depth_residual_input = F.interpolate(current_depth_residual, size=(h_max, w_max), mode='bicubic', align_corners=True)
            depth_residual_input.append(current_depth_residual_input)
        depth_coarse_residual_input = torch.cat(depth_residual_input, dim=1)
        final_depth_residual_output = self.residual_output(depth_coarse_residual_input)
        current_final_depth_output = depth + final_depth_residual_output
        final_depth_output = self.dear_kpn(current_final_depth_output)
        depth_residual_input.append(final_depth_residual_output)
        depth_residual_input.append(final_depth_output - current_final_depth_output)
        return final_depth_output, torch.cat(depth_residual_input, dim=1)