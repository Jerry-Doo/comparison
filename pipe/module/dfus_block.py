import torch
from module.activation import *
from module.conv import conv

def dfus_block(x, flg, regular, i):
    # PyTorch 版本的 dfus_block
    block_filters = [
        24,
        6, 3,
        6, 3,
        8,
    ]
    filter_sizes = [
        1,
        3, 3,
        3, 3,
        1,
    ]
    dilation_sizes = [
        1,
        2, 1,
        1, 2,
        1,
    ]
    # 输入直接用 x
    ae_inputs = x
    # 各分支卷积
    conv_input = conv(previous_pref=None, x=ae_inputs, output_channel=block_filters[0],
                      filter_size=filter_sizes[0], dilation_rate=dilation_sizes[0], trainable=flg,
                      activation=relu, conv_num=0)
    conv_dilation_21_1 = conv(previous_pref=None, x=conv_input, output_channel=block_filters[1],
                      filter_size=filter_sizes[1], dilation_rate=dilation_sizes[1], trainable=flg,
                      activation=relu, conv_num=1)
    conv_dilation_21_2 = conv(previous_pref=None, x=conv_dilation_21_1, output_channel=block_filters[2],
                      filter_size=filter_sizes[2], dilation_rate=dilation_sizes[2], trainable=flg,
                      activation=relu, conv_num=2)
    conv_dilation_12_1 = conv(previous_pref=None, x=conv_input, output_channel=block_filters[3],
                      filter_size=filter_sizes[3], dilation_rate=dilation_sizes[3], trainable=flg,
                      activation=relu, conv_num=3)
    conv_dilation_12_2 = conv(previous_pref=None, x=conv_dilation_12_1, output_channel=block_filters[4],
                      filter_size=filter_sizes[4], dilation_rate=dilation_sizes[4], trainable=flg,
                      activation=relu, conv_num=4)
    # 拼接
    tensor_input = torch.cat([conv_dilation_21_1, conv_dilation_21_2, conv_dilation_12_1, conv_dilation_12_2], dim=1)
    conv_output = conv(previous_pref=None, x=tensor_input, output_channel=block_filters[5],
                      filter_size=filter_sizes[5], dilation_rate=dilation_sizes[5], trainable=flg,
                      activation=relu, conv_num=5)
    tensor_output = torch.cat([ae_inputs, conv_output], dim=1)
    ae_outputs = tensor_output
    return ae_outputs

def dfus_block_add_output_conv(x, flg, regular, i):
    block_filters = [
        24,
        6, 3,
        6, 3,
        8,
        1,
    ]
    filter_sizes = [
        1,
        3, 3,
        3, 3,
        1,
        1,
    ]
    dilation_sizes = [
        1,
        2, 1,
        1, 2,
        1,
        1,
    ]
    ae_inputs = x
    conv_input = conv(previous_pref=None, x=ae_inputs, output_channel=block_filters[0],
                      filter_size=filter_sizes[0], dilation_rate=dilation_sizes[0], trainable=flg,
                      activation=relu, conv_num=0)
    conv_dilation_21_1 = conv(previous_pref=None, x=conv_input, output_channel=block_filters[1],
                      filter_size=filter_sizes[1], dilation_rate=dilation_sizes[1], trainable=flg,
                      activation=relu, conv_num=1)
    conv_dilation_21_2 = conv(previous_pref=None, x=conv_dilation_21_1, output_channel=block_filters[2],
                      filter_size=filter_sizes[2], dilation_rate=dilation_sizes[2], trainable=flg,
                      activation=relu, conv_num=2)
    conv_dilation_12_1 = conv(previous_pref=None, x=conv_input, output_channel=block_filters[3],
                      filter_size=filter_sizes[3], dilation_rate=dilation_sizes[3], trainable=flg,
                      activation=relu, conv_num=3)
    conv_dilation_12_2 = conv(previous_pref=None, x=conv_dilation_12_1, output_channel=block_filters[4],
                      filter_size=filter_sizes[4], dilation_rate=dilation_sizes[4], trainable=flg,
                      activation=relu, conv_num=4)
    tensor_input = torch.cat([conv_dilation_21_1, conv_dilation_21_2, conv_dilation_12_1, conv_dilation_12_2], dim=1)
    conv_output = conv(previous_pref=None, x=tensor_input, output_channel=block_filters[5],
                      filter_size=filter_sizes[5], dilation_rate=dilation_sizes[5], trainable=flg,
                      activation=relu, conv_num=5)
    tensor_output = torch.cat([ae_inputs, conv_output], dim=1)
    conv_final_output = conv(previous_pref=None, x=tensor_input, output_channel=block_filters[6],
                      filter_size=filter_sizes[6], dilation_rate=dilation_sizes[6], trainable=flg,
                      activation=None, conv_num=6)
    ae_outputs = conv_final_output
    return ae_outputs