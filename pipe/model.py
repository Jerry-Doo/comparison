import sys
sys.path.insert(0, './network/')

##################
## Method
from network.dear_kpn_no_rgb_DeepToF import DearKPNNoRGBDeepToF
from network.dear_kpn_no_rgb import DearKPNNoRGB
from network.sample_pyramid_add_kpn import SamplePyramidAddKPN
from network.pyramid_corr_multi_frame_denoising import PyramidCorrMaskMultiFrameDenoising
##################

NETWORK_NAME = {
    'dear_kpn_no_rgb_DeepToF': DearKPNNoRGBDeepToF,
    'dear_kpn_no_rgb': DearKPNNoRGB,
    'sample_pyramid_add_kpn': SamplePyramidAddKPN,
    'pyramid_corr_multi_frame_denoising': PyramidCorrMaskMultiFrameDenoising,
}

ALL_NETWORKS = dict(NETWORK_NAME)

def get_network(name, *args, **kwargs):
    """
    选择并实例化网络结构（PyTorch nn.Module）
    :param name: 网络名称
    :param args: 传递给网络构造函数的参数
    :param kwargs: 传递给网络构造函数的关键字参数flg, regular, batch_size, range
    :return: nn.Module 实例
    """
    if name not in NETWORK_NAME.keys():
        print('Unrecognized network, pick one among: {}'.format(ALL_NETWORKS.keys()))
        raise Exception('Unknown network selected')
    selected_network = ALL_NETWORKS[name]
    return selected_network(*args, **kwargs)