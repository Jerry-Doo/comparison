import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.utils.data as data
PI = 3.14159265358979323846
flg = False
dtype = torch.float32

def plane_correction(fov, h_max, w_max, fov_flag=True):
    """Calculate plane correction for depth images"""
    w_pos, h_pos = torch.meshgrid(torch.arange(w_max), torch.arange(h_max))
    
    w_max_tensor = torch.tensor(w_max, dtype=dtype)
    h_max_tensor = torch.tensor(h_max, dtype=dtype)
    
    w_pos = w_pos.unsqueeze(-1)
    h_pos = h_pos.unsqueeze(-1)
    w_pos = w_pos.float()
    h_pos = h_pos.float()
    
    if fov_flag:
        fov_pi = 63.5 * PI / 180.0
        flen_h = (h_max_tensor / 2.0) / torch.tan(torch.tensor(fov_pi / 2.0))
        flen_w = (w_max_tensor / 2.0) / torch.tan(torch.tensor(fov_pi / 2.0))
    else:
        flen_h = fov
        flen_w = fov
    
    h = (w_pos - w_max_tensor / 2.) / flen_w
    w = (h_pos - h_max_tensor / 2.) / flen_h
    norm = 1. / torch.sqrt(h ** 2 + w ** 2 + 1.)
    
    return norm

def colorize_img(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function that maps a grayscale image to a matplotlib colormap.
    By default it will normalize the input value to the range 0..1 before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size, channels, height, width]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    Returns a 3D tensor of shape [batch_size, height, width, 3].
    """
    # normalize
    vmin = torch.min(value) if vmin is None else vmin
    vmax = torch.max(value) if vmax is None else vmax
    msk = (value > vmax).float()
    value = (value - value * msk) + vmax * msk
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    
    # quantize
    indices = torch.clamp(torch.round(value[:, 0] * 255).long(), 0, 255)
    
    # gather
    color_map = plt.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = torch.tensor(color_map(np.arange(256))[:, :3], dtype=dtype)
    value = torch.index_select(colors, 0, indices.flatten()).view(indices.shape + (3,))
    return value

def preprocessing_deeptof(features, labels):
    """Basic preprocessing for DeepToF dataset"""
    return features, labels

def preprocessing_tof_FT3D2F(features, labels):
    """Preprocessing for FT3D2F dataset"""
    rgb_p = features['rgb']
    noisy_p = features['noisy']
    intensity_p = features['intensity']
    gt_p = labels['gt']
    rgb_p_2 = features['rgb_2']
    noisy_p_2 = features['noisy_2']
    intensity_p_2 = features['intensity_2']
    gt_p_2 = labels['gt_2']
    
    # RGB normalization
    rgb_list = []
    for i in range(3):
        rgb_list.append(rgb_p[:,:,i] - torch.mean(rgb_p[:,:,i]))
    rgb_list_2 = []
    for i in range(3):
        rgb_list_2.append(rgb_p_2[:,:,i] - torch.mean(rgb_p_2[:,:,i]))
    
    rgb_p = torch.stack(rgb_list, dim=-1)
    rgb_p = rgb_p[48:-48,64:-64,:]
    features['rgb'] = rgb_p
    
    rgb_p_2 = torch.stack(rgb_list_2, dim=-1)
    rgb_p_2 = rgb_p_2[48:-48,64:-64,:]
    features['rgb_2'] = rgb_p_2
    
    # Intensity
    intensity_p = intensity_p[48:-48,64:-64,:]
    features['intensity'] = intensity_p
    intensity_p_2 = intensity_p_2[48:-48,64:-64,:]
    features['intensity_2'] = intensity_p_2
    
    # Noisy depth
    noisy_p = noisy_p[48:-48,64:-64,:]
    features['noisy'] = noisy_p
    noisy_p_2 = noisy_p_2[48:-48,64:-64,:]
    features['noisy_2'] = noisy_p_2
    
    # Ground truth
    gt_p = gt_p[48:-48,64:-64,:]
    gt_p = gt_p * 2.0
    labels['gt'] = gt_p
    gt_p_2 = gt_p_2[48:-48,64:-64,:]
    gt_p_2 = gt_p_2 * 2.0
    labels['gt_2'] = gt_p_2
    
    return features, labels

def preprocessing_tof_FT3D2F_T3(features, labels):
    """Preprocessing for FT3D2F_T3 dataset"""
    gt_p = labels['gt']
    gt_p = gt_p * 2.0
    labels['gt'] = gt_p
    return features, labels

def preprocessing_tof_HAMMER(features, labels):
    """Preprocessing for HAMMER dataset"""
    return features, labels

def preprocessing_tof_FT3(features, labels):
    """Preprocessing for FT3 dataset"""
    rgb_p = features['rgb']
    noisy_p = features['noisy']
    intensity_p = features['intensity']
    gt_p = labels['gt']
    
    # RGB normalization
    rgb_list = []
    for i in range(3):
        rgb_list.append(rgb_p[:,:,i] - torch.mean(rgb_p[:,:,i]))
    
    rgb_p = torch.stack(rgb_list, dim=-1)
    rgb_p = rgb_p[48:-48,64:-64,:]
    features['rgb'] = rgb_p
    
    # Intensity
    intensity_p = intensity_p[48:-48,64:-64,:]
    features['intensity'] = intensity_p
    
    # Noisy depth
    noisy_p = noisy_p[48:-48,64:-64,:]
    features['noisy'] = noisy_p
    
    # Ground truth
    gt_p = gt_p[48:-48,64:-64,:]
    gt_p = gt_p * 2.0
    labels['gt'] = gt_p
    
    return features, labels

def preprocessing_cornellbox_2F(features, labels):
    """Preprocessing for CornellBox 2F dataset"""
    noisy_p = features['noisy']
    amplitude_p = features['amplitude']
    noisy_p_2 = features['noisy_2']
    amplitude_p_2 = features['amplitude_2']
    
    # Crop images
    noisy_p = noisy_p[108:-108, 44:-44, :]
    amplitude_p = amplitude_p[108:-108, 44:-44, :]
    gt_p = labels['gt'][108:-108, 44:-44, :]
    
    noisy_p_2 = noisy_p_2[108:-108, 44:-44, :]
    amplitude_p_2 = amplitude_p_2[108:-108, 44:-44, :]
    
    features['amplitude'] = amplitude_p
    features['noisy'] = noisy_p
    labels['gt'] = gt_p
    
    features['amplitude_2'] = amplitude_p_2
    features['noisy_2'] = noisy_p_2
    
    return features, labels

def preprocessing_cornellbox_SN(features, labels):
    """Preprocessing for CornellBox SN dataset"""
    gt_p = labels['gt'][108:-108, 44:-44, :]
    labels['gt'] = gt_p
    return features, labels

def preprocessing_cornellbox(features, labels):
    """Preprocessing for CornellBox dataset"""
    noisy_p = features['noisy']
    amplitude_p = features['amplitude']
    
    # Crop images
    noisy_p = noisy_p[44:-44, 44:-44, :]
    amplitude_p = amplitude_p[44:-44, 44:-44, :]
    gt_p = labels['gt'][44:-44, 44:-44, :]
    
    features['amplitude'] = amplitude_p
    features['noisy'] = noisy_p
    labels['gt'] = gt_p
    
    return features, labels

def preprocessing_Agresti_S1(features, labels):
    """Preprocessing for Agresti S1 dataset"""
    noisy_p = features['noisy']
    amplitude_p = features['amplitude']
    intensity_p = features['intensity']
    
    # Crop images
    noisy_p = noisy_p[7:-8, :, :]
    amplitude_p = amplitude_p[7:-8, :, :]
    gt_p = labels['gt'][7:-8, :, :]
    intensity_p = intensity_p[7:-8, :, :]
    
    features['intensity'] = intensity_p
    features['amplitude'] = amplitude_p
    features['noisy'] = noisy_p
    labels['gt'] = gt_p
    
    return features, labels

def preprocessing_RGBDD(features, labels):
    """Preprocessing for RGBDD dataset"""
    rgb_p = features['rgb']
    noisy_p = features['noisy']
    gt_p = labels['gt']
    
    # RGB normalization
    rgb_list = []
    for i in range(3):
        rgb_list.append(rgb_p[:,:,i] - torch.mean(rgb_p[:,:,i]))
    
    rgb_p = torch.stack(rgb_list, dim=-1)
    rgb_p = rgb_p.unsqueeze(0)
    rgb_p = F.interpolate(rgb_p.permute(0, 3, 1, 2), size=(144, 192), mode='bicubic', align_corners=True)
    rgb_p = rgb_p.permute(0, 2, 3, 1).squeeze(0)
    features['rgb'] = rgb_p
    features['intensity'] = rgb_p
    
    # Resize ground truth
    gt_p = gt_p.unsqueeze(0)
    gt_p = F.interpolate(gt_p.permute(0, 3, 1, 2), size=(144, 192), mode='bicubic', align_corners=True)
    gt_p = gt_p.permute(0, 2, 3, 1).squeeze(0)
    labels['gt'] = gt_p
    
    return features, labels

def preprocessing_FLAT(features, labels):
    """Preprocessing for FLAT dataset"""
    noisy_p = features['noisy']
    amplitude_p = features['amplitude']
    
    # Crop images
    noisy_p = noisy_p[20:-20, :, :]
    amplitude_p = amplitude_p[20:-20, :, :]
    gt_p = labels['gt'][20:-20, :, :]
    
    features['amplitude'] = amplitude_p
    features['noisy'] = noisy_p
    labels['gt'] = gt_p
    
    return features, labels

def preprocessing_TB(features, labels):
    """Preprocessing for TB dataset"""
    rgb_p = features['rgb']
    noisy_p = features['noisy']
    intensity_p = features['intensity']
    gt_p = labels['gt']
    
    # Crop images
    rgb_p = rgb_p[7:-8, :, :]
    features['rgb'] = rgb_p
    
    intensity_p = intensity_p[7:-8, :, :]
    features['intensity'] = intensity_p
    
    noisy_p = noisy_p[7:-8, :, :]
    features['noisy'] = noisy_p
    
    gt_p = gt_p[7:-8, :, :]
    labels['gt'] = gt_p
    
    return features, labels

def bilinear_interpolation(input, offsets, N, deformable_range):
    """
    Bilinear interpolation in PyTorch for depth sampling.
    input: [B, H, W] or [B, 1, H, W]
    offsets: [B, H, W, 2*N]
    return: [B, H, W, N]
    """
    if input.dim() == 4:
        input = input.squeeze(1)  # [B, H, W]

    B, H, W = input.shape
    _, H_o, W_o, _ = offsets.shape

    offsets = offsets.view(B, H_o, W_o, 2, N)
    coords_h = offsets[:, :, :, 0, :]  # [B, H, W, N]
    coords_w = offsets[:, :, :, 1, :]

    h0 = torch.floor(coords_h)
    w0 = torch.floor(coords_w)
    h1 = h0 + 1
    w1 = w0 + 1

    wt_h0 = h1 - coords_h
    wt_h1 = coords_h - h0
    wt_w0 = w1 - coords_w
    wt_w1 = coords_w - w0

    def safe_index(x, y, z):
        """x: B, H, W, N (float), clamp and long"""
        return torch.clamp(x, 0, z - 1).long()

    h0_idx = safe_index(h0, coords_h, H)
    h1_idx = safe_index(h1, coords_h, H)
    w0_idx = safe_index(w0, coords_w, W)
    w1_idx = safe_index(w1, coords_w, W)

    # broadcast for batch indexing
    batch_idx = torch.arange(B).view(B, 1, 1, 1).expand(-1, H, W, N)

    def get_vals(h_idx, w_idx):
        return input[batch_idx, h_idx, w_idx]  # [B, H, W, N]

    im00 = get_vals(h0_idx, w0_idx)
    im01 = get_vals(h0_idx, w1_idx)
    im10 = get_vals(h1_idx, w0_idx)
    im11 = get_vals(h1_idx, w1_idx)

    w00 = wt_h0 * wt_w0
    w01 = wt_h0 * wt_w1
    w10 = wt_h1 * wt_w0
    w11 = wt_h1 * wt_w1

    output = im00 * w00 + im01 * w01 + im10 * w10 + im11 * w11

    coords_h_pos = coords_h
    coords_w_pos = coords_w

    # mask: only allow coords within valid range
    inside_mask = (h0 >= 0) & (h1 < H) & (w0 >= 0) & (w1 < W) & \
                  (coords_h.abs() <= deformable_range) & \
                  (coords_w.abs() <= deformable_range)

    output = output * inside_mask.float()

    return output, coords_h_pos, coords_w_pos

def im2col(input, kernel_size=3):
    """
    PyTorch version of im2col for a 4D input tensor [B, H, W, C=1]
    """
    B, C, H, W = input.shape  
    assert C == 1, "Only supports single channel input for now."
    pad = (kernel_size - 1) // 2
    # Padding
    input_pad = F.pad(input, (pad, pad, pad, pad))  # pad = (left, right, top, bottom)
    # Unfold to patches: [B, C * K*K, H*W]
    patches = F.unfold(input_pad, kernel_size=kernel_size)
    # Reshape to [B, C * K*K, H, W]
    patches = patches.view(B, C * kernel_size * kernel_size, H, W)
    return patches

class DeepToFDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, height, width, transform=None):
        self.height = height
        self.width = width
        self.description = {
            "amps": "byte",
            "depth": "byte",
            "depth_ref": "byte",
        }
        self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        amps = np.frombuffer(sample['amps'], dtype=np.float32).reshape((self.height, self.width, 1))
        depth = np.frombuffer(sample['depth'], dtype=np.float32).reshape((self.height, self.width, 1))
        depth_ref = np.frombuffer(sample['depth_ref'], dtype=np.float32).reshape((self.height, self.width, 1))

        features = {
            'amps': torch.from_numpy(amps).float(),
            'depth': torch.from_numpy(depth).float()
        }
        labels = {
            'depth_ref': torch.from_numpy(depth_ref).float()
        }

        if self.transform:
            features, labels = self.transform(features, labels)

        return features, labels

def imgs_input_fn_deeptof(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    dataset = DeepToFDataset(filenames, height, width, transform=preprocessing_deeptof)

    if shuffle:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io

# CROP_TOP, CROP_BOTTOM = 48, 48
# CROP_LEFT, CROP_RIGHT = 64, 64

CROP_TOP, CROP_BOTTOM = 0, 0
CROP_LEFT, CROP_RIGHT = 0, 0

def _center_crop_hw(arr):
    return arr

class FT3Dataset(data.Dataset):
    def __init__(self, filenames, height, width, transform=None):
        self.filenames = filenames
        self.height = height
        self.width = width
        self.transform = transform

        list_path = os.path.join(filenames, 'list.txt')
        with open(list_path, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # 预先计算裁剪后的尺寸（用于占位张量）
        self.Hc = max(1, self.height - CROP_TOP - CROP_BOTTOM)
        self.Wc = max(1, self.width  - CROP_LEFT - CROP_RIGHT)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]

        # 1) noisy depth (必须)
        noisy_path = os.path.join(self.filenames, 'nToF', f'{name}_noisy_depth.mat')
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"Missing noisy depth: {noisy_path}")
        noisy_data = scipy.io.loadmat(noisy_path)['ndepth']  # [H,W] 或 [H,W,1]
        if noisy_data.ndim == 3:
            noisy_data = noisy_data[..., 0]
        # noisy = _center_crop_hw(noisy_data).astype(np.float32)                  # [Hc,Wc]
        noisy = noisy_data.astype(np.float32)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)                           # [1,Hc,Wc]

        # 2) intensity (必须)
        intensity_path = os.path.join(self.filenames, 'nToF', f'{name}_noisy_intensity.png')
        if not os.path.exists(intensity_path):
            raise FileNotFoundError(f"Missing intensity: {intensity_path}")
        inten_img = Image.open(intensity_path).convert('L').resize((self.width, self.height))
        intensity = np.array(inten_img, dtype=np.float32)                        # [H,W]
        # intensity = _center_crop_hw(intensity)
        intensity = intensity.astype(np.float32)                                   # [Hc,Wc]
        intensity_t = torch.from_numpy(intensity).unsqueeze(0)                   # [1,Hc,Wc]

        # 3) RGB（可缺省：缺就返回全 0，占位，确保键一致）
        rgb_path = os.path.join(self.filenames, 'gt_depth_rgb', f'{name}_rgb.png')
        if os.path.exists(rgb_path):
            rgb_img = Image.open(rgb_path).convert('RGB').resize((self.width, self.height))
            rgb_np = np.array(rgb_img, dtype=np.float32)                         # [H,W,3]
            for c in range(3):   # 每通道减均值（与你原逻辑一致）
                rgb_np[:, :, c] -= np.mean(rgb_np[:, :, c])
            rgb_np = _center_crop_hw(rgb_np)                                     # [Hc,Wc,3]
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1) / 255.0            # [3,Hc,Wc]
        else:
            rgb_t = torch.zeros(3, self.Hc, self.Wc, dtype=torch.float32)        # [3,Hc,Wc]

        # 4) conf（可缺省：缺就占位 0）
        conf_path = os.path.join(self.filenames, 'gt_depth_rgb', f'{name}_gt_conf.mat')
        if os.path.exists(conf_path):
            conf_data = scipy.io.loadmat(conf_path)['conf']                      # [H,W] 或 [H,W,1]
            if conf_data.ndim == 3:
                conf_data = conf_data[..., 0]
            conf = _center_crop_hw(conf_data).astype(np.float32)                 # [Hc,Wc]
            conf_t = torch.from_numpy(conf).unsqueeze(0)                         # [1,Hc,Wc]
        else:
            conf_t = torch.zeros(1, self.Hc, self.Wc, dtype=torch.float32)       # [1,Hc,Wc]

        # 5) GT（可缺省：缺就占位 0；output 模式不会用）
        gt_path = os.path.join(self.filenames, 'gt_depth_rgb', f'{name}_gt_depth.mat')
        if os.path.exists(gt_path):
            gt_data = scipy.io.loadmat(gt_path)['gt_depth']                      # [H,W] 或 [H,W,1]
            if gt_data.ndim == 3:
                gt_data = gt_data[..., 0]
            gt = _center_crop_hw(gt_data).astype(np.float32) * 2.0               # [Hc,Wc]
            gt_t = torch.from_numpy(gt).unsqueeze(0)                             # [1,Hc,Wc]
        else:
            gt_t = torch.zeros(1, self.Hc, self.Wc, dtype=torch.float32)         # [1,Hc,Wc]

        features = {
            'noisy': noisy_t,           # [1,Hc,Wc]
            'intensity': intensity_t,   # [1,Hc,Wc]
            'rgb': rgb_t,               # [3,Hc,Wc]（可能是占位）
            'conf': conf_t              # [1,Hc,Wc]（可能是占位）
        }
        labels = {'gt': gt_t}           # [1,Hc,Wc]（可能是占位）

        return features, labels

def imgs_input_fn_FT3(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    dataset = FT3Dataset(
        filenames=filenames,
        height=height,
        width=width,
        transform=None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if 4 > 0 else False
    )

# class FT3DualFrameDataset(torch.utils.data.Dataset):
#     def __init__(self, filenames, height, width, transform=None):
#         self.height = height
#         self.width = width
#         self.transform = transform

#         self.description = {
#             'noisy': 'byte',
#             'intensity': 'byte',
#             'rgb': 'byte',
#             'gt': 'byte',
#             'noisy_2': 'byte',
#             'intensity_2': 'byte',
#             'rgb_2': 'byte',
#             'gt_2': 'byte'
#         }

#         self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]

#         def decode(key, channels):
#             return np.frombuffer(example[key], dtype=np.float32).reshape(self.height, self.width, channels)

#         noisy = decode('noisy', 1)
#         intensity = decode('intensity', 1)
#         rgb = decode('rgb', 3)
#         gt = decode('gt', 1)

#         noisy_2 = decode('noisy_2', 1)
#         intensity_2 = decode('intensity_2', 1)
#         rgb_2 = decode('rgb_2', 3)
#         gt_2 = decode('gt_2', 1)

#         features = {
#             'noisy': torch.from_numpy(noisy).float(),
#             'intensity': torch.from_numpy(intensity).float(),
#             'rgb': torch.from_numpy(rgb).float(),
#             'noisy_2': torch.from_numpy(noisy_2).float(),
#             'intensity_2': torch.from_numpy(intensity_2).float(),
#             'rgb_2': torch.from_numpy(rgb_2).float()
#         }

#         labels = {
#             'gt': torch.from_numpy(gt).float(),
#             'gt_2': torch.from_numpy(gt_2).float()
#         }

#         if self.transform:
#             features, labels = self.transform(features, labels)

#         return features, labels

# def imgs_input_fn_FT3_2F(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     dataset = FT3DualFrameDataset(filenames, height, width, transform=preprocessing_tof_FT3D2F)

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle
#     )

#     return dataloader

# class FT3_2F_T3_Dataset(Dataset):
#     def __init__(self, filenames, height, width, transform=None):
#         self.filenames = filenames
#         self.height = height
#         self.width = width
#         self.transform = transform

#         self.description = {
#             "noisy": "byte",
#             "intensity": "byte",
#             "gt": "byte",
#             "noisy_2": "byte",
#             "intensity_2": "byte",
#         }

#         self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         sample = self.dataset[idx]

#         H, W = self.height, self.width

#         def decode_and_reshape(key, channels=1):
#             array = np.frombuffer(sample[key], dtype=np.float32)
#             return torch.from_numpy(array).view(H, W, channels)

#         features = {
#             'noisy': decode_and_reshape("noisy", 1),
#             'intensity': decode_and_reshape("intensity", 1),
#             'noisy_2': decode_and_reshape("noisy_2", 1),
#             'intensity_2': decode_and_reshape("intensity_2", 1),
#         }

#         labels = {
#             'gt': decode_and_reshape("gt", 1)
#         }

#         if self.transform:
#             features, labels = self.transform(features, labels)

#         return features, labels

# def imgs_input_fn_FT3_2F_T3(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     dataset = FT3_2F_T3_Dataset(
#         filenames=filenames,
#         height=height,
#         width=width,
#         transform=preprocessing_tof_FT3D2F_T3
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,
#         drop_last=True
#     )

#     return loader

# class HAMMERDataset(Dataset):
#     def __init__(self, filenames, height, width, transform=None):
#         self.height = height
#         self.width = width
#         self.transform = transform

#         self.description = {
#             'noisy': 'byte',
#             'intensity': 'byte',
#             'gt': 'byte',
#             'noisy_2': 'byte',
#             'intensity_2': 'byte',
#         }

#         self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]
#         H, W = self.height, self.width

#         def decode_and_reshape(key, channels=1):
#             data = np.frombuffer(example[key], dtype=np.float32)
#             return torch.from_numpy(data).view(H, W, channels)

#         features = {
#             'noisy': decode_and_reshape('noisy'),
#             'intensity': decode_and_reshape('intensity'),
#             'noisy_2': decode_and_reshape('noisy_2'),
#             'intensity_2': decode_and_reshape('intensity_2'),
#         }

#         labels = {
#             'gt': decode_and_reshape('gt')
#         }

#         if self.transform:
#             features, labels = self.transform(features, labels)

#         return features, labels

# def imgs_input_fn_HAMMER(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     dataset = HAMMERDataset(
#         filenames=filenames,
#         height=height,
#         width=width,
#         transform=preprocessing_tof_HAMMER
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,
#         drop_last=True
#     )

#     return dataloader

# class CornellBoxDataset(Dataset):
#     def __init__(self, filenames, height, width, transform=None):
#         self.height = height
#         self.width = width
#         self.transform = transform

#         self.description = {
#             'noisy_20MHz': 'byte',
#             'amplitude_20MHz': 'byte',
#             'noisy_50MHz': 'byte',
#             'amplitude_50MHz': 'byte',
#             'noisy_70MHz': 'byte',
#             'amplitude_70MHz': 'byte',
#             'gt': 'byte'
#         }

#         self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]
#         H, W = self.height, self.width

#         def decode(key, channels=1):
#             return torch.from_numpy(
#                 np.frombuffer(example[key], dtype=np.float32).reshape(H, W, channels)
#             )

#         # Load and stack noisy channels
#         noisy = torch.cat([
#             decode('noisy_20MHz'),
#             decode('noisy_50MHz'),
#             decode('noisy_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         # Load and stack amplitude channels
#         amplitude = torch.cat([
#             decode('amplitude_20MHz'),
#             decode('amplitude_50MHz'),
#             decode('amplitude_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         gt = decode('gt')  # (H, W, 1)

#         features = {'noisy': noisy, 'amplitude': amplitude}
#         labels = {'gt': gt}

#         if self.transform:
#             features, labels = self.transform(features, labels)

#         return features, labels

# def imgs_input_fn_cornellbox(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     dataset = CornellBoxDataset(
#         filenames=filenames,
#         height=height,
#         width=width,
#         transform=preprocessing_cornellbox  # your preprocessing function
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,
#         drop_last=True
#     )

#     return dataloader

# class CornellBox2FDataset(Dataset):
#     def __init__(self, filenames, height, width, transform=None):
#         self.height = height
#         self.width = width
#         self.transform = transform

#         self.description = {
#             'noisy_20MHz': 'byte',
#             'amplitude_20MHz': 'byte',
#             'noisy_50MHz': 'byte',
#             'amplitude_50MHz': 'byte',
#             'noisy_70MHz': 'byte',
#             'amplitude_70MHz': 'byte',
#             'gt': 'byte',
#             'noisy_2_20MHz': 'byte',
#             'amplitude_2_20MHz': 'byte',
#             'noisy_2_50MHz': 'byte',
#             'amplitude_2_50MHz': 'byte',
#             'noisy_2_70MHz': 'byte',
#             'amplitude_2_70MHz': 'byte',
#             # 'gt_2': 'byte',  # 可选
#         }

#         self.dataset = TFRecordDataset(filenames, index_path=None, description=self.description)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]
#         H, W = self.height, self.width

#         def decode(key, channels=1):
#             return torch.from_numpy(
#                 np.frombuffer(example[key], dtype=np.float32).reshape(H, W, channels)
#             )

#         # 第一帧 noisy + amplitude
#         noisy = torch.cat([
#             decode('noisy_20MHz'),
#             decode('noisy_50MHz'),
#             decode('noisy_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         amplitude = torch.cat([
#             decode('amplitude_20MHz'),
#             decode('amplitude_50MHz'),
#             decode('amplitude_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         # 第二帧 noisy_2 + amplitude_2
#         noisy_2 = torch.cat([
#             decode('noisy_2_20MHz'),
#             decode('noisy_2_50MHz'),
#             decode('noisy_2_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         amplitude_2 = torch.cat([
#             decode('amplitude_2_20MHz'),
#             decode('amplitude_2_50MHz'),
#             decode('amplitude_2_70MHz')
#         ], dim=-1)  # (H, W, 3)

#         # Ground truth
#         gt = decode('gt')  # (H, W, 1)

#         features = {
#             'noisy': noisy,
#             'amplitude': amplitude,
#             'noisy_2': noisy_2,
#             'amplitude_2': amplitude_2
#         }

#         labels = {'gt': gt}

#         if self.transform:
#             features, labels = self.transform(features, labels)

#         return features, labels

# def imgs_input_fn_cornellbox_2F(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     dataset = CornellBox2FDataset(
#         filenames=filenames,
#         height=height,
#         width=width,
#         transform=preprocessing_cornellbox_2F  # 注意调用的是 _2F 版本
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,
#         drop_last=True
#     )

#     return dataloader

# def imgs_input_fn_cornellbox_SN(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for CornellBox SN dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy_50MHz': torch.FixedLenFeature([], torch.string),
#             'amplitude_50MHz': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string),
#             'noisy_2_50MHz': torch.FixedLenFeature([], torch.string),
#             'amplitude_2_50MHz': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([height, width, 1])
#         intensity_shape = torch.tensor([height, width, 1])
#         gt_shape = torch.tensor([height, width, 1])
        
#         noisy_20MHz_raw = parsed_example['noisy_50MHz']
#         amplitude_20MHz_raw = parsed_example['amplitude_50MHz']
#         gt_raw = parsed_example['gt']
#         noisy_20MHz_raw_2 = parsed_example['noisy_2_50MHz']
#         amplitude_20MHz_raw_2 = parsed_example['amplitude_2_50MHz']
        
#         # Decode the raw bytes
#         noisy_20MHz = torch.decode_raw(noisy_20MHz_raw, torch.float32)
#         noisy_20MHz = noisy_20MHz.float()
#         noisy_20MHz = noisy_20MHz.view(noisy_shape)
        
#         amplitude_20MHz = torch.decode_raw(amplitude_20MHz_raw, torch.float32)
#         amplitude_20MHz = amplitude_20MHz.float()
#         amplitude_20MHz = amplitude_20MHz.view(intensity_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         noisy_20MHz_2 = torch.decode_raw(noisy_20MHz_raw_2, torch.float32)
#         noisy_20MHz_2 = noisy_20MHz_2.float()
#         noisy_20MHz_2 = noisy_20MHz_2.view(noisy_shape)
        
#         amplitude_20MHz_2 = torch.decode_raw(amplitude_20MHz_raw_2, torch.float32)
#         amplitude_20MHz_2 = amplitude_20MHz_2.float()
#         amplitude_20MHz_2 = amplitude_20MHz_2.view(intensity_shape)
        
#         noisy = noisy_20MHz
#         amplitude = amplitude_20MHz
#         noisy_2 = noisy_20MHz_2
#         amplitude_2 = amplitude_20MHz_2
        
#         features = {
#             'noisy': noisy, 'amplitude': amplitude,
#             'noisy_2': noisy_2, 'amplitude_2': amplitude_2
#         }
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_tof_HAMMER(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

# def imgs_input_fn_Agresti_S1(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for Agresti S1 dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy': torch.FixedLenFeature([], torch.string),
#             'intensity': torch.FixedLenFeature([], torch.string),
#             'amplitude': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([height, width, 3])
#         intensity_shape = torch.tensor([height, width, 3])
#         amplitude_shape = torch.tensor([height, width, 3])
#         gt_shape = torch.tensor([height, width, 1])
        
#         noisy_raw = parsed_example['noisy']
#         intensity_raw = parsed_example['intensity']
#         amplitude_raw = parsed_example['amplitude']
#         gt_raw = parsed_example['gt']
        
#         # Decode the raw bytes
#         noisy = torch.decode_raw(noisy_raw, torch.float32)
#         noisy = noisy.float()
#         noisy = noisy.view(noisy_shape)
        
#         intensity = torch.decode_raw(intensity_raw, torch.float32)
#         intensity = intensity.float()
#         intensity = intensity.view(intensity_shape)
        
#         amplitude = torch.decode_raw(amplitude_raw, torch.float32)
#         amplitude = amplitude.float()
#         amplitude = amplitude.view(amplitude_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         features = {'noisy': noisy, 'intensity': intensity, 'amplitude': amplitude}
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_Agresti_S1(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

# def imgs_input_fn_RGBDD(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for RGBDD dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy': torch.FixedLenFeature([], torch.string),
#             'rgb': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([144, 192, 1])
#         rgb_shape = torch.tensor([384, 512, 3])
#         gt_shape = torch.tensor([384, 512, 1])
        
#         noisy_raw = parsed_example['noisy']
#         rgb_raw = parsed_example['rgb']
#         gt_raw = parsed_example['gt']
        
#         # Decode the raw bytes
#         noisy = torch.decode_raw(noisy_raw, torch.float32)
#         noisy = noisy.float()
#         noisy = noisy.view(noisy_shape)
        
#         rgb = torch.decode_raw(rgb_raw, torch.float32)
#         rgb = rgb.float()
#         rgb = rgb.view(rgb_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         features = {'noisy': noisy, 'intensity': rgb, 'rgb': rgb}
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_RGBDD(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

# def imgs_input_fn_TB(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for TB dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy': torch.FixedLenFeature([], torch.string),
#             'intensity': torch.FixedLenFeature([], torch.string),
#             'rgb': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([height, width, 1])
#         intensity_shape = torch.tensor([height, width, 1])
#         rgb_shape = torch.tensor([height, width, 1])
#         gt_shape = torch.tensor([height, width, 1])
        
#         noisy_raw = parsed_example['noisy']
#         intensity_raw = parsed_example['intensity']
#         rgb_raw = parsed_example['rgb']
#         gt_raw = parsed_example['gt']
        
#         # Decode the raw bytes
#         noisy = torch.decode_raw(noisy_raw, torch.float32)
#         noisy = noisy.float()
#         noisy = noisy.view(noisy_shape)
        
#         intensity = torch.decode_raw(intensity_raw, torch.float32)
#         intensity = intensity.float()
#         intensity = intensity.view(intensity_shape)
        
#         rgb = torch.decode_raw(rgb_raw, torch.float32)
#         rgb = rgb.float()
#         rgb = rgb.view(rgb_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         features = {'noisy': noisy, 'intensity': intensity, 'rgb': rgb}
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_TB(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

# def imgs_input_fn_TrueBox(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for TrueBox dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy': torch.FixedLenFeature([], torch.string),
#             'amplitude': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([height, width, 1])
#         amplitude_shape = torch.tensor([height, width, 1])
#         gt_shape = torch.tensor([height, width, 1])
        
#         noisy_raw = parsed_example['noisy']
#         amplitude_raw = parsed_example['amplitude']
#         gt_raw = parsed_example['gt']
        
#         # Decode the raw bytes
#         noisy = torch.decode_raw(noisy_raw, torch.float32)
#         noisy = noisy.float()
#         noisy = noisy.view(noisy_shape)
        
#         amplitude = torch.decode_raw(amplitude_raw, torch.float32)
#         amplitude = amplitude.float()
#         amplitude = amplitude.view(amplitude_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         features = {'noisy': noisy, 'amplitude': amplitude}
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_tof_HAMMER(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

# def imgs_input_fn_FLAT(filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
#     """Create input function for FLAT dataset"""
#     def _parse_function(serialized, height=height, width=width):
#         features = {
#             'noisy': torch.FixedLenFeature([], torch.string),
#             'amplitude': torch.FixedLenFeature([], torch.string),
#             'gt': torch.FixedLenFeature([], torch.string)
#         }
        
#         parsed_example = torch.parse_single_example(serialized=serialized, features=features)
        
#         noisy_shape = torch.tensor([height, width, 3])
#         amplitude_shape = torch.tensor([height, width, 3])
#         gt_shape = torch.tensor([height, width, 1])
        
#         noisy_raw = parsed_example['noisy']
#         amplitude_raw = parsed_example['amplitude']
#         gt_raw = parsed_example['gt']
        
#         # Decode the raw bytes
#         noisy = torch.decode_raw(noisy_raw, torch.float32)
#         noisy = noisy.float()
#         noisy = noisy.view(noisy_shape)
        
#         amplitude = torch.decode_raw(amplitude_raw, torch.float32)
#         amplitude = amplitude.float()
#         amplitude = amplitude.view(amplitude_shape)
        
#         gt = torch.decode_raw(gt_raw, torch.float32)
#         gt = gt.float()
#         gt = gt.view(gt_shape)
        
#         features = {'noisy': noisy, 'amplitude': amplitude}
#         labels = {'gt': gt}
        
#         return features, labels
    
#     dataset = torch.data.TFRecordDataset(filenames=filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda features, labels: preprocessing_FLAT(features, labels)
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
    
#     dataset = dataset.repeat(repeat_count)
#     batch_dataset = dataset.batch(batch_size)
#     batch_dataset = batch_dataset.prefetch(2)
    
#     return batch_dataset

ALL_INPUT_FN = {
    # 'FLAT': imgs_input_fn_FLAT,
    # 'FLAT_reflection_s5': imgs_input_fn_FLAT,
    # 'FLAT_full_s5': imgs_input_fn_FLAT,
    # 'deeptof_reflection': imgs_input_fn_deeptof,
    'tof_FT3': imgs_input_fn_FT3,
    # 'tof_FT3D2F': imgs_input_fn_FT3_2F,
    # 'tof_FT3D2F_T1': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T3': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T3_F': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T5': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T7': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T9': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T11': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T13': imgs_input_fn_FT3_2F_T3,
    # 'tof_FT3D2F_T15': imgs_input_fn_FT3_2F_T3,
    # 'RGBDD': imgs_input_fn_RGBDD,
    # 'TB': imgs_input_fn_TB,
    # 'TrueBox': imgs_input_fn_TrueBox,
    # 'cornellbox': imgs_input_fn_cornellbox,
    # 'cornellbox_2F': imgs_input_fn_cornellbox_2F,
    # 'cornellbox_SN': imgs_input_fn_cornellbox_SN,
    # 'cornellbox_SN_F': imgs_input_fn_cornellbox_SN,
    # 'Agresti_S1': imgs_input_fn_Agresti_S1,
    # 'HAMMER': imgs_input_fn_HAMMER,
    # 'HAMMER_A': imgs_input_fn_HAMMER,
    # 'HAMMER_A_D': imgs_input_fn_HAMMER,
    # 'HAMMER_A_F': imgs_input_fn_HAMMER,
}

def get_input_fn(training_set, filenames, height, width, shuffle=False, repeat_count=1, batch_size=32):
    """Get input function for specified dataset"""
    base_input_fn = ALL_INPUT_FN[training_set]
    return base_input_fn(filenames, height, width, shuffle=shuffle, repeat_count=repeat_count, batch_size=batch_size)


