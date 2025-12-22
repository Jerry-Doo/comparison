# Recurrent Cross-Modality Fusion for ToF Depth Denoising (PyTorch Version)

This repository is a PyTorch reimplementation of the original code from [Recurrent Cross-Modality Fusion for ToF Depth Denoising](https://github.com/gtdong-ustc/recurrent_tof_denoising).

## Description

- **Original Author:** GT Dong et al.
- **Original Repository:** https://github.com/gtdong-ustc/recurrent_tof_denoising

## My Modifications

- Complete migration of the original TensorFlow codebase to PyTorch
- Adapted network architecture and training pipeline for PyTorch
- Preserved the core algorithm and experimental settings from the original work

## Usage

This project is under active development. 

## Dataset Download

The TFT3D dataset used in this project can be downloaded from the following link:
https://drive.google.com/drive/folders/1XASaOfcp3TzQJ0A2fMaXex-0eihha0vg

After downloading, please place the dataset under the following directory:
data/dataset/tof_FT3/
```
data/
└── dataset/
    └── tof_FT3/
        ├── tof_FT3_train/
        │   ├── gt_depth_rgb/
        │   ├── gt_depth_rgb_test_small_pt/
        │   ├── nToF/
        │   └── list.txt
        └── tof_FT3_test/
            ├── gt_depth_rgb/
            ├── gt_depth_rgb_test_small_pt/
            ├── nToF/
            └── list.txt
```
## Environment Setup
```bash
conda create -n tof_denoising python=3.8 -y
conda activate tof_denoising
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## How to train
```bash
python pipe/start.py \
  -b 2 \
  -s 200000 \
  -m pyramid_corr_multi_frame_denoising \
  -p size384 \
  -k depth_kinect_with_gt_msk \
  -l 0.0004 \
  -t tof_FT3 \
  -i 480 640 \
  -o mean_l1 \
  --addGradient sobel_gradient \
  -g 4 \
  -e 1200
```

## How to Inference
```bash
python pipe/start.py \
  -f output \
  -m dear_kpn_no_rgb_DeepToF \
  -t tof_FT3 \
  -o mean_l1 \
  -p size384 \
  -c 199036 \
  -b 2 \
  -i 480 640 \
  -g 1
```

## Source of comparison methods
There are four methods in the network folder, each corresponding to a different paper:

<table>
  <thead>
    <tr>
      <th align="center">Code File</th>
      <th align="center">Full Paper Title</th>
      <th align="center">Method Abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><code>dear_kpn_no_rgb.py</code></td>
      <td align="center">Burst Denoising with<br>Kernel Prediction Networks</td>
      <td align="center"><b>KPN</b></td>
    </tr>
    <tr>
      <td align="center"><code>dear_kpn_no_rgb_DeepToF.py</code></td>
      <td align="center">DeepToF: Off-the-Shelf Real-Time Correction of Multipath Interference<br>in Time-of-Flight Imaging</td>
      <td align="center"><b>DeepToF</b></td>
    </tr>
    <tr>
      <td align="center"><code>pyramid_corr_multi_frame_denoising.py</code></td>
      <td align="center">Deep End-to-End Alignment and Refinement<br>for Time-of-Flight RGB-D Module</td>
      <td align="center"><b>E2E Alignment &amp; Refinement</b></td>
    </tr>
    <tr>
      <td align="center"><code>sample_pyramid_add_kpn.py</code></td>
      <td align="center">Spatial Hierarchy Aware Residual Pyramid Network<br>for Time-of-Flight Depth Denoising</td>
      <td align="center"><b>SHARP-Net</b></td>
    </tr>
  </tbody>
</table>