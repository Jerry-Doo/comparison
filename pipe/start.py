# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import sys
import os
import argparse
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

# ---- add pipe dir ----
pipe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipe_dir not in sys.path:
    sys.path.insert(0, pipe_dir)

from loss import *
from model import *
from dataset import *
from metric import *

# ---- ablation support (AG-NAFNet only) ----
_ABLATION_IMPORT_ERROR = None
try:
    from ablation_agnafnet import build_model_for_ablation, ablation_variants
    _ABLATION_AVAILABLE = True
except Exception as _e:
    build_model_for_ablation = None  # type: ignore
    ablation_variants = None  # type: ignore
    _ABLATION_AVAILABLE = False
    _ABLATION_IMPORT_ERROR = _e


# -------------------------
# small helpers
# -------------------------

def stats_graph(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {total_params}')


def _select_amp_dtype():
    # Inference AMP dtype choice: bf16 if truly supported, else fp16
    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@contextmanager
def autocast_infer(enabled: bool, dtype: torch.dtype):
    """
    Cross-version autocast wrapper.
    - torch.amp.autocast needs device_type in many versions
    - torch.cuda.amp.autocast works without device_type but may warn deprecation
    """
    if not enabled or (not torch.cuda.is_available()):
        yield
        return

    # try new API first
    try:
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            yield
    except Exception:
        from torch.cuda.amp import autocast
        with autocast(dtype=dtype, enabled=True):
            yield


def _safe_load_state_dict(model: nn.Module, state_dict: dict):
    """
    Load DP/non-DP checkpoints robustly (handle 'module.' prefix).
    """
    # 直接尝试加载 state_dict（非DP情况下会成功）
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception as e:
        print(f"Initial load failed: {e}")

    # 尝试去掉 'module.' 前缀（适用于 DataParallel 保存的模型）
    if len(state_dict) > 0:
        k0 = next(iter(state_dict.keys()))
        if k0.startswith("module."):
            stripped = {k[7:]: v for k, v in state_dict.items()}  # 去掉 'module.'
            try:
                model.load_state_dict(stripped, strict=True)
                print("Successfully loaded without 'module.' prefix")
                return
            except Exception as e:
                print(f"Load without 'module.' failed: {e}")
                model.load_state_dict(stripped, strict=False)
                print("Loaded without 'module.' prefix (non-strict)")
                return
        else:
            # 如果没有 'module.' 前缀，尝试添加 'module.'（适用于数据并行模型）
            added = {("module." + k): v for k, v in state_dict.items()}  # 添加 'module.'
            try:
                model.load_state_dict(added, strict=True)
                print("Successfully loaded with 'module.' prefix")
                return
            except Exception as e:
                print(f"Load with 'module.' failed: {e}")
                model.load_state_dict(added, strict=False)
                print("Loaded with 'module.' prefix (non-strict)")
                return

    # 最后回退方法（不强制匹配，可以处理丢失的参数）
    print("Final load attempt (strict=False)")
    model.load_state_dict(state_dict, strict=False)


def _load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    return ckpt, {"model_state_dict": ckpt}


def _maybe_dataparallel_train(model, gpu_number: int):
    # training uses DP exactly as you used before
    if gpu_number > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= gpu_number:
        print(f"Using {gpu_number} GPUs for DataParallel.")
        return nn.DataParallel(model, device_ids=list(range(gpu_number)))
    print("Running on single GPU or CPU.")
    return model


def _inference_use_single_gpu_for_speed(model, gpu_number: int):
    # For eval/output with small batch, DP usually slows down.
    # Keep single GPU by default for speed.
    # If you really want DP in inference, set env FORCE_DP_INFER=1
    force_dp = os.environ.get("FORCE_DP_INFER", "0") == "1"
    if (not force_dp) or (gpu_number <= 1) or (not torch.cuda.is_available()):
        return model, 1

    avail = torch.cuda.device_count()
    use_gpus = min(gpu_number, avail)
    if use_gpus <= 1:
        return model, 1
    print(f"[Infer] FORCE_DP_INFER=1, using {use_gpus} GPUs for DataParallel.")
    return nn.DataParallel(model, device_ids=list(range(use_gpus))), use_gpus


# ============================================================
# Ablation helpers (AG-NAFNet only; opt-in)
# ============================================================

_AGNAF_MOD = None

_AG_MODEL_NAMES = {
    "ag_nafnet_itof",
    "ag_nafnet_itof_v3",
    "amp_guided_nafnet",
    "ag_nafnet",
    "nafnet_amp",
}


def _require_ablation_available():
    if not _ABLATION_AVAILABLE:
        raise ImportError(
            "Ablation support is not available. "
            "Please ensure `pipe/ablation_agnafnet.py` exists and is importable. "
            f"Original import error: {repr(_ABLATION_IMPORT_ERROR)}"
        )


def _find_agnaf_file() -> str:
    """
    Try to find network/ag_nafnet_itof.py in common locations.
    """
    cand1 = os.path.join(pipe_dir, "network", "ag_nafnet_itof.py")  # project_root/network/...
    cand2 = os.path.join(os.path.dirname(__file__), "network", "ag_nafnet_itof.py")  # pipe/network/...
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError(f"Cannot find ag_nafnet_itof.py at: {cand1} or {cand2}")


def _load_agnaf_module():
    """
    Load ag_nafnet_itof module with stable identity for patching (even if network isn't a package).
    """
    global _AGNAF_MOD
    if _AGNAF_MOD is not None:
        return _AGNAF_MOD

    # Try standard import if available (network is a package)
    try:
        import network.ag_nafnet_itof as agmod
        _AGNAF_MOD = agmod
        return _AGNAF_MOD
    except Exception:
        pass

    ag_py = _find_agnaf_file()
    spec = importlib.util.spec_from_file_location("ag_nafnet_itof_mod", ag_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {ag_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _AGNAF_MOD = mod
    return _AGNAF_MOD


def build_model_with_ablation(
    model_name: str,
    ablation: str,
    flg: bool,
    regular: float,
    batch_size: int,
    deformable_range: float,
    device: torch.device,
    init_ckpt: str = "",
    init_mode: str = "filter",
):
    """
    - Default behavior (ablation=base, init_ckpt empty): exactly the same as your original get_network().
    - If ablation != base (AG-NAFNet only): build from ag_nafnet_itof.py and apply patch.
    - If init_ckpt provided in training (AG-NAFNet only): warm-start weights with flexible loader.
    """
    mn = model_name.lower()

    # If user didn't request ablation/warm-start, keep ORIGINAL semantics.
    if (ablation == "base") and (not init_ckpt or str(init_ckpt).strip() == ""):
        return get_network(name=model_name, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range).to(device)

    # Ablation / warm-start path: only implemented for AG models
    if mn not in _AG_MODEL_NAMES:
        if ablation != "base":
            raise ValueError(f"Ablation '{ablation}' is only supported for AG-NAFNet models. Got model_name={model_name}")
        # For non-AG model warm-start, fall back to original behavior (no init logic here to keep semantics).
        return get_network(name=model_name, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range).to(device)

    _require_ablation_available()
    agmod = _load_agnaf_module()

    base_kwargs = dict(
        flg=flg,
        regular=regular,
        batch_size=batch_size,
        deformable_range=deformable_range,
    )

    use_init = bool(flg) and (init_ckpt is not None) and (str(init_ckpt).strip() != "")

    model, meta = build_model_for_ablation(
        model_mod=agmod,
        variant_name=ablation,
        model_name=model_name,
        base_net_kwargs=base_kwargs,
        init_ckpt=init_ckpt if use_init else None,
        init_mode=init_mode,
        device=device,
    )
    print("[AB_META]", meta)
    return model


# -------------------------
# process_inputs (unchanged from your logic)
# -------------------------

def process_inputs(features, labels, params, mode, device):
    depth_kinect = None
    amplitude_kinect = None
    rgb_kinect = None
    gt_msk = None
    loss_mask_dict = {}

    for key in features:
        features[key] = features[key].to(device)
    for key in labels:
        if labels[key] is not None:
            labels[key] = labels[key].to(device)

    if params['training_set'] == 'tof_FT3':
        if params['output_flg']:
            gt = None
            gt_msk = (features['noisy'] > 10.0).float()
            loss_mask_dict['gt_msk'] = gt_msk
        else:
            gt = labels['gt']
            gt_msk = (gt > 1e-4).float()
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = ((depth_kinect > 10) & (depth_kinect < 4095)).float()

    elif params['training_set'] == 'TB':
        if params['output_flg']:
            gt = None
            gt_msk = None
            loss_mask_dict['gt_msk'] = gt_msk
        else:
            gt = labels['gt']
            gt_msk = (gt > 1e-4).float()
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = (depth_kinect < 2.0).float()
        depth_kinect_msk_tmp = (depth_kinect > 1e-4).float()
        depth_kinect_msk = depth_kinect_msk * depth_kinect_msk_tmp

    elif params['training_set'] == 'RGBDD':
        if params['output_flg']:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = (gt > 1e-4).float()
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        intensity = features['intensity']
        rgb = features['rgb']

        depth_kinect = full
        amplitude_kinect = intensity
        rgb_kinect = rgb
        depth_kinect_msk = (depth_kinect < 1.0).float()
        depth_kinect_msk_tmp = (depth_kinect > 10.0/4095.0).float()
        depth_kinect_msk = depth_kinect_msk * depth_kinect_msk_tmp

    elif params['training_set'] == 'cornellbox':
        if params['output_flg']:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = ((gt < 1e3).float() * (gt > 0.0000001).float())
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        amplitude = features['amplitude']

        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk

    elif params['training_set'] == 'FLAT':
        if params['output_flg']:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = ((gt < 1e3).float() * (gt > 0.00001).float())
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        amplitude = features['amplitude']

        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk

    elif params['training_set'] == 'Agresti_S1':
        if params['output_flg']:
            gt = None
        else:
            gt = labels['gt']
            gt_msk = ((gt < 1e3).float() * (gt > 0.0000001).float())
            loss_mask_dict['gt_msk'] = gt_msk

        full = features['noisy']
        amplitude = features['amplitude']
        depth_kinect = full
        amplitude_kinect = amplitude
        depth_kinect_msk = gt_msk

    else:
        gt = labels['gt'] if not params['output_flg'] else None
        depth_kinect = features['noisy']
        amplitude_kinect = features.get('amplitude', None)
        rgb_kinect = features.get('rgb', None)
        depth_kinect_msk = None

    loss_mask_dict['depth_kinect_msk'] = depth_kinect_msk
    if gt_msk is not None:
        loss_mask_dict['depth_kinect_with_gt_msk'] = gt_msk * depth_kinect_msk
        loss_mask_dict['gt_msk'] = gt_msk
    else:
        loss_mask_dict['depth_kinect_with_gt_msk'] = depth_kinect_msk

    if params['training_set'] in ['tof_FT3', 'TB', 'RGBDD']:
        inputs = torch.cat([depth_kinect, amplitude_kinect, rgb_kinect], dim=1)
    elif params['training_set'] == 'FLAT':
        inputs = torch.cat([
            depth_kinect[:, 1:2],
            depth_kinect[:, 2:3] - depth_kinect[:, 1:2],
            depth_kinect[:, 0:1] - depth_kinect[:, 1:2],
            amplitude_kinect[:, 2:3] / (amplitude_kinect[:, 1:2] + 1e-8),
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 1:2] + 1e-8)
        ], dim=1)
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1

    elif params['training_set'] == 'cornellbox':
        inputs = torch.cat([
            depth_kinect[:, 2:3],
            depth_kinect[:, 0:1] - depth_kinect[:, 2:3],
            depth_kinect[:, 1:2] - depth_kinect[:, 2:3],
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 2:3] + 1e-8),
            amplitude_kinect[:, 1:2] / (amplitude_kinect[:, 2:3] + 1e-8)
        ], dim=1)
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1

    elif params['training_set'] == 'Agresti_S1':
        inputs = torch.cat([
            depth_kinect[:, 2:3],
            depth_kinect[:, 0:1] - depth_kinect[:, 2:3],
            depth_kinect[:, 1:2] - depth_kinect[:, 2:3],
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 2:3] + 1e-8),
            amplitude_kinect[:, 1:2] / (amplitude_kinect[:, 2:3] + 1e-8)
        ], dim=1)
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1
    else:
        inputs = depth_kinect

    final_loss_msk = loss_mask_dict[params['loss_mask']]
    return inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect


# -------------------------
# TRAIN (keep your original semantics)
# -------------------------

def dataset_training(train_data_path, evaluate_data_path, model_dir, loss_fn, learning_rate, batch_size, traing_steps,
                     evaluate_steps, deformable_range, model_name, checkpoint_steps, loss_mask, gpu_Number,
                     training_set, image_shape, samples_number, add_gradient, decay_epoch,
                     ablation="base", init_ckpt="", init_mode="filter"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model_with_ablation(
        model_name=model_name,
        ablation=ablation,
        flg=True,
        regular=0.1,
        batch_size=batch_size,
        deformable_range=deformable_range,
        device=device,
        init_ckpt=init_ckpt,
        init_mode=init_mode,
    )

    if model_name.lower() == 'sample_pyramid_add_kpn':
        dummy_input = torch.randn(batch_size, 2, 384, 512).to(device)
        _ = model(dummy_input)

    model = _maybe_dataparallel_train(model, gpu_Number)
    stats_graph(model)

    train_loader = get_input_fn(training_set=training_set, filenames=train_data_path, height=image_shape[0], width=image_shape[1],
                                shuffle=True, repeat_count=1, batch_size=batch_size)
    eval_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                               shuffle=False, repeat_count=1, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batches_per_epoch = len(train_loader) if len(train_loader) > 0 else 1
    decay_interval_steps = decay_epoch * batches_per_epoch

    global_step = 0
    num_epochs_to_run = (traing_steps + batches_per_epoch - 1) // batches_per_epoch

    main_pbar = tqdm(total=traing_steps, desc="Training Progress", unit="step")

    for epoch in range(num_epochs_to_run):
        model.train()
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_to_run}", leave=False)
        for batch_idx, (features, labels) in enumerate(epoch_pbar):
            if global_step >= traing_steps:
                break

            inputs, gt, final_loss_msk, _, _, _ = process_inputs(
                features, labels,
                {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask},
                'train', device
            )
            optimizer.zero_grad()

            depth_outs, _ = model(inputs)

            if add_gradient == 'sobel_gradient':
                loss_1 = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                loss_2 = get_supervised_loss('sobel_gradient', depth_outs * final_loss_msk, gt * final_loss_msk, final_loss_msk)
                loss = loss_1 + 1.0 * loss_2
            else:
                loss = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)

            loss.backward()
            optimizer.step()

            main_pbar.update(1)
            epoch_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.8f}',
                'Step': f'{global_step+1}/{traing_steps}'
            })

            if (global_step + 1) % 20 == 0:
                print(f'Global Step: {global_step+1}/{traing_steps}, Epoch: {epoch+1}, Batch: {batch_idx+1}/{batches_per_epoch}, Training Loss: {loss.item():.6f}, Current LR: {optimizer.param_groups[0]["lr"]:.8f}')

            global_step += 1
            if (global_step % decay_interval_steps == 0) and (global_step < traing_steps):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']} at step {global_step}")

        epoch_pbar.close()
        if global_step >= traing_steps:
            break

        # (keep your original eval strategy)
        eval_interval = max(evaluate_steps // batches_per_epoch, 1)
        if (epoch + 1) % eval_interval == 0:
            print(f'\n--- Starting Evaluation for Epoch {epoch+1} ---')
            model.eval()
            eval_loss_total = 0
            ori_mae_total, pre_mae_total = 0.0, 0.0
            pre_mae_25_total, pre_mae_50_total, pre_mae_75_total = 0.0, 0.0, 0.0
            count = 0

            with torch.no_grad():
                eval_pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
                for eval_batch_idx, (features, labels) in enumerate(eval_pbar):
                    inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect = process_inputs(
                        features, labels,
                        {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask},
                        'eval', device
                    )

                    depth_outs, _ = model(inputs)

                    if add_gradient == 'sobel_gradient':
                        loss_1 = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                        loss_2 = get_supervised_loss('sobel_gradient', depth_outs * final_loss_msk, gt * final_loss_msk, final_loss_msk)
                        eval_loss = loss_1 + 1.0 * loss_2
                    else:
                        eval_loss = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                    eval_loss_total += eval_loss.item()

                    if training_set == 'tof_FT3':
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)
                    elif training_set == 'RGBDD':
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)
                    elif training_set == 'cornellbox':
                        depth_kinect_m = depth_kinect[:, 2:3] * 100.0
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs*100.0, depth_kinect_m, gt*100.0, final_loss_msk)
                    elif training_set == 'FLAT':
                        depth_kinect_m = depth_kinect[:, 1:2] * 100.0
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs*100.0, depth_kinect_m, gt*100.0, final_loss_msk)
                    elif training_set == 'Agresti_S1':
                        depth_kinect_m = depth_kinect[:, 2:3] * 100.0
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs*100.0, depth_kinect_m, gt*100.0, final_loss_msk)
                    elif training_set == 'TB':
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs*200.0, depth_kinect*200.0, gt*200.0, final_loss_msk)
                    else:
                        ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)

                    ori_mae_total += ori_mae.item()
                    pre_mae_total += pre_mae.item()
                    pre_mae_25_total += p25.item()
                    pre_mae_50_total += p50.item()
                    pre_mae_75_total += p75.item()
                    count += 1

                    eval_pbar.set_postfix({'EvalLoss': f'{eval_loss.item():.6f}', 'PreMAE': f'{pre_mae.item():.6f}'})
                eval_pbar.close()

            avg_eval_loss = eval_loss_total / count if count > 0 else 0
            print(f'Epoch {epoch+1} Evaluation Avg Loss: {avg_eval_loss:.6f}')

            checkpoint_save_step = (epoch + 1) * batches_per_epoch
            if checkpoint_save_step >= int(checkpoint_steps) or (epoch + 1) == num_epochs_to_run:
                checkpoint_path = os.path.join(model_dir, f'model.ckpt-{checkpoint_save_step}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_eval_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    main_pbar.close()
    print(f"\nTraining finished after {global_step} steps.")


# -------------------------
# TEST / EVAL (unchanged)
# -------------------------

def dataset_testing(evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                    loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape, add_gradient,
                    ablation="base"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model_with_ablation(
        model_name=model_name,
        ablation=ablation,
        flg=False,
        regular=0.1,
        batch_size=batch_size,
        deformable_range=deformable_range,
        device=device,
    )
    model, _ = _inference_use_single_gpu_for_speed(model, gpu_Number)

    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, f'model.ckpt-{checkpoint_steps}.pth')
    if not os.path.exists(checkpoint_path):
        ckpts = [f for f in os.listdir(model_dir) if f.startswith('model.ckpt-') and f.endswith('.pth')]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
        latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(model_dir, latest_ckpt)
        print(f"Warning: ckpt {checkpoint_steps} not found. Loading latest: {latest_ckpt}")

    state_dict, _ = _load_checkpoint(checkpoint_path, device)
    _safe_load_state_dict(model, state_dict)
    model.eval()

    eval_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                               shuffle=False, repeat_count=1, batch_size=batch_size)

    amp_dtype = _select_amp_dtype()
    use_amp = (device.type == "cuda")

    eval_loss_total = 0.0
    ori_mae_total, pre_mae_total = 0.0, 0.0
    pre_mae_25_total, pre_mae_50_total, pre_mae_75_total = 0.0, 0.0, 0.0
    count = 0

    print(f'\n--- Starting Testing from checkpoint {checkpoint_steps} ---')
    with torch.inference_mode():
        test_pbar = tqdm(eval_loader, desc="Testing", leave=False)
        for features, labels in test_pbar:
            inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect = process_inputs(
                features, labels,
                {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask},
                'eval', device
            )

            with autocast_infer(use_amp, amp_dtype):
                depth_outs, _ = model(inputs)

            # compute loss in fp32 for stable numbers
            depth_outs_f = depth_outs.float()
            gt_f = gt.float()
            msk_f = final_loss_msk.float()

            if add_gradient == 'sobel_gradient':
                loss_1 = get_supervised_loss(loss_fn, depth_outs_f, gt_f, msk_f)
                loss_2 = get_supervised_loss('sobel_gradient', depth_outs_f * msk_f, gt_f * msk_f, msk_f)
                eval_loss = loss_1 + 1.0 * loss_2
            else:
                eval_loss = get_supervised_loss(loss_fn, depth_outs_f, gt_f, msk_f)

            eval_loss_total += float(eval_loss.item())

            if training_set == 'tof_FT3':
                ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs_f, depth_kinect.float(), gt_f, msk_f)
            elif training_set == 'TB':
                ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs_f*200.0, depth_kinect.float()*200.0, gt_f*200.0, msk_f)
            else:
                ori_mae, pre_mae, p25, p50, p75 = get_metrics_mae(depth_outs_f, depth_kinect.float(), gt_f, msk_f)

            ori_mae_total += float(ori_mae.item())
            pre_mae_total += float(pre_mae.item())
            pre_mae_25_total += float(p25.item())
            pre_mae_50_total += float(p50.item())
            pre_mae_75_total += float(p75.item())
            count += 1

            test_pbar.set_postfix({'EvalLoss': f'{eval_loss.item():.4f}', 'PreMAE': f'{pre_mae.item():.4f}'})
        test_pbar.close()

    avg_eval_loss = eval_loss_total / count if count > 0 else 0
    avg_ori_mae = ori_mae_total / count if count > 0 else 0
    avg_pre_mae = pre_mae_total / count if count > 0 else 0
    avg_p25 = pre_mae_25_total / count if count > 0 else 0
    avg_p50 = pre_mae_50_total / count if count > 0 else 0
    avg_p75 = pre_mae_75_total / count if count > 0 else 0

    print(f'Testing Results from checkpoint {checkpoint_steps}:')
    print(f'  Avg Loss: {avg_eval_loss:.6f}')
    print(f'  Ori MAE:  {avg_ori_mae:.6f}')
    print(f'  Pre MAE:  {avg_pre_mae:.6f}')
    print(f'  Pre MAE 25%: {avg_p25:.6f}')
    print(f'  Pre MAE 50%: {avg_p50:.6f}')
    print(f'  Pre MAE 75%: {avg_p75:.6f}')
    print(f'--- End Testing ---')


# -------------------------
# OUTPUT (unchanged)
# -------------------------

def dataset_output(result_path, evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                   loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape,
                   ablation="base"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model_with_ablation(
        model_name=model_name,
        ablation=ablation,
        flg=False,
        regular=0.1,
        batch_size=batch_size,
        deformable_range=deformable_range,
        device=device,
    )
    model, _ = _inference_use_single_gpu_for_speed(model, gpu_Number)

    checkpoint_path = os.path.join(model_dir, f'model.ckpt-{checkpoint_steps}.pth')
    if not os.path.exists(checkpoint_path):
        ckpts = [f for f in os.listdir(model_dir) if f.startswith('model.ckpt-') and f.endswith('.pth')]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
        latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(model_dir, latest_ckpt)
        print(f"Warning: ckpt {checkpoint_steps} not found. Loading latest: {latest_ckpt}")

    state_dict, _ = _load_checkpoint(checkpoint_path, device)
    _safe_load_state_dict(model, state_dict)
    model.eval()

    output_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                                 shuffle=False, repeat_count=1, batch_size=batch_size)

    # dirs
    os.makedirs(result_path, exist_ok=True)
    pre_depth_dir = os.path.join(result_path, 'pre_depth')
    depth_input_dir = os.path.join(result_path, 'depth_input')
    amplitude_dir = os.path.join(result_path, 'amplitude')
    depth_input_png_dir = os.path.join(result_path, 'depth_input_png')
    pre_depth_png_dir = os.path.join(result_path, 'pre_depth_png')
    amplitude_png_dir = os.path.join(result_path, 'amplitude_png')

    for d in [pre_depth_dir, depth_input_dir, amplitude_dir, depth_input_png_dir, pre_depth_png_dir, amplitude_png_dir]:
        os.makedirs(d, exist_ok=True)

    # colormap LUT (faster than calling colorize_img each time)
    def _lut(cmap_name: str):
        cmap = plt.cm.get_cmap(cmap_name)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        return lut
    JET = _lut("jet")
    VIR = _lut("viridis")

    def save_color(arr2d: np.ndarray, save_path: str, lut: np.ndarray):
        a = arr2d.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-6:
            idx = np.zeros_like(a, dtype=np.uint8)
        else:
            idx = np.clip(np.round((a - mn) / (mx - mn) * 255.0), 0, 255).astype(np.uint8)
        rgb = lut[idx]  # [H,W,3]
        Image.fromarray(rgb).save(save_path)

    amp_dtype = _select_amp_dtype()
    use_amp = (device.type == "cuda")

    print(f'\n--- Starting Output Generation from checkpoint {checkpoint_steps} ---')
    idx_global = 0
    with torch.inference_mode():
        pbar = tqdm(output_loader, desc="Output Generation", leave=False)
        for features, labels in pbar:
            inputs, _, _, depth_kinect, amplitude_kinect, _ = process_inputs(
                features, labels,
                {'training_set': training_set, 'output_flg': True, 'loss_mask': loss_mask},
                'predict', device
            )

            with autocast_infer(use_amp, amp_dtype):
                depth_outs, _ = model(inputs)

            # move to cpu numpy
            pre_depths = depth_outs.float().cpu().numpy()
            in_depths = depth_kinect.float().cpu().numpy()
            amps = amplitude_kinect.float().cpu().numpy()

            B = pre_depths.shape[0]
            for b in range(B):
                i = idx_global
                idx_global += 1

                pre_depth = np.squeeze(pre_depths[b])
                input_depth = np.squeeze(in_depths[b])
                amplitude = np.squeeze(amps[b])

                # raw
                pre_depth.astype(np.float32).tofile(os.path.join(pre_depth_dir, str(i)))
                input_depth.astype(np.float32).tofile(os.path.join(depth_input_dir, str(i)))
                amplitude.astype(np.float32).tofile(os.path.join(amplitude_dir, str(i)))

                # png
                save_color(input_depth, os.path.join(depth_input_png_dir, f"{i}.png"), JET)
                save_color(pre_depth, os.path.join(pre_depth_png_dir, f"{i}.png"), JET)
                save_color(amplitude, os.path.join(amplitude_png_dir, f"{i}.png"), VIR)

        pbar.close()

    print(f'--- Output Generation Finished. Results saved to {result_path} ---')


# -------------------------
# main
# -------------------------

if __name__ == '__main__':
    # speed hints (safe)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description='Script for training / eval / output')
    parser.add_argument("-t", "--trainingSet", default='FLAT_reflection_s5', type=str)
    parser.add_argument("-m", "--modelName", default="deformable_kpn", type=str)
    parser.add_argument("-l", "--lr", default=1e-5, type=float)
    parser.add_argument("-i", "--imageShape", nargs='+', type=int, default=[239, 320])
    parser.add_argument("-b", "--batchSize", type=int, default=4)
    parser.add_argument("-s", "--steps", type=int, default=4000)
    parser.add_argument("-e", "--evalSteps", default=400, type=int)
    parser.add_argument("-o", "--lossType", default="mean_l2", type=str)
    parser.add_argument("-d", "--deformableRange", default=192.0, type=float)
    parser.add_argument("-f", "--flagMode", default='train', type=str)
    parser.add_argument("-p", "--postfix", default=None, type=str)
    parser.add_argument("-c", "--checkpointSteps", default="199200", type=str)
    parser.add_argument("-k", "--lossMask", default='gt_msk', type=str)
    parser.add_argument("-g", "--gpuNumber", default=1, type=int)
    parser.add_argument("--samplesNumber", default=5800, type=int)
    parser.add_argument("--addGradient", default='sobel_gradient', type=str)
    parser.add_argument("--decayEpoch", default=2, type=int)
    parser.add_argument("--shmFlag", default=False, type=bool)

    # ---- ablation args (optional; do not affect old commands) ----
    if _ABLATION_AVAILABLE:
        _ab_choices = list(ablation_variants().keys())
    else:
        _ab_choices = ["base"]
    parser.add_argument("--ablation", type=str, default="base", choices=_ab_choices,
                        help="(AG-NAFNet only) ablation variant name. default=base")
    parser.add_argument("--initCkpt", type=str, default="",
                        help="(train only) warm-start from a baseline checkpoint (.pth)")
    parser.add_argument("--initMode", type=str, default="filter", choices=["strict", "nonstrict", "filter", "auto"],
                        help="how to load initCkpt (filter recommended for structure-changing variants)")

    args = parser.parse_args()

    # create separate folder for ablation variants
    if args.ablation != "base":
        _require_ablation_available()
        tag = f"ab_{args.ablation}"
        if args.postfix is None:
            args.postfix = tag
        else:
            # avoid double append
            if tag not in args.postfix:
                args.postfix = args.postfix + "_" + tag

    if args.shmFlag:
        dataset_dir = '/dev/shm/dataset/tfrecords'
    else:
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/dataset')

    model_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/models')
    model_dir = os.path.join(model_base_dir, args.modelName)
    os.makedirs(model_dir, exist_ok=True)

    if args.modelName.startswith('deformable'):
        mkdir_name = args.trainingSet + '_' + args.lossType + '_dR' + str(args.deformableRange)
    else:
        mkdir_name = args.trainingSet + '_' + args.lossType

    if args.postfix is not None:
        model_dir = os.path.join(model_dir, mkdir_name + '_' + args.postfix)
    else:
        model_dir = os.path.join(model_dir, mkdir_name)
    os.makedirs(model_dir, exist_ok=True)

    output_dir = os.path.join(model_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = os.path.join(dataset_dir, args.trainingSet)
    train_data_path = os.path.join(dataset_path, args.trainingSet + '_train')
    evaluate_data_path = os.path.join(dataset_path, args.trainingSet + '_test')

    # run
    if args.flagMode == 'train':
        dataset_training(
            train_data_path=train_data_path,
            evaluate_data_path=evaluate_data_path,
            loss_fn=args.lossType,
            model_dir=model_dir,
            learning_rate=args.lr,
            batch_size=args.batchSize,
            traing_steps=args.steps,
            evaluate_steps=args.evalSteps,
            deformable_range=args.deformableRange,
            model_name=args.modelName,
            checkpoint_steps=args.checkpointSteps,
            loss_mask=args.lossMask,
            gpu_Number=args.gpuNumber,
            training_set=args.trainingSet,
            image_shape=args.imageShape,
            samples_number=args.samplesNumber,
            add_gradient=args.addGradient,
            decay_epoch=args.decayEpoch,
            ablation=args.ablation,
            init_ckpt=args.initCkpt,
            init_mode=args.initMode,
        )
    elif args.flagMode in ['eval_TD', 'eval_ED']:
        dataset_testing(
            evaluate_data_path=evaluate_data_path if args.flagMode == 'eval_ED' else train_data_path,
            model_dir=model_dir,
            loss_fn=args.lossType,
            batch_size=args.batchSize,
            checkpoint_steps=args.checkpointSteps,
            deformable_range=args.deformableRange,
            model_name=args.modelName,
            loss_mask=args.lossMask,
            gpu_Number=args.gpuNumber,
            training_set=args.trainingSet,
            image_shape=args.imageShape,
            add_gradient=args.addGradient,
            ablation=args.ablation,
        )
    else:  # output
        dataset_output(
            result_path=output_dir,
            evaluate_data_path=evaluate_data_path,
            model_dir=model_dir,
            loss_fn=args.lossType,
            batch_size=args.batchSize,
            checkpoint_steps=args.checkpointSteps,
            deformable_range=args.deformableRange,
            model_name=args.modelName,
            loss_mask=args.lossMask,
            gpu_Number=args.gpuNumber,
            training_set=args.trainingSet,
            image_shape=args.imageShape,
            ablation=args.ablation,
        )
