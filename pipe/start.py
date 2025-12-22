# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
# 获取 pipe 目录的绝对路径
pipe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将 pipe 目录加入 sys.path，这样 pipe 下的模块都能直接导入
if pipe_dir not in sys.path:
    sys.path.insert(0, pipe_dir)
from loss import *
from model import *
from dataset import *
from metric import *

def stats_graph(model):
    """Calculate trainable parameters of the model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {total_params}')

def process_inputs(features, labels, params, mode, device):
    """Helper function to process inputs and masks, similar to original tof_net_func's input part"""
    depth_kinect = None
    amplitude_kinect = None
    rgb_kinect = None
    gt_msk = None
    loss_mask_dict = {}

    # Move all features and labels to the device
    for key in features:
        features[key] = features[key].to(device)
    for key in labels:
        if labels[key] is not None:
            labels[key] = labels[key].to(device)


    if params['training_set'] == 'tof_FT3':
        if params['output_flg']:
            gt = None # Only needed for loss/metrics, not for output mode
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
        amplitude_kinect = features.get('amplitude', None) # Safely get amplitude
        rgb_kinect = features.get('rgb', None) # Safely get rgb
        depth_kinect_msk = None # Default if not explicitly set above

    # Handle mask dependencies
    loss_mask_dict['depth_kinect_msk'] = depth_kinect_msk
    if gt_msk is not None:
        loss_mask_dict['depth_kinect_with_gt_msk'] = gt_msk * depth_kinect_msk
        loss_mask_dict['gt_msk'] = gt_msk
    else:
        loss_mask_dict['depth_kinect_with_gt_msk'] = depth_kinect_msk

    # Prepare inputs for the network
    if params['training_set'] == 'tof_FT3':
        inputs = torch.cat([depth_kinect, amplitude_kinect, rgb_kinect], dim=1)
    elif params['training_set'] == 'TB':
        inputs = torch.cat([depth_kinect, amplitude_kinect, rgb_kinect], dim=1)
    elif params['training_set'] == 'RGBDD':
        inputs = torch.cat([depth_kinect, amplitude_kinect, rgb_kinect], dim=1)
    elif params['training_set'] == 'FLAT':
        # Ensure proper indexing for [B, C, H, W]
        inputs = torch.cat([
            depth_kinect[:, 1:2], # Assuming noisy has C=3
            depth_kinect[:, 2:3] - depth_kinect[:, 1:2],
            depth_kinect[:, 0:1] - depth_kinect[:, 1:2],
            amplitude_kinect[:, 2:3] / (amplitude_kinect[:, 1:2] + 1e-8), # TF had -1, but this matches the TF data processing
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 1:2] + 1e-8) # TF had -1
        ], dim=1)
        # Re-adding -1 to match original TF logic, which was done in start.py for FLAT, cornellbox, Agresti_S1
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1
    elif params['training_set'] == 'cornellbox':
        inputs = torch.cat([
            depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,2]
            depth_kinect[:, 0:1] - depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,0] - depth_kinect[:,:,:,2]
            depth_kinect[:, 1:2] - depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,1] - depth_kinect[:,:,:,2]
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 2:3] + 1e-8), # TF amplitude_kinect[:,:,:,0] / amplitude_kinect[:,:,:,2]
            amplitude_kinect[:, 1:2] / (amplitude_kinect[:, 2:3] + 1e-8) # TF amplitude_kinect[:,:,:,1] / amplitude_kinect[:,:,:,2]
        ], dim=1)
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1
    elif params['training_set'] == 'Agresti_S1':
        inputs = torch.cat([
            depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,2]
            depth_kinect[:, 0:1] - depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,0] - depth_kinect[:,:,:,2]
            depth_kinect[:, 1:2] - depth_kinect[:, 2:3], # TF depth_kinect[:,:,:,1] - depth_kinect[:,:,:,2]
            amplitude_kinect[:, 0:1] / (amplitude_kinect[:, 2:3] + 1e-8), # TF amplitude_kinect[:,:,:,0] / amplitude_kinect[:,:,:,2]
            amplitude_kinect[:, 1:2] / (amplitude_kinect[:, 2:3] + 1e-8) # TF amplitude_kinect[:,:,:,1] / amplitude_kinect[:,:,:,2]
        ], dim=1)
        inputs[:, 3:4] = inputs[:, 3:4] - 1
        inputs[:, 4:5] = inputs[:, 4:5] - 1
    else:
        inputs = depth_kinect # Assuming depth_kinect is [B, C, H, W] already

    # Get the appropriate loss mask for current mode
    final_loss_msk = loss_mask_dict[params['loss_mask']] # Renamed to avoid clash with function param `loss_mask`

    return inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect


def dataset_training(train_data_path, evaluate_data_path, model_dir, loss_fn, learning_rate, batch_size, traing_steps,
                     evaluate_steps, deformable_range, model_name, checkpoint_steps, loss_mask, gpu_Number,
                     training_set, image_shape, samples_number, add_gradient, decay_epoch):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = get_network(name=model_name, flg=True, regular=0.1, batch_size=batch_size, deformable_range=deformable_range).to(device)
    
    if model_name.lower() == 'sample_pyramid_add_kpn':
        dummy_input = torch.randn(batch_size, 2, 384, 512).to(device)
        _ = model(dummy_input)

    if gpu_Number > 1 and torch.cuda.device_count() >= gpu_Number:
        print(f"Using {gpu_Number} GPUs for DataParallel.")
        model = nn.DataParallel(model, device_ids=list(range(gpu_Number)))
    else:
        print(f"Running on single GPU or CPU: {device}. Check GPU setup if multi-GPU intended.")

    stats_graph(model)

    train_loader = get_input_fn(training_set=training_set, filenames=train_data_path, height=image_shape[0], width=image_shape[1],
                                shuffle=True, repeat_count=1, batch_size=batch_size) # Set repeat_count=1 for proper epoch control
    eval_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                               shuffle=False, repeat_count=1, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    batches_per_epoch = len(train_loader) if len(train_loader) > 0 else 1 # Avoid division by zero
    decay_interval_steps = decay_epoch * batches_per_epoch

    global_step = 0
    # Determine number of epochs needed to reach traing_steps (total iterations)
    num_epochs_to_run = (traing_steps + batches_per_epoch - 1) // batches_per_epoch

    # Main progress bar for overall training
    main_pbar = tqdm(total=traing_steps, desc="Training Progress", unit="step")

    for epoch in range(num_epochs_to_run):
        model.train()
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_to_run}", leave=False)
        for batch_idx, (features, labels) in enumerate(epoch_pbar):
            if global_step >= traing_steps:
                break

            inputs, gt, final_loss_msk, _, _, _ = process_inputs(features, labels, {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask}, 'train', device)
            optimizer.zero_grad()
            
            depth_outs, depth_residual_every_scale = model(inputs)
            #fig, axs = plt.subplots(1, 3, figsize=(16, 5))  # 1行2列子图

            # GT 图像
            #im0 = axs[0].imshow(gt[0, 0].cpu(), cmap='jet')
            #axs[0].set_title('GT')
            #plt.colorbar(im0, ax=axs[0])

            # output 图像
            #im1 = axs[1].imshow(depth_outs[0, 0].detach().cpu(), cmap='jet')
            #axs[1].set_title('depth_outs')
            #plt.colorbar(im1, ax=axs[1])

            # Input 图像
            #im1 = axs[2].imshow(inputs[0, 0].detach().cpu(), cmap='jet')
            #axs[1].set_title('inputs')
            #plt.colorbar(im1, ax=axs[1])
            # 保存整个图像为一个文件
            #plt.tight_layout()
            #plt.savefig("gt_in_output_comparison.png")
            #plt.close()
            
            if add_gradient == 'sobel_gradient':
                loss_1 = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                loss_2 = get_supervised_loss('sobel_gradient', depth_outs * final_loss_msk, gt * final_loss_msk, final_loss_msk)
                loss = loss_1 + 1.0 * loss_2
            else:
                loss = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
            
            loss.backward()
            optimizer.step()

            # Update progress bars
            main_pbar.update(1)
            epoch_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.8f}',
                'Step': f'{global_step+1}/{traing_steps}'
            })

            if (global_step + 1) % 20 == 0: # log_step_count_steps
                print(f'Global Step: {global_step+1}/{traing_steps}, Epoch: {epoch+1}, Batch: {batch_idx+1}/{batches_per_epoch}, Training Loss: {loss.item():.6f}, Current LR: {optimizer.param_groups[0]["lr"]:.8f}')

            global_step += 1
            if (global_step % decay_interval_steps == 0) and (global_step < traing_steps):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']} at step {global_step}")

        epoch_pbar.close()
        if global_step >= traing_steps:
            break

        # Evaluation
        eval_interval = max(evaluate_steps // batches_per_epoch, 1)
        if (epoch + 1) % eval_interval == 0:
            # do evaluation
            print(f'\n--- Starting Evaluation for Epoch {epoch+1} ---')
            model.eval()
            eval_loss_total = 0
            ori_mae_total, pre_mae_total = 0.0, 0.0
            pre_mae_25_total, pre_mae_50_total, pre_mae_75_total = 0.0, 0.0, 0.0
            count = 0

            with torch.no_grad():
                eval_pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
                for eval_batch_idx, (features, labels) in enumerate(eval_pbar):
                    inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect = process_inputs(features, labels, {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask}, 'eval', device)

                    depth_outs, _ = model(inputs)

                    # Calculate loss
                    if add_gradient == 'sobel_gradient':
                        loss_1 = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                        loss_2 = get_supervised_loss('sobel_gradient', depth_outs * final_loss_msk, gt * final_loss_msk, final_loss_msk)
                        eval_loss = loss_1 + 1.0 * loss_2
                    else:
                        eval_loss = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                    eval_loss_total += eval_loss.item()

                    # Calculate metrics
                    if training_set == 'tof_FT3':
                        depth_outs_m = depth_outs
                        depth_kinect_m = depth_kinect
                        gt_m = gt
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
                    elif training_set == 'RGBDD':
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)
                    elif training_set == 'cornellbox':
                        depth_kinect_test1 = depth_kinect[:, 2:3] * 100.0 # TF depth_kinect[:, :, :, 2]
                        depth_outs_m = depth_outs * 100.0
                        gt_m = gt * 100.0
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_test1, gt_m, final_loss_msk)
                    elif training_set == 'FLAT':
                        depth_kinect_m = depth_kinect[:, 1:2] * 100.0 # TF depth_kinect[:,:,:,1]
                        depth_outs_m = depth_outs * 100.0
                        gt_m = gt * 100.0
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
                    elif training_set == 'Agresti_S1':
                        depth_kinect_m = depth_kinect[:, 2:3] * 100.0 # TF depth_kinect[:,:,:,2]
                        depth_outs_m = depth_outs * 100.0
                        gt_m = gt * 100.0
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
                    elif training_set == 'TB':
                        depth_outs_m = depth_outs * 200.0
                        depth_kinect_m = depth_kinect * 200.0
                        gt_m = gt * 200.0
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
                    else:
                        ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)

                    ori_mae_total += ori_mae.item()
                    pre_mae_total += pre_mae.item()
                    pre_mae_25_total += pre_mae_percent_25.item()
                    pre_mae_50_total += pre_mae_percent_50.item()
                    pre_mae_75_total += pre_mae_percent_75.item()
                    count += 1
                    
                    # Update evaluation progress bar
                    eval_pbar.set_postfix({
                        'Eval Loss': f'{eval_loss.item():.6f}',
                        'Pre MAE': f'{pre_mae.item():.6f}'
                    })
                
                eval_pbar.close()
            avg_eval_loss = eval_loss_total / count if count > 0 else 0
            avg_ori_mae = ori_mae_total / count if count > 0 else 0
            avg_pre_mae = pre_mae_total / count if count > 0 else 0
            avg_pre_mae_25 = pre_mae_25_total / count if count > 0 else 0
            avg_pre_mae_50 = pre_mae_50_total / count if count > 0 else 0
            avg_pre_mae_75 = pre_mae_75_total / count if count > 0 else 0

            print(f'Epoch {epoch+1} Evaluation:')
            print(f'  Avg Loss: {avg_eval_loss:.6f}')
            print(f'  Ori MAE: {avg_ori_mae:.6f}')
            print(f'  Pre MAE: {avg_pre_mae:.6f}')
            print(f'  Pre MAE 25%: {avg_pre_mae_25:.6f}')
            print(f'  Pre MAE 50%: {avg_pre_mae_50:.6f}')
            print(f'  Pre MAE 75%: {avg_pre_mae_75:.6f}')
            print(f'--- End Evaluation ---')

            # Save checkpoint
            checkpoint_save_step = (epoch + 1) * batches_per_epoch # Save based on actual steps completed
            if checkpoint_save_step >= int(checkpoint_steps) or (epoch + 1) == num_epochs_to_run: # Save if current step exceeds/equals target, or it's the last epoch
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


def dataset_testing(evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                    loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape, add_gradient):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = get_network(name=model_name, flg=False, regular=0.1, batch_size=batch_size, deformable_range=deformable_range).to(device)   
    if gpu_Number > 1 and torch.cuda.device_count() >= gpu_Number:
        print(f"Using {gpu_Number} GPUs for DataParallel during testing.")
        model = nn.DataParallel(model, device_ids=list(range(gpu_Number)))
    else:
        print(f"Running on single GPU or CPU: {device} during testing. Check GPU setup if multi-GPU intended.")

    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, f'model.ckpt-{checkpoint_steps}.pth')
    if not os.path.exists(checkpoint_path):
        ckpts = [f for f in os.listdir(model_dir) if f.startswith('model.ckpt-') and f.endswith('.pth')]
        if ckpts:
            # Sort by the step number in the filename
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
            checkpoint_path = os.path.join(model_dir, latest_ckpt)
            print(f"Warning: Checkpoint '{checkpoint_steps}' not found. Loading latest checkpoint: {latest_ckpt}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir} for step {checkpoint_steps}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()

    eval_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                               shuffle=False, repeat_count=1, batch_size=batch_size)

    eval_loss_total = 0
    ori_mae_total, pre_mae_total = 0.0, 0.0
    pre_mae_25_total, pre_mae_50_total, pre_mae_75_total = 0.0, 0.0, 0.0
    count = 0

    print(f'\n--- Starting Testing from checkpoint {checkpoint_steps} ---')
    with torch.no_grad():
        test_pbar = tqdm(eval_loader, desc="Testing", leave=False)
        for eval_batch_idx, (features, labels) in enumerate(test_pbar):
            inputs, gt, final_loss_msk, depth_kinect, amplitude_kinect, rgb_kinect = process_inputs(features, labels, {'training_set': training_set, 'output_flg': False, 'loss_mask': loss_mask}, 'eval', device)

            depth_outs, depth_residual_every_scale = model(inputs)
            # Calculate loss
            if add_gradient == 'sobel_gradient':
                loss_1 = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
                loss_2 = get_supervised_loss('sobel_gradient', depth_outs * final_loss_msk, gt * final_loss_msk, final_loss_msk)
                eval_loss = loss_1 + 1.0 * loss_2
            else:
                eval_loss = get_supervised_loss(loss_fn, depth_outs, gt, final_loss_msk)
            eval_loss_total += eval_loss.item()

            # Calculate metrics (same logic as in training)
            if training_set == 'tof_FT3':
                depth_outs_m = depth_outs 
                depth_kinect_m = depth_kinect 
                gt_m = gt 
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
            elif training_set == 'RGBDD':
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)
            elif training_set == 'cornellbox':
                depth_kinect_test1 = depth_kinect[:, 2:3] * 100.0
                depth_outs_m = depth_outs * 100.0
                gt_m = gt * 100.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_test1, gt_m, final_loss_msk)
            elif training_set == 'FLAT':
                depth_kinect_m = depth_kinect[:, 1:2] * 100.0
                depth_outs_m = depth_outs * 100.0
                gt_m = gt * 100.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
            elif training_set == 'Agresti_S1':
                depth_kinect_m = depth_kinect[:, 2:3] * 100.0
                depth_outs_m = depth_outs * 100.0
                gt_m = gt * 100.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
            elif training_set == 'TB':
                depth_outs_m = depth_outs * 200.0
                depth_kinect_m = depth_kinect * 200.0
                gt_m = gt * 200.0
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs_m, depth_kinect_m, gt_m, final_loss_msk)
            else:
                ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75 = get_metrics_mae(depth_outs, depth_kinect, gt, final_loss_msk)

            ori_mae_total += ori_mae.item()
            pre_mae_total += pre_mae.item()
            pre_mae_25_total += pre_mae_percent_25.item()
            pre_mae_50_total += pre_mae_percent_50.item()
            pre_mae_75_total += pre_mae_percent_75.item()
            count += 1
            
            # Update test progress bar
            test_pbar.set_postfix({
                'Test Loss': f'{eval_loss.item():.6f}',
                'Pre MAE': f'{pre_mae.item():.6f}'
            })
        
        test_pbar.close()
    avg_eval_loss = eval_loss_total / count if count > 0 else 0
    avg_ori_mae = ori_mae_total / count if count > 0 else 0
    avg_pre_mae = pre_mae_total / count if count > 0 else 0
    avg_pre_mae_25 = pre_mae_25_total / count if count > 0 else 0
    avg_pre_mae_50 = pre_mae_50_total / count if count > 0 else 0
    avg_pre_mae_75 = pre_mae_75_total / count if count > 0 else 0

    print(f'Testing Results from checkpoint {checkpoint_steps}:')
    print(f'  Avg Loss: {avg_eval_loss:.6f}')
    print(f'  Ori MAE: {avg_ori_mae:.6f}')
    print(f'  Pre MAE: {avg_pre_mae:.6f}')
    print(f'  Pre MAE 25%: {avg_pre_mae_25:.6f}')
    print(f'  Pre MAE 50%: {avg_pre_mae_50:.6f}')
    print(f'  Pre MAE 75%: {avg_pre_mae_75:.6f}')
    print(f'--- End Testing ---')


def dataset_output(result_path, evaluate_data_path, model_dir, batch_size, checkpoint_steps, deformable_range,
                   loss_fn, model_name, loss_mask, gpu_Number, training_set, image_shape):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = get_network(name=model_name, flg=False, regular=0.1, batch_size=batch_size, deformable_range=deformable_range).to(device)

    if gpu_Number > 1 and torch.cuda.device_count() >= gpu_Number:
        print(f"Using {gpu_Number} GPUs for DataParallel during output.")
        model = nn.DataParallel(model, device_ids=list(range(gpu_Number)))
    else:
        print(f"Running on single GPU or CPU: {device} during output. Check GPU setup if multi-GPU intended.")

    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, f'model.ckpt-{checkpoint_steps}.pth')
    if not os.path.exists(checkpoint_path):
        ckpts = [f for f in os.listdir(model_dir) if f.startswith('model.ckpt-') and f.endswith('.pth')]
        if ckpts:
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
            checkpoint_path = os.path.join(model_dir, latest_ckpt)
            print(f"Warning: Checkpoint '{checkpoint_steps}' not found. Loading latest checkpoint: {latest_ckpt}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir} for step {checkpoint_steps}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()

    output_loader = get_input_fn(training_set=training_set, filenames=evaluate_data_path, height=image_shape[0], width=image_shape[1],
                                 shuffle=False, repeat_count=1, batch_size=batch_size)

    all_depth_outs = []
    all_depth_kinect = []
    all_amplitude_kinect = []

    print(f'\n--- Starting Output Generation from checkpoint {checkpoint_steps} ---')
    with torch.no_grad():
        output_pbar = tqdm(output_loader, desc="Output Generation", leave=False)
        for batch_idx, (features, labels) in enumerate(output_pbar):
            # output_flg is True for output mode
            inputs, gt_ignored, loss_msk_ignored, depth_kinect, amplitude_kinect, rgb_kinect = \
                process_inputs(features, labels, {'training_set': training_set, 'output_flg': True, 'loss_mask': loss_mask}, 'predict', device)

            depth_outs, _ = model(inputs)

            all_depth_outs.append(depth_outs.cpu().numpy())
            all_depth_kinect.append(depth_kinect.cpu().numpy())
            all_amplitude_kinect.append(amplitude_kinect.cpu().numpy())
            
            # Update output progress bar
            output_pbar.set_postfix({
                'Batch': f'{batch_idx+1}/{len(output_loader)}'
            })
        
        output_pbar.close()

    # Concatenate all batches
    pre_depths = np.concatenate(all_depth_outs, axis=0)
    input_depths = np.concatenate(all_depth_kinect, axis=0)
    amplitudes = np.concatenate(all_amplitude_kinect, axis=0)
    # Create output directories
    os.makedirs(result_path, exist_ok=True)
    pre_depth_dir = os.path.join(result_path, 'pre_depth')
    depth_input_dir = os.path.join(result_path, 'depth_input')
    amplitude_dir = os.path.join(result_path, 'amplitude')
    depth_input_png_dir = os.path.join(result_path, 'depth_input_png')
    pre_depth_png_dir = os.path.join(result_path, 'pre_depth_png')
    amplitude_png_dir = os.path.join(result_path, 'amplitude_png')

    os.makedirs(pre_depth_dir, exist_ok=True)
    os.makedirs(depth_input_dir, exist_ok=True)
    os.makedirs(amplitude_dir, exist_ok=True)
    os.makedirs(depth_input_png_dir, exist_ok=True)
    os.makedirs(pre_depth_png_dir, exist_ok=True)
    os.makedirs(amplitude_png_dir, exist_ok=True)

    # Progress bar for saving files
    save_pbar = tqdm(range(len(pre_depths)), desc="Saving Files")
    for i in save_pbar:
        pre_depth_path = os.path.join(pre_depth_dir, str(i))
        depth_input_path = os.path.join(depth_input_dir, str(i))
        amplitude_path = os.path.join(amplitude_dir, str(i))
        depth_input_png_path = os.path.join(depth_input_png_dir, str(i)+'.png')
        pre_depth_png_path = os.path.join(pre_depth_png_dir, str(i) + '.png')
        amplitude_png_path = os.path.join(amplitude_png_dir, str(i) + '.png')
        
        pre_depth = np.squeeze(pre_depths[i])
        input_depth = np.squeeze(input_depths[i])
        amplitude = np.squeeze(amplitudes[i])

        # Convert to appropriate scale for saving as PNG
        # Assuming these are 0-1 normalized values, scaling to 0-255 for 8-bit PNG

        dtype = torch.float32
        def to_colorized_png(np_array, save_path, cmap='jet'):
            # 转成tensor，添加batch和channel维度
            tensor = torch.from_numpy(np_array).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            print(tensor.min(), tensor.max())
            # 调用colorize_img，得到彩色tensor [1,H,W,3]
            colorized = colorize_img(tensor, cmap=cmap).squeeze(0)  # [H,W,3], float
            # 转成uint8 numpy
            colorized_uint8 = (colorized.numpy() * 255).astype(np.uint8)
            # 保存彩色png
            Image.fromarray(colorized_uint8).save(save_path)
        # 归一化到0~1，避免colorize_img异常
        def normalize_to_01(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min < 1e-6:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)
        # 先归一化原始数据
        input_depth_norm = normalize_to_01(input_depth)
        pre_depth_norm = normalize_to_01(pre_depth)
        amplitude_norm = normalize_to_01(amplitude)
        # 用colorize_img转成彩色png保存
        to_colorized_png(input_depth_norm, depth_input_png_path, cmap='jet')
        to_colorized_png(pre_depth_norm, pre_depth_png_path, cmap='jet')
        to_colorized_png(amplitude_norm, amplitude_png_path, cmap='viridis')  

        # Save raw float data
        pre_depth.astype(np.float32).tofile(pre_depth_path)
        input_depth.astype(np.float32).tofile(depth_input_path)
        amplitude.astype(np.float32).tofile(amplitude_path)
        
        # Update save progress bar
        save_pbar.set_postfix({
            'File': f'{i+1}/{len(pre_depths)}'
        })
    
    save_pbar.close()

    print(f'--- Output Generation Finished. Results saved to {result_path} ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deformable KPN Network')
    parser.add_argument("-t", "--trainingSet", help='the name to the list file with training set', default = 'FLAT_reflection_s5', type=str)
    parser.add_argument("-m", "--modelName", help="name of the denoise model to be used", default="deformable_kpn")
    parser.add_argument("-l", "--lr", help="initial value for learning rate", default=1e-5, type=float)
    parser.add_argument("-i", "--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[239, 320])
    parser.add_argument("-b", "--batchSize", help='batch size to use during training', type=int, default=4)
    parser.add_argument("-s", "--steps", help='number of training steps', type=int, default=4000)
    parser.add_argument("-e", "--evalSteps", help='after the number of training steps to eval', type=int, default=400)
    parser.add_argument("-o", '--lossType', help="Type of supervised loss to use, such as mean_l2, mean_l1, sum_l2, sum_l1, smoothness, SSIM", default="mean_l2", type=str)
    parser.add_argument("-d", "--deformableRange", help="the range of deformable kernel", default=192.0, type=float)
    parser.add_argument("-f", '--flagMode', help="The flag that select the runing mode, such as train, eval, output", default='train', type=str)
    parser.add_argument("-p", '--postfix', help="the postfix of the training task", default=None, type=str)
    parser.add_argument("-c", '--checkpointSteps', help="select the checkpoint of the model", default="800", type=str)
    parser.add_argument("-k", '--lossMask', help="the mask used in compute loss", default='gt_msk', type=str)
    parser.add_argument("-g", '--gpuNumber', help="The number of GPU used in training", default=1, type=int)
    parser.add_argument('--samplesNumber', help="samples number in one epoch", default=5800, type=int)
    parser.add_argument('--addGradient', help="add the gradient loss function", default='sobel_gradient', type=str)
    parser.add_argument('--decayEpoch', help="after n epoch, decay the learning rate", default=2, type=int)
    parser.add_argument('--shmFlag', help="using shm increase the training speed", default=False, type=bool)
    args = parser.parse_args()

    if args.shmFlag == True:
        dataset_dir = '/dev/shm/dataset/tfrecords'
    else:
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/dataset')
    
    model_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/models')
    model_dir = os.path.join(model_base_dir, args.modelName)
    os.makedirs(model_dir, exist_ok=True)

    if args.modelName[0:10] == 'deformable':
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
    
    # Correct the evaluate_data_path for 'eval_ED' or 'output' mode, which uses '_test' originally in pytorch version
    if args.flagMode == 'eval_ED' or args.flagMode == 'output':
         evaluate_data_path = os.path.join(dataset_path, args.trainingSet + '_test')


    if args.flagMode == 'train':
        dataset_training(train_data_path=train_data_path, evaluate_data_path=evaluate_data_path, loss_fn=args.lossType,
                         model_dir=model_dir, learning_rate=args.lr, batch_size=args.batchSize, traing_steps=args.steps,
                         evaluate_steps=args.evalSteps, deformable_range = args.deformableRange, model_name=args.modelName,
                         checkpoint_steps=args.checkpointSteps, loss_mask = args.lossMask, gpu_Number = args.gpuNumber,
                         training_set = args.trainingSet, image_shape = args.imageShape, samples_number = args.samplesNumber,
                         add_gradient = args.addGradient, decay_epoch=args.decayEpoch)
    elif args.flagMode == 'eval_TD':
        dataset_testing(evaluate_data_path=train_data_path, model_dir=model_dir, loss_fn=args.lossType,batch_size=args.batchSize,
                        checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange, model_name = args.modelName,
                        loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet, image_shape = args.imageShape,
                        add_gradient=args.addGradient)
    elif args.flagMode == 'eval_ED':
        dataset_testing(evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        batch_size=args.batchSize,
                        checkpoint_steps=args.checkpointSteps, deformable_range=args.deformableRange,
                        model_name=args.modelName,
                        loss_mask=args.lossMask, gpu_Number=args.gpuNumber, training_set=args.trainingSet,
                        image_shape=args.imageShape,
                        add_gradient=args.addGradient)
    else: # output mode
        dataset_output(result_path=output_dir,evaluate_data_path=evaluate_data_path, model_dir=model_dir, loss_fn=args.lossType,
                        batch_size=args.batchSize, checkpoint_steps=args.checkpointSteps, deformable_range = args.deformableRange,
                        model_name = args.modelName, loss_mask = args.lossMask, gpu_Number = args.gpuNumber, training_set = args.trainingSet,
                        image_shape = args.imageShape)