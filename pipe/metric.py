import torch
import torch.nn.functional as F

def SSIM(x, y):
    # x, y: (B, C, H, W)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = SSIM_n / (SSIM_d + 1e-8)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

def mean_SSIM(x, y):
    return torch.mean(SSIM(x, y))

def get_metrics_psnr(depth, ori_depth, gt, msk):
    # PSNR = 10 * log10(MAX^2 / MSE)
    # MAX=5 (TF代码用25.0)
    MAX = 5.0
    def mse(a, b):
        return torch.mean((a - b) ** 2)
    def psnr(a, b):
        mse_val = mse(a, b)
        return 10 * torch.log10(MAX ** 2 / (mse_val + 1e-8))
    ori_psnr = psnr(gt, ori_depth)
    pre_psnr = psnr(gt, depth)
    # masked
    depth_m = depth * msk
    ori_depth_m = ori_depth * msk
    gt_m = gt * msk
    def safe_psnr(a, b, mask):
        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            return torch.tensor(0.0, device=a.device)
        mse_val = torch.sum((a - b) ** 2) / (mask_sum + 1e-8)
        return 10 * torch.log10(MAX ** 2 / (mse_val + 1e-8))
    ori_psnr_dm = safe_psnr(gt_m, ori_depth_m, msk)
    pre_psnr_dm = safe_psnr(gt_m, depth_m, msk)
    return ori_psnr, pre_psnr, ori_psnr_dm, pre_psnr_dm

def get_metrics_mae(depth, ori_depth, gt, msk):
    # 返回 ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75
    if msk is None:
        ori_mae = torch.mean(torch.abs(gt - ori_depth))
        pre_mae = torch.mean(torch.abs(gt - depth))
        pre_mae_percent_25 = pre_mae
        pre_mae_percent_50 = pre_mae
        pre_mae_percent_75 = pre_mae
    else:
        msk_one = torch.ones_like(gt)
        gt_tmp = gt * msk
        depth_tmp = depth * msk
        ori_depth_tmp = ori_depth * msk
        msk_one_sum = torch.sum(msk_one)
        msk_sum = torch.sum(msk)
        msk_coeff = msk_one_sum / (msk_sum + 1e-8)
        # 展平成一维
        error_array = torch.abs(ori_depth_tmp - gt_tmp).reshape(-1)
        error_array_sorted, _ = torch.sort(error_array)
        msk_one_sum = int(msk_one_sum.item())
        msk_sum = int(msk_sum.item())
        percent_25 = msk_sum // 4
        percent_50 = percent_25 * 2
        percent_75 = percent_25 * 3
        msk_sum_diff = msk_one_sum - msk_sum
        min_error_value = error_array_sorted[msk_sum_diff]
        percent_25_error_value = error_array_sorted[msk_sum_diff + percent_25]
        percent_50_error_value = error_array_sorted[msk_sum_diff + percent_50]
        percent_75_error_value = error_array_sorted[msk_sum_diff + percent_75]
        msk_percent_25 = ((error_array > min_error_value) & (error_array < percent_25_error_value)).float().reshape(gt.shape)
        msk_percent_50 = ((error_array > percent_25_error_value) & (error_array < percent_50_error_value)).float().reshape(gt.shape)
        msk_percent_75 = ((error_array > percent_50_error_value) & (error_array < percent_75_error_value)).float().reshape(gt.shape)
        msk_coeff_percent_25 = 4 * msk_coeff
        msk_coeff_percent_50 = 4 * msk_coeff
        msk_coeff_percent_75 = 4 * msk_coeff
        gt_tmp_percent_25 = gt * msk_percent_25
        depth_tmp_percent_25 = depth * msk_percent_25
        gt_tmp_percent_50 = gt * msk_percent_50
        depth_tmp_percent_50 = depth * msk_percent_50
        gt_tmp_percent_75 = gt * msk_percent_75
        depth_tmp_percent_75 = depth * msk_percent_75
        ori_mae = torch.mean(torch.abs(gt_tmp * msk_coeff - ori_depth_tmp * msk_coeff))
        pre_mae = torch.mean(torch.abs(gt_tmp * msk_coeff - depth_tmp * msk_coeff))
        pre_mae_percent_25 = torch.mean(torch.abs(gt_tmp_percent_25 * msk_coeff_percent_25 - depth_tmp_percent_25 * msk_coeff_percent_25))
        pre_mae_percent_50 = torch.mean(torch.abs(gt_tmp_percent_50 * msk_coeff_percent_50 - depth_tmp_percent_50 * msk_coeff_percent_50))
        pre_mae_percent_75 = torch.mean(torch.abs(gt_tmp_percent_75 * msk_coeff_percent_75 - depth_tmp_percent_75 * msk_coeff_percent_75))
    return ori_mae, pre_mae, pre_mae_percent_25, pre_mae_percent_50, pre_mae_percent_75
