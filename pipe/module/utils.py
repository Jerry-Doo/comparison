import torch
import torch.nn.functional as F
from functools import partial

def pad2d(x, vpad, hpad):
    # PyTorch 的 pad 格式是 [left, right, top, bottom]
    return F.pad(x, [hpad[0], hpad[1], vpad[0], vpad[1]])

def crop2d(x, vcrop, hcrop):
    # PyTorch 的 crop 格式是 [top, bottom, left, right]
    return x[:, :, vcrop[0]:-vcrop[1], hcrop[0]:-hcrop[1]]

def get_cost(features_0, features_0from1, shift):
    """
    Calculate cost volume for specific shift

    - inputs
    features_0 (batch, nch, h, w): feature maps at time slice 0
    features_0from1 (batch, nch, h, w): feature maps at time slice 0 warped from 1
    shift (2): spatial (vertical and horizontal) shift to be considered

    - output
    cost (batch, h, w): cost volume map for the given shift
    """
    v, h = shift  # vertical/horizontal element
    vt, vb = max(v,0), abs(min(v,0))  # top/bottom
    hl, hr = max(h,0), abs(min(h,0))  # left/right
    
    f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
    f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
    cost_pad = f_0_pad * f_0from1_pad
    return torch.mean(crop2d(cost_pad, [vt, vb], [hl, hr]), dim=1)

def costvolumelayer(f1, f2, search_range=4):
    b, c, h, w = f1.shape
    cost_length = (2*search_range+1)**2
    get_c = partial(get_cost, f1, f2)
    cv = []
    
    for v in range(-search_range, search_range+1):
        for h in range(-search_range, search_range+1):
            cv.append(get_c(shift=[v,h]))
    
    cv = torch.stack(cv, dim=1)
    cv = F.leaky_relu(cv, 0.1)
    return cv

def get_grid(x):
    batch_size, _, height, width = x.shape
    Bg, Yg, Xg = torch.meshgrid(
        torch.arange(batch_size, device=x.device),
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing='ij'
    )
    return Bg, Yg, Xg

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = flow.long()

    warped_gy = grid_y + flow[:, 1]  # flow_y
    warped_gx = grid_x + flow[:, 0]  # flow_x
    
    # clip value by height/width limitation
    _, _, h, w = x.shape
    warped_gy = torch.clamp(warped_gy, 0, h-1)
    warped_gx = torch.clamp(warped_gx, 0, w-1)

    warped_indices = torch.stack([grid_b, warped_gy, warped_gx], dim=3)
    warped_x = torch.gather(x.view(x.shape[0], -1, x.shape[1]), 1, 
                           warped_indices.view(x.shape[0], -1, 1).expand(-1, -1, x.shape[1]))
    return warped_x.view(x.shape)

def bilinear_warp(x, flow):
    _, _, h, w = x.shape
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = grid_b.float()
    grid_y = grid_y.float()
    grid_x = grid_x.float()

    fx, fy = flow[:, 0], flow[:, 1]
    fx_0 = torch.floor(fx)
    fx_1 = fx_0 + 1
    fy_0 = torch.floor(fy)
    fy_1 = fy_0 + 1

    # warping indices
    h_lim = float(h - 1)
    w_lim = float(w - 1)
    gy_0 = torch.clamp(grid_y + fy_0, 0., h_lim)
    gy_1 = torch.clamp(grid_y + fy_1, 0., h_lim)
    gx_0 = torch.clamp(grid_x + fx_0, 0., w_lim)
    gx_1 = torch.clamp(grid_x + fx_1, 0., w_lim)

    g_00 = torch.stack([grid_b, gy_0, gx_0], dim=3).long()
    g_01 = torch.stack([grid_b, gy_0, gx_1], dim=3).long()
    g_10 = torch.stack([grid_b, gy_1, gx_0], dim=3).long()
    g_11 = torch.stack([grid_b, gy_1, gx_1], dim=3).long()

    # gather contents
    x_00 = torch.gather(x.view(x.shape[0], -1, x.shape[1]), 1, 
                       g_00.view(x.shape[0], -1, 1).expand(-1, -1, x.shape[1]))
    x_01 = torch.gather(x.view(x.shape[0], -1, x.shape[1]), 1, 
                       g_01.view(x.shape[0], -1, 1).expand(-1, -1, x.shape[1]))
    x_10 = torch.gather(x.view(x.shape[0], -1, x.shape[1]), 1, 
                       g_10.view(x.shape[0], -1, 1).expand(-1, -1, x.shape[1]))
    x_11 = torch.gather(x.view(x.shape[0], -1, x.shape[1]), 1, 
                       g_11.view(x.shape[0], -1, 1).expand(-1, -1, x.shape[1]))

    # coefficients
    c_00 = ((fy_1 - fy) * (fx_1 - fx)).unsqueeze(1)
    c_01 = ((fy_1 - fy) * (fx - fx_0)).unsqueeze(1)
    c_10 = ((fy - fy_0) * (fx_1 - fx)).unsqueeze(1)
    c_11 = ((fy - fy_0) * (fx - fx_0)).unsqueeze(1)

    return (c_00 * x_00 + c_01 * x_01 + c_10 * x_10 + c_11 * x_11).view(x.shape)