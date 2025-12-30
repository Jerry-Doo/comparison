from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Stable LayerNorm2d (FP32 stats)
# -----------------------------

class LayerNorm2d(nn.Module):
    """Normalize over channel dimension for each spatial location. Compute stats in FP32 for AMP stability."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=1, keepdim=True)
        x32 = (x32 - mean) * torch.rsqrt(var + self.eps)
        x32 = x32.to(dtype)

        w = self.weight.to(dtype).view(1, -1, 1, 1)
        b = self.bias.to(dtype).view(1, -1, 1, 1)
        return x32 * w + b


class SimpleGate(nn.Module):
    """NAFNet SimpleGate: split channels into two halves and multiply."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention: y = x * Conv1x1(AvgPool(x))"""
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv(self.pool(x))
        return x * w


# -----------------------------
# Conditioning pyramid: [amp_feat, amp_conf]
# -----------------------------

class AmpCondPyramid(nn.Module):
    """
    Returns cond pyramid (per scale):
      cond = cat([amp_feat, amp_conf], dim=1)  # [B,2,h,w]
    where:
      amp_conf: raw normalized amplitude in [0,1] (better as confidence)
      amp_feat: gamma-corrected amplitude in [0,1] (better as feature)
    """
    def __init__(self, amp_gamma: float = 0.5, amp_is_uint8: bool = True, eps: float = 1e-6):
        super().__init__()
        self.amp_gamma = float(amp_gamma)
        self.amp_is_uint8 = bool(amp_is_uint8)
        self.eps = float(eps)

    def _normalize_raw(self, amp: torch.Tensor) -> torch.Tensor:
        if self.amp_is_uint8:
            return torch.clamp(amp, 0.0, 255.0) / 255.0
        amp = torch.clamp(amp, min=0.0)
        amax = torch.amax(amp, dim=(2, 3), keepdim=True)  # per-sample max
        need_norm = (amax > 1.5).to(amp.dtype)
        amp01 = (amp / (amax + self.eps)) * need_norm + amp * (1.0 - need_norm)
        return torch.clamp(amp01, 0.0, 1.0)

    def forward(self, amp: torch.Tensor, num_scales: int) -> List[torch.Tensor]:
        amp_conf = self._normalize_raw(amp)
        amp_feat = amp_conf
        if abs(self.amp_gamma - 1.0) > 1e-6:
            amp_feat = torch.pow(torch.clamp(amp_feat, 0.0, 1.0) + self.eps, self.amp_gamma)

        cond0 = torch.cat([amp_feat, amp_conf], dim=1)
        conds = [cond0]
        for _ in range(1, num_scales):
            conds.append(F.avg_pool2d(conds[-1], kernel_size=2, stride=2))
        return conds


# -----------------------------
# Confidence-weighted normalized convolution (depth hint)
# -----------------------------

class NormalizedBoxFilter(nn.Module):
    """
    hint = sum(depth * conf) / (sum(conf) + eps)
    """
    def __init__(self, k: int = 5, eps: float = 1e-6, pad_mode: str = "reflect"):
        super().__init__()
        assert k % 2 == 1 and k >= 3
        self.k = int(k)
        self.eps = float(eps)
        self.pad_mode = str(pad_mode)
        self.register_buffer("kernel", torch.ones(1, 1, k, k), persistent=False)

    def forward(self, depth01: torch.Tensor, conf01: torch.Tensor) -> torch.Tensor:
        dtype = depth01.dtype
        d = depth01.float()
        c = torch.clamp(conf01.float(), 0.0, 1.0)

        pad = self.k // 2
        mode = self.pad_mode
        if d.shape[-2] <= pad or d.shape[-1] <= pad:
            mode = "replicate"

        num = F.conv2d(F.pad(d * c, (pad, pad, pad, pad), mode=mode), self.kernel)
        den = F.conv2d(F.pad(c,     (pad, pad, pad, pad), mode=mode), self.kernel) + self.eps
        return (num / den).to(dtype)


# -----------------------------
# FiLM with 2-ch cond (bounded)
# -----------------------------

class AmpFiLM(nn.Module):
    """
    gamma, beta = f(cond) and apply: x <- (1+gamma)*x + beta
    cond: [B,2,H,W]
    """
    def __init__(self, channels: int, cond_ch: int = 2, hidden: int = 16,
                 gamma_scale: float = 0.5, beta_scale: float = 0.5):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)

        self.net = nn.Sequential(
            nn.Conv2d(cond_ch, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2 * channels, 1, padding=0, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.gamma_scale
        beta = torch.tanh(beta) * self.beta_scale
        return x * (1.0 + gamma) + beta


# -----------------------------
# Anti-aliased Down / Up
# -----------------------------

class BlurPool2d(nn.Module):
    """Fixed low-pass filter then strided sampling (anti-alias)."""
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        assert filt_size in (3, 5)
        self.channels = int(channels)
        self.stride = int(stride)
        if filt_size == 3:
            a = torch.tensor([1., 2., 1.])
        else:
            a = torch.tensor([1., 4., 6., 4., 1.])
        k = a[:, None] * a[None, :]
        k = k / k.sum()
        weight = k[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=False)
        self.pad = filt_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = "reflect"
        if x.shape[-2] <= self.pad or x.shape[-1] <= self.pad:
            mode = "replicate"
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=mode)
        return F.conv2d(x, self.weight, stride=self.stride, padding=0, groups=self.channels)


class DownsampleAA(nn.Module):
    """BlurPool + Conv to reduce aliasing / grid artifacts."""
    def __init__(self, in_ch: int, out_ch: int, filt_size: int = 3):
        super().__init__()
        self.blur = BlurPool2d(in_ch, filt_size=filt_size, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.blur(x))


class UpsampleBilinear(nn.Module):
    """Interpolate + Conv (less periodic artifacts than PixelShuffle)."""
    def __init__(self, in_ch: int, out_ch: int, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if self.mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=size_hw, mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, size=size_hw, mode=self.mode)
        return self.conv(x)


# -----------------------------
# Anti-gridding multi-scale refine (NO dilation)
# -----------------------------

class AmpMSKRefine(nn.Module):
    """
    Multi-kernel depthwise conv mixture with amplitude-dependent gating.
    (Avoids dilated-conv gridding artifacts.)
    Uses amp_conf (cond[:,1]) to control strength: low conf => stronger refine.

    All conv weights are zero-init => y=0 at init => identity.
    """
    def __init__(self, channels: int, cond_ch: int = 2, hidden: int = 16, ks: Tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        k1, k2, k3 = ks
        self.dw1 = nn.Conv2d(channels, channels, k1, padding=k1 // 2, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, k2, padding=k2 // 2, groups=channels, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, k3, padding=k3 // 2, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

        self.w_net = nn.Sequential(
            nn.Conv2d(cond_ch, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, padding=0, bias=True),
        )
        nn.init.zeros_(self.w_net[-1].weight)
        nn.init.zeros_(self.w_net[-1].bias)

        # start from identity
        nn.init.zeros_(self.dw1.weight)
        nn.init.zeros_(self.dw2.weight)
        nn.init.zeros_(self.dw3.weight)
        nn.init.zeros_(self.pw.weight)

        self.scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(scale) in (0,1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        logits = self.w_net(cond)
        w = torch.softmax(logits, dim=1)

        y = (
            w[:, 0:1] * self.dw1(x) +
            w[:, 1:2] * self.dw2(x) +
            w[:, 2:3] * self.dw3(x)
        )
        y = self.pw(y)

        conf = cond[:, 1:2]
        strength = (1.0 - conf)
        return x + torch.sigmoid(self.scale) * (y * strength)


# -----------------------------
# NAFBlock with conditioning
# -----------------------------

class AmpNAFBlock(nn.Module):
    def __init__(self, channels: int, cond_ch: int = 2, dw_expand: int = 2, ffn_expand: int = 2, film_hidden: int = 16):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.film1 = AmpFiLM(channels, cond_ch=cond_ch, hidden=film_hidden, gamma_scale=0.5, beta_scale=0.5)

        mid_ch = channels * dw_expand
        self.pw1 = nn.Conv2d(channels, mid_ch, 1, bias=True)
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=True)
        self.sg = SimpleGate()
        self.sca = SCA(mid_ch // 2)
        self.pw2 = nn.Conv2d(mid_ch // 2, channels, 1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.film2 = AmpFiLM(channels, cond_ch=cond_ch, hidden=film_hidden, gamma_scale=0.5, beta_scale=0.5)

        ffn_ch = channels * ffn_expand
        self.pw3 = nn.Conv2d(channels, ffn_ch * 2, 1, bias=True)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(ffn_ch, channels, 1, bias=True)

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.film1(y, cond)
        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.pw2(y)
        x = x + y * self.beta

        z = self.norm2(x)
        z = self.film2(z, cond)
        z = self.pw3(z)
        z = self.sg2(z)
        z = self.pw4(z)
        x = x + z * self.gamma
        return x


# -----------------------------
# Skip fusion
# -----------------------------

class SkipFuse(nn.Module):
    """Concat skip + decoder then 1x1 reduce. Cond-aware gate on skip (init identity)."""
    def __init__(self, ch: int, cond_ch: int = 2):
        super().__init__()
        self.reduce = nn.Conv2d(ch * 2, ch, 1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(cond_ch, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, ch, 1, padding=0, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, dec: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        factor = 2.0 * torch.sigmoid(self.gate(cond))  # (0,2), init 1
        skip = skip * factor
        return self.reduce(torch.cat([dec, skip], dim=1))


# -----------------------------
# Main Network
# -----------------------------

@dataclass
class AGNAFNetConfig:
    base_ch: int = 32
    enc_blocks: Tuple[int, int, int, int] = (2, 2, 4, 6)
    dec_blocks: Tuple[int, int, int] = (4, 2, 2)
    dw_expand: int = 2
    ffn_expand: int = 2
    film_hidden: int = 16

    amp_gamma: float = 0.5
    amp_is_uint8: bool = True

    # anti-grid
    use_aa_down: bool = True
    upsample_mode: str = "bilinear"

    use_ms_refine: bool = True
    use_ms_refine0: bool = False
    ms_kernel_sizes: Tuple[int, int, int] = (3, 5, 7)

    # hint / mask
    use_nconv_hint: bool = True
    nconv_ksize: int = 5
    use_invalid_mask: bool = True

    max_depth: float = 4095.0
    residual_scale: float = 0.2
    residual_highconf_mul: float = 0.5
    residual_lowconf_mul: float = 2.0


class AmpGuidedNAFNet(nn.Module):
    """
    Input: x [B,5,H,W], uses only depth(0) and amplitude(1)
    Output: [B,1,H,W]
    Returns: (depth_output, current_output)
    """
    def __init__(self, cfg: AGNAFNetConfig = AGNAFNetConfig(),
                 flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.cfg = cfg

        C0 = cfg.base_ch
        C1 = C0 * 2
        C2 = C0 * 4
        C3 = C0 * 8

        self.cond_pyr = AmpCondPyramid(amp_gamma=cfg.amp_gamma, amp_is_uint8=cfg.amp_is_uint8)

        self.use_nconv_hint = bool(cfg.use_nconv_hint)
        if self.use_nconv_hint:
            self.nconv = NormalizedBoxFilter(k=cfg.nconv_ksize)

        self.use_invalid_mask = bool(cfg.use_invalid_mask)

        # in_proj input channels:
        # depth01(1) + cond0(2) + (hint01, detail01)(2 optional) + invalid(1 optional)
        in_ch = 1 + 2
        if self.use_nconv_hint:
            in_ch += 2
        if self.use_invalid_mask:
            in_ch += 1
        self.in_proj = nn.Conv2d(in_ch, C0, 3, padding=1, bias=True)

        cond_ch = 2
        self.enc0 = nn.ModuleList([AmpNAFBlock(C0, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.enc_blocks[0])])

        if cfg.use_aa_down:
            self.down1 = DownsampleAA(C0, C1)
            self.down2 = DownsampleAA(C1, C2)
            self.down3 = DownsampleAA(C2, C3)
        else:
            self.down1 = nn.Conv2d(C0, C1, 3, stride=2, padding=1, bias=True)
            self.down2 = nn.Conv2d(C1, C2, 3, stride=2, padding=1, bias=True)
            self.down3 = nn.Conv2d(C2, C3, 3, stride=2, padding=1, bias=True)

        self.enc1 = nn.ModuleList([AmpNAFBlock(C1, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.enc_blocks[1])])
        self.enc2 = nn.ModuleList([AmpNAFBlock(C2, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.enc_blocks[2])])
        self.enc3 = nn.ModuleList([AmpNAFBlock(C3, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.enc_blocks[3])])

        self.use_ms_refine = bool(cfg.use_ms_refine)
        if self.use_ms_refine:
            self.ms2 = AmpMSKRefine(C2, cond_ch=cond_ch, hidden=cfg.film_hidden, ks=cfg.ms_kernel_sizes)
            self.ms3 = AmpMSKRefine(C3, cond_ch=cond_ch, hidden=cfg.film_hidden, ks=cfg.ms_kernel_sizes)

        self.use_ms_refine0 = bool(cfg.use_ms_refine0)
        if self.use_ms_refine0:
            self.ms0 = AmpMSKRefine(C0, cond_ch=cond_ch, hidden=cfg.film_hidden, ks=cfg.ms_kernel_sizes)

        self.up2 = UpsampleBilinear(C3, C2, mode=cfg.upsample_mode)
        self.fuse2 = SkipFuse(C2, cond_ch=cond_ch)
        self.dec2 = nn.ModuleList([AmpNAFBlock(C2, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.dec_blocks[0])])

        self.up1 = UpsampleBilinear(C2, C1, mode=cfg.upsample_mode)
        self.fuse1 = SkipFuse(C1, cond_ch=cond_ch)
        self.dec1 = nn.ModuleList([AmpNAFBlock(C1, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.dec_blocks[1])])

        self.up0 = UpsampleBilinear(C1, C0, mode=cfg.upsample_mode)
        self.fuse0 = SkipFuse(C0, cond_ch=cond_ch)
        self.dec0 = nn.ModuleList([AmpNAFBlock(C0, cond_ch, cfg.dw_expand, cfg.ffn_expand, cfg.film_hidden) for _ in range(cfg.dec_blocks[2])])

        self.head = nn.Sequential(
            nn.Conv2d(C0, C0, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(C0, 1, 1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # compatibility args
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def _apply_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for b in blocks:
            x = b(x, cond)
        return x

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(1) >= 2, f"Expected [B,>=2,H,W], got {tuple(x.shape)}"
        depth = x[:, 0:1]
        amp = x[:, 1:2]

        conds = self.cond_pyr(amp, num_scales=4)
        c0, c1, c2, c3 = conds
        conf0 = c0[:, 1:2]

        dmax = float(self.cfg.max_depth) if self.cfg.max_depth is not None else 4095.0
        dmax = max(dmax, 1.0)
        depth01 = torch.clamp(depth, 0.0, dmax) / dmax

        parts = [depth01]
        if self.use_nconv_hint:
            hint01 = self.nconv(depth01, conf0)
            detail01 = torch.clamp(depth01 - hint01, -1.0, 1.0)
            parts += [hint01, detail01]
        parts += [c0]
        if self.use_invalid_mask:
            invalid = (depth01 <= 0.0).to(depth01.dtype)
            parts += [invalid]

        inp = torch.cat(parts, dim=1)
        f0 = self.in_proj(inp)

        f0 = self._apply_blocks(self.enc0, f0, c0)
        if self.use_ms_refine0:
            f0 = self.ms0(f0, c0)
        skip0 = f0

        f1 = self.down1(f0)
        f1 = self._apply_blocks(self.enc1, f1, c1)
        skip1 = f1

        f2 = self.down2(f1)
        f2 = self._apply_blocks(self.enc2, f2, c2)
        if self.use_ms_refine:
            f2 = self.ms2(f2, c2)
        skip2 = f2

        f3 = self.down3(f2)
        f3 = self._apply_blocks(self.enc3, f3, c3)
        if self.use_ms_refine:
            f3 = self.ms3(f3, c3)

        u2 = self.up2(f3, size_hw=skip2.shape[-2:])
        u2 = self.fuse2(u2, skip2, c2)
        u2 = self._apply_blocks(self.dec2, u2, c2)

        u1 = self.up1(u2, size_hw=skip1.shape[-2:])
        u1 = self.fuse1(u1, skip1, c1)
        u1 = self._apply_blocks(self.dec1, u1, c1)

        u0 = self.up0(u1, size_hw=skip0.shape[-2:])
        u0 = self.fuse0(u0, skip0, c0)
        u0 = self._apply_blocks(self.dec0, u0, c0)

        residual01 = torch.tanh(self.head(u0))  # [-1,1]

        scale_map = float(self.cfg.residual_scale) * (
            float(self.cfg.residual_highconf_mul) * conf0 +
            float(self.cfg.residual_lowconf_mul) * (1.0 - conf0)
        )
        residual01 = residual01 * scale_map

        depth_pred01 = torch.clamp(depth01 + residual01, 0.0, 1.0)
        depth_pred = depth_pred01 * dmax

        return depth_pred, depth_pred


def get_network(name: str = "ag_nafnet_itof", **kwargs) -> nn.Module:
    name = name.lower()
    if name in ["ag_nafnet_itof", "ag_nafnet_itof_v3", "amp_guided_nafnet", "ag_nafnet", "nafnet_amp"]:
        cfg = AGNAFNetConfig(
            base_ch=int(kwargs.get("base_ch", 32)),
            enc_blocks=tuple(kwargs.get("enc_blocks", (2, 2, 4, 6))),
            dec_blocks=tuple(kwargs.get("dec_blocks", (4, 2, 2))),
            dw_expand=int(kwargs.get("dw_expand", 2)),
            ffn_expand=int(kwargs.get("ffn_expand", 2)),
            film_hidden=int(kwargs.get("film_hidden", 16)),
            amp_gamma=float(kwargs.get("amp_gamma", 0.5)),
            amp_is_uint8=bool(kwargs.get("amp_is_uint8", True)),
            use_aa_down=bool(kwargs.get("use_aa_down", True)),
            upsample_mode=str(kwargs.get("upsample_mode", "bilinear")),
            use_ms_refine=bool(kwargs.get("use_ms_refine", True)),
            use_ms_refine0=bool(kwargs.get("use_ms_refine0", False)),
            ms_kernel_sizes=tuple(kwargs.get("ms_kernel_sizes", (3, 5, 7))),
            use_nconv_hint=bool(kwargs.get("use_nconv_hint", True)),
            nconv_ksize=int(kwargs.get("nconv_ksize", 5)),
            use_invalid_mask=bool(kwargs.get("use_invalid_mask", True)),
            max_depth=float(kwargs.get("max_depth", 4095.0)),
            residual_scale=float(kwargs.get("residual_scale", 0.2)),
            residual_highconf_mul=float(kwargs.get("residual_highconf_mul", 0.5)),
            residual_lowconf_mul=float(kwargs.get("residual_lowconf_mul", 2.0)),
        )
        return AmpGuidedNAFNet(
            cfg,
            flg=kwargs.get("flg", None),
            regular=kwargs.get("regular", None),
            batch_size=kwargs.get("batch_size", None),
            deformable_range=kwargs.get("deformable_range", None),
        )
    raise ValueError(f"Unknown model name: {name}")
