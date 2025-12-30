from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Ablation Variants (train-level)
# ============================================================

@dataclass
class AblationVariant:
    name: str
    desc: str
    # passed into get_network(**kwargs)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # patch(model, model_module)
    patches: List[Callable[[nn.Module, Any], None]] = field(default_factory=list)


# ----------------------------
# Patches (remove modules)
# ----------------------------

class _NoFiLM(nn.Module):
    """Drop FiLM: forward(x, cond) -> x"""
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return x


class _SkipFuseNoGate(nn.Module):
    """Keep concat+reduce but remove cond gate."""
    def __init__(self, reduce: nn.Conv2d):
        super().__init__()
        self.reduce = reduce

    def forward(self, dec: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.reduce(torch.cat([dec, skip], dim=1))


def _replace_modules(root: nn.Module, target_type: type, factory: Callable[[nn.Module], nn.Module]) -> int:
    n = 0
    for name, child in list(root.named_children()):
        if isinstance(child, target_type):
            setattr(root, name, factory(child))
            n += 1
        else:
            n += _replace_modules(child, target_type, factory)
    return n


def patch_disable_film(model: nn.Module, model_mod: Any) -> None:
    """Replace all AmpFiLM with identity."""
    if not hasattr(model_mod, "AmpFiLM"):
        raise AttributeError("model_mod has no AmpFiLM class (cannot patch).")
    AmpFiLM = getattr(model_mod, "AmpFiLM")
    n = _replace_modules(model, AmpFiLM, lambda old: _NoFiLM())
    print(f"[ablation] no_film: replaced {n} AmpFiLM modules")


def patch_disable_skip_gate(model: nn.Module, model_mod: Any) -> None:
    """Replace all SkipFuse with no-gate version (keep reduce conv weights)."""
    if not hasattr(model_mod, "SkipFuse"):
        raise AttributeError("model_mod has no SkipFuse class (cannot patch).")
    SkipFuse = getattr(model_mod, "SkipFuse")

    def _factory(old: nn.Module) -> nn.Module:
        reduce = getattr(old, "reduce", None)
        if not isinstance(reduce, nn.Conv2d):
            raise TypeError("SkipFuse.reduce is not nn.Conv2d; cannot patch safely.")
        return _SkipFuseNoGate(reduce=reduce)

    n = _replace_modules(model, SkipFuse, _factory)
    print(f"[ablation] no_skip_gate: replaced {n} SkipFuse modules")


# ----------------------------
# Variant registry
# ----------------------------

def ablation_variants() -> Dict[str, AblationVariant]:
    """
    Each row changes ONE thing for clean interpretation.
    """
    variants = [
        AblationVariant(
            name="base",
            desc="Baseline (original AG-NAFNet)",
        ),
        AblationVariant(
            name="no_film",
            desc="w/o FiLM (AmpFiLM -> identity)",
            patches=[patch_disable_film],
        ),
        AblationVariant(
            name="no_skip_gate",
            desc="w/o skip gate (SkipFuse gate removed, keep concat+reduce)",
            patches=[patch_disable_skip_gate],
        ),
        AblationVariant(
            name="no_ms_refine",
            desc="w/o multi-scale refine (use_ms_refine=False)",
            model_kwargs={"use_ms_refine": False},
        ),
        AblationVariant(
            name="no_nconv_hint",
            desc="w/o normalized conv hint/detail (use_nconv_hint=False)  [in_proj shape changes]",
            model_kwargs={"use_nconv_hint": False},
        ),
        AblationVariant(
            name="no_invalid_mask",
            desc="w/o invalid mask input (use_invalid_mask=False)  [in_proj shape changes]",
            model_kwargs={"use_invalid_mask": False},
        ),
        AblationVariant(
            name="no_aa_down",
            desc="w/o anti-aliased downsample (use_aa_down=False)",
            model_kwargs={"use_aa_down": False},
        ),
    ]
    return {v.name: v for v in variants}


# ============================================================
# Checkpoint loading helpers (for finetune / warm-start)
# ============================================================

def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Robustly locate a tensor state_dict in common checkpoint formats.
    """
    if isinstance(ckpt_obj, dict):
        candidates = [
            "state_dict", "model_state_dict", "model", "net", "network",
            "params", "params_ema", "ema", "model_ema", "ema_state_dict",
        ]
        for k in candidates:
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and v and all(isinstance(tv, torch.Tensor) for tv in v.values()):
                return v

        # dict itself might be state_dict
        if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj

    raise ValueError("Cannot find tensor state_dict in checkpoint.")


def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip common prefixes like 'module.' 'model.' 'net.' 'network.' repeatedly.
    """
    prefixes = ("module.", "model.", "net.", "network.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def load_ckpt_flexible(
    model: nn.Module,
    ckpt_path: str,
    mode: str = "filter",
    map_location: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    mode:
      - "strict": strict=True (structure must match exactly)
      - "nonstrict": strict=False (ignore missing/unexpected, but shape mismatch still errors)
      - "filter": load only keys that exist AND shapes match (recommended for structure ablations)
      - "auto": try strict first, fallback to filter
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = _strip_prefixes(_extract_state_dict(ckpt))

    info: Dict[str, Any] = {"mode": mode, "ckpt_path": ckpt_path}

    if mode == "auto":
        try:
            model.load_state_dict(sd, strict=True)
            info["loaded"] = "strict"
            if verbose:
                print(f"[init] loaded STRICT from {ckpt_path}")
            return info
        except Exception as e:
            if verbose:
                print(f"[init] strict load failed, fallback to FILTER. reason: {repr(e)}")
            mode = "filter"

    if mode == "strict":
        model.load_state_dict(sd, strict=True)
        info["loaded"] = "strict"
        if verbose:
            print(f"[init] loaded STRICT from {ckpt_path}")
        return info

    if mode == "nonstrict":
        incompat = model.load_state_dict(sd, strict=False)
        info["loaded"] = "nonstrict"
        info["missing_keys"] = list(incompat.missing_keys)
        info["unexpected_keys"] = list(incompat.unexpected_keys)
        if verbose:
            print(f"[init] loaded NONSTRICT from {ckpt_path}")
            if incompat.missing_keys:
                print(f"  missing_keys({len(incompat.missing_keys)}): {incompat.missing_keys[:20]} ...")
            if incompat.unexpected_keys:
                print(f"  unexpected_keys({len(incompat.unexpected_keys)}): {incompat.unexpected_keys[:20]} ...")
        return info

    if mode == "filter":
        msd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if (k in msd) and (tuple(v.shape) == tuple(msd[k].shape))}

        incompat = model.load_state_dict(filtered, strict=False)
        info["loaded"] = "filter"
        info["loaded_keys"] = len(filtered)
        info["missing_keys"] = list(incompat.missing_keys)
        info["unexpected_keys"] = list(incompat.unexpected_keys)

        if verbose:
            print(f"[init] loaded FILTERED({len(filtered)}/{len(msd)}) from {ckpt_path}")
        return info

    raise ValueError(f"Unknown load mode: {mode}")


# ============================================================
# Main entry: build model for a given ablation variant
# ============================================================

def build_model_for_ablation(
    model_mod: Any,
    variant_name: str,
    model_name: str = "ag_nafnet_itof",
    base_net_kwargs: Optional[Dict[str, Any]] = None,
    init_ckpt: Optional[str] = None,
    init_mode: str = "filter",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    model_mod: imported module object of network/ag_nafnet_itof.py
              (must include get_network / AmpFiLM / SkipFuse)

    base_net_kwargs: kwargs passed to get_network (flg/regular/batch_size/deformable_range ...)
    init_ckpt: optional warm-start weights (baseline ckpt)
    init_mode: recommended "filter"
    """
    variants = ablation_variants()
    if variant_name not in variants:
        raise KeyError(f"Unknown variant '{variant_name}'. Available: {list(variants.keys())}")

    v = variants[variant_name]
    base_net_kwargs = dict(base_net_kwargs or {})

    net_kwargs = {**base_net_kwargs, **v.model_kwargs}

    # build model
    model = model_mod.get_network(name=model_name, **net_kwargs)

    # apply patches BEFORE optimizer creation
    for p in v.patches:
        p(model, model_mod)

    # optional init
    init_info: Dict[str, Any] = {}
    if init_ckpt is not None and str(init_ckpt).strip():
        init_info = load_ckpt_flexible(model, init_ckpt, mode=init_mode, map_location="cpu", verbose=True)

    if device is not None:
        model.to(device)

    meta = {
        "variant": v.name,
        "desc": v.desc,
        "model_name": model_name,
        "net_kwargs": net_kwargs,
        "init": init_info,
    }
    return model, meta
