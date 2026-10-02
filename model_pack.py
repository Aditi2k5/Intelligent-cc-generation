"""8-bit weight packing for the bundled models.

Large weight matrices are stored as int8 with one float16 scale per group of
128 consecutive weights in a row (symmetric), which halves fp16 checkpoints and
quarters fp32 ones; the scales add ~1.5%.
At load time they are expanded back to float, so the models run exactly as
before, only with weights rounded to 8 bits. Small tensors (biases, norms,
BatchNorm statistics) are kept as they are.
"""
from __future__ import annotations

import torch

SCALE_SUFFIX = ".__scale__"
MIN_QUANT_NUMEL = 4096
GROUP = 128


def _groups(tensor: torch.Tensor) -> torch.Tensor:
    """View as (rows, groups, group size); rows whose length isn't a multiple of GROUP form one group."""
    rows = tensor.reshape(tensor.shape[0], -1)
    size = GROUP if rows.shape[1] % GROUP == 0 else rows.shape[1]
    return rows.reshape(rows.shape[0], -1, size)


def pack_state_dict(state_dict: dict) -> dict:
    packed = {}
    seen = {}
    for name, tensor in state_dict.items():
        tensor = tensor.detach().cpu()
        if tensor.data_ptr() in seen and tensor.numel():
            continue  # tied weight; model.tie_weights() restores it after loading
        seen[tensor.data_ptr()] = name
        if tensor.is_floating_point() and tensor.dim() >= 2 and tensor.numel() >= MIN_QUANT_NUMEL:
            groups = _groups(tensor.float())
            scale = (groups.abs().amax(dim=2) / 127).clamp(min=1e-12).to(torch.float16)
            packed[name] = torch.round(groups / scale.float()[..., None]).clamp(-127, 127).to(torch.int8).reshape(tensor.shape)
            packed[name + SCALE_SUFFIX] = scale
        else:
            packed[name] = tensor.contiguous()
    return packed


def unpack_state_dict(packed: dict, dtype: torch.dtype = torch.float32) -> dict:
    state = {}
    for name, tensor in packed.items():
        if name.endswith(SCALE_SUFFIX):
            continue
        scale = packed.get(name + SCALE_SUFFIX)
        if scale is not None:
            tensor = (_groups(tensor).to(dtype) * scale.to(dtype)[..., None]).reshape(tensor.shape)
        elif tensor.is_floating_point():
            tensor = tensor.to(dtype)
        state[name] = tensor
    return state


def save(state_dict: dict, path) -> None:
    from safetensors.torch import save_file
    save_file(pack_state_dict(state_dict), str(path))


def load(path, dtype: torch.dtype = torch.float32) -> dict:
    from safetensors.torch import load_file
    return unpack_state_dict(load_file(str(path)), dtype)
