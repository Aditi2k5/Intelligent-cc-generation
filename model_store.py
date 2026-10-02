"""Load the models bundled by tools/pack_models.py.

`bundled(name)` returns the model's folder when the app ships it (the
PyInstaller build, or a dev checkout after running pack_models.py), else None;
callers then load from the Hugging Face hub as before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

import model_pack

MODELS_DIR = Path(os.environ.get("PLANETREAD_MODELS_DIR")
                  or Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "models")


def bundled(name: str) -> Path | None:
    path = MODELS_DIR / name
    return path if path.is_dir() else None


def load_packed_hf(path: Path, auto_class, dtype: torch.dtype = torch.float32):
    """Build a (remote-code) transformers model from its config and load 8-bit packed weights."""
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with no_init_weights():
        model = auto_class.from_config(config, trust_remote_code=True, torch_dtype=dtype)
    result = model.load_state_dict(model_pack.load(path / "weights.int8.safetensors", dtype), strict=False)
    model.tie_weights()
    # Tied weights are stored once; every other missing key would be left uninitialised.
    state = model.state_dict()
    loaded = {state[k].data_ptr() for k in state if k not in result.missing_keys}
    missing = [k for k in result.missing_keys if state[k].data_ptr() not in loaded]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"{path.name}: weights do not match the model "
                           f"(missing {missing[:5]}, unexpected {result.unexpected_keys[:5]})")
    return model.eval()


def load_panns_state(path: Path) -> dict:
    return model_pack.load(path / "Cnn14_mAP=0.431.int8.safetensors")
