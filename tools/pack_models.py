"""Build step: download every model the pipeline uses and write compact copies
to models/, which planetread.spec bundles. The app then starts offline, with
nothing to download.

Florence-2, MERT and PANNs are stored with 8-bit weights (see model_pack.py);
MiniLM is stored in float16. Revisions are pinned so every build ships the
same weights.

    python tools/pack_models.py [--out models]
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model_pack  # noqa: E402

# (repo, revision, files to copy besides the weights)
FLORENCE = ("microsoft/Florence-2-large", "21a599d414c4d928c9032694c424fb94458e3594",
            ["config.json", "configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py",
             "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json"])
MERT = ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
        ["config.json", "configuration_MERT.py", "modeling_MERT.py", "preprocessor_config.json"])
MINILM = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
PANNS_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
SILERO_ZIP = "https://github.com/snakers4/silero-vad/archive/refs/tags/v6.2.1.zip"


def pack_hf(repo, revision, files, weights_file, out: Path) -> None:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copyfile(hf_hub_download(repo, name, revision=revision), out / name)
    path = hf_hub_download(repo, weights_file, revision=revision)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model_pack.save(state, out / "weights.int8.safetensors")


def pack_minilm(out: Path) -> None:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    repo, revision = MINILM
    src = Path(snapshot_download(repo, revision=revision, allow_patterns=[
        "*.json", "vocab.txt", "model.safetensors", "1_Pooling/*"]))
    shutil.copytree(src, out, dirs_exist_ok=True)
    weights = out / "model.safetensors"
    state = {k: v.half() if v.is_floating_point() else v for k, v in load_file(weights).items()}
    weights.unlink()
    save_file(state, weights, metadata={"format": "pt"})  # transformers requires the format tag


def pack_panns(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PANNS_URL, timeout=120) as resp:
        checkpoint = torch.load(io.BytesIO(resp.read()), map_location="cpu", weights_only=False)
    model_pack.save(checkpoint["model"], out / "Cnn14_mAP=0.431.int8.safetensors")


def pack_silero(out: Path) -> None:
    """Only the torch.hub entry point, the Python sources and the JIT model."""
    with urllib.request.urlopen(SILERO_ZIP, timeout=120) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    prefix = zf.namelist()[0].split("/")[0] + "/"
    for name in zf.namelist():
        rel = name[len(prefix):]
        keep = rel == "hubconf.py" or (rel.startswith("src/silero_vad/") and (
            rel.endswith(".py") or rel == "src/silero_vad/data/silero_vad.jit"))
        if keep and not name.endswith("/"):
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "models"))
    out = Path(parser.parse_args().out)
    shutil.rmtree(out, ignore_errors=True)
    steps = [
        ("Florence-2-large", lambda: pack_hf(*FLORENCE, "model.safetensors", out / "florence-2-large")),
        ("MERT-v1-95M", lambda: pack_hf(*MERT, "pytorch_model.bin", out / "mert-v1-95m")),
        ("all-MiniLM-L6-v2", lambda: pack_minilm(out / "all-MiniLM-L6-v2")),
        ("PANNs Cnn14", lambda: pack_panns(out / "panns")),
        ("Silero VAD", lambda: pack_silero(out / "silero-vad")),
    ]
    for name, step in steps:
        print(f"Packing {name} ...", flush=True)
        step()
    for sub in sorted(out.iterdir()):
        size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
        print(f"  {sub.name:20s} {size / 1e6:7.1f} MB")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  {'total':20s} {total / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
