"""Load the course kernels: the reference solutions, or the ones you wrote.

    GK_KERNELS=solution   (default)  labs/*/solution.py
    GK_KERNELS=mine                  labs/*/kernel.py

So once your labs pass, `GK_KERNELS=mine python train/train.py` trains HSTU
on your own kernels.
"""
import importlib.util
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_cache = {}


def lab(name):
    which = "kernel" if os.environ.get("GK_KERNELS", "solution") == "mine" else "solution"
    key = (name, which)
    if key not in _cache:
        path = ROOT / "labs" / name / f"{which}.py"
        spec = importlib.util.spec_from_file_location(f"gk_{name}_{which}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[key] = mod
    return _cache[key]


def ln_mul():
    return lab("03_layernorm_mul").ln_mul


def matmul_tma():
    return lab("04_matmul_silu").triton_matmul_tma


def hstu_attention():
    return lab("07_hstu_attention_bwd").hstu_attention
