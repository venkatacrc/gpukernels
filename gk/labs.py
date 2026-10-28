"""Load a lab's kernel.py (your code) or solution.py by lab name.

Later labs and the HSTU model reuse earlier kernels through this loader, so the
model you train in Lab 09 runs the kernels you wrote. Set GK_IMPL=solution to
use the reference kernels instead (the default), or GK_IMPL=kernel for yours.
"""
import importlib.util
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_cache = {}


def lab_module(lab, which=None):
    which = which or os.environ.get("GK_IMPL", "solution")
    key = (lab, which)
    if key not in _cache:
        path = ROOT / "labs" / lab / f"{which}.py"
        spec = importlib.util.spec_from_file_location(f"gk_lab_{lab}_{which}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[key] = mod
    return _cache[key]
