"""Timing, bandwidth and result-printing helpers used by every lab.

Every lab prints the same four blocks so you can compare your run with the
reference numbers in the lesson:

    CLAIM     what the lesson says should happen
    MEASURED  what your GPU did
    VERDICT   whether the claim held
    NEXT      what to try next
"""
import os
import pathlib
import site

import torch
import triton
import triton.testing


def ensure_cuda_home():
    """Point CUDA_HOME at the CUDA 13 runtime that ships inside the torch wheel.

    torch.cuda._compile_kernel uses NVRTC, which looks for CUDA_HOME. There is
    no system CUDA toolkit on this machine, but the wheel bundles libnvrtc and
    the headers under site-packages/nvidia/cu13.
    """
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    for sp in site.getsitepackages():
        cand = pathlib.Path(sp) / "nvidia" / "cu13"
        if (cand / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_HOME"] = str(cand)
            return str(cand)
    raise RuntimeError("no CUDA headers found; set CUDA_HOME")


def bench_ms(fn, warmup=25, rep=100):
    """Median kernel time in milliseconds (L2 is flushed between runs)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def peak_mem_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def device_summary():
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.multi_processor_count} SMs, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GiB visible"


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def report(claim, measured, verdict, next_step):
    print()
    print(f"CLAIM     {claim}")
    print(f"MEASURED  {measured}")
    print(f"VERDICT   {verdict}")
    print(f"NEXT      {next_step}")


def load_impl(lab_file, use_solution):
    """Import kernel.py (your code) or solution.py from the lab's directory."""
    import importlib.util

    lab_dir = pathlib.Path(lab_file).resolve().parent
    name = "solution" if use_solution else "kernel"
    spec = importlib.util.spec_from_file_location(f"{lab_dir.name}_{name}", lab_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[{lab_dir.name}] using {name}.py on {device_summary()}")
    return mod
