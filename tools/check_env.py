"""Lesson 00: check the toolchain and measure this GPU's two ceilings.

    python tools/check_env.py

Every later lab compares a kernel against one of two roofs: how fast the GPU
can move bytes (HBM bandwidth) and how fast it can do bf16 matrix math (tensor
core throughput). This script measures both with library kernels, so you know
what "fast" means on your machine before you write anything.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import triton  # noqa: E402

from gk.bench import bench_ms, device_summary, ensure_cuda_home, gbps, table, tflops  # noqa: E402


def main():
    assert torch.cuda.is_available(), "no CUDA device visible"
    p = torch.cuda.get_device_properties(0)
    print(device_summary())
    rows = [
        ("torch", torch.__version__),
        ("triton", triton.__version__),
        ("CUDA (torch build)", torch.version.cuda),
        ("SMs", p.multi_processor_count),
        ("L2 cache", f"{p.L2_cache_size / 2**20:.0f} MiB"),
        ("shared memory / block (opt-in)", f"{p.shared_memory_per_block_optin / 1024:.0f} KB"),
        ("registers / SM", p.regs_per_multiprocessor),
        ("warp size", p.warp_size),
    ]
    try:
        from triton.experimental import gluon  # noqa: F401
        rows.append(("gluon", "available"))
    except ImportError:
        rows.append(("gluon", "missing (labs 01 needs it)"))
    try:
        rows.append(("CUDA_HOME for NVRTC", ensure_cuda_home()))
        k = torch.cuda._compile_kernel('extern "C" __global__ void one(float* x) { x[threadIdx.x] = 1.f; }', "one")
        x = torch.zeros(32, device="cuda")
        k(grid=(1, 1, 1), block=(32, 1, 1), args=[x])
        rows.append(("NVRTC compile + launch", "ok" if x.sum().item() == 32 else "WRONG RESULT"))
    except Exception as e:  # noqa: BLE001
        rows.append(("NVRTC compile + launch", f"failed: {e}"))
    table(rows, ["item", "value"])

    print("\nceilings (library kernels)")
    n = 1 << 28
    src = torch.empty(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    copy_ms = bench_ms(lambda: dst.copy_(src))
    copy_bw = gbps(2 * src.numel() * 4, copy_ms)
    size = 8192
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    mm_ms = bench_ms(lambda: a @ b)
    mm_tf = tflops(2 * size**3, mm_ms)
    tiny = torch.empty(1, device="cuda")
    launch_us = bench_ms(lambda: tiny.add_(1)) * 1e3
    table([
        ("HBM copy, 1 GiB -> 1 GiB", f"{copy_bw:,.0f} GB/s"),
        ("bf16 matmul 8192^3", f"{mm_tf:,.0f} TFLOP/s"),
        ("smallest kernel (1 element)", f"{launch_us:.1f} us"),
        ("ridge point (FLOP per byte)", f"{mm_tf * 1e12 / (copy_bw * 1e9):.0f}"),
    ], ["measurement", "value"])
    print("\nA kernel doing fewer FLOPs per byte than the ridge point is memory-bound: judge it in GB/s.")
    print("Above the ridge it is compute-bound: judge it in TFLOP/s.")


if __name__ == "__main__":
    main()
