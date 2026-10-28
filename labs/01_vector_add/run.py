"""Check and benchmark Lab 01.  Usage: python labs/01_vector_add/run.py [--solution]"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch

from gk.bench import bench_ms, gbps, load_impl, report, table

ap = argparse.ArgumentParser()
ap.add_argument("--solution", action="store_true")
args = ap.parse_args()
m = load_impl(__file__, args.solution)

impls = {
    "torch x+y": lambda x, y: x + y,
    "triton": m.triton_add,
    "gluon": m.gluon_add,
    "cuda scalar": m.cuda_add,
    "cuda float4": m.cuda_add_float4,
}

# correctness, including a size that is not a multiple of the block
for n in (1, 1000, 98_432):
    x = torch.randn(n * 4, device="cuda")
    y = torch.randn_like(x)
    for name, fn in impls.items():
        torch.testing.assert_close(fn(x, y), x + y, msg=f"{name} wrong for n={n * 4}")
print("correctness: all implementations match x + y")

rows = []
best = {}
for n in (2**16, 2**20, 2**24, 2**28):
    x = torch.randn(n, device="cuda")
    y = torch.randn_like(x)
    moved = 3 * x.numel() * x.element_size()  # read x, read y, write out
    for name, fn in impls.items():
        ms = bench_ms(lambda: fn(x, y))
        bw = gbps(moved, ms)
        rows.append((f"2^{n.bit_length() - 1}", name, f"{ms * 1e3:9.1f}", f"{bw:7.0f}"))
        if n == 2**28:
            best[name] = bw
table(rows, ["n", "impl", "time us", "GB/s"])

top = max(best.values())
vectorized = [best[k] for k in ("torch x+y", "triton", "gluon", "cuda float4")]
report(
    "vector add is memory bound: implementations that keep many bytes in flight per thread reach the same "
    "HBM bandwidth at large n, one 4-byte load per CUDA thread falls well short, and small n is launch-latency bound",
    "at n=2^28: " + ", ".join(f"{k} {v:.0f} GB/s" for k, v in best.items()),
    "PASS" if min(vectorized) > 0.9 * top and best["cuda scalar"] < 0.8 * top
    else "CHECK: compare your numbers with the lesson's reference table",
    "change BLOCK (256..8192) and num_warps in triton_add and see which settings move the 2^20 row; "
    "then set vec=1 in gluon_add: it stays fast because each thread still issues BLOCK/(32*num_warps)=8 "
    "independent loads; finally make the CUDA scalar kernel handle 4 strided elements per thread",
)
