"""Check and benchmark Lab 05.  Usage: python labs/05_cuda_reduction/run.py [--solution]"""
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
    "torch.sum": lambda x: x.sum(dim=1),
    "cuda v1 thread/row": m.cuda_rowsum_thread,
    "cuda v2 smem tree": m.cuda_rowsum_smem,
    "cuda v3 shuffle+float4": m.cuda_rowsum_shfl,
    "triton": m.triton_rowsum,
}

torch.manual_seed(0)
for shape in ((3, 8), (1000, 1000), (37, 4100)):
    x = torch.randn(*shape, device="cuda")
    for name, fn in impls.items():
        if "float4" in name and shape[1] % 4:
            continue
        torch.testing.assert_close(fn(x), x.sum(1), atol=1e-3, rtol=1e-4, msg=f"{name} wrong for {shape}")
print("correctness: every implementation matches torch.sum")

rows = []
res = {}
for M, N in ((16384, 1024), (16384, 8192), (2048, 65536)):
    x = torch.randn(M, N, device="cuda")
    for name, fn in impls.items():
        ms = bench_ms(lambda: fn(x))
        res[(M, N, name)] = gbps(x.numel() * 4, ms)
        rows.append((f"{M}x{N}", name, f"{ms * 1e3:8.1f}", f"{res[(M, N, name)]:6.0f}"))
table(rows, ["shape", "impl", "time us", "GB/s"])

v1, v3 = res[(16384, 8192, "cuda v1 thread/row")], res[(16384, 8192, "cuda v3 shuffle+float4")]
report(
    "the thread-per-row kernel is uncoalesced and far below bandwidth; block-per-row with coalesced loads "
    "is several times faster but still pays for scalar loads and eight __syncthreads rounds; shuffles + "
    "float4 reach the same bandwidth as torch and Triton",
    f"16384x8192: v1 {v1:.0f} GB/s, v3 {v3:.0f} GB/s, triton {res[(16384, 8192, 'triton')]:.0f} GB/s, "
    f"torch {res[(16384, 8192, 'torch.sum')]:.0f} GB/s",
    "PASS" if v3 > 5 * v1 else "CHECK: v3 should be many times faster than v1",
    "look at the 2048x65536 row: only 2048 blocks of 256 threads. Raise threads to 1024 for v2/v3 and "
    "re-measure; then explain the change with the occupancy section of the lesson",
)
