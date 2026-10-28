"""Check and benchmark Lab 04.  Usage: python labs/04_matmul_silu/run.py [--solution]"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch

from gk.bench import bench_ms, load_impl, report, table, tflops

ap = argparse.ArgumentParser()
ap.add_argument("--solution", action="store_true")
args = ap.parse_args()
m = load_impl(__file__, args.solution)

torch.manual_seed(0)
for M, N, K in ((1, 1, 1), (129, 257, 63), (1000, 512, 256)):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(m.triton_matmul(a, b), a @ b, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(m.triton_matmul_silu(a, b), m.torch_matmul_silu(a, b), atol=5e-2, rtol=2e-2)
# TMA needs 16-byte aligned row strides, so K and N must be multiples of 8 in bf16
for M, N, K in ((129, 264, 64), (1000, 512, 256)):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bt = b.t().contiguous()
    torch.testing.assert_close(m.triton_matmul_tma(a, bt), a @ b, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(m.triton_matmul_tma(a, bt, "silu"), m.torch_matmul_silu(a, b), atol=6e-2, rtol=2e-2)
print("correctness: step 1 and step 2 kernels match cuBLAS in bf16, with and without SiLU")

rows = []
res = {}
# (tokens, 4*d, d): the HSTU UVQK projection shapes, then a square GEMM
for M, N, K in ((65536, 1024, 256), (65536, 2048, 512), (8192, 8192, 8192)):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bt = b.t().contiguous()
    flops = 2 * M * N * K
    t = {
        "cuBLAS a@b": bench_ms(lambda: a @ b),
        "cuBLAS + F.silu": bench_ms(lambda: m.torch_matmul_silu(a, b)),
        "step1 tiles": bench_ms(lambda: m.triton_matmul(a, b)),
        "step1 tiles + silu": bench_ms(lambda: m.triton_matmul_silu(a, b)),
        "step2 tma-ws": bench_ms(lambda: m.triton_matmul_tma(a, bt)),
        "step2 tma-ws + silu": bench_ms(lambda: m.triton_matmul_tma(a, bt, "silu")),
    }
    for name, ms in t.items():
        rows.append((f"{M}x{N}x{K}", name, f"{ms * 1e3:8.1f}", f"{tflops(flops, ms):6.0f}"))
    res[(M, N, K)] = t
table(rows, ["MxNxK", "impl", "time us", "TFLOP/s"])

t = res[(65536, 1024, 256)]
gain = t["cuBLAS + F.silu"] / t["step2 tma-ws + silu"]
step = t["step1 tiles"] / t["step2 tma-ws"]
report(
    "pointer-based tiles leave Blackwell's tensor cores idle waiting on loads; TMA + a persistent grid + "
    "warp specialization closes most of the gap to cuBLAS, and then fusing SiLU beats cuBLAS + a separate "
    "SiLU pass on HSTU's skinny projection",
    f"65536x1024x256: step2 is {step:.2f}x faster than step1; fused step2 {t['step2 tma-ws + silu'] * 1e3:.0f} us "
    f"vs cuBLAS+silu {t['cuBLAS + F.silu'] * 1e3:.0f} us ({gain:.2f}x)",
    "PASS" if gain > 1.2 and step > 1.3 else "CHECK: compare with the lesson's reference table",
    "print matmul_tma_kernel.best_config per shape; then set warp_specialize=False and re-measure "
    "to see how much of step 2's gain comes from the producer/consumer warp split",
)
