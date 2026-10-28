"""Check and benchmark Lab 03.  Usage: python labs/03_layernorm_mul/run.py [--solution]"""
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

torch.manual_seed(0)
for D in (64, 200, 512):
    x = torch.randn(33, D, device="cuda", requires_grad=True)
    u = torch.randn(33, D, device="cuda", requires_grad=True)
    w = torch.randn(D, device="cuda", requires_grad=True)
    b = torch.randn(D, device="cuda", requires_grad=True)
    ref = m.torch_ln_mul(x, u, w, b)
    out = m.ln_mul(x, u, w, b)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    g = torch.randn_like(ref)
    grads_ref = torch.autograd.grad(ref, (x, u, w, b), g)
    grads = torch.autograd.grad(out, (x, u, w, b), g)
    for name, a, r in zip("xuwb", grads, grads_ref):
        torch.testing.assert_close(a, r, atol=1e-3, rtol=1e-3, msg=f"d{name} wrong for D={D}")
print("correctness: forward and all four gradients match PyTorch autograd")

rows = []
res = {}
for M, D in ((65536, 256), (65536, 512), (262144, 512)):
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(M, D, device="cuda", dtype=dt)
        u = torch.randn_like(x)
        w = torch.randn(D, device="cuda", dtype=dt)
        b = torch.randn(D, device="cuda", dtype=dt)
        fused_bytes = 3 * x.numel() * x.element_size()  # read x, read u, write y
        t_ref = bench_ms(lambda: m.torch_ln_mul(x, u, w, b))
        t_tri = bench_ms(lambda: m.triton_ln_mul_fwd(x, u, w, b))
        res[(M, D, dt)] = t_ref / t_tri
        rows.append((f"{M}x{D}", str(dt).split(".")[-1], f"{t_ref * 1e3:7.1f}", f"{t_tri * 1e3:7.1f}",
                     f"{t_ref / t_tri:4.2f}x", f"{gbps(fused_bytes, t_tri):6.0f}"))
table(rows, ["shape", "dtype", "torch us", "triton us", "speedup", "triton GB/s"])

s = res[(262144, 512, torch.bfloat16)]
report(
    "unfused LayerNorm then multiply moves 5 tensors through HBM (read x, write ln, read ln, read u, "
    "write y); fused moves 3, so traffic alone predicts about 5/3 = 1.7x, more if the torch kernels "
    "themselves run below peak bandwidth",
    f"262144x512 bf16 speedup {s:.2f}x",
    "PASS" if s > 1.3 else "CHECK: expected a clear speedup on large bf16 inputs",
    "bf16 reaches only about half the fp32 GB/s because a 1 KB row is too little work per program: "
    "process ROWS=4 rows per program with a 2-D block and re-measure. Stretch goal: write the backward "
    "in Triton (dw and db need partial sums per program plus a second small reduction kernel)",
)
