"""Check and benchmark Lab 06.  Usage: python labs/06_hstu_attention_fwd/run.py [--solution]"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch

from gk.bench import bench_ms, load_impl, peak_mem_mb, report, table
from gk.hstu_ref import hstu_attention_dense, hstu_attention_ref, make_batch, padded_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--solution", action="store_true")
args = ap.parse_args()
m = load_impl(__file__, args.solution)

P, NT = 256, 32
torch.manual_seed(0)
pos_w = (torch.randn(P, device="cuda") * 0.5)
ts_w = (torch.randn(NT, device="cuda") * 0.5)

for B, N, H, D in ((1, 1, 1, 32), (3, 50, 2, 32), (8, 200, 2, 64), (4, 257, 4, 128)):
    q, k, v, ts, off = make_batch(B, N, H, D, seed=B + N)
    alpha = D ** -0.5
    ref = hstu_attention_ref(q, k, v, ts, off, N, pos_w, ts_w, alpha)
    out = m.hstu_attention_fwd(q, k, v, ts, off, N, pos_w, ts_w, alpha)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2, msg=f"B={B} N={N} H={H} D={D}")
print("correctness: matches the dense PyTorch reference on jagged batches (lengths 1..N, causal, pos+time bias)")

rows = []
res = {}
for B, N, H, D in ((128, 200, 2, 64), (64, 512, 4, 64), (32, 1024, 4, 128), (16, 2048, 4, 128)):
    q, k, v, ts, off = make_batch(B, N, H, D, seed=1)
    alpha = D ** -0.5
    T = q.shape[0]
    padded = padded_inputs(q, k, v, ts, off, N)
    pad_frac = 1 - T / (B * N)
    fused = lambda: m.hstu_attention_fwd(q, k, v, ts, off, N, pos_w, ts_w, alpha)
    dense = lambda: hstu_attention_dense(*padded, pos_w, ts_w, alpha)
    t_f = bench_ms(fused)
    mem_f = peak_mem_mb(fused)
    t_d = bench_ms(dense)
    mem_d = peak_mem_mb(dense)
    res[(B, N)] = (t_d, t_f, mem_d, mem_f)
    rows.append((f"B={B} N={N} H={H} D={D}", T, f"{pad_frac:4.0%}", f"{t_d * 1e3:8.0f}", f"{t_f * 1e3:7.0f}",
                 f"{t_d / t_f:5.1f}x", f"{mem_d:8.0f}", f"{mem_f:6.1f}"))
table(rows, ["shape", "tokens", "padding", "dense us", "fused us", "speedup", "dense MB", "fused MB"])

t_d, t_f, mem_d, mem_f = res[(64, 512)]
report(
    "the fused jagged kernel skips padding, never materialises the B x H x N x N score, bias and mask "
    "tensors, and skips the upper (non-causal) triangle, so it is several times faster than padded dense "
    "bf16 attention and its extra memory is just the output",
    f"B=64 N=512: dense {t_d * 1e3:.0f} us / {mem_d:.0f} MB, fused {t_f * 1e3:.0f} us / {mem_f:.1f} MB "
    f"({t_d / t_f:.1f}x faster, {mem_d / mem_f:.0f}x less memory)",
    "PASS" if t_d / t_f > 3 and mem_f < 0.1 * mem_d else "CHECK: compare with the lesson's reference table",
    "rerun with full=True in make_batch (no padding) to separate the jagged win from the fusion win; "
    "then Lab 07 writes the backward",
)
