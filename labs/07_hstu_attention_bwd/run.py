"""Check and benchmark Lab 07.  Usage: python labs/07_hstu_attention_bwd/run.py [--solution]"""
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


def tables():
    return ((torch.randn(P, device="cuda") * 0.5).requires_grad_(), (torch.randn(NT, device="cuda") * 0.5).requires_grad_())


for B, N, H, D in ((1, 1, 1, 32), (3, 50, 2, 32), (8, 200, 2, 64), (4, 257, 2, 128)):
    q, k, v, ts, off = make_batch(B, N, H, D, seed=B + N, dtype=torch.float32)
    alpha = D ** -0.5
    pos_w, ts_w = tables()
    leaves = [x.clone().requires_grad_() for x in (q, k, v)]
    ref = hstu_attention_ref(*leaves, ts, off, N, pos_w, ts_w, alpha)
    g = torch.randn_like(ref)
    gref = torch.autograd.grad(ref, leaves + [pos_w, ts_w], g)
    leaves2 = [x.clone().requires_grad_() for x in (q, k, v)]
    out = m.hstu_attention(*leaves2, ts, off, N, pos_w, ts_w, alpha)
    gout = torch.autograd.grad(out, leaves2 + [pos_w, ts_w], g)
    for name, a, r in zip(["dq", "dk", "dv", "dpos", "dts"], gout, gref):
        # fp32 inputs, but tl.dot uses TF32 tensor cores, so allow TF32-level error
        scale = r.abs().max().clamp(min=1e-6)
        err = ((a - r).abs().max() / scale).item()
        assert err < 2e-2, f"{name} relative error {err:.3g} for B={B} N={N} H={H} D={D}"
print("correctness: dq, dk, dv and both bias-table gradients match autograd through the reference")

rows = []
res = {}
for B, N, H, D in ((128, 200, 2, 64), (64, 512, 4, 64), (16, 2048, 4, 128)):
    q, k, v, ts, off = make_batch(B, N, H, D, seed=1)
    alpha = D ** -0.5
    pos_w, ts_w = tables()
    qf, kf, vf = (x.clone().requires_grad_() for x in (q, k, v))
    out = m.hstu_attention(qf, kf, vf, ts, off, N, pos_w, ts_w, alpha)
    g = torch.randn_like(out)
    fused = lambda: torch.autograd.grad(m.hstu_attention(qf, kf, vf, ts, off, N, pos_w, ts_w, alpha),
                                        (qf, kf, vf, pos_w, ts_w), g)
    qp, kp, vp, tsp, lengths = padded_inputs(q, k, v, ts, off, N)
    qp, kp, vp = (x.requires_grad_() for x in (qp, kp, vp))
    gp = torch.randn(B, H, N, D, device="cuda", dtype=q.dtype)
    dense = lambda: torch.autograd.grad(hstu_attention_dense(qp, kp, vp, tsp, lengths, pos_w, ts_w, alpha),
                                        (qp, kp, vp, pos_w, ts_w), gp)
    t_f, t_d = bench_ms(fused), bench_ms(dense)
    mem_f, mem_d = peak_mem_mb(fused), peak_mem_mb(dense)
    res[(B, N)] = (t_d, t_f, mem_d, mem_f)
    rows.append((f"B={B} N={N} H={H} D={D}", f"{t_d * 1e3:8.0f}", f"{t_f * 1e3:7.0f}", f"{t_d / t_f:5.1f}x",
                 f"{mem_d:7.0f}", f"{mem_f:6.1f}"))
table(rows, ["shape (fwd+bwd)", "dense us", "fused us", "speedup", "dense MB", "fused MB"])

t_d, t_f, mem_d, mem_f = res[(64, 512)]
report(
    "fused forward+backward recomputes the scores instead of storing them, so training memory stays "
    "O(tokens) instead of O(B * H * N^2), and it is several times faster than padded dense autograd",
    f"B=64 N=512 fwd+bwd: dense {t_d * 1e3:.0f} us / {mem_d:.0f} MB, fused {t_f * 1e3:.0f} us / {mem_f:.1f} MB",
    "PASS" if t_d / t_f > 3 and mem_f < 0.1 * mem_d else "CHECK: compare with the lesson's reference table",
    "time the two backward kernels separately with the profiler (Lab 11). Then replace the time-bucket "
    "atomics with an in-register reduction over the NT buckets and see whether contention was costing you",
)
