"""Check and benchmark Lab 08.  Usage: python labs/08_hstu_model/run.py [--solution]"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch

from gk.bench import bench_ms, load_impl, peak_mem_mb, report, table
from gk.hstu_ref import make_batch

ap = argparse.ArgumentParser()
ap.add_argument("--solution", action="store_true")
args = ap.parse_args()
m = load_impl(__file__, args.solution)

NUM_ITEMS = 3706  # MovieLens-1M


def batch(B, N, seed):
    _, _, _, ts, off = make_batch(B, N, 1, 16, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = int(off[-1])
    ids = torch.randint(1, NUM_ITEMS + 1, (T,), generator=g).cuda()
    tgt = torch.randint(1, NUM_ITEMS + 1, (T,), generator=g).cuda()
    return ids, ts, off, tgt


torch.manual_seed(0)
model = m.HSTU(NUM_ITEMS, d=64, layers=2, heads=2, head_dim=32, dropout=0.0).cuda()
with torch.no_grad():
    for blk in model.blocks:  # non-zero bias tables so the rab path is exercised
        blk.pos_w.normal_(0, 0.3)
        blk.ts_w.normal_(0, 0.3)

ids, ts, off, tgt = batch(16, 200, seed=3)
s_t = model.encode(ids, ts, off, 200, impl="triton")
s_r = model.encode(ids, ts, off, 200, impl="torch")
fwd_err = (s_t - s_r).abs().max().item()
grads = {}
for impl in ("triton", "torch"):
    model.zero_grad()
    torch.manual_seed(1)  # same negatives for both paths
    loss = model.sampled_softmax_loss(model.encode(ids, ts, off, 200, impl=impl), tgt)
    loss.backward()
    grads[impl] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
worst = max(((grads["triton"][n] - g).norm() / g.norm().clamp(min=1e-8)).item() for n, g in grads["torch"].items())
print(f"forward max |diff| = {fwd_err:.4f} (bf16), worst relative gradient error over all parameters = {worst:.4f}")
assert fwd_err < 5e-2 and worst < 5e-2, "triton and torch paths disagree"
print("correctness: the Triton-kernel model and the plain PyTorch model agree in forward and backward")

rows = []
res = {}
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
for B, N in ((128, 200), (64, 512), (32, 1024)):
    ids, ts, off, tgt = batch(B, N, seed=B)
    for impl in ("torch", "triton"):
        def step():
            opt.zero_grad(set_to_none=True)
            model.sampled_softmax_loss(model.encode(ids, ts, off, N, impl=impl), tgt).backward()
            opt.step()
        ms = bench_ms(step, warmup=5, rep=30)
        mem = peak_mem_mb(step)
        res[(B, N, impl)] = (ms, mem)
        rows.append((f"B={B} N={N}", int(off[-1]), impl, f"{ms:7.2f}", f"{int(off[-1]) / ms / 1e3:7.2f}", f"{mem:7.0f}"))
table(rows, ["batch", "tokens", "impl", "step ms", "Mtok/s", "peak MB"])

(t_ms, t_mem), (r_ms, r_mem) = res[(64, 512, "triton")], res[(64, 512, "torch")]
report(
    "with all three fused kernels the full training step (forward, backward, AdamW) is several times "
    "faster than the same model in PyTorch, and peak memory no longer grows with N^2",
    f"B=64 N=512: torch {r_ms:.1f} ms / {r_mem:.0f} MB, triton {t_ms:.1f} ms / {t_mem:.0f} MB "
    f"({r_ms / t_ms:.1f}x faster, {r_mem / t_mem:.1f}x less memory)",
    "PASS" if r_ms / t_ms > 2 else "CHECK: compare with the lesson's reference table",
    "profile one triton step (Lab 11) and list the kernels that now dominate; Lab 09 trains this model "
    "on MovieLens",
)
