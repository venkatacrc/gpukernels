"""Lab 11: system optimisations around the kernels.  Usage: python labs/11_system_opt/run.py

Four experiments, each answering one question:
  A. Is the training step GPU-bound or CPU-bound?   (profiler: GPU busy time vs wall time)
  B. How does throughput scale with batch size?     (amortising per-step launch overhead)
  C. What do CUDA graphs buy at inference?          (eager vs graph replay, by batch size)
  D. What does a warm Triton cache save?            (cold vs warm compile + autotune, in subprocesses)
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from torch.profiler import ProfilerActivity, profile

from gk import kernels
from gk.bench import bench_ms, device_summary, report, table
from gk.hstu_ref import make_batch

NUM_ITEMS, MAX_LEN = 3706, 200
HSTU = kernels.lab("08_hstu_model").HSTU


def batch(B, seed=0):
    _, _, _, ts, off = make_batch(B, MAX_LEN, 1, 16, seed=seed)
    T = int(off[-1])
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(1, NUM_ITEMS + 1, (T,), generator=g).cuda(), ts, off,
            torch.randint(1, NUM_ITEMS + 1, (T,), generator=g).cuda())


def train_step_fn(model, opt, b, impl):
    ids, ts, off, tgt = b

    def step():
        opt.zero_grad(set_to_none=True)
        model.sampled_softmax_loss(model.encode(ids, ts, off, MAX_LEN, impl), tgt, 0).backward()
        opt.step()
    return step


def gpu_busy_fraction(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = sum(e.device_time for e in events) * 1e-6
    per_kernel = {}
    for e in events:
        per_kernel[e.name] = per_kernel.get(e.name, 0.0) + e.device_time / iters
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:8]
    return busy / wall, wall / iters * 1e3, len(events) / iters, top


def main():
    print(f"[11_system_opt] {device_summary()}")
    torch.manual_seed(0)
    model = HSTU(NUM_ITEMS, d=64, layers=2, heads=2, head_dim=32, dropout=0.2, max_len=MAX_LEN).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    # A. GPU-bound or CPU-bound?
    rows = []
    a_res = {}
    for impl in ("torch", "triton"):
        frac, ms, launches, top = gpu_busy_fraction(train_step_fn(model, opt, batch(128), impl))
        a_res[impl] = (frac, ms)
        rows.append((impl, f"{ms:6.2f}", f"{frac:5.0%}", f"{launches:5.0f}"))
        if impl == "triton":
            top_kernels = [(name[:70], f"{us:8.1f}") for name, us in top]
    print("\nA. training step, B=128 users, N=200 (ML-1M shape)")
    table(rows, ["impl", "step ms", "GPU busy", "kernels/step"])
    print("\n   top GPU kernels per triton step (us):")
    table(top_kernels, ["kernel", "us/step"])

    # B. batch-size scaling
    rows = []
    b_res = {}
    for B in (32, 128, 512, 2048):
        b = batch(B, seed=B)
        ms = bench_ms(train_step_fn(model, opt, b, "triton"), warmup=5, rep=30)
        T = b[0].numel()
        b_res[B] = T / ms / 1e3
        rows.append((B, T, f"{ms:7.2f}", f"{b_res[B]:6.2f}"))
    print("\nB. triton training step vs batch size")
    table(rows, ["users", "tokens", "step ms", "Mtok/s"])

    # C. CUDA graphs at inference
    from cuda_graphs import GraphedTopK

    model.eval()
    graphed = GraphedTopK(model, MAX_LEN)
    rows = []
    c_res = {}
    for B in (1, 8, 64):
        ids, ts, off, _ = batch(B, seed=100 + B)
        eager_ms = bench_ms(lambda: model.topk(ids, ts, off, MAX_LEN))
        graph_ms = bench_ms(lambda: graphed(ids, ts, off))
        same = torch.equal(model.topk(ids, ts, off, MAX_LEN)[1], graphed(ids, ts, off)[1])
        c_res[B] = (eager_ms, graph_ms)
        rows.append((B, ids.numel(), f"{eager_ms * 1e3:7.0f}", f"{graph_ms * 1e3:7.0f}", f"{eager_ms / graph_ms:4.1f}x", same))
    print("\nC. inference top-10: eager vs CUDA graph replay")
    table(rows, ["users", "tokens", "eager us", "graph us", "speedup", "same top-10"])

    # D. cold vs warm Triton cache, measured in fresh processes
    probe = (
        "import sys, time; sys.path.insert(0, %r); t0 = time.perf_counter(); import torch\n"
        "from gk import kernels\n"
        "from gk.hstu_ref import make_batch\n"
        "m = kernels.lab('08_hstu_model').HSTU(3706, max_len=200).cuda()\n"
        "_, _, _, ts, off = make_batch(128, 200, 1, 16)\n"
        "ids = torch.randint(1, 3707, (int(off[-1]),), device='cuda')\n"
        "m.sampled_softmax_loss(m.encode(ids, ts, off, 200), ids, 0).backward(); torch.cuda.synchronize()\n"
        "print(time.perf_counter() - t0)\n" % str(ROOT)
    )
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, TRITON_CACHE_DIR=cache)
        runs = [float(subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                                     check=True).stdout.strip().splitlines()[-1]) for _ in range(2)]
        env_at = dict(env, TRITON_CACHE_AUTOTUNING="1")
        with tempfile.TemporaryDirectory() as cache2:
            env_at["TRITON_CACHE_DIR"] = cache2
            runs_at = [float(subprocess.run([sys.executable, "-c", probe], env=env_at, capture_output=True, text=True,
                                            check=True).stdout.strip().splitlines()[-1]) for _ in range(2)]
    print("\nD. first training step in a fresh process (seconds)")
    table([("kernel cache only", f"{runs[0]:6.1f}", f"{runs[1]:6.1f}"),
           ("kernel + autotune cache", f"{runs_at[0]:6.1f}", f"{runs_at[1]:6.1f}")],
          ["cache", "cold", "warm"])

    report(
        "at the ML-1M batch size the fused model is launch-bound (GPU idle much of the step), so bigger "
        "batches raise tokens/s almost for free, CUDA graphs cut small-batch inference latency, and caching "
        "autotune results removes most of the start-up cost",
        f"GPU busy {a_res['triton'][0]:.0%} at B=128; {b_res[32]:.1f} -> {b_res[2048]:.1f} Mtok/s from B=32 to 2048; "
        f"graphs {c_res[1][0] / c_res[1][1]:.1f}x at B=1; start-up {runs_at[0]:.0f}s cold -> {runs_at[1]:.0f}s warm",
        "PASS" if c_res[1][0] > c_res[1][1] and b_res[2048] > b_res[32] else "CHECK",
        "serve through GraphedTopK in serving/hstu_backend.py and rerun serving/client.py; "
        "then try torch.compile on the torch impl and compare with the hand-written kernels",
    )


if __name__ == "__main__":
    main()
