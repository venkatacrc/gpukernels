"""Check and benchmark Lab 02.  Usage: python labs/02_fused_softmax/run.py [--solution]"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch

from gk.bench import bench_ms, gbps, load_impl, peak_mem_mb, report, table

ap = argparse.ArgumentParser()
ap.add_argument("--solution", action="store_true")
args = ap.parse_args()
m = load_impl(__file__, args.solution)

for shape in ((1, 1), (7, 1000), (257, 4097)):
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(m.triton_softmax(x), torch.softmax(x, dim=1))
xb = torch.randn(64, 3000, device="cuda", dtype=torch.bfloat16)
torch.testing.assert_close(m.triton_softmax(xb), torch.softmax(xb.float(), dim=1).to(torch.bfloat16))
print("correctness: matches torch.softmax in fp32 and bf16")

impls = {"naive (5 kernels)": m.naive_softmax, "torch.softmax": lambda x: torch.softmax(x, dim=1), "triton fused": m.triton_softmax}
rows = []
at = {}
n_rows = 16384
for n_cols in (256, 1024, 4096, 16384):
    x = torch.randn(n_rows, n_cols, device="cuda")
    ideal = 2 * x.numel() * x.element_size()  # read x once, write out once
    for name, fn in impls.items():
        ms = bench_ms(lambda: fn(x))
        mem = peak_mem_mb(lambda: fn(x))
        rows.append((n_cols, name, f"{ms * 1e3:8.1f}", f"{gbps(ideal, ms):6.0f}", f"{mem:7.0f}"))
        at[(n_cols, name)] = ms
table(rows, ["cols", "impl", "time us", "effective GB/s", "peak extra MB"])

speedup = at[(4096, "naive (5 kernels)")] / at[(4096, "triton fused")]
report(
    "fusing the five passes into one read and one write makes softmax several times faster than the naive "
    "version, matches or beats torch.softmax, and needs no intermediate tensors",
    f"at 16384x4096 fp32: naive {at[(4096, 'naive (5 kernels)')] * 1e3:.0f} us, torch {at[(4096, 'torch.softmax')] * 1e3:.0f} us, "
    f"triton {at[(4096, 'triton fused')] * 1e3:.0f} us (fused is {speedup:.1f}x faster than naive)",
    "PASS" if speedup > 2.5 else "CHECK: fused kernel should be several times faster than naive",
    "add n_cols=65536 to the sweep: a whole row (256 KB fp32) no longer fits in one program's registers "
    "and the kernel spills. Fix it with the two-pass online softmax described in the lesson",
)
