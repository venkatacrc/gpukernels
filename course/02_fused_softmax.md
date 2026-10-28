---
id: 02
title: Fused softmax
lab: labs/02_fused_softmax
minutes: 45
track: kernels
summary: kernel fusion = fewer HBM round trips; one row per program, stable max, register limits
---
Softmax along a row: \(y_i = e^{x_i - m} / \sum_j e^{x_j - m}\) with \(m = \max_j x_j\).

Written in PyTorch as separate ops (`max`, `sub`, `exp`, `sum`, `div`), every step reads the full matrix from HBM and writes a temporary back. That is about five reads and three writes per element. A **fused** kernel loads each row once into registers, does all of it there, and writes once: one read, one write.

That is the entire idea of kernel fusion: count HBM round trips and remove them.

::: write
`labs/02_fused_softmax/kernel.py`, TODOs 1–5. One program per row:

```python
row = tl.program_id(0)
cols = tl.arange(0, BLOCK)
mask = cols < n_cols
x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)
num = tl.exp(x)
out = num / tl.sum(num, axis=0)
tl.store(out_ptr + row * stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)
```

`other=-inf` matters: padded lanes must not win the max, and `exp(-inf) = 0` keeps them out of the sum. `BLOCK = triton.next_power_of_2(n_cols)`; bump `num_warps` as rows grow so each thread holds fewer elements.
:::

::: run
```bash
python labs/02_fused_softmax/run.py
```
:::

::: measure
- naive_us | naive (5 PyTorch ops), 16384×4096 | 558 | µs
- torch_us | torch.softmax, 16384×4096 | 182 | µs
- triton_us | your Triton, 16384×4096 | 82 | µs
- triton_16k | Triton at 16384 columns | 7038 | GB/s
- torch_16k | torch.softmax at 16384 columns | 4406 | GB/s
- triton_64k | Triton at 65536 columns | 3951 | GB/s
:::

::: explain
**Naive vs fused: 6.8×.** The 16384×4096 fp32 matrix is 256 MiB. The naive version moves about 8 × 256 MiB; the fused one moves 2 × 256 MiB. Time ratio 558 / 82 ≈ 6.8, which is the round-trip ratio plus the saved launches.

**Fused vs torch.softmax: 2.2×.** PyTorch's softmax is already one fused kernel, so why is ours faster at this shape? Its generic kernel makes more than one pass over the row (max pass, then sum-and-write) and is tuned for a wide range of shapes. Ours holds the entire row in registers and reads it once. At 16384 columns we reach 7.0 TB/s, essentially HBM speed, vs 4.4 TB/s for torch.

**Where it breaks: 65536 columns.** A 64K-column fp32 row is 256 KB. With 32 warps that is 64 values per thread, more than the register file can hold alongside everything else, so the compiler **spills** to local memory, which really is HBM. Bandwidth drops to 3.95 TB/s and torch catches up (3.7 TB/s). The fix for very long rows is the *online softmax*: loop over chunks keeping a running max and a rescaled running sum. That is exactly the trick flash attention uses, and you will use its relative in Lab 06.

**Numerical safety.** Subtracting the row max keeps `exp` from overflowing in fp32 (it overflows above about 88) without changing the result, because the factor \(e^{-m}\) cancels.
:::

::: cards
Q: What is kernel fusion, in terms of memory traffic?
A: Combining several elementwise or reduction steps into one kernel so intermediates stay in registers or shared memory instead of making HBM round trips. Count bytes moved before and after.
Q: Why load masked lanes with `other=-inf` in softmax?
A: So padded lanes never become the row max, and exp(-inf) = 0 keeps them out of the denominator.
Q: Why does the one-row-per-program softmax slow down at 65536 columns?
A: The row no longer fits in registers; the compiler spills to local memory (which lives in HBM), adding traffic. Use an online (chunked) softmax instead.
Q: What is the online softmax?
A: Process a row in chunks keeping running max m and sum s; when the max grows, rescale: s ← s·e^(m_old − m_new) + Σ e^(x − m_new). One pass, bounded registers; the core of flash attention.
Q: Why subtract the max before exp?
A: To avoid overflow (exp overflows fp32 above about 88). The common factor e^(−max) cancels between numerator and denominator.
:::
