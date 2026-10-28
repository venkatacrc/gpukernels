---
id: 03
title: LayerNorm × gate, forward and backward
lab: labs/03_layernorm_mul
minutes: 60
track: kernels
summary: fuse a normalisation with HSTU's gating multiply; save mean/rstd; write the backward
---
HSTU's output path computes `y = LayerNorm(attn) * u`, where `u` is a gate projected from the input. In PyTorch that is a LayerNorm kernel, which writes its output to HBM, and then a multiply kernel that reads it back. Fusing them saves one full read and write of the activation. It is also the first kernel in the course with a **backward**, so the forward must save what the backward needs.

::: write
`labs/03_layernorm_mul/kernel.py`.

**Forward (TODOs 1–6), one row per program:**

```python
x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / D
xc = tl.where(mask, x - mean, 0.0)
var = tl.sum(xc * xc, axis=0) / D
rstd = 1.0 / tl.sqrt(var + eps)
y = (xc * rstd * w + b) * u
tl.store(mean_ptr + row, mean); tl.store(rstd_ptr + row, rstd)
```

Accumulate in fp32 even when inputs are bf16.

**Backward (TODO 7)**, written in PyTorch inside the `autograd.Function`, using the saved mean and rstd. With \(\hat{x} = (x-\mu)\,r\), \(\ell = \hat{x} w + b\) and \(y = \ell u\):

- \(\partial u = g \cdot \ell\)
- \(g_\ell = g \cdot u\), so \(\partial w = \sum_{rows} g_\ell \hat{x}\) and \(\partial b = \sum_{rows} g_\ell\)
- \(g_{\hat{x}} = g_\ell w\), then \(\partial x = r\,(g_{\hat{x}} - \overline{g_{\hat{x}}} - \hat{x}\,\overline{g_{\hat{x}} \hat{x}})\), where the overline is the mean over the row.
:::

::: run
```bash
python labs/03_layernorm_mul/run.py
```
:::

::: measure
- speed_min | smallest fused speedup in the table | 2.09 | ×
- speed_max | largest fused speedup | 2.57 | ×
- bf16_bw | bf16, 262144×512 | 3737 | GB/s
- fp32_bw | fp32, 262144×512 | 6086 | GB/s
:::

::: explain
**Why about 2×+.** Unfused: LayerNorm reads x and writes ℓ; the multiply reads ℓ and u and writes y. That is 5 activation-sized transfers. Fused: read x and u, write y, so 3 transfers, plus 8 bytes per row for mean and rstd. 5/3 ≈ 1.7× from bytes alone. The measured 2.1–2.6× is higher because PyTorch's LayerNorm also does a separate pass to compute statistics, and we save a launch.

**bf16 is slower in GB/s, not in time.** The same shape in bf16 moves half the bytes, but the kernel still does the same number of reductions and the same per-row work with half as many bytes to hide them behind. With D=512, a row is only 1 KB in bf16; each program is short and spends proportionally more time in the two tree reductions and the launch tail. The lesson: for narrow rows, put several rows in one program (a 2-D block) so each program moves more bytes.

**Save statistics, not activations.** Saving mean and rstd costs 8 bytes per row; saving ℓ would cost D·2 bytes per row. The backward recomputes \(\hat{x}\) from x and the stats. This *recompute instead of store* trade comes back in Lab 07 (attention backward) and Lab 08 (GEMM+SiLU backward).

**Why the dx formula has two mean terms.** Both μ and r depend on every element in the row, so each input's gradient is reduced by the gradient carried through the mean (the \(\overline{g_{\hat{x}}}\) term) and through the variance (the \(\hat{x}\,\overline{g_{\hat{x}}\hat{x}}\) term).
:::

::: cards
Q: Why save mean and rstd in LayerNorm's forward rather than the normalised output?
A: They cost 8 bytes per row instead of D values per row; the backward recomputes x̂ = (x − mean)·rstd cheaply from x.
Q: How many activation-sized HBM transfers does LayerNorm-then-multiply take unfused vs fused?
A: Unfused 5 (read x, write ℓ, read ℓ, read u, write y); fused 3 (read x, read u, write y).
Q: Give dx for LayerNorm in terms of g_x̂ = g·w.
A: dx = rstd · (g_x̂ − mean(g_x̂) − x̂ · mean(g_x̂ · x̂)), means taken over the row.
Q: Why accumulate LayerNorm statistics in fp32 for bf16 inputs?
A: bf16 has only 8 bits of mantissa; summing hundreds of values loses precision and the variance can even go negative from cancellation.
Q: The bf16 version reaches lower GB/s than fp32 at D=512. What would you change?
A: Process several rows per program (2-D block) so each program moves more bytes and the fixed per-program reduction and launch costs amortise.
:::
