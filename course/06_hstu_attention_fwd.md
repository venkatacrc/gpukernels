---
id: 06
title: HSTU attention forward (jagged, fused)
lab: labs/06_hstu_attention_fwd
minutes: 120
track: hstu
summary: pointwise SiLU attention with a time-and-position bias, over jagged sequences, without ever storing N×N
---
## What HSTU attention computes

HSTU (Hierarchical Sequential Transduction Units, Zhai et al., 2024) is the generative recommender architecture behind Meta's trillion-parameter recommendation models. Each user is a sequence of interactions (here: movies, with timestamps). Its attention differs from a transformer's in three ways:

1. **Pointwise, not softmax.** \(A = \mathrm{SiLU}(\alpha\,QK^\top + \mathrm{rab}) / N\). No row normalisation, so no running max or sum, and each score is independent.
2. **Relative attention bias from position and time.**
   \(\mathrm{rab}_{ij} = w_{pos}[\mathrm{clamp}(i-j, 0, P-1)] + w_{ts}[\lfloor \log_2 \max(|t_i - t_j|, 1) \rfloor]\).
   How long ago you watched something matters, on a log scale.
3. **Jagged batches.** Users have different history lengths. Instead of padding to the longest, sequences are concatenated into one `(T, H, D)` tensor with `offsets[b] .. offsets[b+1]` marking user b.

Causal masking keeps \(j \le i\).

## The fusion

A PyTorch implementation pads to `(B, H, N, D)`, materialises the `(B, H, N, N)` score matrix, gathers the bias into another N×N tensor, applies SiLU, masks, and multiplies by V. That is several N² tensors in HBM, plus compute wasted on padding.

The fused kernel gives each program BM query rows of one (user, head). It streams that user's key/value blocks through shared memory, builds scores *and* bias in registers, and accumulates \(A V\) directly. **Nothing N×N ever touches HBM**, and padding never exists.

::: write
`labs/06_hstu_attention_fwd/kernel.py`.

**The bias helper (TODOs 1–3):**

```python
rel = offs_m[:, None] - offs_n[None, :]
pos_b = tl.minimum(tl.maximum(rel, 0), P - 1)
dt = tl.maximum(tl.abs(ts_m[:, None] - ts_n[None, :]).to(tl.float32), 1.0)
t_b = ((dt.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127      # floor(log2(dt)) from the exponent bits
t_b = tl.minimum(tl.maximum(t_b, 0), NT - 1)
bias = tl.load(pos_w_ptr + pos_b) + tl.load(ts_w_ptr + t_b)
```

**The kernel (TODOs 4–8).** Grid `(cdiv(max_len, BM), B * H)`. Read `start = offsets[b]` and `L = offsets[b+1] - start`; programs with `pid_m * BM >= L` return immediately (that is how jaggedness costs nothing). Then:

```python
hi = tl.minimum((pid_m + 1) * BM, L)          # causal: no key beyond this block's last query
for n0 in range(0, hi, BN):
    k, v, ts_n = ...                          # masked loads of BN keys
    s = tl.dot(q, tl.trans(k)) * alpha + bias
    a = s * tl.sigmoid(s) * inv_n
    a = tl.where((rel >= 0) & m_mask[:, None] & n_mask[None, :], a, 0.0)
    acc += tl.dot(a.to(v.dtype), v)
```

Use int64 for `start * stride_t`: with tens of millions of tokens × H × D, 32-bit offsets overflow.
:::

::: run
```bash
python labs/06_hstu_attention_fwd/run.py
```
The baseline is a *fair* dense PyTorch version: inputs already padded once (`padded_inputs`) so it is not charged for a Python loop, and bf16 throughout.
:::

::: measure
- s200 | speedup at B=128, N≤200 | 10.4 | ×
- s200_dense_mb | dense peak memory, B=128 N≤200 | 186 | MB
- s200_fused_mb | fused peak memory, B=128 N≤200 | 3 | MB
- s512 | speedup at B=64, N≤512 | 31.9 | ×
- s512_fused_us | fused time, B=64 N≤512 | 121 | µs
- s2048 | speedup at N≤2048 | 12.9 | ×
:::

::: explain
**Where the 10–32× comes from.** Three sources, roughly multiplicative:

- **Memory traffic.** Dense writes and rereads B·H·N² scores, plus bias tensors and masks: 994 MB at B=64, N=512. The fused kernel only reads Q, K, V once per query block and writes the output: 8.7 MB peak.
- **Padding.** Realistic histories are 45–52% padding in these batches. Dense computes all of it; fused programs past a user's length exit on their first instruction.
- **Causality.** Dense computes the full N×N and masks half away. The fused loop stops at `hi`, skipping blocks above the diagonal, which is about half the FLOPs.

The speedup peaks near N=512 and settles at 12–13× for long sequences. At large N the dense version becomes efficient big-GEMM work, while the fused kernel becomes compute-bound on its own `tl.dot`s.

**The exponent trick is exact.** A positive fp32 number is stored as \(2^{e-127} \times 1.m\), so `(bits >> 23) - 127` is exactly \(\lfloor \log_2 x \rfloor\). The reference uses `torch.frexp`, whose exponent is one larger; the course reference subtracts 1, and the two agree on every input. No `log2`, no rounding risk at exact powers of two.

**Why pointwise attention fuses so well.** Softmax attention (flash attention) must keep a running max and rescale the accumulator whenever the max grows. HSTU's SiLU score is independent per element, so each block's contribution simply adds to `acc`. That makes it simpler than flash attention, and it is why the backward in Lab 07 needs no stored row statistics at all.

**Why 1/N with N = max_len.** HSTU normalises by sequence length rather than by a softmax. Using the batch's max_len (as the reference does) keeps the scale identical between the dense and fused paths.
:::

::: cards
Q: Write the HSTU attention formula.
A: A = SiLU(α·QKᵀ + rab) / N, causal (j ≤ i), output A·V; rab = w_pos[clamp(i−j)] + w_ts[floor(log2 max(|t_i − t_j|, 1))].
Q: What is a jagged batch?
A: Variable-length sequences concatenated into one (T, …) tensor with an offsets array: user b owns rows offsets[b] … offsets[b+1]. No padding.
Q: How does the fused kernel avoid wasting work on short users?
A: The grid covers max_len query blocks for every user; programs whose block starts at or after the user's length L return immediately.
Q: How do you compute floor(log2(x)) for positive fp32 x without calling log2?
A: Reinterpret the bits as int32: ((bits >> 23) & 0xFF) − 127, the unbiased exponent.
Q: Why is HSTU attention easier to fuse than softmax attention?
A: Scores are pointwise (SiLU), with no row-wise normalisation, so there is no running max or sum to maintain and no rescaling; block contributions just add.
Q: Name the three sources of the fused kernel's speedup over dense PyTorch.
A: No N×N tensors in HBM, no compute on padding, and skipping key blocks above the causal diagonal.
Q: Why compute the base address in int64?
A: start·stride_t can exceed 2³¹ for large jagged batches (tokens × H × D), silently wrapping a 32-bit offset.
:::
