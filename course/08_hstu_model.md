---
id: 08
title: Stitch the kernels into an HSTU model
lab: labs/08_hstu_model
minutes: 90
track: hstu
summary: one HSTU layer from Labs 03, 04, 06 and 07; autograd glue; check against plain PyTorch
---
Everything so far was one kernel at a time. Now they become a model. One HSTU layer on the jagged token matrix `x` of shape `(T, d)`:

```text
h          = LayerNorm(x)
u, v, q, k = split(SiLU(h @ W_uvqk))          Lab 04: one fused GEMM + SiLU for all four projections
a          = HSTUAttention(q, k, v, ts, rab)  Labs 06 + 07: jagged fused attention, fwd + bwd
y          = LayerNorm(a) * u                 Lab 03: fused norm + gate
x          = x + Dropout(y @ W_o)
```

The full model embeds item ids (plus an optional learned position embedding), stacks layers, L2-normalises the output, and scores the next item by cosine similarity against the item embedding table. Item and output embeddings are tied.

`gk/kernels.py` loads your `kernel.py` from each lab when `GK_KERNELS=mine` is set, and the reference `solution.py` otherwise. So you can run the model on your own kernels end to end.

::: write
`labs/08_hstu_model/kernel.py`.

**TODO 1: MatmulSiLU backward.** The forward uses the Lab 04 kernel and saves only `a` and `w`. In backward, recompute the pre-activation with cuBLAS:

```python
z = (a @ w_t.t()).float()
sig = torch.sigmoid(z)
dz = (dy.float() * sig * (1 + z * (1 - sig))).to(a.dtype)
return dz @ w_t, dz.t() @ a
```

**TODOs 2–4: the block forward.**

```python
uvqk = MatmulSiLU.apply(h.to(torch.bfloat16), self.w_uvqk.to(torch.bfloat16))
u, v, q, k = uvqk.split(H * dh, dim=1)
a = kernels.hstu_attention()(q, k, v, ts, offsets, max_len, self.pos_w, self.ts_w, alpha)
y = kernels.ln_mul()(a.reshape(T, H * dh), u.contiguous(), self.norm_w, self.norm_b)
```

**TODO 5: the sampled softmax loss.** Score each position's state against its true next item and `num_neg` shared random negatives, divide by a temperature (0.05), and take cross-entropy with the positive at index 0. Mask a negative that happens to equal the target. `num_neg=0` means full softmax over the catalogue.
:::

::: run
```bash
python labs/08_hstu_model/run.py
GK_KERNELS=mine python labs/08_hstu_model/run.py     # the model on *your* kernels from labs 03/04/07
```
:::

::: measure
- fwd_diff | forward max abs diff vs PyTorch | 0.0045 |
- grad_err | worst relative gradient error | 0.0111 |
- b128_torch | step time torch, B=128 N=200 | 27.1 | ms
- b128_triton | step time triton, B=128 N=200 | 6.5 | ms
- b64_torch_mb | peak memory torch, B=64 N=512 | 1042 | MB
- b64_triton_mb | peak memory triton, B=64 N=512 | 144 | MB
- b32_triton | step time triton, B=32 N=1024 | 12.1 | ms
:::

::: explain
**Correctness at bf16 precision.** Forward outputs agree to about 4.5e-3 and every parameter gradient to about 1% relative error. That is what bf16 matmuls in two different orders produce. The check compares *every* parameter's gradient, including the bias tables, which is where a wrong atomic or a missing `alpha` shows up first.

**4× faster, 7× less memory.** The step includes forward, backward, the loss and AdamW. Memory is the clearer story: the dense path's peak grows with N² (365 MB at N=200 → 1042 MB at 512 → 2006 MB at 1024) while the fused path stays near 120–150 MB, because nothing N×N exists. Step-time ratios move around between runs (this GPU is shared), but the triton step is consistently 4–5× faster at N=200–512.

**Why the speedup shrinks at N=1024 (about 1.3×).** At B=32 the dense path is a few large batched GEMMs, efficient on tensor cores. The fused attention backward loops over 1024 keys per program and recomputes scores twice, which is the compute-bound regime from Lab 07.

**What the step is made of now.** At the ML-1M shape the attention kernels are no longer the bottleneck; Lab 11's profile shows the fp32 loss GEMMs, softmax, AdamW and launch overhead on top. Fusion moves the bottleneck; it never removes it.

**Recompute in MatmulSiLU.** Saving `z` (T × 4·H·dh) per layer would cost more memory than all the attention state. Recomputing it is one skinny cuBLAS GEMM, about 37 µs at the Lab 04 shape.

**Why the dropout and residual stay in PyTorch.** They are cheap elementwise ops on (T, d) with d=64. Fusing them would save microseconds. Always fuse in the order of bytes saved, which is the order this course followed.
:::

::: cards
Q: List the four stages of one HSTU layer and which lab's kernel implements each.
A: SiLU(LN(x)·W_uvqk) (Lab 04 fused GEMM+SiLU) → jagged HSTU attention (Labs 06/07) → LayerNorm(a)·u (Lab 03) → residual + dropout + output projection (PyTorch).
Q: How do you check a model built on custom kernels is correct?
A: Run the same weights through a plain-PyTorch implementation and compare forward outputs and every parameter's gradient (relative error), with a tolerance matched to the precision (about 1e-2 for bf16).
Q: Why does the fused model's peak memory stay flat as N grows, while dense grows quadratically?
A: Nothing of size N×N is ever materialised: attention scores live in registers, and backward recomputes them.
Q: What does sampled softmax with shared negatives compute?
A: For each position: logits = [⟨s, e_target⟩, ⟨s, e_neg1⟩ … ⟨s, e_negK⟩] / τ, cross-entropy with label 0. One random set of negatives is shared by the whole batch, so it is a single GEMM.
Q: Why recompute z = a·Wᵀ in MatmulSiLU's backward instead of saving it?
A: z is a (T × 4·H·dh) activation per layer; one extra skinny GEMM is cheaper than that much memory, and memory is what limits batch size.
:::
