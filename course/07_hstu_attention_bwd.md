---
id: 07
title: HSTU attention backward
lab: labs/07_hstu_attention_bwd
minutes: 120
track: hstu
summary: recompute instead of store, two-kernel FA2 split, and why naive atomics cost 70×
---
Training needs gradients for Q, K, V and for the two bias tables. The forward saved only its inputs (no N×N matrices), so the backward **recomputes** scores tile by tile, exactly like flash attention.

With \(s = \alpha\, q\cdot k + \mathrm{rab}\), \(\sigma = \mathrm{sigmoid}(s)\), \(A = \mathrm{SiLU}(s)/N\) (masked):

| Gradient | Formula |
|---|---|
| dV | \(A^\top dO\) |
| dA | \(dO\, V^\top\) |
| dS | \(dA \cdot \sigma (1 + s(1-\sigma)) / N\), since \(\mathrm{SiLU}'(s) = \sigma(1 + s(1-\sigma))\) |
| dQ | \(\alpha\, dS\, K\) |
| dK | \(\alpha\, dS^\top Q\) |
| dpos[b], dts[b] | sum of dS over entries that fall in bucket b |

## Two kernels, no atomics on Q/K/V

dK and dV are sums over *query* rows; dQ is a sum over *key* rows. The FlashAttention-2 split:

- **Kernel 1** fixes a block of BN keys and loops over query blocks \(i \ge j\), accumulating dK and dV in registers. It also produces the bias-table gradients.
- **Kernel 2** fixes a block of BM queries and loops over key blocks \(j \le i\), accumulating dQ.

Each output block has exactly one owner, so dQ, dK and dV are plain stores.

::: write
`labs/07_hstu_attention_bwd/kernel.py`.

**The shared tile (TODOs 1–3)** in `_ds_tile`: recompute `s`, the bias, `sig`, the validity mask and `a`, then

```python
da = tl.dot(do, tl.trans(v))
ds = tl.where(valid, da * sig * (1.0 + s * (1.0 - sig)) * inv_n, 0.0)
```

**Kernel 1 (TODOs 4–5):** `dv += tl.dot(tl.trans(a.to(do.dtype)), do)`, `dk += tl.dot(tl.trans(ds.to(q.dtype)), q)`, and scale dk by alpha once at the end.

For the bias tables, **first write the obvious version**: `tl.atomic_add(dpos_ptr + pos_b, ds, mask=valid)` into one shared table of P floats. Run it and record the time. Then give every program its own private row:

```python
row = (pid_bh * tl.num_programs(0) + pid_n).to(tl.int64)
tl.atomic_add(dpos_ptr + row * P + pos_b, ds, mask=valid, sem="relaxed")
```

and sum the rows on the host side with `dpos.sum(0)`. Size the buffer for the smallest BN in the autotune list, and add `reset_to_zero=["dpos_ptr", "dts_ptr"]` to the autotuner. Otherwise each benchmarking trial adds into the same buffer.

**Kernel 2 (TODO 6):** `dq += tl.dot(ds.to(k.dtype), k)`.

**Autograd (TODO 7):** return one gradient per forward input, `None` for ts, offsets, max_len and alpha.
:::

::: run
```bash
python labs/07_hstu_attention_bwd/run.py
```
:::

::: measure
- naive_atomic_us | backward with one shared bias table (your first version) | 33285 | µs
- private_us | backward with private rows | 474 | µs
- fb200 | forward+backward speedup, B=128 N≤200 | 20.4 | ×
- fb512 | forward+backward speedup, B=64 N≤512 | 31.7 | ×
- fb512_mb | fused fwd+bwd peak memory, B=64 N≤512 | 48 | MB
- fb2048 | forward+backward speedup, N≤2048 | 6.5 | ×
:::

::: explain
**The atomics cliff: 33 ms vs 0.47 ms.** The position table has 256 entries and the time table 32. With one shared copy, every one of thousands of programs fires BM·BN atomic adds per tile into the same 288 addresses. Atomics on the same address serialise at the L2 slice that owns that cache line, so the whole GPU queues behind a few cache lines. Private rows remove all cross-program contention. Within a program, the compiler can combine lanes that hit the same address. The final `sum(0)` over a few thousand rows of 256 floats is a trivial reduction.

The general rule: **atomics are cheap when addresses are spread out and catastrophic when they are hot.** Make reductions hierarchical: private, then local, then global.

**Recompute vs store.** Storing A for backward would cost B·H·N² bf16 values, about 1 GB at B=64, N=512. Recomputing it costs one extra `QKᵀ` per tile, which the tensor cores do far faster than HBM could reload it. Fused peak memory for forward+backward is 48 MB vs 996 MB dense.

**Why the speedup is larger than the forward alone (20–32×).** The dense backward through PyTorch autograd also materialises dS, dA and the gathered bias gradient as N² tensors. The bias backward through advanced indexing originally took 1.26 s until the course reference switched to `bincount` (`TableGather` in `gk/hstu_ref.py`). The fused version adds two kernels and no new N² traffic.

**Why it shrinks at N=2048 (6.5×).** Each fused backward program now loops over long key or query ranges and is compute-bound, recomputing `QKᵀ` twice (once per kernel). The dense path, meanwhile, is doing large efficient GEMMs. Production kernels fuse dQ into kernel 1 with atomics on dQ, or use warp specialization, to claw this back.

**`reset_to_zero` matters.** The autotuner runs each config several times on the same buffers. Without zeroing, the accumulated bias gradients grow with every trial and your correctness check fails, but only the first time you run.
:::

::: cards
Q: Give the derivative of SiLU(s) in terms of σ = sigmoid(s).
A: SiLU'(s) = σ·(1 + s·(1 − σ)).
Q: Why does the HSTU backward use two kernels?
A: dK/dV reduce over queries and dQ reduces over keys. One kernel per reduction direction gives each output block a single owner, so no atomics on Q/K/V gradients (the FlashAttention-2 split).
Q: Why were atomic adds into one shared bias table about 70× slower than private rows?
A: Thousands of programs hammered the same 288 addresses; same-address atomics serialise at their L2 slice. Private rows remove contention, then one cheap sum reduces them.
Q: What is "recompute instead of store" in attention backward?
A: Do not save the N×N attention matrix; rebuild each score tile from Q, K and the bias inside the backward kernel. Costs FLOPs, saves O(N²) memory and traffic.
Q: Why add `reset_to_zero` to the autotuner of a kernel that accumulates with atomics?
A: Autotuning runs the kernel many times on the same output buffer; without zeroing between trials, the accumulated values are wrong.
Q: Rule of thumb for atomics on GPUs?
A: Cheap when addresses are spread out, catastrophic when they are hot. Reduce hierarchically: registers → private buffers → one final reduction.
:::
