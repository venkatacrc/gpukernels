---
id: 04
title: GEMM + SiLU, from tiles to TMA
lab: labs/04_matmul_silu
minutes: 90
track: kernels
summary: tiled GEMM, grouped ordering, epilogue fusion, then Blackwell TMA + persistent + warp specialization
---
Every HSTU layer starts with one GEMM that produces four projections at once, followed by SiLU: `UVQK = SiLU(LN(x) @ W)`. The HSTU shape is tall and skinny: M = tokens (tens of thousands), N = 4·d, K = d.

This lab has two steps:

1. A **pointer-based tiled GEMM**, the classic Triton matmul, with the SiLU fused into the epilogue.
2. The **Blackwell version**: TMA descriptors move tiles between HBM and shared memory, a persistent grid runs one program per SM, and warp specialization splits producer and consumer warps.

::: write
`labs/04_matmul_silu/kernel.py`.

**Step 1 (TODOs 1–5).** Each program computes one BM×BN tile of C:

```python
a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BK)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)          # tensor cores, fp32 accumulate
    a_ptrs += BK * sak; b_ptrs += BK * sbk
if ACT == "silu":
    acc = acc * tl.sigmoid(acc)      # epilogue fusion: free, the tile is in registers
```

**Grouped ordering (TODO 3).** Map `pid` so that GROUP consecutive programs walk down a column of M-tiles before moving to the next N-tile. They reuse the same B tiles while those are still in L2.

**Step 2 (TODOs 6–9).** B is passed transposed (N×K) so both operands are K-contiguous:

```python
a_d = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
b_d = tl.make_tensor_descriptor(bt_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
for tile in tl.range(tl.program_id(0), num_m * num_n, NUM_SMS, flatten=True, warp_specialize=True):
    ...
    for k in range(k_tiles):
        acc = tl.dot(a_d.load([pid_m * BM, k * BK]), b_d.load([pid_n * BN, k * BK]).T, acc)
    acc = acc / (1.0 + tl.exp2(-1.4426950408889634 * acc))     # SiLU via exp2
    c_d.store([pid_m * BM, pid_n * BN], acc.to(c_ptr.dtype.element_ty))
```

Descriptors need scratch memory: call `triton.set_allocator(...)` once. TMA requires 16-byte-aligned strides, so K and N must be multiples of 8 in bf16.

**Autotune key.** Key on `M_BUCKET = triton.next_power_of_2(M)`, not M. In training M is the token count of a jagged batch, which changes every step, and keying on raw M re-runs the autotuner every step. That bug made the first version of Lab 09 hang.
:::

::: run
```bash
python labs/04_matmul_silu/run.py
```
:::

::: measure
- cublas_tf | cuBLAS GEMM, 65536×1024×256 | 925 | TFLOP/s
- cublas_silu_us | cuBLAS GEMM, then F.silu | 95 | µs
- step1_tf | step 1 pointer GEMM | 453 | TFLOP/s
- step2_tf | step 2 TMA/persistent/WS GEMM | 763 | TFLOP/s
- step2_silu_us | step 2 GEMM+SiLU fused | 59 | µs
- big_cublas | cuBLAS at 8192³ | 1778 | TFLOP/s
- big_step2 | step 2 at 8192³ | 1567 | TFLOP/s
:::

::: explain
**The fused epilogue is the win.** For the HSTU shape, cuBLAS does the GEMM in 37 µs, but the separate SiLU reads and writes the 65536×1024 bf16 output (128 MiB each way) and the pair takes 95 µs. Our step-2 kernel does both in about 59 µs: **1.6× faster than cuBLAS + SiLU**, even though our plain GEMM (763 TFLOP/s) is slower than cuBLAS's (925). On a skinny GEMM the epilogue is a large fraction of total bytes, and fusing it beats a better GEMM.

**Why step 1 is slow (453 TFLOP/s).** With pointer loads, the same warps compute addresses, issue loads, wait, and run MMAs. Software pipelining (`num_stages`) helps, but every warp still spends registers and issue slots on address math and waiting.

**What step 2 changes.**

- **TMA** (Tensor Memory Accelerator): one thread hands a descriptor and a coordinate to a hardware copy engine, which moves the whole tile into shared memory asynchronously. No per-thread address math, bounds checks done in hardware.
- **Persistent grid**: launch exactly 152 programs, one per SM. Each loops over tiles, so the pipeline stays warm across tiles and there is no tail of half-empty waves.
- **Warp specialization** (`warp_specialize=True`): the compiler splits the program into producer warps that only issue TMA loads and consumer warps that only issue MMAs, connected by shared-memory barriers. Loads for tile k+1 overlap MMAs for tile k. An ablation on this machine: the same TMA kernel without WS reached 636 TFLOP/s; with WS, 790.
- **EPI=2**: store the output tile in two halves so fewer accumulator registers are live during the store.

Total: 1.67× over step 1.

**Large square GEMMs.** At 8192³, cuBLAS (1778 TFLOP/s) still leads our 1567. It uses Blackwell's 2-CTA tcgen05 MMAs and hand-tuned tile schedulers. Matching it is a research project; beating it on *fused* workloads is the practical goal, and you just did.

**The SiLU formula.** \(x \cdot \sigma(x) = x / (1 + e^{-x})\), and \(e^{-x} = 2^{-x \log_2 e}\). `exp2` maps to a single hardware instruction (MUFU.EX2), while `tl.sigmoid` may lower to a slower exp plus a full-precision divide.
:::

::: cards
Q: Why does GEMM+SiLU fused beat cuBLAS GEMM followed by F.silu on the HSTU shape even though cuBLAS's GEMM is faster?
A: The separate SiLU reads and writes the whole output (128 MiB each way at 65536×1024 bf16). On a skinny GEMM that traffic rivals the GEMM itself; fusing it into the epilogue removes it.
Q: What does TMA do?
A: The Tensor Memory Accelerator copies a whole tile between global and shared memory asynchronously, from a descriptor (base, shape, strides, block shape) and a coordinate. No per-thread address arithmetic, hardware bounds handling.
Q: What is a persistent kernel?
A: A grid of exactly one program per SM; each program loops over many output tiles. Removes wave-quantization tails and keeps pipelines warm across tiles.
Q: What does `warp_specialize=True` do in a Triton `tl.range` loop?
A: Splits warps into producers (issue TMA loads) and consumers (issue MMAs), communicating through shared-memory barriers, so loads for the next tile overlap math on the current one.
Q: Why must the GEMM autotune key be bucketed for jagged training?
A: The token count M changes every batch; keying on raw M re-runs autotuning (dozens of compile+bench runs) on every step. Key on next_power_of_2(M).
Q: What does grouped tile ordering buy?
A: L2 reuse: GROUP consecutive programs share the same B column-tiles (and nearby A row-tiles) while those are still cached, cutting HBM reads.
Q: Why SiLU as `x / (1 + exp2(-x·log2 e))`?
A: exp2 is a single MUFU instruction; it avoids a slower exp and a separate sigmoid then multiply.
:::
