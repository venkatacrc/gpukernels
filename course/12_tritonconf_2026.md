---
id: 12
title: Field notes from Triton Developer Conference 2026
minutes: 45
track: frontier
summary: twelve posters mapped onto what you built: autotuning cost, compilers under Triton, Gluon, multi-GPU, AOT serving
---
The conference posters cluster around four problems you have already hit in this course. Each section links a poster to the lab where you met the problem.

## 1. Autotuning is too slow (Labs 04, 09, 11 D)

You saw it three times: an autotune key that re-tuned every step (Lab 09), 17 s of cold start (Lab 11), and a 1.5 s cold p99 in serving (Lab 10). Five posters attack it:

| Poster | Idea | Reported result |
|---|---|---|
| **Helion warm-start + shared caches** (Red Hat) | Helion's exact cache key hashes the kernel *source*, so any edit discards tuning. A *loose* key (dtypes, shapes, strides, hardware, plus a structural hash of the config space) seeds the search with cached winners (`FROM_BEST_AVAILABLE`), the default config stays pinned, and a `RemoteCache` ABC (get / put / list over Redis, S3 or HTTP) shares winners across developers and CI. | 1238 s → 92 s (13×) autotuning with the same winning quality. Enable with `HELION_AUTOTUNE_CACHE=…` |
| **TritonPPM** (Meta) | Static analysis of the Triton kernel computes symbolic work per block (bytes, FLOPs, registers, spills, pipelining); about 15 roofline-style features feed an XGBoost runtime model trained on about 250K B200 launches. Rank configs, benchmark only the top k. | < 0.5 s per kernel; at k=10, < 6% regret at p90; on unseen kernels, 10% better geomean by searching 30× larger spaces with 2.5× less resource |
| **Diode** (Meta, PT2 Inductor) | A residual MLP on the CPU predicts log-runtime of each GEMM config from shape, dtype, strides and GPU; Inductor compiles and benchmarks only the top-K against cuBLAS. | Compile+autotune time cut 14–57% on production workloads with QPS and NE preserved; per-shape tuning time 80–87% lower at K=1 |
| **CompileIQ** (NVIDIA) | Autotuners search tiles, warps and stages; CompileIQ searches *PTXAS's* internal controls (register allocation, scheduling, unrolling) with an evolutionary search and ships the result as an "advanced control file". | 5% geomean over 12 TritonBench kernels on B200 vs default PTXAS. In Triton: `ptx_options="--apply-controls=best.acf"`; `pip install compileiq` |
| **AOT-Tensor** (Meta) | Record each kernel's specialisations and autotune picks during one forward pass, compile them ahead of time into one `.so` per kernel with generated dispatch guards, and rewrite call sites into torch ops so a C++-only server can run them. | In production for large ranking models, 70+ kernels, parity with JIT Triton. Open source in `meta-pytorch/torchinfer` |

The common thread: **measuring is the expensive part, so predict, cache, or move it offline.** Your `TRITON_CACHE_AUTOTUNING=1` result in Lab 11 is the simplest member of this family.

## 2. Lower-level control: Gluon and correct-by-construction pipelines (Labs 01, 04)

- **TokenSpeed-Kernel** (AMD / LightSeek) writes LLM attention and MoE kernels in Gluon for AMD CDNA4: explicit `async_copy` double buffering, sub-tiled loads to cut register pressure, `convert_layout(..., assert_trivial=True)` to *prove* a layout conversion moves no data, serpentine CTA ordering to balance causal attention, and warp-pipeline stages. That is the same layout control you used in Lab 01's `BlockedLayout`, at production scale.
- **A provably-correct pipeline DSL** lowered to Gluon: a human writes the algorithm, an LLM or human supplies the *schedule* (stages), and the compiler injects waits, sizes buffers, and peels prologue and epilogue. Races are ruled out at compile time. That is the hand-written version of what `warp_specialize=True` did for you in Lab 04.
- **Beyond powers of two** (Meta): mixed-radix `LinearLayout`s let tiles like N=192 or K=96 compile exactly instead of padding to 256. Example: an N=192 WGMMA tile fits a 3-stage pipeline in 192 KB of shared memory, where padded 256 needs 256 KB (over the 227 KB limit) and drops to 2 stages. Exact tiles gave 1.2–1.3× for head-dim-192 attention on H100.

## 3. Copy and launch overhead in real serving (Labs 01, 10, 11)

- **vLLM hybrid-model state** (IBM): SSM state copies for prefix caching plus speculative decoding put the CPU on the decode critical path. Fixes, merged in vLLM PRs #40172, #48110 and #49436: a fused Triton kernel that does the copy logic *and* the copies on the GPU (about 30% decode throughput), 8-byte vectorised copies instead of byte copies (Lab 01's lesson exactly), and a 3-D-grid tiled copy that splits one state across SMs (over 80% of H200 bandwidth at just 4 requests).

## 4. Beyond one GPU (forward-looking)

- **Iris on AMD Helios** (AMD): symmetric heaps make peer GPU memory addressable from Triton (`iris.store(remote_ptr, …)`). MoE dispatch becomes "compute where expert 47's buffer lives and store there". Push beats pull by 3.4–4.3× on Helios.
- **Triton IR → Chakra → ASTRA-sim** (Marvell): convert compiled Triton IR into a portable execution trace and simulate multi-GPU timing for hardware that doesn't exist yet.
- **FlagOS KernelGen**: an execution-driven agent loop (generate → compile → validate → benchmark → profile → record evidence) that produces Triton kernels for 20+ chip vendors.

::: write
Pick one exercise and apply a conference idea to this course's code:

1. **Loose-key autotune cache (Helion/Diode).** Record the winning config of `matmul_tma_kernel` for each `(M_BUCKET, N, K)` in a JSON file; on start-up, pass only the cached winner plus the default as `configs`. Measure first-step time against Lab 11 D.
2. **Predict, then benchmark (TritonPPM).** For the six `TMA_CONFIGS`, compute a roofline estimate (bytes per tile ÷ bandwidth vs FLOPs ÷ peak) and check whether it ranks the configs the way the autotuner did.
3. **AOT for serving (AOT-T).** List every Triton kernel and autotune key the serving model hits in `_warmup`, and extend the warm-up so the Lab 10 cold p99 disappears.
:::

::: note
The numbers in this lesson are the posters' claims, as presented. They were not reproduced on the course GPU. Treat them like any CLAIM line: measure before you rely on them.
:::

::: cards
Q: Why does an exact-match autotune cache miss after a trivial source edit in Helion, and what is the fix?
A: The key includes a hash of the kernel source. The loose key uses specialisation (dtypes, shapes, strides), hardware and a structural config-space hash, then warm-starts the search from cached winners (FROM_BEST_AVAILABLE): 1238 s → 92 s.
Q: What does CompileIQ tune that Triton's autotuner does not?
A: PTXAS's internal controls (register allocation, instruction scheduling, unrolling), searched evolutionarily and applied with ptx_options="--apply-controls=best.acf": 5% geomean on B200.
Q: How do TritonPPM and Diode cut autotuning cost?
A: Both predict runtime per config without running it (TritonPPM: static-analysis features + XGBoost; Diode: an MLP on shape/config features) and benchmark only the top-k.
Q: What problem does AOT-Tensor solve?
A: JIT Triton needs Python and compile time at serving; AOT-T records specialisations and autotune picks in one forward pass, compiles cubins ahead of time into one .so per kernel with guards, for C++ serving.
Q: Why do non-power-of-two tiles (e.g. N=192) help on Hopper and Blackwell?
A: Model shapes like head_dim 192 waste MMA work and shared memory when padded to 256; exact tiles let a 3-stage pipeline fit in 192 KB, where padded 256 needs 256 KB (over the limit).
Q: Which course lesson does vLLM's 8-byte-copy fix mirror?
A: Lab 01: wider loads per thread put more bytes in flight (float4 vs scalar), turning a latency-bound copy into a bandwidth-bound one.
Q: What does `convert_layout(..., assert_trivial=True)` buy a Gluon programmer?
A: A compile-time guarantee that the layout conversion moves no data (no shared-memory round trip); it errors otherwise.
:::
