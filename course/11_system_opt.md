---
id: 11
title: System optimisation
lab: labs/11_system_opt
minutes: 90
track: systems
summary: profile the whole step, find the launch-bound regime, batch scaling, CUDA graphs for jagged shapes, compile caches
---
Fast kernels are necessary, not sufficient. Once the big kernels are fused, the step is made of everything *between* them: the CPU issuing launches, small PyTorch ops, the optimizer, the loss, and start-up compilation. This lab measures all of it in four parts:

- **A. GPU-bound or CPU-bound?** Profile 20 training steps; GPU busy time ÷ wall time; kernels per step; the kernels that dominate.
- **B. Batch scaling.** Step time and tokens/s from 32 to 2048 users.
- **C. CUDA graphs for jagged inference.** Record the whole top-k forward once and replay it with one launch.
- **D. Compile caches.** Time to the first training step in a fresh process, cold vs warm `TRITON_CACHE_DIR`, with and without `TRITON_CACHE_AUTOTUNING=1`.

::: write
Read `labs/11_system_opt/cuda_graphs.py` (`GraphedTopK`). CUDA graphs need **static shapes and static addresses**, but jagged batches change every call. The trick:

- Capture one graph per **bucket** (B users, T tokens): (1, 256), (8, 2048), (64, 16384).
- Copy a real batch into the bucket's static buffers; unused user slots get length 0 (`offsets` repeat).
- One **dummy user** at index B owns the unused token tail, so the jagged kernels see a consistent layout. Its outputs are thrown away.
- Run once *before* capture so Triton compiles and autotunes outside the graph (compilation cannot be captured).

Exercise after the run: wire `GraphedTopK` into `serving/hstu_backend.py` and rerun the Lab 10 load test.
:::

::: run
```bash
python labs/11_system_opt/run.py
```
Part D spawns fresh Python processes with empty and then warm cache directories, so it takes a minute.
:::

::: measure
- busy | GPU busy fraction, triton step, B=128 | 22 | %
- kernels | kernels launched per triton step | 278 |
- step_ms | triton step time, B=128 | 10.5 | ms
- tok32 | tokens/s at 32 users | 0.40 | Mtok/s
- tok2048 | tokens/s at 2048 users | 7.65 | Mtok/s
- eager_us | eager top-10, 1 user | 2617 | µs
- graph_us | CUDA graph top-10, 1 user | 174 | µs
- cold_s | first step, cold cache | 17.0 | s
- warm_s | first step, warm kernel + autotune cache | 2.7 | s
:::

::: explain
**A. The GPU is idle almost 80% of the step.** 278 kernels in about 10 ms, while summed GPU kernel time is about 2.3 ms. The rest is the CPU: Python, autograd bookkeeping, and about 5–10 µs to issue each launch. This is the **launch-bound** regime, and at B=128 on ML-1M the model lives in it. (Timings under the profiler, and on a shared GPU, wobble by ±30% between runs; the busy fraction is the stable signal.)

The top GPU kernels are now *not* ours: the softmax backward and SIMT `sgemm` kernels of the **fp32 loss**, then AdamW. The `simt_sgemm` name is the tell: these are fp32 GEMMs on CUDA cores, not tensor cores, because the logits are computed in fp32. Next steps, in order of payoff: compute the logits GEMM in bf16 (or allow TF32), fuse the loss into a cross-entropy kernel that never writes the (T × 3706) logits, and keep using the fused AdamW (`fused=True`).

**B. Batch scaling is almost free throughput.** From 32 to 512 users, tokens per step grow 16× while step time barely moves (about 7.3–7.5 ms), so tokens/s grow about 16×. Only at 2048 users (204K tokens) does the GPU finally become the bottleneck: 26.7 ms per step, 7.65 M tokens/s, 19× the 32-user rate. For recommendation training, the batch size is a systems knob as much as an optimisation one. Larger batches need learning-rate retuning, but the hardware wants them.

**C. CUDA graphs: about 15× lower latency for one user.** Eager top-10 for one user takes about 2.6 ms, nearly all CPU launch overhead across 4 layers. Replaying the graph issues one launch for the whole forward: 174 µs. At 64 users the gain shrinks (7.5×), because more of the time is real GPU work. Results are identical (`same top-10 = True`). The padding-to-buckets cost is invisible next to the launch savings. This is why production inference servers capture graphs per shape bucket.

**D. Start-up: 17 s → 2.7 s.** A cold process compiles every Triton kernel and autotunes each config set: about 17 s before the first step. A warm `TRITON_CACHE_DIR` skips compilation but still benchmarks configs (6.8 s). Adding `TRITON_CACHE_AUTOTUNING=1` caches the autotune decision too (2.7 s). In serving, the same cache is the difference between a 1.5 s cold p99 and a 10 ms p99 (Lab 10). Lesson 12 shows the conference work that pushes this further: shared remote autotune caches and learned config predictors.
:::

::: cards
Q: How do you tell whether a training step is launch-bound?
A: Profile it: if summed GPU kernel time is far below wall time (low GPU-busy fraction) and the step has hundreds of small kernels, the CPU's launch rate is the limit.
Q: Name three cures for a launch-bound model.
A: Bigger batches (more work per launch), CUDA graphs (one launch per captured sequence), and fusing small ops into fewer kernels (or torch.compile).
Q: How do you use CUDA graphs with jagged, variable-shape inputs?
A: Capture one graph per (users, tokens) bucket with static buffers; copy each batch into the smallest bucket that fits, give unused users length 0, and let a dummy user absorb the unused token tail.
Q: Why must you run a Triton model once before capturing a CUDA graph?
A: JIT compilation and autotuning (allocations, benchmarking, host syncs) cannot happen inside graph capture; they must be done before.
Q: What do TRITON_CACHE_DIR and TRITON_CACHE_AUTOTUNING=1 save?
A: The cache dir stores compiled kernels (skip compilation); TRITON_CACHE_AUTOTUNING also stores the chosen autotune config (skip re-benchmarking): 17 s → 2.7 s to first step here.
Q: A profile shows `cutlass…simt_sgemm_f32…` near the top. What does that tell you?
A: An fp32 GEMM is running on CUDA cores rather than tensor cores; switch it to bf16/TF32 or fuse it away (here: the loss logits).
:::
