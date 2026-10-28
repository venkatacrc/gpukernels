---
id: 00
title: Setup and the GPU mental model
minutes: 30
track: foundations
summary: check the toolchain, measure the two roofs every later lab is judged against
---
## How this course works

Every lab follows the same loop, and the app follows it too:

1. **Write**: open `labs/NN_name/kernel.py` and fill in the numbered `TODO`s. `solution.py` next to it is the reference; peek only when stuck.
2. **Run**: `python labs/NN_name/run.py` checks your kernel against PyTorch, then benchmarks it. `--solution` runs the reference instead.
3. **Measure**: type the numbers your GPU printed into the lesson. The app shows the reference from the course GB200 next to yours.
4. **Explain**: the explanation unlocks once you have measured. Predict first, then read why.

Every `run.py` ends with four lines: `CLAIM` (what the lesson asserts), `MEASURED`, `VERDICT` and `NEXT`. When a claim fails on your hardware, that is a finding, not a bug in your reading.

The path through the course:

| Part | Labs | You build |
|---|---|---|
| Kernels | 01–05 | vector add (Triton, Gluon, CUDA), fused softmax, LayerNorm·mul, GEMM+SiLU with TMA, CUDA reductions |
| HSTU | 06–08 | jagged attention forward, backward, then the full model |
| Train | 09 | HSTU on MovieLens-1M |
| Serve | 10 | Triton Inference Server with dynamic batching |
| Systems | 11 | profiling, batch scaling, CUDA graphs, compile caches |
| Field notes | 12 | what the 2026 Triton conference added |

## A GPU in five sentences

A GPU is a grid of **streaming multiprocessors** (SMs; 152 on GB200). Each SM runs groups of 32 threads called **warps** in lockstep, and keeps many warps resident so that while one waits on memory another computes. Data moves through a hierarchy: registers (per thread) → shared memory (per block, up to 227 KB) → L2 (shared, 129 MiB) → HBM. A kernel launch is a grid of **blocks** (Triton calls them **programs**); every block runs on one SM. Performance is almost always about keeping enough bytes in flight to cover HBM latency, or enough math in flight to keep the tensor cores busy.

::: write
Nothing to write yet. Create the environment once:

```bash
python -m venv venv && source venv/bin/activate
python -m pip install torch triton numpy
git clone https://github.com/venkatacrc/gpukernels && cd gpukernels
```

The CUDA labs compile with NVRTC through `torch.cuda._compile_kernel`, so no `nvcc` is needed; `gk/bench.py` finds the CUDA headers that ship inside the torch wheel.
:::

::: run
```bash
python tools/check_env.py
```
:::

::: measure
- copy_bw | HBM copy bandwidth | 6637 | GB/s
- mm_tf | bf16 matmul 8192³ | 1741 | TFLOP/s
- launch | smallest kernel as timed | 6.9 | µs
- sms | number of SMs | 152 | SMs
:::

::: explain
**Two roofs.** The copy reads 1 GiB and writes 1 GiB, so it reports achievable HBM bandwidth (GB200 HBM3e is rated near 8 TB/s; reaching about 85% with a plain copy is normal). The 8192³ matmul does \(2 \cdot 8192^3\) FLOPs and is bound by the tensor cores.

**Arithmetic intensity.** Divide the two: 1741 TFLOP/s ÷ 6.6 TB/s ≈ 262 FLOPs per byte. That is the *ridge point* of the roofline. A vector add does 1 FLOP per 12 bytes, far below the ridge, so labs 01–03 and 05 are judged in GB/s. A large GEMM does hundreds of FLOPs per byte, so Lab 04 is judged in TFLOP/s. Attention (labs 06–07) sits in between, and the whole game there is not writing the \(N^2\) matrix to memory.

**The launch floor.** Even a 1-element kernel costs several microseconds end to end. A model step that launches 300 small kernels pays that floor 300 times. Remember this number for Lab 11.
:::

::: cards
Q: What is the ridge point of a roofline, and how do you compute it?
A: Peak FLOP/s divided by peak bytes/s. Kernels below it (in FLOPs per byte moved) are memory-bound, above it compute-bound. On the course GB200: about 1741 TFLOP/s ÷ 6.6 TB/s ≈ 262 FLOP/byte.
Q: What does one Triton *program* correspond to in CUDA?
A: One thread block. It runs on a single SM; the threads inside it are managed by the compiler according to `num_warps`.
Q: Why does a GPU keep many warps resident on each SM?
A: To hide memory latency: while one warp waits hundreds of cycles for HBM, the scheduler issues instructions from another warp that is ready.
Q: List the GPU memory hierarchy from fastest to slowest.
A: Registers (per thread) → shared memory / L1 (per SM, up to 227 KB per block on GB200) → L2 (129 MiB, shared by all SMs) → HBM.
Q: Which four lines does every lab's run.py print at the end?
A: CLAIM, MEASURED, VERDICT, NEXT.
:::
