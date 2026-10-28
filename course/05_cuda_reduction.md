---
id: 05
title: Reductions in raw CUDA
lab: labs/05_cuda_reduction
minutes: 60
track: kernels
summary: coalescing, shared-memory trees, warp shuffles, float4; compiled at runtime with NVRTC
---
Triton hides two things you must understand to read any kernel profile: **coalescing** (how the 32 threads of a warp turn their addresses into memory transactions) and **intra-block communication** (shared memory and warp shuffles). This lab sums each row of an M×N fp32 matrix three ways in CUDA, then once more in Triton.

The CUDA source is a Python string compiled at runtime with NVRTC:

```python
k = torch.cuda._compile_kernel(CUDA_SRC, "rowsum_shfl")
k(grid=(m, 1, 1), block=(256, 1, 1), args=[x, out, m, n])
```

::: write
`labs/05_cuda_reduction/kernel.py`.

**v1, one thread per row** (given). Each thread loops over its row.

**v2, one block per row with a shared-memory tree (TODOs 1–4):**

```cpp
float s = 0.f;
for (int j = threadIdx.x; j < n; j += blockDim.x) s += r[j];     // coalesced: lane i reads r[base + i]
buf[threadIdx.x] = s; __syncthreads();
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
    __syncthreads();
}
if (threadIdx.x == 0) out[blockIdx.x] = buf[0];
```

**v3, float4 loads and warp shuffles (TODOs 5–7):**

```cpp
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
```

Each thread accumulates float4s, reduces within its warp by shuffles (no shared memory, no barriers), and lane 0 of each warp writes one partial. Warp 0 then reduces the partials.

**Triton (TODO 8):** loop over the row in BLOCK chunks into a `(BLOCK,)` fp32 accumulator, then `tl.sum` once at the end.
:::

::: run
```bash
python labs/05_cuda_reduction/run.py
```
:::

::: measure
- v1 | v1 thread-per-row, 16384×8192 | 522 | GB/s
- v2 | v2 shared-memory tree | 2362 | GB/s
- v3 | v3 float4 + shuffles | 5823 | GB/s
- triton | Triton rowsum | 5533 | GB/s
- torch | torch.sum(dim=1) | 4480 | GB/s
- v1_wide | v1 at 2048×65536 | 30 | GB/s
:::

::: explain
**v1 (522 GB/s, and 30 GB/s on wide rows): uncoalesced.** At step j, thread t reads `x[t·n + j]`. The 32 lanes of a warp touch 32 different rows, n·4 bytes apart, so each lane needs its own 32-byte sector: 32 transactions to deliver 128 useful bytes. Only 1/8 of every sector is used. On wide, short matrices (2048 rows) it gets worse: only 2048 threads exist, 64 warps for 152 SMs, so most SMs sit idle: almost no parallelism *and* no coalescing. 30 GB/s is 0.5% of the hardware.

**v2 (2362 GB/s): coalesced but under-fed.** Lane i reads `r[j + i]`, so a warp's 32 loads hit 128 contiguous bytes, perfectly coalesced. But each thread keeps only one 4-byte load in flight per loop iteration (lesson 01's Little's law again), and the tree spends log₂(256) = 8 barrier-separated steps, in which most threads sit idle.

**v3 (5823 GB/s): enough bytes in flight, cheap reduction.** float4 quadruples bytes per load. `__shfl_xor_sync` exchanges registers between lanes of a warp directly: 5 steps, no shared memory, no `__syncthreads`. Shared memory carries only 8 partials (one per warp). This is how production reductions (CUB, PyTorch) are written.

**Triton (5533 GB/s) gets close with four lines.** The `(BLOCK,)` accumulator means each thread holds several independent partial sums, which gives instruction-level parallelism and wide loads, and `tl.sum` lowers to the same shuffle tree. Torch's generic reduction (4480) handles every dtype, layout and dimension, and pays for that generality.

**Takeaway.** When a profile shows low bandwidth, check in this order: are warps coalesced? Are there enough bytes in flight per thread? Is the reduction serialised behind barriers?
:::

::: cards
Q: What does it mean for a warp's loads to be coalesced?
A: The 32 lanes access contiguous (or at least the same few 32-byte sectors of) memory, so the hardware serves the warp with a handful of transactions instead of 32.
Q: Why is thread-per-row summation slow on a row-major matrix?
A: At each step neighbouring threads read addresses a full row apart: every lane needs its own sector, wasting 7/8 of each 32-byte transaction.
Q: What does `__shfl_xor_sync(0xffffffff, v, off)` do?
A: Each lane receives v from lane (laneId XOR off) in the same warp, register to register, no shared memory. Five rounds (16, 8, 4, 2, 1) give a full warp sum in every lane.
Q: Why does a shared-memory tree reduction leave performance on the table?
A: log₂(blockDim) steps each separated by __syncthreads, with half the remaining threads idle at each step; plus a shared-memory round trip per step.
Q: How does `torch.cuda._compile_kernel` let you write CUDA without nvcc?
A: It compiles a CUDA C++ source string at runtime with NVRTC and returns a launchable kernel: k(grid=..., block=..., args=[...], shared_mem=...).
Q: Why does the Triton rowsum keep a (BLOCK,) accumulator instead of summing each chunk to a scalar?
A: Independent per-lane partial sums keep many adds and loads in flight (ILP) and defer the expensive cross-lane reduction to a single tl.sum at the end.
:::
