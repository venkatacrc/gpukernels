---
id: 01
title: Vector add three ways
lab: labs/01_vector_add
minutes: 60
track: kernels
summary: the same kernel in Triton, Gluon and CUDA; masks, layouts, vector loads, Little's law
---
The simplest kernel there is: `out = x + y`. You will write it three times, at three levels of control:

- **Triton**: you describe what one *program* does with a whole block of elements; the compiler decides how threads split it.
- **Gluon**: Triton's lower-level dialect. Same language, but *you* choose the layout: which thread owns which elements.
- **CUDA**: you write what one *thread* does. Everything else is your job.

Vector add is purely memory-bound (12 bytes moved per FLOP), so the only score is GB/s, against the copy bandwidth you measured in lesson 00.

::: write
Open `labs/01_vector_add/kernel.py`.

**Triton (TODOs 1–4).** Each program owns `BLOCK` consecutive elements:

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n                       # the last block runs past the end
x = tl.load(x_ptr + offs, mask=mask)
y = tl.load(y_ptr + offs, mask=mask)
tl.store(out_ptr + offs, x + y, mask=mask)
```

Launch with `grid = (triton.cdiv(n, BLOCK),)`.

**Gluon (TODOs 5–6).** The body is identical except `gl.arange` needs an explicit layout. Build it in the wrapper:

```python
layout = gl.BlockedLayout(size_per_thread=[vec], threads_per_warp=[32],
                          warps_per_cta=[num_warps], order=[0])
```

`size_per_thread=[4]` means each thread owns 4 *consecutive* fp32 values, which the compiler can fetch with one 16-byte load. If `BLOCK` is larger than `vec · 32 · num_warps`, the layout repeats and each thread owns several such chunks.

**CUDA (TODOs 7–9).** A scalar kernel (one float per thread) and a `float4` kernel (16 bytes per thread):

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n4) { float4 a = x[i], b = y[i]; out[i] = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
```
:::

::: run
```bash
python labs/01_vector_add/run.py           # your kernel.py
python labs/01_vector_add/run.py --solution
```
The table sweeps sizes from 2¹⁶ to 2²⁸ elements and a Gluon `vec` sweep.
:::

::: measure
- torch_big | torch add, 2²⁸ elements | 7142 | GB/s
- triton_big | Triton add, 2²⁸ | 7149 | GB/s
- gluon_big | Gluon add, 2²⁸ | 7142 | GB/s
- cuda_scalar | CUDA scalar, 2²⁸ | 4075 | GB/s
- cuda_f4 | CUDA float4, 2²⁸ | 7215 | GB/s
- small_us | any kernel at 2¹⁶ elements | 6.6 | µs
- gluon_vec1 | Gluon vec=1, 2²⁸ | 7208 | GB/s
:::

::: explain
**Everyone hits the same wall except scalar CUDA.** Torch, Triton, Gluon and float4 CUDA all land near 7.1–7.2 TB/s, above the 6.6 TB/s copy because a copy moves 2 bytes per byte read while add moves 3, and more concurrent streams keep HBM busier. Scalar CUDA tops out near 4.1 TB/s.

**Little's law explains why.** To sustain bandwidth \(B\) against memory latency \(L\), you need \(B \times L\) bytes in flight. At roughly 7 TB/s and roughly 1 µs that is several MB across the chip, tens of KB per SM. A scalar CUDA thread issues one 4-byte load per input and then waits. Even with the SM full of threads, there aren't enough bytes outstanding. The float4 version puts 16 bytes per load in flight, four times more per instruction.

**Then why is Gluon vec=1 still fast?** Because the layout *repeats*. With `BLOCK=4096`, 4 warps and vec=1, each thread owns 4096 / 128 = 32 elements and issues 32 independent loads before waiting. Width per load matters less than *total* bytes in flight per thread. The vec sweep (7208 / 7174 / 7133 / 7134 GB/s for vec 1/2/4/8) shows it: once enough loads are outstanding, the width barely matters. Triton's default layout makes this choice for you; Gluon lets you see and change it.

**Small sizes measure launch, not bandwidth.** At 2¹⁶ elements (768 KB) every implementation takes about 6.6 µs, the launch-and-sync floor from lesson 00. The data sits in L2 anyway. Never quote GB/s for a kernel that runs in under about 10 µs.

**The mask.** Without `mask=offs < n` the last program reads and writes past the end of the buffer. On a GPU that is silent memory corruption, or an illegal-address fault if you are lucky.
:::

::: cards
Q: In Triton, what does `tl.arange(0, BLOCK)` inside a program represent?
A: The vector of BLOCK lane offsets that this one program processes. The compiler maps those lanes onto the program's threads; you never index threads directly.
Q: What does Gluon's `BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0])` mean?
A: Each thread owns 4 consecutive elements, 32 threads per warp, 4 warps per program: one layout tile covers 4·32·4 = 512 elements, repeated to fill BLOCK.
Q: State Little's law for memory bandwidth.
A: Bytes in flight = bandwidth × latency. To reach peak HBM bandwidth you must keep about (TB/s × µs) bytes outstanding, which means many wide or independent loads per thread.
Q: Why does scalar CUDA add reach only about 4.1 TB/s while float4 reaches 7.2 TB/s?
A: Each scalar thread keeps only 4 bytes per stream in flight. float4 issues 16-byte loads, four times more bytes outstanding per instruction, enough to cover HBM latency.
Q: Why is GB/s meaningless for a 2¹⁶-element add?
A: The kernel finishes in about 6.6 µs, dominated by launch and synchronization overhead, and the data is L2-resident; you are measuring the launch floor.
Q: What happens if you drop the mask on the last Triton block?
A: Out-of-bounds loads and stores: silent memory corruption or an illegal memory access error.
:::
