"""Reference solution for Lab 05: row sums in CUDA, three ways, compiled with NVRTC."""
import torch
import triton
import triton.language as tl

from gk.bench import ensure_cuda_home

CUDA_SRC = r"""
// v1: one thread per row. Each thread walks its row serially, so neighbouring
// threads read addresses n floats apart: every load is a separate transaction.
extern "C" __global__ void rowsum_thread(const float* x, float* out, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float s = 0.f;
    for (int j = 0; j < n; ++j) s += x[(long)row * n + j];
    out[row] = s;
}

// v2: one block per row, coalesced strided loads, tree reduction in shared memory.
extern "C" __global__ void rowsum_smem(const float* x, float* out, int m, int n) {
    extern __shared__ float buf[];
    const float* r = x + (long)blockIdx.x * n;
    float s = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) s += r[j];
    buf[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = buf[0];
}

// v3: one block per row, float4 loads, warp shuffles inside each warp and
// shared memory only for the 32 per-warp partials.
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
extern "C" __global__ void rowsum_shfl(const float* x, float* out, int m, int n) {
    __shared__ float partial[32];
    const float4* r = reinterpret_cast<const float4*>(x + (long)blockIdx.x * n);
    float s = 0.f;
    for (int j = threadIdx.x; j < n / 4; j += blockDim.x) {
        float4 v = r[j];
        s += (v.x + v.y) + (v.z + v.w);
    }
    s = warp_sum(s);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = lane < (blockDim.x >> 5) ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) out[blockIdx.x] = s;
    }
}
"""
_k = {}


def _kernel(name):
    if name not in _k:
        ensure_cuda_home()
        _k[name] = torch.cuda._compile_kernel(CUDA_SRC, name)
    return _k[name]


def cuda_rowsum_thread(x, threads=256):
    m, n = x.shape
    out = torch.empty(m, device=x.device, dtype=torch.float32)
    _kernel("rowsum_thread")(grid=((m + threads - 1) // threads, 1, 1), block=(threads, 1, 1), args=[x, out, m, n])
    return out


def cuda_rowsum_smem(x, threads=256):
    m, n = x.shape
    out = torch.empty(m, device=x.device, dtype=torch.float32)
    _kernel("rowsum_smem")(grid=(m, 1, 1), block=(threads, 1, 1), args=[x, out, m, n], shared_mem=threads * 4)
    return out


def cuda_rowsum_shfl(x, threads=256):
    m, n = x.shape
    assert n % 4 == 0
    out = torch.empty(m, device=x.device, dtype=torch.float32)
    _kernel("rowsum_shfl")(grid=(m, 1, 1), block=(threads, 1, 1), args=[x, out, m, n])
    return out


@triton.jit
def rowsum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + row * n + cols, mask=cols < n, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def triton_rowsum(x, BLOCK=1024):
    m, n = x.shape
    out = torch.empty(m, device=x.device, dtype=torch.float32)
    rowsum_kernel[(m,)](x, out, n, BLOCK=BLOCK, num_warps=4)
    return out
