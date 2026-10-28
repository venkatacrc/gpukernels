"""Lab 05: row sums in raw CUDA (compiled at runtime with NVRTC), then in Triton.

Run:  python labs/05_cuda_reduction/run.py      (your code)
      python labs/05_cuda_reduction/run.py --solution
"""
import torch
import triton
import triton.language as tl

from gk.bench import ensure_cuda_home

CUDA_SRC = r"""
// v1 (given): one thread per row. Run it first and note how slow it is.
extern "C" __global__ void rowsum_thread(const float* x, float* out, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float s = 0.f;
    for (int j = 0; j < n; ++j) s += x[(long)row * n + j];
    out[row] = s;
}

// v2: one block per row.
extern "C" __global__ void rowsum_smem(const float* x, float* out, int m, int n) {
    extern __shared__ float buf[];
    const float* r = x + (long)blockIdx.x * n;
    // TODO 1: each thread sums r[threadIdx.x], r[threadIdx.x + blockDim.x], ...
    //         so a warp reads 32 consecutive floats per step (coalesced).
    // TODO 2: write the partial to buf[threadIdx.x]; __syncthreads();
    // TODO 3: tree reduction: for stride = blockDim.x/2 down to 1, threads below
    //         stride add buf[t + stride] into buf[t]; __syncthreads() each step.
    // TODO 4: thread 0 writes buf[0] to out[blockIdx.x].
}

// v3: warp shuffles instead of shared memory for most of the reduction.
__device__ __forceinline__ float warp_sum(float v) {
    // TODO 5: butterfly: for off = 16, 8, 4, 2, 1:  v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
extern "C" __global__ void rowsum_shfl(const float* x, float* out, int m, int n) {
    __shared__ float partial[32];
    const float4* r = reinterpret_cast<const float4*>(x + (long)blockIdx.x * n);
    float s = 0.f;
    // TODO 6: loop j = threadIdx.x .. n/4 step blockDim.x, load float4 v = r[j],
    //         s += (v.x + v.y) + (v.z + v.w);
    // TODO 7: s = warp_sum(s); lane 0 of each warp writes partial[warp];
    //         __syncthreads(); warp 0 reduces the partials and lane 0 writes out.
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
    # TODO 8: the same reduction in Triton. Keep a (BLOCK,) fp32 accumulator,
    #         loop over the row in BLOCK-wide masked chunks, then tl.sum once.
    #         Triton chooses the shuffles and shared memory for you.
    pass


def triton_rowsum(x, BLOCK=1024):
    m, n = x.shape
    out = torch.empty(m, device=x.device, dtype=torch.float32)
    rowsum_kernel[(m,)](x, out, n, BLOCK=BLOCK, num_warps=4)
    return out
