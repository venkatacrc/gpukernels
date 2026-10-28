"""Reference solution for Lab 01: one memory-bound kernel in three languages."""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gk.bench import ensure_cuda_home


# ---------------------------------------------------------------- Triton
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- Gluon
@gluon.jit
def gluon_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    x = gl.load(x_ptr + offs, mask=mask)
    y = gl.load(y_ptr + offs, mask=mask)
    gl.store(out_ptr + offs, x + y, mask=mask)


def gluon_add(x, y, BLOCK=1024, vec=4, num_warps=4):
    out = torch.empty_like(x)
    n = x.numel()
    layout = gl.BlockedLayout(size_per_thread=[vec], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    gluon_add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK, layout, num_warps=num_warps)
    return out


# ---------------------------------------------------------------- CUDA (NVRTC)
CUDA_SRC = r"""
extern "C" __global__ void add_scalar(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

extern "C" __global__ void add_float4(const float4* x, const float4* y, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = x[i], b = y[i];
        out[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}
"""
_cuda = {}


def _cuda_kernel(name):
    if name not in _cuda:
        ensure_cuda_home()
        _cuda[name] = torch.cuda._compile_kernel(CUDA_SRC, name)
    return _cuda[name]


def cuda_add(x, y, threads=256):
    out = torch.empty_like(x)
    n = x.numel()
    _cuda_kernel("add_scalar")(grid=((n + threads - 1) // threads, 1, 1), block=(threads, 1, 1), args=[x, y, out, n])
    return out


def cuda_add_float4(x, y, threads=256):
    assert x.numel() % 4 == 0
    out = torch.empty_like(x)
    n4 = x.numel() // 4
    _cuda_kernel("add_float4")(grid=((n4 + threads - 1) // threads, 1, 1), block=(threads, 1, 1), args=[x, y, out, n4])
    return out
