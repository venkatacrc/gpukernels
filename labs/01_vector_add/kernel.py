"""Lab 01: write vector add three times (Triton, Gluon, CUDA).

Fill in every TODO, then run:
    python labs/01_vector_add/run.py
Compare with the reference:
    python labs/01_vector_add/run.py --solution
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gk.bench import ensure_cuda_home


# ---------------------------------------------------------------- Triton
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # TODO 1: which block of BLOCK elements does this program own?
    #   pid  = tl.program_id(0)
    #   offs = pid * BLOCK + tl.arange(0, BLOCK)
    # TODO 2: the last block runs past n. Build mask = offs < n.
    # TODO 3: tl.load x and y with the mask, tl.store x + y into out_ptr.
    pass


def triton_add(x, y, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    # TODO 4: launch one program per BLOCK elements: grid = (triton.cdiv(n, BLOCK),)
    raise NotImplementedError("launch add_kernel")
    return out


# ---------------------------------------------------------------- Gluon
@gluon.jit
def gluon_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: gl.constexpr, layout: gl.constexpr):
    # TODO 5: same body as the Triton kernel, but gl.arange needs the layout:
    #   offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    pass


def gluon_add(x, y, BLOCK=1024, vec=4, num_warps=4):
    out = torch.empty_like(x)
    n = x.numel()
    # TODO 6: describe how BLOCK elements map onto threads. Each thread owns
    #   `vec` contiguous floats, 32 threads per warp, num_warps warps:
    #   layout = gl.BlockedLayout(size_per_thread=[vec], threads_per_warp=[32],
    #                             warps_per_cta=[num_warps], order=[0])
    # Then launch: gluon_add_kernel[grid](x, y, out, n, BLOCK, layout, num_warps=num_warps)
    raise NotImplementedError("launch gluon_add_kernel")
    return out


# ---------------------------------------------------------------- CUDA (NVRTC)
CUDA_SRC = r"""
extern "C" __global__ void add_scalar(const float* x, const float* y, float* out, int n) {
    // TODO 7: global thread index i = blockIdx.x * blockDim.x + threadIdx.x;
    //         guard with if (i < n) and write out[i] = x[i] + y[i];
}

extern "C" __global__ void add_float4(const float4* x, const float4* y, float4* out, int n4) {
    // TODO 8: same, but each thread moves one float4 (16 bytes) per load.
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
    # TODO 9: launch add_float4 with enough blocks to cover n4 float4 elements.
    raise NotImplementedError("launch add_float4")
    return out
