"""Reference solution for Lab 02: row softmax as one fused kernel."""
import torch
import triton
import triton.language as tl


def naive_softmax(x):
    """Five separate kernels; every intermediate goes back to HBM."""
    row_max = x.max(dim=1, keepdim=True).values
    z = x - row_max
    num = torch.exp(z)
    den = num.sum(dim=1, keepdim=True)
    return num / den


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_x, stride_out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def triton_softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    softmax_kernel[(n_rows,)](x, out, x.stride(0), out.stride(0), n_cols, BLOCK=BLOCK, num_warps=num_warps)
    return out
