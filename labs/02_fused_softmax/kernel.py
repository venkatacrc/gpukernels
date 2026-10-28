"""Lab 02: fuse max, subtract, exp, sum and divide into one pass over each row.

Run:  python labs/02_fused_softmax/run.py      (your code)
      python labs/02_fused_softmax/run.py --solution
"""
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
    # TODO 1: load one row. Masked-off lanes must not win the max, so use
    #         other=-float("inf"). Upcast to fp32 with .to(tl.float32).
    # TODO 2: subtract the row max (tl.max(x, axis=0)) for numerical safety.
    # TODO 3: num = tl.exp(x); out = num / tl.sum(num, axis=0)
    # TODO 4: store, casting back with .to(out_ptr.dtype.element_ty).
    pass


def triton_softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # TODO 5: one program per row. BLOCK must be a power of two >= n_cols
    #         (triton.next_power_of_2). Pick num_warps from BLOCK: 4 up to 2048,
    #         8 up to 8192, 16 beyond.
    raise NotImplementedError("launch softmax_kernel")
    return out
