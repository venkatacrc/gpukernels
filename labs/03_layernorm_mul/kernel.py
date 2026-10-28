"""Lab 03: fuse LayerNorm and the HSTU gate,  y = LayerNorm(x) * u.

Run:  python labs/03_layernorm_mul/run.py      (your code)
      python labs/03_layernorm_mul/run.py --solution
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def torch_ln_mul(x, u, w, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps) * u


@triton.jit
def ln_mul_fwd_kernel(x_ptr, u_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr, stride, D, eps, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    # TODO 1: load the row of x as fp32 (other=0.0 for masked lanes).
    # TODO 2: mean = sum(x) / D. Centre: xc = tl.where(mask, x - mean, 0.0)
    #         (the where matters: padded lanes would otherwise add mean^2 to var).
    # TODO 3: var = sum(xc * xc) / D; rstd = 1 / sqrt(var + eps)
    # TODO 4: load w, b (length D) and the row of u. y = (xc * rstd * w + b) * u
    # TODO 5: store y (cast to y's dtype), and store mean and rstd for this row;
    #         backward needs them so it does not recompute the statistics.
    pass


def triton_ln_mul_fwd(x, u, w, b, eps=1e-5):
    shape = x.shape
    x2, u2 = x.reshape(-1, shape[-1]).contiguous(), u.reshape(-1, shape[-1]).contiguous()
    M, D = x2.shape
    y = torch.empty_like(x2)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    # TODO 6: one program per row, BLOCK_D = next power of two >= D,
    #         num_warps 4 up to 1024 columns, 8 beyond.
    raise NotImplementedError("launch ln_mul_fwd_kernel")
    return y.reshape(shape), mean, rstd


class LayerNormMul(torch.autograd.Function):
    """Forward in Triton. Backward reuses the saved mean/rstd and runs in PyTorch."""

    @staticmethod
    def forward(ctx, x, u, w, b, eps=1e-5):
        y, mean, rstd = triton_ln_mul_fwd(x, u, w, b, eps)
        ctx.save_for_backward(x, u, w, b, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, u, w, b, mean, rstd = ctx.saved_tensors
        D = x.shape[-1]
        x2 = x.reshape(-1, D).float()
        u2 = u.reshape(-1, D).float()
        dy2 = dy.reshape(-1, D).float()
        xhat = (x2 - mean[:, None]) * rstd[:, None]
        ln = xhat * w.float() + b.float()
        # TODO 7: derive the gradients of y = ln * u, with ln = xhat * w + b.
        #   du = dy * ln          dln = dy * u
        #   dw = sum_rows(dln * xhat)      db = sum_rows(dln)
        #   dxhat = dln * w
        #   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))   (row means)
        raise NotImplementedError("write the backward")


def ln_mul(x, u, w, b, eps=1e-5):
    return LayerNormMul.apply(x, u, w, b, eps)
