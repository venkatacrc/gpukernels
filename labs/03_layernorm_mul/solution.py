"""Reference solution for Lab 03: HSTU's  y = LayerNorm(x) * u  in one kernel.

HSTU applies this after attention: the attention output x is normalised and
gated elementwise by u, one of the four SiLU projections (U, V, Q, K).
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
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * w + b) * u
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def triton_ln_mul_fwd(x, u, w, b, eps=1e-5):
    shape = x.shape
    x2, u2 = x.reshape(-1, shape[-1]).contiguous(), u.reshape(-1, shape[-1]).contiguous()
    M, D = x2.shape
    y = torch.empty_like(x2)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    BLOCK_D = triton.next_power_of_2(D)
    ln_mul_fwd_kernel[(M,)](x2, u2, w, b, y, mean, rstd, x2.stride(0), D, eps, BLOCK_D=BLOCK_D,
                            num_warps=4 if BLOCK_D <= 1024 else 8)
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
        du = dy2 * ln
        dln = dy2 * u2
        dw = (dln * xhat).sum(0)
        db = dln.sum(0)
        dxhat = dln * w.float()
        dx = rstd[:, None] * (dxhat - dxhat.mean(-1, keepdim=True) - xhat * (dxhat * xhat).mean(-1, keepdim=True))
        return dx.to(x.dtype).reshape(x.shape), du.to(u.dtype).reshape(u.shape), dw.to(w.dtype), db.to(b.dtype), None


def ln_mul(x, u, w, b, eps=1e-5):
    return LayerNormMul.apply(x, u, w, b, eps)
