"""Reference solution for Lab 06: fused, jagged HSTU attention forward in Triton.

One program owns BM query rows of one (user, head). It streams the user's key
and value blocks j <= i through shared memory, builds the relative bias in
registers from two tiny tables, applies SiLU and the 1/N scale, and
accumulates A @ V. The (N x N) score matrix never touches HBM.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rab_tile(offs_m, offs_n, ts_m, ts_n, pos_w_ptr, ts_w_ptr, P: tl.constexpr, NT: tl.constexpr):
    """Relative bias for a (BM, BN) tile plus the bucket indices backward needs."""
    rel = offs_m[:, None] - offs_n[None, :]
    pos_b = tl.minimum(tl.maximum(rel, 0), P - 1)
    dt = tl.abs(ts_m[:, None] - ts_n[None, :]).to(tl.float32)
    dt = tl.maximum(dt, 1.0)
    # floor(log2(dt)) straight from the fp32 exponent bits: exact and cheap
    t_b = ((dt.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    t_b = tl.minimum(tl.maximum(t_b, 0), NT - 1)
    bias = tl.load(pos_w_ptr + pos_b) + tl.load(ts_w_ptr + t_b)
    return bias, rel, pos_b, t_b


@triton.autotune(
    configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
             for bm, bn, w, s in [(64, 64, 4, 2), (64, 32, 4, 2), (128, 64, 8, 2), (32, 32, 4, 2)]],
    key=["MAX_LEN", "D"],
)
@triton.jit
def hstu_fwd_kernel(q_ptr, k_ptr, v_ptr, ts_ptr, off_ptr, pos_w_ptr, ts_w_ptr, out_ptr,
                    stride_t, stride_h, alpha, inv_n,
                    H: tl.constexpr, D: tl.constexpr, MAX_LEN: tl.constexpr, P: tl.constexpr, NT: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    start = tl.load(off_ptr + b)
    L = tl.load(off_ptr + b + 1) - start
    if pid_m * BM >= L:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < L
    base = start.to(tl.int64) * stride_t + h * stride_h
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    ts_m = tl.load(ts_ptr + start + offs_m, mask=m_mask, other=0)
    acc = tl.zeros((BM, D), dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, L)  # causal: keys up to the last query row of this block
    for n0 in range(0, hi, BN):
        offs_n = n0 + tl.arange(0, BN)
        n_mask = offs_n < L
        k = tl.load(k_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        ts_n = tl.load(ts_ptr + start + offs_n, mask=n_mask, other=0)
        s = tl.dot(q, tl.trans(k)) * alpha
        bias, rel, _, _ = rab_tile(offs_m, offs_n, ts_m, ts_n, pos_w_ptr, ts_w_ptr, P, NT)
        s = s + bias
        a = s * tl.sigmoid(s) * inv_n
        a = tl.where((rel >= 0) & m_mask[:, None] & n_mask[None, :], a, 0.0)
        acc += tl.dot(a.to(v.dtype), v)
    tl.store(out_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None])


def hstu_attention_fwd(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
    T, H, D = q.shape
    assert q.is_contiguous() and k.stride() == q.stride() and v.stride() == q.stride()
    out = torch.empty_like(q)
    B = offsets.numel() - 1
    grid = lambda meta: (triton.cdiv(max_len, meta["BM"]), B * H)
    hstu_fwd_kernel[grid](q, k, v, ts, offsets, pos_w, ts_w, out, q.stride(0), q.stride(1), alpha, 1.0 / max_len,
                          H=H, D=D, MAX_LEN=max_len, P=pos_w.numel(), NT=ts_w.numel())
    return out
