"""Reference solution for Lab 07: HSTU attention backward and the autograd wrapper.

With s = alpha * q.k + rab,  sig = sigmoid(s),  A = SiLU(s) / N  (masked):
    dV   = A^T dO
    dA   = dO V^T
    dS   = dA * sig * (1 + s * (1 - sig)) / N          (SiLU'(s) = sig * (1 + s(1 - sig)))
    dQ   = alpha * dS K        dK = alpha * dS^T Q
    dpos[b] += sum of dS over entries in position bucket b; same for dts.

Two kernels, as in FlashAttention-2: one walks key blocks and accumulates dK,
dV (plus the bias-table gradients); the other walks query blocks and
accumulates dQ. Each output block is owned by exactly one program, so dQ, dK
and dV need no atomics. Only the tiny bias tables use atomic adds.
"""
import importlib.util
import pathlib

import torch
import triton
import triton.language as tl

_fwd_path = pathlib.Path(__file__).resolve().parents[1] / "06_hstu_attention_fwd" / "solution.py"
_spec = importlib.util.spec_from_file_location("lab06_solution", _fwd_path)
fwd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fwd)
rab_tile = fwd.rab_tile


@triton.jit
def _ds_tile(q, k, do, v, offs_m, offs_n, ts_m, ts_n, m_mask, n_mask, pos_w_ptr, ts_w_ptr, alpha, inv_n,
             P: tl.constexpr, NT: tl.constexpr):
    s = tl.dot(q, tl.trans(k)) * alpha
    bias, rel, pos_b, t_b = rab_tile(offs_m, offs_n, ts_m, ts_n, pos_w_ptr, ts_w_ptr, P, NT)
    s = s + bias
    sig = tl.sigmoid(s)
    valid = (rel >= 0) & m_mask[:, None] & n_mask[None, :]
    a = tl.where(valid, s * sig * inv_n, 0.0)
    da = tl.dot(do, tl.trans(v))
    ds = tl.where(valid, da * sig * (1.0 + s * (1.0 - sig)) * inv_n, 0.0)
    return a, ds, valid, pos_b, t_b


BWD_CONFIGS = [triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
               for bm, bn, w, s in [(64, 64, 4, 2), (32, 64, 4, 2), (64, 32, 4, 2), (32, 32, 4, 1)]]


@triton.autotune(configs=BWD_CONFIGS, key=["MAX_LEN", "D"], reset_to_zero=["dpos_ptr", "dts_ptr"])
@triton.jit
def hstu_bwd_dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, ts_ptr, off_ptr, pos_w_ptr, ts_w_ptr,
                         dk_ptr, dv_ptr, dpos_ptr, dts_ptr, stride_t, stride_h, alpha, inv_n,
                         H: tl.constexpr, D: tl.constexpr, MAX_LEN: tl.constexpr, P: tl.constexpr, NT: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    start = tl.load(off_ptr + b)
    L = tl.load(off_ptr + b + 1) - start
    if pid_n * BN >= L:
        return
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < L
    base = start.to(tl.int64) * stride_t + h * stride_h
    kv_off = base + offs_n[:, None] * stride_t + offs_d[None, :]
    k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
    v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
    ts_n = tl.load(ts_ptr + start + offs_n, mask=n_mask, other=0)
    dk = tl.zeros((BN, D), dtype=tl.float32)
    dv = tl.zeros((BN, D), dtype=tl.float32)
    lo = (pid_n * BN // BM) * BM  # causal: only query rows i >= first key of this block
    for m0 in range(lo, L, BM):
        offs_m = m0 + tl.arange(0, BM)
        m_mask = offs_m < L
        q_off = base + offs_m[:, None] * stride_t + offs_d[None, :]
        q = tl.load(q_ptr + q_off, mask=m_mask[:, None], other=0.0)
        do = tl.load(do_ptr + q_off, mask=m_mask[:, None], other=0.0)
        ts_m = tl.load(ts_ptr + start + offs_m, mask=m_mask, other=0)
        a, ds, valid, pos_b, t_b = _ds_tile(q, k, do, v, offs_m, offs_n, ts_m, ts_n, m_mask, n_mask,
                                            pos_w_ptr, ts_w_ptr, alpha, inv_n, P, NT)
        dv += tl.dot(tl.trans(a.to(do.dtype)), do)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        # private row per program: no cross-SM contention on the tiny tables
        row = (pid_bh * tl.num_programs(0) + pid_n).to(tl.int64)
        tl.atomic_add(dpos_ptr + row * P + pos_b, ds, mask=valid, sem="relaxed")
        tl.atomic_add(dts_ptr + row * NT + t_b, ds, mask=valid, sem="relaxed")
    tl.store(dk_ptr + kv_off, (dk * alpha).to(dk_ptr.dtype.element_ty), mask=n_mask[:, None])
    tl.store(dv_ptr + kv_off, dv.to(dv_ptr.dtype.element_ty), mask=n_mask[:, None])


@triton.autotune(configs=BWD_CONFIGS, key=["MAX_LEN", "D"])
@triton.jit
def hstu_bwd_dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, ts_ptr, off_ptr, pos_w_ptr, ts_w_ptr, dq_ptr,
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
    q_off = base + offs_m[:, None] * stride_t + offs_d[None, :]
    q = tl.load(q_ptr + q_off, mask=m_mask[:, None], other=0.0)
    do = tl.load(do_ptr + q_off, mask=m_mask[:, None], other=0.0)
    ts_m = tl.load(ts_ptr + start + offs_m, mask=m_mask, other=0)
    dq = tl.zeros((BM, D), dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, L)
    for n0 in range(0, hi, BN):
        offs_n = n0 + tl.arange(0, BN)
        n_mask = offs_n < L
        kv_off = base + offs_n[:, None] * stride_t + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        ts_n = tl.load(ts_ptr + start + offs_n, mask=n_mask, other=0)
        _, ds, _, _, _ = _ds_tile(q, k, do, v, offs_m, offs_n, ts_m, ts_n, m_mask, n_mask,
                                  pos_w_ptr, ts_w_ptr, alpha, inv_n, P, NT)
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(dq_ptr + q_off, (dq * alpha).to(dq_ptr.dtype.element_ty), mask=m_mask[:, None])


def hstu_attention_bwd(do, q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
    T, H, D = q.shape
    do = do.contiguous()
    B = offsets.numel() - 1
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    # one partial row per (key block, user, head); sized for the smallest BN in BWD_CONFIGS
    rows = triton.cdiv(max_len, 32) * B * H
    dpos = torch.zeros(rows, pos_w.numel(), device=q.device, dtype=torch.float32)
    dts = torch.zeros(rows, ts_w.numel(), device=q.device, dtype=torch.float32)
    common = dict(H=H, D=D, MAX_LEN=max_len, P=pos_w.numel(), NT=ts_w.numel())
    grid_n = lambda meta: (triton.cdiv(max_len, meta["BN"]), B * H)
    hstu_bwd_dkdv_kernel[grid_n](q, k, v, do, ts, offsets, pos_w, ts_w, dk, dv, dpos, dts,
                                 q.stride(0), q.stride(1), alpha, 1.0 / max_len, **common)
    dpos, dts = dpos.sum(0), dts.sum(0)
    grid_m = lambda meta: (triton.cdiv(max_len, meta["BM"]), B * H)
    hstu_bwd_dq_kernel[grid_m](q, k, v, do, ts, offsets, pos_w, ts_w, dq,
                               q.stride(0), q.stride(1), alpha, 1.0 / max_len, **common)
    return dq, dk, dv, dpos.to(pos_w.dtype), dts.to(ts_w.dtype)


class HSTUAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = fwd.hstu_attention_fwd(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha)
        ctx.save_for_backward(q, k, v, ts, offsets, pos_w, ts_w)
        ctx.max_len, ctx.alpha = max_len, alpha
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, ts, offsets, pos_w, ts_w = ctx.saved_tensors
        dq, dk, dv, dpos, dts = hstu_attention_bwd(do, q, k, v, ts, offsets, ctx.max_len, pos_w, ts_w, ctx.alpha)
        return dq, dk, dv, None, None, None, dpos, dts, None


def hstu_attention(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
    return HSTUAttention.apply(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha)
