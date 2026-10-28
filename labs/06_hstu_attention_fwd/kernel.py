"""Lab 06: fused, jagged HSTU attention forward.

Run:  python labs/06_hstu_attention_fwd/run.py      (your code)
      python labs/06_hstu_attention_fwd/run.py --solution

Math, for query row i and key row j <= i of the same user:
    s      = alpha * q_i . k_j + pos_w[min(i - j, P-1)] + ts_w[bucket(ts_i - ts_j)]
    A[i,j] = SiLU(s) / N
    out_i  = sum_j A[i, j] v_j
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rab_tile(offs_m, offs_n, ts_m, ts_n, pos_w_ptr, ts_w_ptr, P: tl.constexpr, NT: tl.constexpr):
    """Relative bias for a (BM, BN) tile plus the bucket indices backward needs."""
    rel = offs_m[:, None] - offs_n[None, :]
    # TODO 1: pos_b = clamp(rel, 0, P - 1)   (tl.maximum / tl.minimum)
    # TODO 2: time bucket. dt = |ts_m[:, None] - ts_n[None, :]| as fp32, at least 1.0.
    #         floor(log2(dt)) is the fp32 exponent: bitcast to int32, >> 23, & 0xFF, - 127.
    #         Clamp to [0, NT - 1].
    # TODO 3: bias = tl.load(pos_w_ptr + pos_b) + tl.load(ts_w_ptr + t_b)
    #         (a gather from tables small enough to stay in L1).
    raise NotImplementedError
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
    # TODO 4: this user's tokens are rows start .. start + L of the jagged tensor.
    #         start = tl.load(off_ptr + b); L = tl.load(off_ptr + b + 1) - start
    #         The grid is sized for MAX_LEN, so programs past L return immediately.
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    # TODO 5: base = start (as int64) * stride_t + h * stride_h.
    #         Load the (BM, D) q tile and the BM query timestamps, masked by offs_m < L.
    acc = tl.zeros((BM, D), dtype=tl.float32)
    # TODO 6: causal loop. Keys beyond the last query row of this block can never
    #         be attended to, so loop n0 over range(0, min((pid_m + 1) * BM, L), BN).
    #   In each step: load k, v (BN, D) and ts_n; s = tl.dot(q, tl.trans(k)) * alpha
    #   s += bias from rab_tile; a = s * sigmoid(s) * inv_n
    #   zero a where rel < 0 or a row/column is past L;  acc += tl.dot(a.to(v.dtype), v)
    # TODO 7: store acc into out (same addressing as q), masked by offs_m < L.
    pass


def hstu_attention_fwd(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
    T, H, D = q.shape
    assert q.is_contiguous() and k.stride() == q.stride() and v.stride() == q.stride()
    out = torch.empty_like(q)
    B = offsets.numel() - 1
    # TODO 8: 2-D grid: (query blocks for the longest user, B * H)
    grid = lambda meta: (triton.cdiv(max_len, meta["BM"]), B * H)
    hstu_fwd_kernel[grid](q, k, v, ts, offsets, pos_w, ts_w, out, q.stride(0), q.stride(1), alpha, 1.0 / max_len,
                          H=H, D=D, MAX_LEN=max_len, P=pos_w.numel(), NT=ts_w.numel())
    return out
