"""Dense PyTorch reference for HSTU attention, used to test the Triton kernels.

HSTU (Zhai et al., "Actions Speak Louder than Words", 2024) replaces softmax
attention with pointwise SiLU attention:

    A[i, j] = SiLU(alpha * q_i . k_j + rab(i, j)) / N        for j <= i
    out_i   = sum_j A[i, j] * v_j

rab(i, j) is a relative attention bias that is the sum of a position term,
indexed by the distance i - j, and a time term, indexed by a log bucket of the
timestamp gap ts_i - ts_j. N is the maximum sequence length, so the scale does
not depend on how many items a user has.

Sequences are jagged: a batch of B users is stored as one (T, H, D) tensor of
all their tokens back to back, with seq_offsets[b] .. seq_offsets[b+1] marking
user b. Nothing is padded, so no FLOPs or bytes are spent on padding.
"""
import torch
import torch.nn.functional as F

def time_bucket(dt, num_buckets):
    """floor(log2(gap in seconds)), clamped to [0, num_buckets-1].

    Computed from the fp32 exponent (frexp) so it is bit-identical to the
    kernels, which read the exponent bits directly instead of calling log2.
    """
    e = torch.frexp(dt.float().clamp(min=1.0)).exponent - 1
    return e.long().clamp(0, num_buckets - 1)


def jagged_to_padded(x, offsets, max_len):
    B = offsets.numel() - 1
    out = x.new_zeros((B, max_len) + x.shape[1:])
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    for b, L in enumerate(lengths):
        out[b, :L] = x[offsets[b] : offsets[b] + L]
    return out


def padded_to_jagged(x, offsets):
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    return torch.cat([x[b, :L] for b, L in enumerate(lengths)], dim=0)


def hstu_attention_ref(q, k, v, ts, offsets, max_len, pos_w, ts_w, alpha):
    """q, k, v: (T, H, D); ts: (T,) int64 seconds; pos_w: (P,); ts_w: (NT,)."""
    B = offsets.numel() - 1
    qp = jagged_to_padded(q, offsets, max_len).float()  # (B, N, H, D)
    kp = jagged_to_padded(k, offsets, max_len).float()
    vp = jagged_to_padded(v, offsets, max_len).float()
    tsp = jagged_to_padded(ts, offsets, max_len)  # (B, N)
    lengths = offsets[1:] - offsets[:-1]
    N = max_len

    s = torch.einsum("bihd,bjhd->bhij", qp, kp) * alpha
    i = torch.arange(N, device=q.device)
    rel = i[:, None] - i[None, :]
    pos_b = rel.clamp(0, pos_w.numel() - 1)
    t_b = time_bucket((tsp[:, :, None] - tsp[:, None, :]).abs(), ts_w.numel())
    rab = pos_w.float()[pos_b][None] + ts_w.float()[t_b]  # (B, N, N)
    s = s + rab[:, None]
    valid = (rel >= 0)[None] & (i[None, :] < lengths[:, None])[:, :, None] & (i[None, :] < lengths[:, None])[:, None, :]
    a = F.silu(s) / N * valid[:, None].float()
    out = torch.einsum("bhij,bjhd->bihd", a, vp)
    return padded_to_jagged(out, offsets).to(q.dtype)


def padded_inputs(q, k, v, ts, offsets, max_len):
    """Pad once, outside any timed region: (B, H, N, D) tensors, (B, N) ts, (B,) lengths."""
    pad = lambda x: jagged_to_padded(x, offsets, max_len)
    qp, kp, vp = (pad(x).transpose(1, 2).contiguous() for x in (q, k, v))
    return qp, kp, vp, pad(ts), (offsets[1:] - offsets[:-1])


class TableGather(torch.autograd.Function):
    """w[idx] whose backward is a bincount.

    PyTorch's default indexing backward sorts the indices and reduces each
    run of equal indices serially. With only ~20 distinct time buckets over
    millions of entries that takes over a second at B=64, N=512; bincount
    takes a few milliseconds.
    """

    @staticmethod
    def forward(ctx, w, idx):
        ctx.save_for_backward(idx)
        ctx.n = w.numel()
        return w[idx]

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        dw = torch.bincount(idx.flatten(), weights=g.flatten().float(), minlength=ctx.n)
        return dw.to(g.dtype), None


def hstu_attention_dense(qp, kp, vp, tsp, lengths, pos_w, ts_w, alpha):
    """Fair dense baseline: what a padded HSTU implementation does on every call.

    bf16 matmuls on (B, H, N, N) scores, bias and mask built as full tensors.
    """
    B, H, N, D = qp.shape
    i = torch.arange(N, device=qp.device)
    rel = i[:, None] - i[None, :]
    t_b = time_bucket((tsp[:, :, None] - tsp[:, None, :]).abs(), ts_w.numel())
    rab = TableGather.apply(pos_w, rel.clamp(0, pos_w.numel() - 1))[None] + TableGather.apply(ts_w, t_b)
    valid = (rel >= 0)[None] & (i[None, :] < lengths[:, None])[:, None, :]
    s = (qp @ kp.transpose(-1, -2)).float() * alpha + rab[:, None]
    a = (F.silu(s) / N).masked_fill(~valid[:, None], 0.0)
    return a.to(vp.dtype) @ vp  # (B, H, N, D)


def make_batch(B, max_len, H, D, dtype=torch.bfloat16, device="cuda", min_len=1, seed=0, full=False):
    """Random jagged batch with realistic timestamps (gaps of seconds to months)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if full:
        lengths = torch.full((B,), max_len)
    else:
        lengths = torch.randint(min_len, max_len + 1, (B,), generator=g)
    offsets = torch.zeros(B + 1, dtype=torch.int32)
    offsets[1:] = lengths.cumsum(0)
    T = int(offsets[-1])
    q = torch.randn(T, H, D, generator=g).to(dtype)
    k = torch.randn(T, H, D, generator=g).to(dtype)
    v = torch.randn(T, H, D, generator=g).to(dtype)
    gaps = torch.exp(torch.rand(T, generator=g) * 16).long()  # 1 s .. ~100 days
    ts = torch.cat([gaps[offsets[b] : offsets[b + 1]].cumsum(0) for b in range(B)]) + 1_600_000_000
    return (q.to(device), k.to(device), v.to(device), ts.to(device), offsets.to(device))
