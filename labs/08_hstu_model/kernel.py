"""Lab 08: stitch your kernels into an HSTU model.

Run:  python labs/08_hstu_model/run.py      (your code)
      python labs/08_hstu_model/run.py --solution

One HSTU layer, for the jagged token matrix x of shape (T, d):

    h          = LayerNorm(x)
    u, v, q, k = split(SiLU(h @ W_uvqk))           Lab 04 (fused GEMM + SiLU)
    a          = HSTUAttention(q, k, v, ts, rab)   Labs 06 + 07 (jagged, fused)
    y          = LayerNorm(a) * u                  Lab 03 (fused norm + gate)
    x          = x + Dropout(y @ W_o)

impl="triton" uses the course kernels; impl="torch" is the same model in plain
PyTorch on padded tensors, used to check correctness and as the baseline.
"""
import math
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from gk import kernels
from gk.hstu_ref import hstu_attention_dense
from gk.jagged import positions, to_jagged, to_padded


class MatmulSiLU(torch.autograd.Function):
    """y = SiLU(a @ w_t.T) with the fused Triton kernel.

    Only a and w_t are saved. Backward recomputes z = a @ w_t.T with cuBLAS
    (one extra GEMM) instead of keeping the (T x 4d) pre-activation alive:
    a memory-for-compute trade.
    """

    @staticmethod
    def forward(ctx, a, w_t):
        ctx.save_for_backward(a, w_t)
        return kernels.matmul_tma()(a, w_t, "silu")

    @staticmethod
    def backward(ctx, dy):
        a, w_t = ctx.saved_tensors
        # TODO 1: recompute z = a @ w_t.T (cuBLAS, in fp32 for the derivative),
        #   dz = dy * SiLU'(z) with SiLU'(z) = sig * (1 + z * (1 - sig)), cast to a.dtype.
        #   Return (da, dw_t): da = dz @ w_t, dw_t = dz.T @ a.
        raise NotImplementedError


class HSTUBlock(nn.Module):
    def __init__(self, d, heads, head_dim, num_pos=256, num_time=32, dropout=0.2):
        super().__init__()
        self.d, self.h, self.dh = d, heads, head_dim
        self.w_uvqk = nn.Parameter(torch.empty(4 * heads * head_dim, d))
        nn.init.xavier_uniform_(self.w_uvqk)
        self.norm_w = nn.Parameter(torch.ones(heads * head_dim))
        self.norm_b = nn.Parameter(torch.zeros(heads * head_dim))
        self.out = nn.Linear(heads * head_dim, d)
        self.pos_w = nn.Parameter(torch.zeros(num_pos))
        self.ts_w = nn.Parameter(torch.zeros(num_time))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ts, offsets, max_len, impl):
        T = x.shape[0]
        H, dh = self.h, self.dh
        h = F.layer_norm(x, (self.d,))
        if impl == "triton":
            # TODO 2: one fused GEMM + SiLU for all four projections, in bf16:
            #   uvqk = MatmulSiLU.apply(h.to(torch.bfloat16), self.w_uvqk.to(torch.bfloat16))
            raise NotImplementedError
        else:
            uvqk = F.silu(h.to(torch.bfloat16) @ self.w_uvqk.to(torch.bfloat16).t())
        u, v, q, k = uvqk.split(H * dh, dim=1)
        q, k, v = (t.reshape(T, H, dh).contiguous() for t in (q, k, v))
        alpha = 1.0 / math.sqrt(dh)
        if impl == "triton":
            # TODO 3: a = kernels.hstu_attention()(q, k, v, ts, offsets, max_len, self.pos_w, self.ts_w, alpha)
            # TODO 4: y = kernels.ln_mul()(a.reshape(T, H * dh), u.contiguous(), self.norm_w, self.norm_b)
            #   (kernels.* picks your lab kernels when GK_KERNELS=mine)
            raise NotImplementedError
        else:
            pad = lambda t: to_padded(t, offsets, max_len).transpose(1, 2)
            lengths = offsets[1:] - offsets[:-1]
            a = hstu_attention_dense(pad(q), pad(k), pad(v), to_padded(ts, offsets, max_len), lengths,
                                     self.pos_w, self.ts_w, alpha)
            a = to_jagged(a.transpose(1, 2), offsets).reshape(T, H * dh)
            y = F.layer_norm(a, (H * dh,), self.norm_w.to(a.dtype), self.norm_b.to(a.dtype)) * u
        return x + self.drop(self.out(y.to(x.dtype)))


class HSTU(nn.Module):
    def __init__(self, num_items, d=64, layers=2, heads=2, head_dim=32, dropout=0.2, temperature=0.05,
                 max_len=0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        # optional learnable absolute position embedding (the HSTU MovieLens setup uses one)
        self.pos_emb = nn.Embedding(max_len, d) if max_len else None
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(HSTUBlock(d, heads, head_dim, dropout=dropout) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.temperature = temperature
        self.d = d

    def encode(self, ids, ts, offsets, max_len, impl="triton"):
        """Jagged item ids (T,) -> L2-normalised user states (T, d), one per prefix."""
        x = self.item_emb(ids) * math.sqrt(self.d)
        if self.pos_emb is not None:
            pos = positions(offsets, ids.numel())[0].clamp(max=self.pos_emb.num_embeddings - 1)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, ts, offsets, max_len, impl)
        return F.normalize(x, dim=-1)

    def item_table(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    def sampled_softmax_loss(self, states, targets, num_neg=512):
        """Next-item loss: the true next item against num_neg shared random negatives.

        num_neg=0 means a full softmax over every item, which is affordable when
        the catalogue is small (MovieLens-1M has 3,706 movies).
        """
        items = self.item_table()
        if num_neg == 0:
            logits = (states @ items.t()).float() / self.temperature
            logits[:, 0] = float("-inf")
            return F.cross_entropy(logits, targets)
        neg = torch.randint(1, items.shape[0], (num_neg,), device=targets.device)
        # TODO 5: pos_logit = <state, item[target]> (T, 1); neg_logit = states @ items[neg].T (T, num_neg)
        #   mask accidental hits (neg == target) with -inf, concatenate [pos | neg],
        #   divide by the temperature and take cross-entropy with label 0 for every row.
        raise NotImplementedError

    @torch.no_grad()
    def topk(self, ids, ts, offsets, max_len, k=10, impl="triton"):
        states = self.encode(ids, ts, offsets, max_len, impl)
        last = states[offsets[1:].long() - 1]
        scores = last @ self.item_table().t()
        scores[:, 0] = float("-inf")
        return scores.topk(k, dim=1)
