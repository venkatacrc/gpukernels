"""CUDA-graph inference for the jagged HSTU model.

A CUDA graph records a whole sequence of kernel launches once and replays it
with a single launch, removing per-kernel CPU overhead. Graphs need static
shapes and static memory addresses, but jagged batches change shape on every
call. The fix is to pad to buckets: each graph is captured for a fixed number
of users B and a fixed token count T. A real batch is copied into the static
buffers, unused user slots get length 0, and one extra dummy user absorbs the
unused token tail. Its outputs are computed and thrown away.
"""
import torch


class GraphedTopK:
    def __init__(self, model, max_len, k=10, buckets=((1, 256), (8, 2048), (64, 16384)), impl="triton"):
        self.model, self.max_len, self.k, self.impl = model, max_len, k, impl
        self.graphs = {}
        for B, T in buckets:
            ids = torch.ones(T, dtype=torch.long, device="cuda")
            ts = torch.arange(T, dtype=torch.long, device="cuda")
            off = torch.zeros(B + 2, dtype=torch.int32, device="cuda")  # B users + 1 dummy
            off[1 : B + 1] = torch.minimum(torch.arange(1, B + 1, device="cuda") * max_len, torch.tensor(T, device="cuda"))
            off[B + 1] = T
            self.model.topk(ids, ts, off, max_len, k, impl)  # compile + autotune outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                vals, idx = self.model.topk(ids, ts, off, max_len, k, impl)
            self.graphs[(B, T)] = (g, ids, ts, off, vals, idx)

    def __call__(self, ids, ts, offsets):
        B, T = offsets.numel() - 1, ids.numel()
        key = min((k for k in self.graphs if k[0] >= B and k[1] >= T), default=None)
        if key is None:
            return self.model.topk(ids, ts, offsets, self.max_len, self.k, self.impl)
        g, s_ids, s_ts, s_off, vals, idx = self.graphs[key]
        s_ids[:T].copy_(ids)
        s_ts[:T].copy_(ts)
        s_off[: B + 1].copy_(offsets)
        s_off[B + 1 : -1].fill_(T)  # empty user slots
        g.replay()
        return vals[:B], idx[:B]
