"""Lab 09: train HSTU on MovieLens with the course kernels.

    python train/data.py --dataset ml-1m
    python train/train.py --dataset ml-1m                 # Triton kernels (solutions)
    GK_KERNELS=mine python train/train.py                 # your kernels
    python train/train.py --impl torch                    # plain PyTorch baseline

Protocol (as in the HSTU paper's MovieLens experiments): each user's last
interaction is the test target; everything before it is training data. The
model predicts the next item at every position of the training sequence
(at most max_len + 1 most recent items). Evaluation ranks the test item
against every movie and reports HR@10 and NDCG@10.

The whole dataset lives on the GPU (1M events = 12 MB), and batches are
assembled with a few vectorised index ops, so there is no DataLoader and no
host-to-device copy in the training loop.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gk import kernels  # noqa: E402


class GPUSequences:
    def __init__(self, path, max_len, device="cuda"):
        d = np.load(path)
        self.items = torch.from_numpy(d["items"].astype(np.int64)).to(device)
        self.ts = torch.from_numpy(d["ts"]).to(device)
        off = torch.from_numpy(d["offsets"]).to(device)
        self.num_items = int(d["raw_ids"].shape[0] - 1)
        self.raw_ids = d["raw_ids"]
        start, length = off[:-1], off[1:] - off[:-1]
        self.max_len = max_len
        n_train = length - 1  # last event is the test target
        # training tail: the max_len + 1 most recent training events -> max_len (input, target) pairs
        tail = n_train.clamp(max=max_len + 1)
        keep = tail >= 2
        self.train_start = (start + n_train - tail)[keep]
        self.train_len = (tail - 1)[keep]
        # evaluation: the max_len events before the test item
        ev = n_train.clamp(max=max_len)
        keep_e = ev >= 1
        self.eval_start = (start + n_train - ev)[keep_e]
        self.eval_len = ev[keep_e]
        self.eval_target = self.items[(start + n_train)[keep_e]]
        self.num_users = int(keep.sum())

    @staticmethod
    def _gather(starts, lengths):
        offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
        offsets[1:] = lengths.cumsum(0)
        T = int(offsets[-1])
        seg = torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths, output_size=T)
        src = starts[seg] + torch.arange(T, device=lengths.device) - offsets[:-1].long()[seg]
        return src, offsets

    def train_batches(self, batch_size, generator):
        perm = torch.randperm(self.train_start.numel(), device=self.items.device, generator=generator)
        for i in range(0, perm.numel(), batch_size):
            u = perm[i : i + batch_size]
            src, offsets = self._gather(self.train_start[u], self.train_len[u])
            yield self.items[src], self.ts[src], offsets, self.items[src + 1]

    def eval_batches(self, batch_size):
        for i in range(0, self.eval_start.numel(), batch_size):
            sl = slice(i, i + batch_size)
            src, offsets = self._gather(self.eval_start[sl], self.eval_len[sl])
            yield self.items[src], self.ts[src], offsets, self.eval_target[sl]


@torch.no_grad()
def evaluate(model, data, impl, k=10):
    model.eval()
    hits = ndcg = n = 0.0
    items = model.item_table()
    for ids, ts, off, target in data.eval_batches(1024):
        states = model.encode(ids, ts, off, data.max_len, impl)
        last = states[off[1:].long() - 1]
        scores = (last @ items.t()).float()
        scores[:, 0] = -float("inf")
        tgt = scores.gather(1, target[:, None])
        rank = (scores > tgt).sum(1).float()
        hits += (rank < k).sum().item()
        ndcg += torch.where(rank < k, 1.0 / torch.log2(rank + 2.0), 0.0).sum().item()
        n += target.numel()
    model.train()
    return hits / n, ndcg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--impl", default="triton", choices=["triton", "torch"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--negatives", type=int, default=512, help="0 = full softmax over all items")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--pos-emb", action="store_true", help="add a learnable absolute position embedding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="checkpoint path (default checkpoints/hstu_<dataset>.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = GPUSequences(ROOT / "data" / f"{args.dataset}.npz", args.max_len)
    HSTU = kernels.lab("08_hstu_model").HSTU
    cfg = dict(num_items=data.num_items, d=args.d, layers=args.layers, heads=args.heads,
               head_dim=args.head_dim, dropout=args.dropout, max_len=args.max_len if args.pos_emb else 0)
    model = HSTU(**cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.dataset}: {data.num_users:,} training users, {data.num_items:,} items | impl={args.impl} | "
          f"{params / 1e6:.2f}M params | max_len={args.max_len} batch={args.batch}")

    history = []
    best = (0.0, 0.0, 0)
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = 0
        loss_sum = torch.zeros((), device="cuda")
        steps = 0
        for ids, ts, off, tgt in data.train_batches(args.batch, gen):
            states = model.encode(ids, ts, off, args.max_len, args.impl)
            loss = model.sampled_softmax_loss(states, tgt, args.negatives)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach()
            tokens += ids.numel()
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        line = (f"epoch {epoch:3d}  loss {loss_sum.item() / steps:.4f}  {dt:5.2f}s  "
                f"{tokens / dt / 1e6:5.2f} Mtok/s  {dt / steps * 1e3:5.2f} ms/step")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            hr, ndcg = evaluate(model, data, args.impl)
            line += f"  HR@10 {hr:.4f}  NDCG@10 {ndcg:.4f}"
            history.append(dict(epoch=epoch, hr10=hr, ndcg10=ndcg))
            if hr > best[0]:
                best = (hr, ndcg, epoch)
                out = pathlib.Path(args.out or ROOT / "checkpoints" / f"hstu_{args.dataset}.pt")
                out.parent.mkdir(exist_ok=True)
                torch.save(dict(cfg=cfg, max_len=args.max_len, state=model.state_dict(),
                                raw_ids=torch.from_numpy(data.raw_ids), dataset=args.dataset), out)
        print(line, flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"best HR@10 {best[0]:.4f}  NDCG@10 {best[1]:.4f} at epoch {best[2]}  | peak GPU memory {peak:.0f} MB")
    print(json.dumps(dict(best_hr10=best[0], best_ndcg10=best[1], best_epoch=best[2], history=history)))


if __name__ == "__main__":
    main()
