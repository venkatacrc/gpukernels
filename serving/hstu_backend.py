"""Triton Python-backend model that serves HSTU top-k recommendations.

Each request is one user: a variable-length history of item ids and unix
timestamps. With dynamic batching Triton hands execute() several queued
requests at once; we concatenate their histories into one jagged batch, which
is exactly the layout the fused attention kernel wants, so batching costs no
padding at all.
"""
import json
import os
import pathlib
import sys

import torch
import triton_python_backend_utils as pb_utils

ROOT = pathlib.Path(os.environ.get("GK_ROOT", pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from gk import kernels  # noqa: E402


class HSTUServer:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        self.k = int(params.get("top_k", "10"))
        ckpt = torch.load(os.environ.get("GK_CHECKPOINT", ROOT / "checkpoints" / "hstu_ml-1m.pt"),
                          map_location="cpu", weights_only=False)
        self.max_len = ckpt["max_len"]
        self.model = kernels.lab("08_hstu_model").HSTU(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state"])
        self.model = self.model.cuda().eval()
        self.raw_ids = ckpt["raw_ids"].cuda()
        self.impl = params.get("impl", "triton")
        self._warmup()
        pb_utils.Logger.log_info(f"HSTU ready: impl={self.impl} max_len={self.max_len} "
                                 f"items={self.raw_ids.numel() - 1} name={cfg['name']}")

    def _warmup(self):
        # compile and autotune the Triton kernels before the first real request
        for b in (1, 8, 64):
            n = b * self.max_len
            ids = torch.randint(1, self.raw_ids.numel(), (n,), device="cuda")
            ts = torch.arange(n, device="cuda", dtype=torch.int64) * 3600 + 1_000_000_000
            off = torch.arange(0, n + 1, self.max_len, device="cuda", dtype=torch.int32)
            self.model.topk(ids, ts, off, self.max_len, self.k, self.impl)
        torch.cuda.synchronize()

    def execute(self, requests):
        # Tensors cross the backend boundary through DLPack, not as_numpy()/pb_utils.Tensor(ndarray):
        # the 2.51 stub is compiled against the numpy 1.x C ABI and misreads numpy 2 arrays.
        ids, ts, lengths = [], [], []
        for r in requests:
            i = _torch_in(r, "item_ids")[-self.max_len:]
            t = _torch_in(r, "timestamps")[-self.max_len:]
            ids.append(i)
            ts.append(t)
            lengths.append(i.shape[0])
        offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        offsets[1:] = torch.tensor(lengths).cumsum(0)
        ids_g = torch.cat(ids).cuda(non_blocking=True)
        ts_g = torch.cat(ts).cuda(non_blocking=True)
        off_g = offsets.cuda(non_blocking=True)
        scores, items = self.model.topk(ids_g, ts_g, off_g, self.max_len, self.k, self.impl)
        movies = self.raw_ids[items].cpu()
        items = items.cpu()
        scores = scores.float().cpu()
        bsz = torch.tensor([[len(requests)]], dtype=torch.int32)
        out = []
        for b in range(len(requests)):
            out.append(pb_utils.InferenceResponse([
                _out("top_items", items[b : b + 1]),
                _out("top_movie_ids", movies[b : b + 1]),
                _out("top_scores", scores[b : b + 1]),
                _out("batch_size", bsz),
            ]))
        return out


def _torch_in(request, name):
    return torch.from_dlpack(pb_utils.get_input_tensor_by_name(request, name).to_dlpack()).reshape(-1)


def _out(name, t):
    return pb_utils.Tensor.from_dlpack(name, torch.utils.dlpack.to_dlpack(t.contiguous().clone()))
