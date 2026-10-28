"""Lab 10 client: check served recommendations, then load-test the server.

    bash serving/start_server.sh &          # wait for "Started GRPCInferenceService"
    python serving/client.py                # correctness + latency/throughput sweep

1. Sends real MovieLens-1M user histories over gRPC and checks the top-10
   items equal what the same checkpoint returns offline.
2. Load test: for each concurrency level, N client threads each send one
   request at a time in a closed loop for a fixed duration. Reports
   throughput, p50/p99 latency, and the mean batch size the server formed.
   Run against "hstu" (dynamic batching) and "hstu_nobatch".
"""
import argparse
import os
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("GK_TRITON_CLIENT", str(pathlib.Path.home() / ".cache/gk_tritonserver/client")))
import tritonclient.grpc as grpcclient  # noqa: E402

from gk.bench import report, table  # noqa: E402


def histories(max_len, n_users):
    d = np.load(ROOT / "data" / "ml-1m.npz")
    items, ts, off = d["items"].astype(np.int64), d["ts"], d["offsets"]
    out = []
    for u in range(min(n_users, off.shape[0] - 1)):
        s, e = off[u], off[u + 1] - 1  # drop the test item, as in evaluation
        s = max(s, e - max_len)
        out.append((items[s:e], ts[s:e]))
    return out


def make_inputs(ids, ts):
    a = grpcclient.InferInput("item_ids", [1, ids.shape[0]], "INT64")
    a.set_data_from_numpy(ids[None])
    b = grpcclient.InferInput("timestamps", [1, ts.shape[0]], "INT64")
    b.set_data_from_numpy(ts[None])
    return [a, b]


def check_correctness(url, model, users):
    import torch

    from gk import kernels

    ckpt = torch.load(ROOT / "checkpoints" / "hstu_ml-1m.pt", map_location="cpu", weights_only=False)
    net = kernels.lab("08_hstu_model").HSTU(**ckpt["cfg"])
    net.load_state_dict(ckpt["state"])
    net = net.cuda().eval()
    c = grpcclient.InferenceServerClient(url)
    agree = 0
    for ids, ts in users:
        res = c.infer(model, make_inputs(ids, ts))
        served = res.as_numpy("top_items")[0]
        off = torch.tensor([0, ids.shape[0]], dtype=torch.int32, device="cuda")
        _, local = net.topk(torch.from_numpy(ids).cuda(), torch.from_numpy(ts).cuda(), off, ckpt["max_len"])
        agree += int((local[0].cpu().numpy() == served).all())
    return agree


def load_test(url, model, users, concurrency, seconds):
    lat, sizes = [], []
    stop = time.perf_counter() + seconds
    lock = threading.Lock()

    def worker(w):
        c = grpcclient.InferenceServerClient(url)
        mine, bs = [], []
        i = w
        while time.perf_counter() < stop:
            ids, ts = users[i % len(users)]
            t0 = time.perf_counter()
            res = c.infer(model, make_inputs(ids, ts))
            mine.append(time.perf_counter() - t0)
            bs.append(int(res.as_numpy("batch_size")[0, 0]))
            i += concurrency
        with lock:
            lat.extend(mine)
            sizes.extend(bs)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    lat = np.array(lat) * 1e3
    return len(lat) / wall, np.percentile(lat, 50), np.percentile(lat, 99), float(np.mean(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"localhost:{os.environ.get('GK_GRPC_PORT', '8001')}")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--concurrency", default="1,8,32,128")
    ap.add_argument("--models", default="hstu_nobatch,hstu")
    args = ap.parse_args()

    users = histories(200, 2000)
    n_check = 50
    agree = check_correctness(args.url, "hstu", users[:n_check])
    print(f"correctness: served top-10 equals offline top-10 for {agree}/{n_check} users")

    rows = []
    res = {}
    for model in args.models.split(","):
        for c in map(int, args.concurrency.split(",")):
            qps, p50, p99, bs = load_test(args.url, model, users, c, args.seconds)
            res[(model, c)] = (qps, p50, p99, bs)
            rows.append((model, c, f"{qps:8.0f}", f"{p50:7.2f}", f"{p99:7.2f}", f"{bs:5.1f}"))
    table(rows, ["model", "clients", "req/s", "p50 ms", "p99 ms", "mean batch"])

    c = max(map(int, args.concurrency.split(",")))
    nb, db = res.get(("hstu_nobatch", c)), res.get(("hstu", c))
    if nb and db:
        report(
            "with many concurrent clients, dynamic batching groups requests into one jagged batch, so "
            "throughput rises several-fold while each request waits at most the queue delay",
            f"{c} clients: no batching {nb[0]:.0f} req/s (p99 {nb[2]:.1f} ms), dynamic batching {db[0]:.0f} req/s "
            f"(p99 {db[2]:.1f} ms, mean batch {db[3]:.1f})",
            "PASS" if agree == n_check and db[0] > 2 * nb[0] else "CHECK",
            "change max_queue_delay_microseconds (200, 2000, 10000) in hstu/config.pbtxt and replot "
            "throughput against p99; then try instance_group count: 2",
        )


if __name__ == "__main__":
    main()
