"""Download and preprocess MovieLens into per-user, time-ordered item sequences.

    python train/data.py --dataset ml-1m     # 6,040 users, 3,706 movies, 1.0M ratings
    python train/data.py --dataset ml-20m    # 138,493 users, 26,744 movies, 20M ratings

Writes data/<dataset>.npz with:
    items    int32 (num_events,)  item ids 1..num_items, users back to back, oldest first
    ts       int64 (num_events,)  unix seconds
    offsets  int64 (num_users+1,)
    raw_ids  int64 (num_items+1,) original MovieLens movieId for each item id (index 0 unused)
"""
import argparse
import io
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}


def load_ratings(name):
    DATA.mkdir(exist_ok=True)
    zpath = DATA / f"{name}.zip"
    if not zpath.exists():
        print(f"downloading {URLS[name]}")
        urllib.request.urlretrieve(URLS[name], zpath)
    with zipfile.ZipFile(zpath) as z:
        if name == "ml-1m":
            raw = z.read("ml-1m/ratings.dat").decode("latin-1")
            df = pd.read_csv(io.StringIO(raw), sep="::", engine="python", names=["user", "movie", "rating", "ts"])
        else:
            df = pd.read_csv(z.open("ml-20m/ratings.csv")).rename(
                columns={"userId": "user", "movieId": "movie", "timestamp": "ts"})
    return df


def build(name):
    df = load_ratings(name)
    movies = np.sort(df["movie"].unique())
    remap = {m: i + 1 for i, m in enumerate(movies)}
    df["item"] = df["movie"].map(remap).astype(np.int32)
    # stable order inside a user: by time, then by movie id for ties
    df = df.sort_values(["user", "ts", "movie"], kind="stable")
    counts = df.groupby("user", sort=True).size().to_numpy()
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    raw_ids = np.zeros(len(movies) + 1, dtype=np.int64)
    raw_ids[1:] = movies
    out = DATA / f"{name}.npz"
    np.savez(out, items=df["item"].to_numpy(np.int32), ts=df["ts"].to_numpy(np.int64), offsets=offsets, raw_ids=raw_ids)
    print(f"{name}: {len(counts):,} users, {len(movies):,} items, {len(df):,} events, "
          f"median length {int(np.median(counts))}, max {counts.max()} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=sorted(URLS))
    build(ap.parse_args().dataset)
