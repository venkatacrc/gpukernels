---
id: 09
title: Train HSTU on MovieLens-1M
lab: train
minutes: 90
track: train
summary: GPU-resident jagged batches, full-ranking HR@10 / NDCG@10, and an ablation you run yourself
---
Now the model learns something real. **MovieLens-1M** has 1,000,209 ratings from 6,040 users on 3,706 movies, each with a timestamp. The task: given a user's history in time order, predict the next movie.

**Protocol** (the one used in the HSTU and SASRec papers):

- The last interaction of each user is the test target; everything before it is history.
- Train on next-item prediction at every position of the history (teacher forcing), with up to 200 most recent events.
- Evaluate with **full ranking**: score all 3,706 movies, then compute **HR@10** (is the true movie in the top 10?) and **NDCG@10** (rewards higher positions: 1/log₂(rank+1)).

**The data pipeline is on the GPU.** `train/train.py` loads the whole dataset as tensors once. Each batch picks random users, slices their tails with vectorised index arithmetic, and builds `offsets`. No DataLoader and no host-to-device copies per step: at about 8 ms per step a CPU pipeline would be the bottleneck.

::: write
Nothing new to write; this lab is about experiments. Read `train/train.py` first: `GPUSequences.batch`, `evaluate`, and the training loop. Then download the data:

```bash
python train/data.py --dataset ml-1m        # into data/ml-1m.npz
```
:::

::: run
Baseline, then the experiments. Each run prints the loss, step time and tokens/s, and HR@10 / NDCG@10 every 10 epochs.

```bash
python train/train.py --epochs 100                                          # baseline: 2 layers, 512 negatives
python train/train.py --epochs 200 --pos-emb --out checkpoints/exp_pos.pt   # + position embedding
python train/train.py --epochs 200 --pos-emb --negatives 0 --out checkpoints/exp_full.pt  # + full softmax
python train/train.py --epochs 200 --pos-emb --negatives 0 --layers 4      # final recipe -> checkpoints/hstu_ml-1m.pt
```
Add `--impl torch` to any run to train the identical model without the custom kernels.
:::

::: measure
- base_hr | baseline HR@10, 100 epochs | 0.2353 |
- base_ndcg | baseline NDCG@10 | 0.1183 |
- full_hr | full softmax + pos emb, HR@10 | 0.2651 |
- l4_hr | 4 layers, full softmax, pos emb, HR@10 | 0.2820 |
- l4_ndcg | 4 layers, NDCG@10 | 0.1557 |
- epoch_s | seconds per epoch, 2 layers | 0.4 | s
:::

::: explain
**The ablation on the course GB200** (best HR@10 / NDCG@10 over the run):

| Change | HR@10 | NDCG@10 |
|---|---|---|
| baseline: 2 layers, d=64, 512 negatives, 100 epochs | 0.2353 | 0.1183 |
| 200 epochs | 0.2502 | 0.1275 |
| + position embedding | 0.2508 | 0.1308 |
| 128 negatives instead of 512 | 0.2253 | — |
| full softmax (all 3,706 items) | 0.2621 | 0.1445 |
| full softmax + position embedding | 0.2651 | 0.1460 |
| d=128 | 0.2687 | 0.1570 |
| **4 layers + full softmax + pos emb** | **0.2820** | **0.1557** |
| 4 layers, d=128 | 0.2750 | 0.1562 |
| *paper: SASRec* | *0.2853* | *0.1603* |
| *paper: HSTU* | *0.3097* | *0.1720* |

**What each change teaches.**

- **Negatives matter most.** The softmax denominator over 128 random negatives is a noisy, biased estimate, because random negatives are mostly easy. Going 128 → 512 → full raises HR by 4 points. With 3,706 items, full softmax is one (T × 3706) GEMM, cheap enough, so use it. At YouTube scale (millions of items) you cannot, and the sampling strategy becomes a research topic.
- **Depth beats width here.** 4 layers gains 1.7 points; doubling d gains less, and 4 layers at d=128 overfits ML-1M's 6,040 users.
- **Timestamps already encode order.** The absolute position embedding adds little on top of the relative time-and-position bias.

The remaining gap to the paper's HSTU (0.31) comes from tuning (learning-rate schedule, longer training, dropout schedule and seeds). The paper reports its best configurations. Treat our numbers as a reproducible, honest baseline, not a leaderboard entry.

**Throughput.** About 0.4 s per epoch with 2 layers (48 steps of 128 users, about 1.6 M tokens/s); 4 layers takes about 0.85 s. The GPU is mostly *idle* during these steps, because the batch is small and each step launches about 280 kernels. Lab 11 measures exactly how idle, and what to do about it.

**The bug this lab found.** The first training run hung for minutes per step. The Lab 04 GEMM's autotune key included M, and the jagged token count changes every batch, so every step re-ran the autotuner. Bucketing M to the next power of two fixed it. Real workloads find bugs that microbenchmarks never do.
:::

::: cards
Q: Define HR@10 and NDCG@10.
A: HR@10: fraction of users whose true next item ranks in the top 10. NDCG@10: average of 1/log₂(rank+1) when the item is in the top 10 (0 otherwise); it rewards higher ranks.
Q: What is the leave-one-out protocol for sequential recommendation?
A: Hold out each user's last interaction as the test target, train on everything before it, and rank the target against the whole catalogue (full ranking, not sampled).
Q: Why did full softmax beat 512 sampled negatives on ML-1M?
A: Sampled negatives give a noisy, biased estimate of the partition function (random negatives are mostly easy); with only 3,706 items the full softmax is one cheap GEMM.
Q: Why keep the training data pipeline on the GPU?
A: Steps take about 8 ms; building jagged batches on the CPU and copying them would take longer than the step. GPU-side slicing keeps the GPU fed.
Q: Why did keying the GEMM autotuner on M make training hang?
A: The token count M of a jagged batch changes every step, so every step triggered a full autotune (compiling and benchmarking every config).
:::
