# GPU Kernels → HSTU: a hands-on course

Write GPU kernels in **Triton**, **Gluon** and **CUDA**, one at a time, and stitch them into an **HSTU** generative recommender in PyTorch. Then train it on MovieLens and serve it with **Triton Inference Server**. In every lab you write the code, run it, measure the result on your GPU, and only then read why it behaves that way.

The lessons live in a small offline web app (`web/`, live at [venkatacrc.github.io/gpukernels](https://venkatacrc.github.io/gpukernels/)) and a native Mac app (`app/`): Lessons, flash Cards (Learn and Test modes), and Progress. Progress is stored on your device and can be exported from one app and imported into the other.

## Syllabus

| # | Lesson | Lab | You measure |
|---|---|---|---|
| 00 | Setup and the GPU mental model | `tools/check_env.py` | HBM bandwidth, bf16 TFLOP/s, launch floor |
| 01 | Vector add three ways | `labs/01_vector_add` | Triton vs Gluon layouts vs CUDA scalar/float4, Little's law |
| 02 | Fused softmax | `labs/02_fused_softmax` | 6.8× over unfused, register spills at 64K columns |
| 03 | LayerNorm × gate, fwd + bwd | `labs/03_layernorm_mul` | 2.1–2.6× from fusion, saved statistics |
| 04 | GEMM + SiLU, tiles to TMA | `labs/04_matmul_silu` | TMA + persistent + warp specialization, 1.6× over cuBLAS+SiLU |
| 05 | Reductions in raw CUDA | `labs/05_cuda_reduction` | coalescing, shared-memory trees, warp shuffles |
| 06 | HSTU attention forward | `labs/06_hstu_attention_fwd` | jagged fused attention, 10–32× over dense |
| 07 | HSTU attention backward | `labs/07_hstu_attention_bwd` | recompute vs store, the 70× atomics cliff |
| 08 | The HSTU model | `labs/08_hstu_model` | 4× faster steps, 7× less memory than PyTorch |
| 09 | Train on MovieLens-1M | `train/` | HR@10 / NDCG@10 ablation (best 0.282 / 0.156) |
| 10 | Serve with Triton Inference Server | `serving/` | dynamic ragged batching: 14.6× throughput |
| 11 | System optimisation | `labs/11_system_opt` | launch-bound profile, batch scaling, CUDA graphs, compile caches |
| 12 | Triton Developer Conference 2026 | — | field notes mapped onto the labs |

Reference numbers in the lessons were measured on one NVIDIA GB200 (152 SMs, sm_100).

## Run the labs

```bash
python -m pip install torch triton numpy        # torch 2.12 / triton 3.7 were used
python tools/check_env.py                       # lesson 00
python labs/01_vector_add/run.py                # your kernel.py (TODO stubs)
python labs/01_vector_add/run.py --solution     # the reference solution
GK_KERNELS=mine python labs/08_hstu_model/run.py   # the model on your own kernels

python train/data.py --dataset ml-1m
python train/train.py --epochs 200 --pos-emb --negatives 0 --layers 4

bash serving/setup_tritonserver.sh              # one-time, no Docker
bash serving/start_server.sh &
python serving/client.py
```

CUDA labs compile with NVRTC (`torch.cuda._compile_kernel`), so no `nvcc` is needed.

## The web app

```bash
python tools/build_course.py                    # course/*.md -> web/course.json
python -m http.server -d web 8080               # open http://localhost:8080
node tools/smoke_web.js                         # headless render check
```

Lessons are Markdown in `course/` with `::: write`, `::: run`, `::: measure`, `::: explain`, `::: cards` and `::: note` blocks (format documented at the top of `tools/build_course.py`). The explanation of each lab unlocks after you record your measurements, and the app compares each one with the reference.

The layout is a single phone-width column: the ☰ menu opens a drawer with the lesson list (tick lessons to choose which cards you practise and what prints), card options (filter, shuffle, restart) and Export / Import / Print / Reset. The buttons at the bottom move between lessons or cards; *Practise N cards* drills just the open lesson.

Keyboard: `1`/`2`/`3` switch views; `←`/`→` previous/next lesson or card; in Cards, `Space` reveals, `G`/`M` got it / missed it, `K`/`R` mark known / review; `Ctrl+P` (`⌘P`) prints the open lesson, or the selected lessons from Cards and Progress. Cards swipe on phones. The app is installable and works offline.

## The Mac app

```bash
./scripts/build_app.sh                          # needs Xcode or the Command Line Tools (macOS 14+)
open build/GPUKernels.app
```

The same course in a native window: the sidebar lists the lessons with progress rings and checkboxes for selection, the toolbar switches Lessons / Cards / Progress, and lessons render with the same highlighting and measurement checks as the web app. Shortcuts: `⌘1`–`⌘3` views, `⌘[` / `⌘]` previous / next lesson, `⌘L` practise this lesson's cards, `⌘P` print or save the lesson as PDF, `⌘⇧P` print the selected lessons, `⌘⇧E` / `⌘⇧I` export / import progress. In Cards, `←`/`→`, `Space`, `G`/`M` and `K`/`R` work as on the web. Progress is saved to `~/Library/Application Support/GPUKernels/progress.json`, in the same format as the web app's export.

GitHub Pages: `.github/workflows/pages.yml` rebuilds `course.json` and publishes `web/` on every push to `main`. Enable it under Settings → Pages → Source: GitHub Actions.
