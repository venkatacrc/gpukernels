---
id: 10
title: Serve with Triton Inference Server
lab: serving
minutes: 90
track: serve
summary: Python-backend model repository, ragged dynamic batching into one jagged batch, load testing
---
Two different things are both called "Triton". **Triton the language** is what you wrote kernels in. **Triton Inference Server** is NVIDIA's model server: it takes HTTP/gRPC requests, queues them, forms batches and runs models through backends (TensorRT, ONNX, PyTorch, Python, and others). This lab serves the trained HSTU through the **Python backend**, so the same Triton kernels run inside the server.

## The model repository

```text
serving/model_repository/
  hstu/            config.pbtxt + 1/model.py    dynamic batching on
  hstu_nobatch/    config.pbtxt + 1/model.py    one request at a time
```

Each request is one user: a variable-length list of item ids and timestamps. The key configuration:

```text
max_batch_size: 128
input [ { name: "item_ids" data_type: TYPE_INT64 dims: [ -1 ] allow_ragged_batch: true }, ... ]
dynamic_batching { max_queue_delay_microseconds: 2000  preferred_batch_size: [ 32, 64, 128 ] }
```

`allow_ragged_batch` lets requests of *different lengths* be grouped. `execute()` (in `serving/hstu_backend.py`) receives the group, concatenates the histories and builds `offsets`. That is a jagged batch, exactly what the attention kernel wants. Batching costs zero padding.

::: write
Read `serving/hstu_backend.py`: `initialize` loads the checkpoint and warms the kernels up; `execute` builds the jagged batch and returns top-10 items per request.

**One sharp edge you will meet.** Tensors cross into and out of the backend through **DLPack** (`pb_utils.Tensor.from_dlpack`, `to_dlpack`), not `as_numpy()` and `pb_utils.Tensor(name, ndarray)`. The server's stub is compiled against the numpy 1.x C ABI. Given numpy 2 arrays, it misreads their element size and every response fails with "output tensor shape does not match size of output". DLPack bypasses numpy entirely.

Set up the server once (no Docker: it unpacks the server binaries shipped in the `nvidia-pytriton` wheel and builds a matching Python 3.13 for the backend stub):

```bash
bash serving/setup_tritonserver.sh
```
:::

::: run
```bash
bash serving/start_server.sh &                 # wait for "Started GRPCInferenceService"
python serving/client.py                       # correctness on 50 users, then the load test
```
The load test runs N closed-loop clients (each sends the next request when the previous returns) for 1, 8, 32 and 128 clients, against both models. Run it twice and keep the second, warm run.
:::

::: measure
- agree | users whose served top-10 equals offline top-10 (of 50) | 50 |
- nb_1 | no batching, 1 client | 181 | req/s
- nb_128 | no batching, 128 clients | 270 | req/s
- nb_128_p99 | no batching, 128 clients, p99 | 633 | ms
- db_128 | dynamic batching, 128 clients | 3940 | req/s
- db_128_p99 | dynamic batching, 128 clients, p99 | 66 | ms
- db_128_bs | mean batch size formed, 128 clients | 48.4 |
:::

::: explain
**Throughput: 270 → 3,940 req/s (14.6×) at 128 clients.** Without batching, each request is one forward pass of about 5 ms: 4 layers of about 10 tiny kernels each, plus top-k, for a single user of at most 200 tokens. The GPU does almost no work per launch, so the server saturates at 1 / 5 ms ≈ 200–270 req/s no matter how many clients wait. Queueing makes p99 explode to 633 ms.

With dynamic batching, the scheduler holds requests up to 2 ms and hands `execute()` up to 128 at once. A batch of 48 users costs barely more GPU time than a batch of 1, because the same launches simply do more work each. That is the same launch-bound regime as training (Lab 11). Throughput grows with the batch size the scheduler forms (mean 48.4 at 128 clients), and p99 *falls* to 66 ms.

**At 1 client batching costs a little.** 181 vs 141 req/s: the batcher waits for companions that never come. `max_queue_delay_microseconds` is the knob between latency at low load and throughput at high load.

**Cold start is real.** The first, cold run showed a p99 of 1.56 s at 1 client: each model instance is its own process, and kernels it had not seen yet (new autotune keys) compiled on the first requests. `initialize()` warms up a few shapes; production servers warm up every shape bucket and persist the Triton cache (Lab 11, part D).

**Correctness first.** The client compares the served top-10 with the offline model for 50 real users: 50/50 identical. Batching changes nothing about each user's result, because the jagged kernel never mixes users.
:::

::: cards
Q: What does `allow_ragged_batch: true` enable in a Triton Inference Server config?
A: The dynamic batcher may group requests whose inputs have different shapes (here, history lengths); the backend receives them as separate tensors and decides how to combine them.
Q: Why does HSTU serving batch without padding?
A: execute() concatenates the ragged histories into one jagged tensor plus offsets, which is exactly the input format of the fused attention kernel.
Q: Why does throughput without batching saturate near 1 / (per-request latency)?
A: One model instance processes one request at a time; each small forward is launch-bound, so extra concurrent clients only queue (p99 grows) without raising throughput.
Q: What trade-off does `max_queue_delay_microseconds` control?
A: How long the scheduler waits to fill a batch: longer waits mean bigger batches and higher throughput under load, at the cost of added latency at low load.
Q: Why use DLPack rather than as_numpy() in the course's Python backend?
A: The bundled backend stub is built against the numpy 1.x C ABI and misreads numpy 2 arrays (wrong element size → "output tensor shape does not match"); DLPack hands buffers across without numpy.
Q: Why can a served model's first requests be very slow?
A: JIT compilation and autotuning of kernels for shapes not covered by warm-up; fix by warming every shape bucket and persisting compile/autotune caches.
:::
