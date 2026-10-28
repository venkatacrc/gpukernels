"""Lab 04: tiled GEMM in Triton with a fused SiLU epilogue.

Run:  python labs/04_matmul_silu/run.py      (your code)
      python labs/04_matmul_silu/run.py --solution
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def torch_matmul_silu(a, b):
    return F.silu(a @ b)


def m_bucket(M):
    """Autotune key for M. Keying on M itself re-runs the autotuner for every new
    token count, which with jagged batches means every training step."""
    return triton.next_power_of_2(M)


CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP": 8}, num_stages=3, num_warps=4),
    triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP": 8}, num_stages=2, num_warps=8),
    triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP": 8}, num_stages=4, num_warps=4),
]


@triton.autotune(configs=CONFIGS, key=["M_BUCKET", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, M_BUCKET, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                  ACT: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    # Start simple: pid_m = pid // num_n ; pid_n = pid % num_n
    # TODO 3 (after it works): grouped ordering so GROUP row-blocks reuse the
    #   same B tiles from L2:
    #   group_size = GROUP * num_n; group_id = pid // group_size
    #   first_m = group_id * GROUP; rows = min(num_m - first_m, GROUP)
    #   pid_m = first_m + (pid % group_size) % rows; pid_n = (pid % group_size) // rows
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    # TODO 1: pointer tiles. a_ptrs is BM x BK, b_ptrs is BK x BN:
    #   a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    #   b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # TODO 2: loop over K in steps of BK. Load masked tiles (other=0.0),
    #   acc = tl.dot(a, b, acc), then advance a_ptrs by BK*sak and b_ptrs by BK*sbk.
    # TODO 4: epilogue. If ACT == "silu": acc = acc * tl.sigmoid(acc)
    # TODO 5: store acc (cast to c's dtype) with an (M, N) bounds mask.
    pass


def triton_matmul(a, b, act=""):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    matmul_kernel[grid](a, b, c, M, N, K, m_bucket(M), a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), ACT=act)
    return c


def triton_matmul_silu(a, b):
    return triton_matmul(a, b, act="silu")


# ---------------------------------------------------------------- step 2: TMA + persistent + warp specialization
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
TMA_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8, "EPI": epi}, num_stages=s, num_warps=w)
    for bm, bn, bk, s, w, epi in [
        (128, 128, 64, 4, 4, 1), (128, 256, 64, 3, 8, 1), (128, 128, 128, 3, 8, 1),
        (128, 256, 64, 4, 4, 2), (256, 128, 64, 3, 8, 1), (128, 128, 64, 5, 8, 2)]
]


def _alloc(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


@triton.autotune(configs=TMA_CONFIGS, key=["M_BUCKET", "N", "K", "ACT"])
@triton.jit
def matmul_tma_kernel(a_ptr, bt_ptr, c_ptr, M, N, K, M_BUCKET,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                      EPI: tl.constexpr, ACT: tl.constexpr, NUM_SMS: tl.constexpr):
    # TODO 6: build three TMA descriptors. The hardware copies whole tiles, so
    #   there are no pointer grids and no masks (out-of-range reads return 0):
    #   a_d = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    #   b_d = ... bt_ptr, shape [N, K], block [BN, BK]
    #   c_d = ... c_ptr,  shape [M, N], block [BM, BN // EPI]
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    # TODO 7: persistent loop. Launch only NUM_SMS programs; each walks tiles
    #   tile = program_id, program_id + NUM_SMS, ...:
    #   for tile in tl.range(tl.program_id(0), num_m * num_n, NUM_SMS, flatten=True, warp_specialize=True):
    #       (grouped pid_m / pid_n from `tile`, exactly as in step 1)
    #       acc = zeros; for k: a = a_d.load([pid_m*BM, k*BK]); b = b_d.load([pid_n*BN, k*BK])
    #                         acc = tl.dot(a, b.T, acc)
    # TODO 8: SiLU without a full-precision divide by sigmoid:
    #       acc = acc / (1.0 + tl.exp2(-1.4426950408889634 * acc))
    # TODO 9: c_d.store([pid_m*BM, pid_n*BN], acc.to(c_ptr.dtype.element_ty)) when EPI == 1;
    #   for EPI == 2 split acc into two (BM, BN//2) halves (reshape, permute, tl.split)
    #   and store each, which keeps fewer registers live.
    pass


def triton_matmul_tma(a, bt, act=""):
    """a: (M, K) row-major, bt: (N, K) row-major (that is, B transposed). Returns a @ bt.T."""
    triton.set_allocator(_alloc)  # TMA descriptors built in-kernel need a small scratch buffer
    M, K = a.shape
    N = bt.shape[0]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"])),)
    matmul_tma_kernel[grid](a, bt, c, M, N, K, m_bucket(M), ACT=act, NUM_SMS=NUM_SMS)
    return c
