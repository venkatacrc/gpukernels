"""Reference solution for Lab 04: tiled GEMM with a fused SiLU epilogue.

HSTU's first step per layer is  UVQK = SiLU(LayerNorm(x) @ W_uvqk), one GEMM
that produces all four projections at once. Fusing the SiLU into the GEMM's
epilogue saves a full read and write of the (tokens x 4*d) output.

Step 1 (matmul_kernel) is the classic pointer-based tiled GEMM.
Step 2 (matmul_tma_kernel) is the Blackwell version: TMA tensor descriptors,
a persistent grid (one program per SM looping over tiles) and warp
specialization (producer warps issue TMA loads while consumer warps run MMAs).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def torch_matmul_silu(a, b):
    return F.silu(a @ b)


# ---------------------------------------------------------------- step 1: pointer-based tiles
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
    # grouped ordering: GROUP row-blocks share the same B column-blocks while they are hot in L2
    group_size = GROUP * num_n
    group_id = pid // group_size
    first_m = group_id * GROUP
    rows_in_group = min(num_m - first_m, GROUP)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    if ACT == "silu":
        acc = acc * tl.sigmoid(acc)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
    # B is passed transposed (N x K, K-major) so both operands are loaded as K-contiguous tiles.
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_d = tl.make_tensor_descriptor(bt_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    c_d = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN // EPI])
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    for tile in tl.range(tl.program_id(0), num_m * num_n, NUM_SMS, flatten=True, warp_specialize=True):
        group_size = GROUP * num_n
        first_m = (tile // group_size) * GROUP
        rows = min(num_m - first_m, GROUP)
        pid_m = first_m + (tile % group_size) % rows
        pid_n = (tile % group_size) // rows
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(k_tiles):
            a = a_d.load([pid_m * BM, k * BK])
            b = b_d.load([pid_n * BN, k * BK])
            acc = tl.dot(a, b.T, acc)
        if ACT == "silu":
            acc = acc / (1.0 + tl.exp2(-1.4426950408889634 * acc))
        if EPI == 1:
            c_d.store([pid_m * BM, pid_n * BN], acc.to(c_ptr.dtype.element_ty))
        else:
            # store the tile in two halves so fewer registers are live in the epilogue
            acc = tl.reshape(acc, (BM, 2, BN // 2)).permute(0, 2, 1)
            c0, c1 = tl.split(acc)
            c_d.store([pid_m * BM, pid_n * BN], c0.to(c_ptr.dtype.element_ty))
            c_d.store([pid_m * BM, pid_n * BN + BN // 2], c1.to(c_ptr.dtype.element_ty))


def triton_matmul_tma(a, bt, act=""):
    """a: (M, K) row-major, bt: (N, K) row-major (that is, B transposed). Returns a @ bt.T."""
    triton.set_allocator(_alloc)
    M, K = a.shape
    N = bt.shape[0]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"])),)
    matmul_tma_kernel[grid](a, bt, c, M, N, K, m_bucket(M), ACT=act, NUM_SMS=NUM_SMS)
    return c
