import math
import torch
import triton
import triton.language as tl


# ----------------------------
# 2-pass: pass1 compute (m,l) per row
# ----------------------------
@triton.jit
def attn_stats_kernel(
    Q_ptr, K_ptr,
    M_ptr, L_ptr,
    stride_qg, stride_qn, stride_qd,
    stride_kg, stride_kn, stride_kd,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row id in [0, BH*N)
    q_idx = pid % N
    g = pid // N

    d = tl.arange(0, D)

    # load q: Q[g, q_idx, :]
    q_ptrs = Q_ptr + g * stride_qg + q_idx * stride_qn + d * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)

    # base pointer for K[g, :, :]
    k_base = K_ptr + g * stride_kg

    m = -float("inf")
    l = 0.0

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        mask_n = n < N

        # load K block: K[g, n, d]
        k_ptrs = k_base + n[:, None] * stride_kn + d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # scores: [BLOCK_N]
        s = tl.sum(k * q[None, :], axis=1)

        # update running max+sumexp (stable)
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    tl.store(M_ptr + pid, m)
    tl.store(L_ptr + pid, l)


# ----------------------------
# 2-pass: pass2 compute output using stored (m,l)
# ----------------------------
@triton.jit
def attn_out_2pass_kernel(
    Q_ptr, K_ptr, V_ptr,
    M_ptr, L_ptr,
    O_ptr,
    stride_qg, stride_qn, stride_qd,
    stride_kg, stride_kn, stride_kd,
    stride_vg, stride_vn, stride_vd,
    stride_og, stride_on, stride_od,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % N
    g = pid // N

    d = tl.arange(0, D)

    q = tl.load(Q_ptr + g * stride_qg + q_idx * stride_qn + d * stride_qd).to(tl.float32)
    k_base = K_ptr + g * stride_kg
    v_base = V_ptr + g * stride_vg

    m = tl.load(M_ptr + pid).to(tl.float32)
    l = tl.load(L_ptr + pid).to(tl.float32)

    o = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        mask_n = n < N

        k = tl.load(k_base + n[:, None] * stride_kn + d[None, :] * stride_kd,
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + n[:, None] * stride_vn + d[None, :] * stride_vd,
                    mask=mask_n[:, None], other=0.0).to(tl.float32)

        s = tl.sum(k * q[None, :], axis=1)
        p = tl.exp(s - m) / l  # [BLOCK_N]
        o += tl.sum(v * p[:, None], axis=0)

    tl.store(O_ptr + g * stride_og + q_idx * stride_on + d * stride_od, o.to(tl.float16))


# ----------------------------
# 1-pass: online softmax + output in one kernel
# ----------------------------
@triton.jit
def attn_out_1pass_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    stride_qg, stride_qn, stride_qd,
    stride_kg, stride_kn, stride_kd,
    stride_vg, stride_vn, stride_vd,
    stride_og, stride_on, stride_od,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % N
    g = pid // N

    d = tl.arange(0, D)

    q = tl.load(Q_ptr + g * stride_qg + q_idx * stride_qn + d * stride_qd).to(tl.float32)
    k_base = K_ptr + g * stride_kg
    v_base = V_ptr + g * stride_vg

    m = -float("inf")
    l = 0.0
    o = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        mask_n = n < N

        k = tl.load(k_base + n[:, None] * stride_kn + d[None, :] * stride_kd,
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + n[:, None] * stride_vn + d[None, :] * stride_vd,
                    mask=mask_n[:, None], other=0.0).to(tl.float32)

        s = tl.sum(k * q[None, :], axis=1)

        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l_new = l * alpha + tl.sum(p, axis=0)

        o = o * alpha + tl.sum(v * p[:, None], axis=0)

        m = m_new
        l = l_new

    o = o / l
    tl.store(O_ptr + g * stride_og + q_idx * stride_on + d * stride_od, o.to(tl.float16))


# ----------------------------
# Python wrappers
# ----------------------------
def triton_attn_2pass(Q, K, V, block_n=128):
    # Q,K,V: [B,H,N,D], fp16
    B, H, N, D = Q.shape
    BH = B * H
    Q2 = Q.reshape(BH, N, D).contiguous()
    K2 = K.reshape(BH, N, D).contiguous()
    V2 = V.reshape(BH, N, D).contiguous()

    # allocate stats and output
    M = torch.empty((BH * N,), device=Q.device, dtype=torch.float32)
    L = torch.empty((BH * N,), device=Q.device, dtype=torch.float32)
    O = torch.empty((BH, N, D), device=Q.device, dtype=torch.float16)

    grid = (BH * N,)

    sqg, sqn, sqd = Q2.stride(0), Q2.stride(1), Q2.stride(2)
    skg, skn, skd = K2.stride(0), K2.stride(1), K2.stride(2)

    attn_stats_kernel[grid](
        Q2, K2,
        M, L,
        sqg, sqn, sqd,
        skg, skn, skd,
        N=N, D=D,
        BLOCK_N=block_n,
        num_warps=4,
    )

    svg, svn, svd = V2.stride(0), V2.stride(1), V2.stride(2)
    sog, son, sod = O.stride(0), O.stride(1), O.stride(2)

    attn_out_2pass_kernel[grid](
        Q2, K2, V2,
        M, L,
        O,
        sqg, sqn, sqd,
        skg, skn, skd,
        svg, svn, svd,
        sog, son, sod,
        N=N, D=D,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return O.reshape(B, H, N, D)


def triton_attn_1pass(Q, K, V, block_n=128):
    B, H, N, D = Q.shape
    BH = B * H
    Q2 = Q.reshape(BH, N, D).contiguous()
    K2 = K.reshape(BH, N, D).contiguous()
    V2 = V.reshape(BH, N, D).contiguous()
    O = torch.empty((BH, N, D), device=Q.device, dtype=torch.float16)

    grid = (BH * N,)

    sqg, sqn, sqd = Q2.stride(0), Q2.stride(1), Q2.stride(2)
    skg, skn, skd = K2.stride(0), K2.stride(1), K2.stride(2)
    svg, svn, svd = V2.stride(0), V2.stride(1), V2.stride(2)
    sog, son, sod = O.stride(0), O.stride(1), O.stride(2)

    attn_out_1pass_kernel[grid](
        Q2, K2, V2,
        O,
        sqg, sqn, sqd,
        skg, skn, skd,
        svg, svn, svd,
        sog, son, sod,
        N=N, D=D,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return O.reshape(B, H, N, D)


def torch_ref(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)


@torch.no_grad()
def benchmark(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    torch.manual_seed(0)
    device = "cuda"

    B, H, N, D = 1, 8, 8192, 64
    Q = torch.randn((B, H, N, D), device=device, dtype=torch.float32)
    K = torch.randn((B, H, N, D), device=device, dtype=torch.float32)
    V = torch.randn((B, H, N, D), device=device, dtype=torch.float32)

    O_ref = torch_ref(Q.float(), K.float(), V.float())
    O_2p = triton_attn_2pass(Q, K, V, block_n=128)
    O_1p = triton_attn_1pass(Q, K, V, block_n=128)

    max_err_2p = (O_2p - O_ref).abs().max().item()
    max_err_1p = (O_1p - O_ref).abs().max().item()
    # print(f"max_err 2-pass: {max_err_2p}")
    # print(f"max_err 1-pass: {max_err_1p}")

    t_ref = benchmark(lambda: torch_ref(Q.float(), K.float(), V.float()), iters=50, warmup=10)
    t_2p = benchmark(lambda: triton_attn_2pass(Q, K, V, block_n=128), iters=50, warmup=10)
    t_1p = benchmark(lambda: triton_attn_1pass(Q, K, V, block_n=128), iters=50, warmup=10)

    print(f"triton 2-pass (stats + out)       : {t_2p:.3f} ms")
    print(f"triton 1-pass (flashattention v1) : {t_1p:.3f} ms")
    print(f"torch ref (dot_product_attention) : {t_ref:.3f} ms")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
