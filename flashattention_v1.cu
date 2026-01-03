// flash_attn_test.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

/*
Teaching implementation of causal self-attention (single batch/head) with head_dim=256.

1) Vanilla baseline (two-pass, materialized scores):
   - Materialize S = (Q K^T) in HBM as float[M, N] (here M=N=rows).
   - For each row i: compute softmax(S[i,:]) and then O[i] = softmax(S[i,:]) * V.
   - This stresses HBM bandwidth because S is O(N^2) and must be written/read.

2) FlashAttention-style (tiled + online softmax, no materialized S):
   - Process keys/values in tiles (BN) and load each tile of K/V into shared memory (SRAM).
   - Maintain numerically-stable online softmax state per query row:
       m = running max, l = running sum of exp, o = running output accumulator.
   - Update per tile:
       m_new = max(m, tile_max)
       l = l * exp(m - m_new) + sum_j exp(x_j - m_new)
       o = o * exp(m - m_new) + sum_j exp(x_j - m_new) * v_j
     Final output: O = o / l

Notes:
- This code is intentionally not fully optimized (no tensor cores, no cp.async double-buffering,
  scalar math, and it recomputes Q·K twice per tile). It is meant to demonstrate the core idea:
  avoiding O(N^2) score materialization by fusing softmax with the attention computation.
*/

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

// ---------------- warp helpers ----------------
__inline__ __device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__inline__ __device__ float block_reduce_max(float v) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  return sdata[0];
}

__inline__ __device__ float block_reduce_sum(float v) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
}

// Vanilla step 1: compute and store the full score matrix S in HBM (O(N^2) memory).
// For rows=4096, S is 4096*4096 floats ≈ 64 MB (just for intermediate scores).
// ---------------- vanilla attention baseline (materialize scores) ----------------
// scores S: float [M, N]
__global__ void vanilla_scores_256_causal(const half* __restrict__ Q,
                                         const half* __restrict__ K,
                                         float* __restrict__ S,
                                         int M, int N, float scale) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;

  if (j > i) {
    S[(long long)i * N + j] = -INFINITY;
    return;
  }

  const half* q = Q + (long long)i * 256;
  const half* k = K + (long long)j * 256;
  float dot = 0.f;
#pragma unroll
  for (int d = 0; d < 256; d++) {
    dot += __half2float(q[d]) * __half2float(k[d]);
  }
  S[(long long)i * N + j] = dot * scale;
}

// Vanilla step 2: per-row softmax over S, then multiply by V (reads S from HBM again).
// There are two variants:
// A) 3-pass variant (max pass, sum pass, PV pass) reading S three times (matches current behavior).
// B) 2-pass variant optimized to read S only twice (first pass max, second pass sumexp+PV together).

// A) 3-pass variant: renamed from original vanilla_softmax_pv_256_causal
__global__ void vanilla_softmax_pv_256_causal_3passS(const float* __restrict__ S,
                                             const half* __restrict__ V,
                                             half* __restrict__ O,
                                             int M, int N) {
  int i = blockIdx.x;
  int d = threadIdx.x; // 0..255
  if (i >= M || d >= 256) return;

  // 1) max over j<=i
  float local_max = -INFINITY;
  for (int j = 0; j <= i; j++) {
    float x = S[(long long)i * N + j];
    local_max = fmaxf(local_max, x);
  }
  float m = block_reduce_max(local_max); // shared float reduction
  __syncthreads();

  // 2) sum exp over j<=i
  float local_sum = 0.f;
  for (int j = 0; j <= i; j++) {
    float x = S[(long long)i * N + j];
    local_sum += expf(x - m);
  }
  float l = block_reduce_sum(local_sum);
  __syncthreads();

  // 3) O[d] = sum softmax * V[j,d]
  float acc = 0.f;
  for (int j = 0; j <= i; j++) {
    float x = S[(long long)i * N + j];
    float p = expf(x - m) / (l + 1e-9f);
    acc += p * __half2float(V[(long long)j * 256 + d]);
  }
  O[(long long)i * 256 + d] = __float2half(acc);
}

// B) 2-pass variant: optimized baseline that reads S only twice
// Vanilla step 2 (optimized): per-row softmax and PV with only TWO passes over S.
// Pass1: max; Pass2: compute sumexp and PV together (each S[i,j] read once in pass2).
__global__ void vanilla_softmax_pv_256_causal_2passS(const float* __restrict__ S,
                                                    const half* __restrict__ V,
                                                    half* __restrict__ O,
                                                    int M, int N) {
  int i = blockIdx.x;
  int d = threadIdx.x; // 0..255
  if (i >= M || d >= 256) return;

  // Pass 1: max over j<=i
  float local_max = -INFINITY;
  for (int j = 0; j <= i; j++) {
    float x = S[(long long)i * N + j];
    local_max = fmaxf(local_max, x);
  }
  float m = block_reduce_max(local_max);
  __syncthreads();

  // Pass 2: compute sumexp and PV together
  float sum = 0.f;
  float acc = 0.f;
  for (int j = 0; j <= i; j++) {
    float x = S[(long long)i * N + j];
    float e = expf(x - m);
    sum += e;
    acc += e * __half2float(V[(long long)j * 256 + d]);
  }
  float l = block_reduce_sum(sum);
  __syncthreads();

  O[(long long)i * 256 + d] = __float2half(acc / (l + 1e-9f));
}

// FlashAttention-style: tile K/V into shared memory and compute attention with online softmax.
// Key idea: never materialize the full S matrix in HBM; compute softmax and PV on the fly.
template<int BM, int BN, bool CAUSAL>
__global__ void flash_attn_fwd_256(
    const half* Q, const half* K, const half* V,
    half* O,
    int M, int N, float scale)
{
  int qb   = blockIdx.x;
  int warp = threadIdx.y;
  int lane = threadIdx.x;

  int q = qb * BM + warp;
  if (q >= M) return;

  const half* q_ptr = Q + q * 256;
  half* o_ptr = O + q * 256;

  float qreg[8];
  float oacc[8] = {0};

#pragma unroll
  for (int i = 0; i < 8; i++)
    qreg[i] = __half2float(q_ptr[lane + 32 * i]);

  float m = -INFINITY;
  float l = 0.f;

  extern __shared__ half smem[];
  half* sK = smem;
  half* sV = smem + BN * 256;

  for (int kb = 0; kb < N; kb += BN) {
    int tn = min(BN, N - kb);
    if (CAUSAL && kb > q) break;

    int tid = threadIdx.y * 32 + lane;
    for (int idx = tid; idx < tn * 256; idx += BM * 32) {
      int j = idx / 256;
      int d = idx % 256;
      sK[j * 256 + d] = K[(kb + j) * 256 + d];
      sV[j * 256 + d] = V[(kb + j) * 256 + d];
    }
    __syncthreads();

    float tile_max = -INFINITY;

    for (int j = 0; j < tn; j++) {
      if (CAUSAL && kb + j > q) break;
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++)
        dot += qreg[i] * __half2float(sK[j * 256 + lane + 32 * i]);

      dot = warp_sum(dot);
      if (lane == 0) tile_max = fmaxf(tile_max, dot * scale);
    }

    tile_max = __shfl_sync(0xffffffff, tile_max, 0);
    float m_new = fmaxf(m, tile_max);
    float alpha = expf(m - m_new);

    l *= alpha;
#pragma unroll
    for (int i = 0; i < 8; i++) oacc[i] *= alpha;

    float l_add = 0.f;

    for (int j = 0; j < tn; j++) {
      if (CAUSAL && kb + j > q) break;
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++)
        dot += qreg[i] * __half2float(sK[j * 256 + lane + 32 * i]);

      dot = warp_sum(dot);
      dot = __shfl_sync(0xffffffff, dot, 0) * scale;

      float p = expf(dot - m_new);
      if (lane == 0) l_add += p;

#pragma unroll
      for (int i = 0; i < 8; i++)
        oacc[i] += p * __half2float(sV[j * 256 + lane + 32 * i]);
    }

    l += __shfl_sync(0xffffffff, l_add, 0);
    m = m_new;
    __syncthreads();
  }

  float inv_l = 1.f / (l + 1e-9f);
#pragma unroll
  for (int i = 0; i < 8; i++)
    o_ptr[lane + 32 * i] = __float2half(oacc[i] * inv_l);
}

// ---------------- main ----------------
int main() {
  int rows = 4096;
  int cols = 256;

  size_t bytes = rows * cols * sizeof(half);

  std::vector<half> hQ(rows * cols);
  std::vector<half> hK(rows * cols);
  std::vector<half> hV(rows * cols);
  std::vector<half> hO(rows * cols);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int i = 0; i < rows * cols; i++) {
    hQ[i] = __float2half(dist(gen));
    hK[i] = __float2half(dist(gen));
    hV[i] = __float2half(dist(gen));
  }

  half *dQ, *dK, *dV, *dO;
  CHECK_CUDA(cudaMalloc(&dQ, bytes));
  CHECK_CUDA(cudaMalloc(&dK, bytes));
  CHECK_CUDA(cudaMalloc(&dV, bytes));
  CHECK_CUDA(cudaMalloc(&dO, bytes));
  float* dS = nullptr;
  CHECK_CUDA(cudaMalloc(&dS, (size_t)rows * rows * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));

  float scale = 1.0f / sqrtf((float)cols);

  constexpr int BM = 8;
  constexpr int BN = 32;
  dim3 block(32, BM);
  dim3 grid((rows + BM - 1) / BM);
  size_t smem = BN * cols * sizeof(half) * 2;

  // ---------------- FlashAttention-style timing ----------------
  // warmup + timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warmup (avoid first-launch overhead)
  for (int i = 0; i < 5; i++) {
    flash_attn_fwd_256<BM, BN, true>
        <<<grid, block, smem>>>(dQ, dK, dV, dO, rows, rows, scale);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // timing (average over multiple runs)
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < 10; i++) {
    flash_attn_fwd_256<BM, BN, true>
        <<<grid, block, smem>>>(dQ, dK, dV, dO, rows, rows, scale);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  printf("Avg FlashAttention kernel time: %.3f ms\n", ms / 10.0f);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // ---------------- vanilla baseline timing (materialized S) ----------------
  // Note: these runs overwrite dO. If you want to print/compare outputs, allocate two output buffers.

  dim3 vblock_scores(16, 16);
  dim3 vgrid_scores((rows + vblock_scores.x - 1) / vblock_scores.x,
                    (rows + vblock_scores.y - 1) / vblock_scores.y);

  dim3 vblock_pv(256);
  dim3 vgrid_pv(rows);

  auto time_kernel_loop = [&](auto launch_fn, const char* name) {
    cudaEvent_t s, t;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&t));

    // warmup
    for (int i = 0; i < 2; i++) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < 10; i++) launch_fn();
    CHECK_CUDA(cudaEventRecord(t));
    CHECK_CUDA(cudaEventSynchronize(t));

    float ms2 = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms2, s, t));
    float avg = ms2 / 10.0f;
    printf("%s: %.3f ms\n", name, avg);

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(t));
    return avg;
  };

  // component timings
  float t_scores = time_kernel_loop([&]() {
    vanilla_scores_256_causal<<<vgrid_scores, vblock_scores>>>(dQ, dK, dS, rows, rows, scale);
  }, "Avg vanilla scores time (write S)");

  float t_pv_3pass = time_kernel_loop([&]() {
    vanilla_softmax_pv_256_causal_3passS<<<vgrid_pv, vblock_pv, 256 * sizeof(float)>>>(dS, dV, dO, rows, rows);
  }, "Avg vanilla softmax+PV time (read S 3-pass)");

  float t_pv_2pass = time_kernel_loop([&]() {
    vanilla_softmax_pv_256_causal_2passS<<<vgrid_pv, vblock_pv, 256 * sizeof(float)>>>(dS, dV, dO, rows, rows);
  }, "Avg vanilla softmax+PV time (read S 2-pass)");

  printf("Avg vanilla total (scores + 3-pass PV): %.3f ms\n", t_scores + t_pv_3pass);
  printf("Avg vanilla total (scores + 2-pass PV): %.3f ms\n", t_scores + t_pv_2pass);

  // compare against flash
  float flash_avg = ms / 10.0f;
  printf("Speedup vs flash (vanilla total 3-pass / flash): %.2fx\n", (t_scores + t_pv_3pass) / flash_avg);
  printf("Speedup vs flash (vanilla total 2-pass / flash): %.2fx\n", (t_scores + t_pv_2pass) / flash_avg);
  printf("Extra cost from reading S 3-pass vs 2-pass (PV only): %.2fx\n", t_pv_3pass / t_pv_2pass);

  // This prints the output currently in dO (after the last kernel run above).
  CHECK_CUDA(cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost));

  printf("O[0][0..7] = ");
  for (int i = 0; i < 8; i++)
    printf("%f ", __half2float(hO[i]));
  printf("\n");

  cudaFree(dS);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
}