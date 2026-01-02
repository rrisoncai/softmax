// ============================================================
// softmax_avx_cpu.cpp
//
// 一个用于 Intel CPU 的 Softmax 性能基准程序：
// - reference：标量 + std::exp（数值最稳，但慢）
// - optimized：AVX2 向量化 + exp 近似（展示 non-linear 可软件优化）
//
// 用途：
// 1) 验证 softmax 在 CPU 上的可优化空间
// 2) 量化“不需要 softmax 专用硬件”的工程事实
//
// Build (GCC/Clang):
//   g++ -O3 -march=native -std=c++17 softmax_avx_cpu.cpp -o softmax
//   clang++ -O3 -march=native -std=c++17 softmax_avx_cpu.cpp -o softmax
//
// Build (Intel oneAPI icx/icpx):
//   icpx -O3 -xHost -std=c++17 softmax_avx_cpu.cpp -o softmax
//
// Run:
//   ./softmax               # defaults: cols=1024 rows=4096
//   ./softmax 2048 8192     # cols rows
// ============================================================

#include <immintrin.h>   // AVX / AVX2 intrinsics
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ------------------------------------------------------------
// Horizontal sum of 8 floats in a __m256
// 用于 reduction：sum(exp(x_i))
// ------------------------------------------------------------
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);       // low 128
    __m128 vhigh = _mm256_extractf128_ps(v, 1);     // high 128
    vlow = _mm_add_ps(vlow, vhigh);                 // 4 lanes

    // 标准 SSE 横向求和套路
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ------------------------------------------------------------
// Horizontal max of 8 floats in a __m256
// 用于 softmax 的 max-subtraction（数值稳定）
// ------------------------------------------------------------
static inline float hmax256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(vlow, vhigh);

    m = _mm_max_ps(m, _mm_movehdup_ps(m));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(m);
}

// ------------------------------------------------------------
// Fast exp2 approximation (AVX2, vectorized)
//
// 思想：
//   exp2(x) = 2^n * 2^f
//   n = round(x), f ∈ [-0.5, 0.5]
//   2^n 通过直接构造 float exponent bits
//   2^f 用低阶多项式近似
//
// 目的：
//   - 避开 libm / SFU 风格的 exp
//   - 展示 non-linear 可以在通用向量 ALU 上跑得很快
// ------------------------------------------------------------
static inline __m256 exp2_approx_ps(__m256 x) {
    // 防止指数位溢出（float exponent 8 bits）
    const __m256 max_x = _mm256_set1_ps( 127.0f);
    const __m256 min_x = _mm256_set1_ps(-126.0f);
    x = _mm256_min_ps(max_x, _mm256_max_ps(min_x, x));

    // n = round(x), f = x - n
    __m256 n = _mm256_round_ps(
        x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    __m256 f = _mm256_sub_ps(x, n);

    // 通过直接设置 exponent bits 构造 2^n
    // float: sign(1) | exponent(8) | mantissa(23)
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 two_n = _mm256_castsi256_ps(ni);

    // 多项式近似 2^f
    // 在 softmax 的 (x - max) 后，f 非常安全
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);   // ln2
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);

    __m256 f2 = _mm256_mul_ps(f, f);
    __m256 f3 = _mm256_mul_ps(f2, f);
    __m256 f4 = _mm256_mul_ps(f2, f2);

    __m256 poly = _mm256_add_ps(
        _mm256_set1_ps(1.0f),
        _mm256_add_ps(_mm256_mul_ps(c1, f),
        _mm256_add_ps(_mm256_mul_ps(c2, f2),
        _mm256_add_ps(_mm256_mul_ps(c3, f3),
                      _mm256_mul_ps(c4, f4))))
    );

    return _mm256_mul_ps(two_n, poly);
}

// ------------------------------------------------------------
// exp(x) = exp2(x * log2(e))
// ------------------------------------------------------------
static inline __m256 exp_approx_ps(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    return exp2_approx_ps(_mm256_mul_ps(x, log2e));
}

// ------------------------------------------------------------
// AVX2 Softmax (one row)
//
// 结构：
//   1) max-reduction（数值稳定）
//   2) exp + sum-reduction
//   3) normalize
//
// 这是“non-GEMM 算子在 CPU 上的最佳实践形态”：
//   - 全 SIMD
//   - 无函数调用
//   - reduction + memory bound
// ------------------------------------------------------------
static void softmax_row_avx2(const float* in, float* out, int N) {
    // ---- 1) max ----
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    float maxv = hmax256_ps(vmax);
    for (; i < N; ++i) maxv = std::max(maxv, in[i]);

    // ---- 2) exp + sum ----
    __m256 vsum = _mm256_setzero_ps();
    const __m256 vmaxv = _mm256_set1_ps(maxv);
    i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(in + i), vmaxv);
        __m256 e = exp_approx_ps(v);
        _mm256_storeu_ps(out + i, e);
        vsum = _mm256_add_ps(vsum, e);
    }

    float sum = hsum256_ps(vsum);
    for (; i < N; ++i) {
        float e = std::exp(in[i] - maxv);   // tail 用高精度
        out[i] = e;
        sum += e;
    }

    // ---- 3) normalize ----
    float inv = 1.0f / sum;
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(out + i), vinv);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < N; ++i) out[i] *= inv;
}

// ------------------------------------------------------------
// Reference softmax（标量 + std::exp）
// 用于 correctness & baseline 对比
// ------------------------------------------------------------
static void softmax_ref(const float* in, float* out, int N) {
    float maxv = in[0];
    for (int i = 1; i < N; ++i) maxv = std::max(maxv, in[i]);

    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = std::exp(double(in[i] - maxv));
        out[i] = float(e);
        sum += e;
    }

    float inv = 1.0f / float(sum);
    for (int i = 0; i < N; ++i) out[i] *= inv;
}

static void online_softmax(const float* in, float* out, int N) {
    /* 
    online softmax implementation: Two Pass Algorithm
    */

    float max_i = in[0];
    float sum_i = std::exp(in[0] - max_i);

    for (int i = 1; i < N; ++i) {
        float max_i_1 = max_i;
        max_i = std::max(max_i, in[i]);
        sum_i = sum_i * std::exp(max_i_1 - max_i) + std::exp(in[i] - max_i);
    }

    float inv = 1.0f / sum_i;;
    for (int i = 0; i < N; ++i) {
        out[i] = std::exp(in[i] - max_i) * inv;
    }
}
// ------------------------------------------------------------
// 最大绝对误差（用于验证近似 exp 的数值影响）
// ------------------------------------------------------------
static float max_abs_diff(const float* a, const float* b, int N) {
    float m = 0.0f;
    for (int i = 0; i < N; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// ------------------------------------------------------------
// main: correctness + benchmark
// ------------------------------------------------------------
int main(int argc, char** argv) {
    int rows = 4096;
    int cols = 256;
    if (argc >= 2) cols = std::atoi(argv[1]);
    if (argc >= 3) rows = std::atoi(argv[2]);

    std::vector<float> x(size_t(rows) * cols);
    std::vector<float> y(size_t(rows) * cols);
    std::vector<float> yref(cols);

    // 随机输入（较宽分布，避免“假快”）
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng) * 3.0f;

    // ---- correctness ----
    softmax_row_avx2(x.data(), y.data(), cols);
    softmax_ref(x.data(), yref.data(), cols);
    float avx_err = max_abs_diff(y.data(), yref.data(), cols);

    online_softmax(x.data(), y.data(), cols);
    softmax_ref(x.data(), yref.data(), cols);
    float online_err = max_abs_diff(y.data(), yref.data(), cols);

    std::printf("cols=%d rows=%d\n", cols, rows);
    std::printf("max_abs_diff(AVX vs Ref) = %.6g\n", avx_err);
    std::printf("max_abs_diff(online vs Ref)   = %.6g\n", online_err);

    // ---- reference benchmark ----
    auto ref_t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rows; ++r) {
        softmax_ref(x.data() + size_t(r) * cols, yref.data(), cols);
    }
    auto ref_t1 = std::chrono::high_resolution_clock::now();
    double ref_ms =
        std::chrono::duration<double, std::milli>(ref_t1 - ref_t0).count();

    std::printf("[ref]   time = %.3f ms\n", ref_ms);

    // ---- online softmax benchmark ----
    auto online_t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rows; ++r) {
        online_softmax(x.data() + size_t(r) * cols, yref.data(), cols);
    }
    auto online_t1 = std::chrono::high_resolution_clock::now();
    double online_ms =
        std::chrono::duration<double, std::milli>(online_t1 - online_t0).count();
    std::printf("[online]time = %.3f ms\n", online_ms);

    // ---- warmup ----
    for (int r = 0; r < std::min(rows, 256); ++r) {
        softmax_row_avx2(
            x.data() + size_t(r) * cols,
            y.data() + size_t(r) * cols,
            cols
        );
    }

    // ---- optimized benchmark ----
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rows; ++r) {
        softmax_row_avx2(
            x.data() + size_t(r) * cols,
            y.data() + size_t(r) * cols,
            cols
        );
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    double elems = double(rows) * double(cols);
    double gelem_s = (elems / (ms / 1e3)) / 1e9;

    std::printf("time = %.3f ms, throughput = %.3f Gelem/s\n",
                ms, gelem_s);

    return 0;
}