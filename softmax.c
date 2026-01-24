#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* -------------Start Softmax------------------ */
int softmax_3pass(float* x, int n, float* y)
{
    float x_max = FLT_MIN;
    float sum = 0;

    /* First Pass */
    for (int i = 0; i < n; ++i) {
        x_max = fmax(x[i], x_max);
    }

    /* Second Pass */
    for (int i = 0; i < n; ++i) {
        sum += exp(x[i] - x_max);
    }

    /* Third Pass */
    for (int i = 0; i < n; ++i) {
        y[i] = exp(x[i] - x_max) / sum;
    }

    return 0;
}

int softmax_2pass(float* x, int n, float* y)
{
    float x_max = FLT_MIN;
    float sum_i = 0;

    /* First Pass */
    for (int i = 0; i < n; ++i) {
        float x_max_1 = x_max;
        x_max = fmax(x_max, x[i]);
        sum_i = sum_i * exp(x_max_1 - x_max) + exp(x[i] - x_max);
    }

    /* Second Pass */
    for (int i = 0; i < n; ++i) {
        y[i] = exp(x[i] - x_max) / sum_i;
    }

    return 0;
}

/* -----------END Softmax------------------ */


#include <emmintrin.h>   // SSE2
#if defined(_MSC_VER)
  #define ALIGN16 __declspec(align(16))
#elif defined(__GNUC__) || defined(__clang__)
  #define ALIGN16 __attribute__((aligned(16)))
#else
  #define ALIGN16
#endif

// SSE2 helpers for SSE3/SSSE3/SSE4.1 intrinsics
static inline __m128i mm_max_epi8_sse2(__m128i a, __m128i b) {
    // signed max for int8 lanes using SSE2
    __m128i gt = _mm_cmpgt_epi8(a, b);
    return _mm_or_si128(_mm_and_si128(gt, a), _mm_andnot_si128(gt, b));
}

static inline __m128i mm_hadd_epi32_sse2(__m128i v) {
    // horizontal add: [a b c d] -> [a+b c+d a+b c+d]
    __m128i t = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i s = _mm_add_epi32(v, t); // [a+c b+d c+a d+b]
    __m128i u = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum = _mm_add_epi32(s, u); // lane0 = a+b+c+d
    // replicate pairwise sums into all lanes like hadd/hadd usage pattern expects
    return _mm_shuffle_epi32(sum, _MM_SHUFFLE(0, 0, 0, 0));
}

static inline __m128i mm_lddqu_sse2(const void *p) {
    // substitute for _mm_lddqu_si128 (SSE3) with SSE2 unaligned load
    return _mm_loadu_si128((const __m128i *)p);
}

static inline void mm_storeu_si32_sse2(void *p, __m128i v) {
    // store low 32 bits safely
    int32_t x = _mm_cvtsi128_si32(v);
    memcpy(p, &x, sizeof(x));
}

#if defined(_MSC_VER)
  #define ALIGN16 __declspec(align(16))
#elif defined(__GNUC__) || defined(__clang__)
  #define ALIGN16 __attribute__((aligned(16)))
#else
  #define ALIGN16
#endif

/* data for softmax1 case */
#define SOFTMAX1_OUT_CH 5
#define SOFTMAX1_IN_CH 5
#define SOFTMAX1_INPUT_W 1
#define SOFTMAX1_INPUT_H 1
#define SOFTMAX1_DST_SIZE 5
#define SOFTMAX1_INPUT_SIZE 5
#define SOFTMAX1_INPUT_BATCHES 1
#define SOFTMAX1_OUTPUT_W 1
#define SOFTMAX1_OUTPUT_H 1
static const int8_t  softmax1_input[5] = {10, 20, 30, 40, 50};
static const int32_t softmax1_input_multiplier = 1717986944;
static const int32_t softmax1_input_left_shift = 23;
static const int32_t softmax1_diff_min = -240;
static const int8_t  softmax1_output_ref[5] = {-125, -120, -106, -68, 35};

typedef struct {
    void *scratch_buf;
    uint32_t scratch_buf_size;
} nn_context;

typedef struct {
    int32_t n, w, h, c;
} nn_dims;

#define SOFTMAX_ACCUM_BITS 12
#define SOFTMAX_MASK_IF_NON_ZERO(x) (x) != 0 ? ~0 : 0
#define SOFTMAX_SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

static inline int32_t bsr_u32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return 31 - __builtin_clz(x);
#else
    int32_t r = 0;
    while (x >>= 1) r++;
    return r;
#endif
}

static inline __m128i mm_loadu_partial_bytes(const void *p, int n_bytes) {
    uint8_t tmp[16] = {0};
    if (n_bytes > 0) memcpy(tmp, p, (size_t)n_bytes);
    return _mm_loadu_si128((const __m128i *)tmp);
}

static inline void mm_masked_store_bytes(void *p, __m128i mask, int n_bytes, __m128i v) {
    uint8_t m[16];
    uint8_t t[16];
    _mm_storeu_si128((__m128i *)m, mask);
    _mm_storeu_si128((__m128i *)t, v);
    uint8_t *dst = (uint8_t *)p;
    for (int i = 0; i < n_bytes; i++) {
        if (m[i]) dst[i] = t[i];
    }
}

static inline __m128i mm_masked_load_bytes(const void *p, __m128i mask, int n_bytes) {
    __m128i v = mm_loadu_partial_bytes(p, n_bytes);
    return _mm_and_si128(v, mask);
}

static inline __m128i mm_pmulds_epi32(__m128i a, __m128i b) {
    // In this file, b is always a broadcast power-of-two (mask = 1<<shift).
    // Implement as saturating left shift by log2(b).
    ALIGN16 int32_t av[4], bv[4], rv[4];
    _mm_storeu_si128((__m128i *)av, a);
    _mm_storeu_si128((__m128i *)bv, b);

    uint32_t m = (uint32_t)bv[0];
    int sh = -1;
    if (m != 0 && (m & (m - 1)) == 0) {
#if defined(__GNUC__) || defined(__clang__)
        sh = __builtin_ctz(m);
#else
        sh = 0; while (((m >> sh) & 1u) == 0u) sh++;
#endif
    }

    for (int i = 0; i < 4; i++) {
        if (sh >= 0) {
            int64_t v = (int64_t)av[i] << sh;
            if (v > INT32_MAX) v = INT32_MAX;
            if (v < INT32_MIN) v = INT32_MIN;
            rv[i] = (int32_t)v;
        } else {
            int64_t prod = (int64_t)av[i] * (int64_t)bv[i];
            rv[i] = (int32_t)prod;
        }
    }
    return _mm_loadu_si128((const __m128i *)rv);
}

static inline __m128i mm_pmuldfrs_epi32(__m128i a, __m128i b) {
    // Fixed-point multiply with rounding, returning (a*b + 2^30) >> 31 per lane.
    ALIGN16 int32_t av[4], bv[4], rv[4];
    _mm_storeu_si128((__m128i *)av, a);
    _mm_storeu_si128((__m128i *)bv, b);
    for (int i = 0; i < 4; i++) {
        int64_t mult = (int64_t)1 << 30;
        if ((av[i] < 0) ^ (bv[i] < 0)) mult = 1 - mult;
        mult += (int64_t)av[i] * (int64_t)bv[i];
        int64_t res = mult / ((int64_t)1 << 31);
        if (av[i] == bv[i] && av[i] == (int32_t)INT32_MIN) {
            rv[i] = INT32_MAX;
        } else {
            rv[i] = (int32_t)res;
        }
    }
    return _mm_loadu_si128((const __m128i *)rv);
}

static int32_t softmax_doubling_high_mult(const int32_t m1, const int32_t m2) {
    int64_t mult = ((int64_t)1 << 30);
    if ((m1 < 0) ^ (m2 < 0)) {
        mult = 1 - mult;
    }
    mult = mult + (int64_t)m1 * (int64_t)m2;
    int32_t result = (int32_t)(mult / ((int64_t)1 << 31));
    if ((m1 == m2) && (m1 == (int32_t)INT32_MIN)) {
        result = INT32_MAX;
    }
    return result;
}

static __m128i softmax_divide_by_power_of_two(const __m128i dividend, const int32_t exponent) {
    const int32_t fixup = 1 << (exponent - 1);
    __m128i fixup_128 = _mm_set1_epi32(fixup);
    __m128i dividend_fixup = _mm_add_epi32(dividend, fixup_128);
    return _mm_srli_epi32(dividend_fixup, exponent);
}

static __m128i softmax_exp_on_negative_values(const __m128i val) {
    int32_t shift = 24;

    const __m128i val_mod_minus_quarter =
        _mm_sub_epi32(_mm_and_si128(val, _mm_set1_epi32((1 << shift) - 1)), _mm_set1_epi32(1 << shift));
    const __m128i remainder = _mm_sub_epi32(val_mod_minus_quarter, val);
    const __m128i x = _mm_add_epi32(_mm_slli_epi32(val_mod_minus_quarter, 5), _mm_set1_epi32(1 << 28));

    const __m128i x2 = mm_pmuldfrs_epi32(x, x);
    const __m128i op_1 = _mm_add_epi32(
        softmax_divide_by_power_of_two(mm_pmuldfrs_epi32(x2, x2), 2),
        mm_pmuldfrs_epi32(x2, x));
    const __m128i op_2 = _mm_add_epi32(
        x,
        softmax_divide_by_power_of_two(_mm_add_epi32(mm_pmuldfrs_epi32(op_1, _mm_set1_epi32(715827883)), x2), 1));

    __m128i result = _mm_add_epi32(
        _mm_set1_epi32(1895147668),
        mm_pmuldfrs_epi32(_mm_set1_epi32(1895147668), op_2));

#define SELECT_IF_NON_ZERO(xc)                                                                                  \
    {                                                                                                            \
        __m128i msk = _mm_xor_si128(                                                                             \
            _mm_cmpeq_epi32(_mm_and_si128(remainder, _mm_set1_epi32(1 << shift++)), _mm_setzero_si128()),        \
            _mm_set1_epi32(-1));                                                                                 \
        result = _mm_xor_si128(                                                                                  \
            _mm_and_si128(msk, mm_pmuldfrs_epi32(result, _mm_set1_epi32(xc))),                                   \
            _mm_and_si128(_mm_xor_si128(msk, _mm_set1_epi32(-1)), result));                                      \
    }

    SELECT_IF_NON_ZERO(1672461947)
    SELECT_IF_NON_ZERO(1302514674)
    SELECT_IF_NON_ZERO(790015084)
    SELECT_IF_NON_ZERO(290630308)
    SELECT_IF_NON_ZERO(39332535)
    SELECT_IF_NON_ZERO(720401)
    SELECT_IF_NON_ZERO(242)

#undef SELECT_IF_NON_ZERO

    __m128i mask0 = _mm_cmpeq_epi32(val, _mm_setzero_si128());
    result = _mm_xor_si128(
        _mm_and_si128(mask0, _mm_set1_epi32(INT32_MAX)),
        _mm_and_si128(_mm_xor_si128(mask0, _mm_set1_epi32(-1)), result));

    return result;
}

static int32_t softmax_mult_by_power_of_two(const int32_t val, const int32_t exp) {
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t result = val << exp;
    result = SOFTMAX_SELECT_USING_MASK(SOFTMAX_MASK_IF_NON_ZERO(val > thresh), INT32_MAX, result);
    result = SOFTMAX_SELECT_USING_MASK(SOFTMAX_MASK_IF_NON_ZERO(val < -thresh), INT32_MIN, result);
    return result;
}

static int32_t softmax_one_over_one_plus_x_for_x_in_0_1(const int32_t val) {
    const int64_t sum = (int64_t)val + (int64_t)INT32_MAX;
    const int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2L);
    int32_t x = 1515870810 + softmax_doubling_high_mult(half_denominator, -1010580540);

    const int32_t shift = (1 << 29);
    x += softmax_mult_by_power_of_two(softmax_doubling_high_mult(x, shift - softmax_doubling_high_mult(half_denominator, x)), 2);
    x += softmax_mult_by_power_of_two(softmax_doubling_high_mult(x, shift - softmax_doubling_high_mult(half_denominator, x)), 2);
    x += softmax_mult_by_power_of_two(softmax_doubling_high_mult(x, shift - softmax_doubling_high_mult(half_denominator, x)), 2);

    return softmax_mult_by_power_of_two(x, 1);
}

int nn_softmax_s8(const nn_context *ctx,
                  const nn_dims *input_dims,
                  const int8_t *input_data,
                  const int32_t input_multiplier,
                  const int32_t input_left_shift,
                  const int32_t diff_min,
                  int8_t *output_data) {
    if (ctx == NULL || input_dims == NULL || input_data == NULL || output_data == NULL) return -1;
    if (ctx->scratch_buf == NULL) return -1;

    const int32_t num_rows = input_dims->n;
    const int32_t row_size = input_dims->c;
    const int32_t mult = input_multiplier;
    const int32_t shift = input_left_shift;

    const int32_t mask = (1 << shift);

    for (int32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        __m128i max_comp_0;
        int32_t *exp_buff = (int32_t *)ctx->scratch_buf;

        uint32_t row_num_16x = (uint32_t)row_size >> 4;
        uint32_t row_num_left = (uint32_t)row_size & 15u;
        __m128i row_left_mask = _mm_setzero_si128();
        if (row_num_left) {
            for (uint32_t i = 0; i < row_num_left; i++) {
                ((int8_t *)&row_left_mask)[i] = -1;
            }
        }

        /* START 求最大值 */
        int8_t *input_row_0 = (int8_t *)input_data + row_idx * row_size;
        if (row_num_16x == 0) {
            __m128i max_comp_tmp = mm_masked_load_bytes(input_row_0, row_left_mask, (int)row_num_left);
            max_comp_0 = _mm_set1_epi8(-128);
            max_comp_0 = _mm_or_si128(_mm_and_si128(max_comp_tmp, row_left_mask), _mm_andnot_si128(row_left_mask, max_comp_0));
        } else {
            __m128i max_comp_1;
            max_comp_0 = mm_lddqu_sse2((const void *)input_row_0);
            input_row_0 += 16;
            for (uint32_t i = 1; i < row_num_16x; i++) {
                max_comp_1 = mm_lddqu_sse2((const void *)input_row_0);
                max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_1);
                input_row_0 += 16;
            }
            if (row_num_left) {
                __m128i max_comp_tmp = mm_masked_load_bytes(input_row_0, row_left_mask, (int)row_num_left);
                max_comp_1 = _mm_set1_epi8(-128);
                max_comp_1 = _mm_or_si128(_mm_and_si128(max_comp_tmp, row_left_mask), _mm_andnot_si128(row_left_mask, max_comp_1));
                max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_1);
            }
        }
        __m128i max_comp_2 = _mm_srli_si128(max_comp_0, 8);
        max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_2);
        max_comp_2 = _mm_srli_si128(max_comp_0, 4);
        max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_2);
        max_comp_2 = _mm_srli_si128(max_comp_0, 2);
        max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_2);
        max_comp_2 = _mm_srli_si128(max_comp_0, 1);
        max_comp_0 = mm_max_epi8_sse2(max_comp_0, max_comp_2);
        int8_t maxv = ((int8_t *)&max_comp_0)[0];
        /* END 求最大值 */

        /* START 指数求和 */
        int32_t sum = 0;
        uint32_t count_4x = (uint32_t)row_size >> 2;
        uint32_t count_left = (uint32_t)row_size & 3u;
        __m128i count_left_mask_8 = _mm_setzero_si128();
        __m128i count_left_mask_32 = _mm_setzero_si128();
        if (count_left) {
            for (uint32_t i = 0; i < count_left; i++) {
                ((int8_t *)&count_left_mask_8)[i] = -1;
                ((int32_t *)&count_left_mask_32)[i] = -1;
            }
        }
        __m128i count_load_mask = _mm_setzero_si128();
        for (uint32_t i = 0; i < 4; i++) {
            ((int8_t *)&count_load_mask)[i] = -1;
        }

        int8_t *input_row_1 = (int8_t *)input_data + row_idx * row_size;
        int32_t *exp_buff_1 = (int32_t *)exp_buff;
        for (uint32_t i = 0; i < count_4x; i++) {
            __m128i input_orig = mm_masked_load_bytes(input_row_1, count_load_mask, 4);
            input_row_1 += 4;
            __m128i input_pack = _mm_unpacklo_epi8(input_orig, input_orig);
            input_pack = _mm_unpacklo_epi16(input_pack, input_pack);
            input_pack = _mm_srai_epi32(input_pack, 24);
            __m128i max_128 = _mm_set1_epi32((int32_t)maxv);
            __m128i diff = _mm_sub_epi32(input_pack, max_128);
            __m128i diff_min_128 = _mm_set1_epi32((int32_t)diff_min);
            __m128i cmp_diff = _mm_cmpgt_epi32(diff, diff_min_128);
            __m128i cmp = _mm_cmpeq_epi32(cmp_diff, _mm_setzero_si128());
            if (_mm_movemask_epi8(cmp) != 0xFFFF) {
                __m128i mask_128 = _mm_set1_epi32((int32_t)mask);
                __m128i result = mm_pmulds_epi32(diff, mask_128);
                __m128i mult_128 = _mm_set1_epi32((int32_t)mult);
                __m128i res = mm_pmuldfrs_epi32(result, mult_128);
                res = softmax_exp_on_negative_values(res); /* exp(x - max) */
                _mm_storeu_si128((__m128i *)exp_buff_1, res);

                res = softmax_divide_by_power_of_two(res, SOFTMAX_ACCUM_BITS);
                res = mm_hadd_epi32_sse2(res);
                sum += ((int32_t *)&res)[0];
            } else {
                _mm_storeu_si128((__m128i *)exp_buff_1, _mm_setzero_si128());
            }
            exp_buff_1 += 4;
        }

        if (count_left) {
            __m128i input_orig = mm_masked_load_bytes(input_row_1, count_left_mask_8, (int)count_left);
            __m128i input_pack = _mm_unpacklo_epi8(input_orig, input_orig);
            input_pack = _mm_unpacklo_epi16(input_pack, input_pack);
            input_pack = _mm_srai_epi32(input_pack, 24);
            __m128i max_128 = _mm_set1_epi32((int32_t)maxv);
            __m128i diff = _mm_sub_epi32(input_pack, max_128);
            __m128i diff_min_128 = _mm_set1_epi32((int32_t)diff_min);
            __m128i cmp_diff = _mm_cmpgt_epi32(diff, diff_min_128);
            __m128i cmp = _mm_cmpeq_epi32(cmp_diff, _mm_setzero_si128());
            if (_mm_movemask_epi8(cmp) != 0xFFFF) {
                __m128i mask_128 = _mm_set1_epi32((int32_t)mask);
                __m128i result = mm_pmulds_epi32(diff, mask_128);
                __m128i mult_128 = _mm_set1_epi32((int32_t)mult);
                __m128i res = mm_pmuldfrs_epi32(result, mult_128);
                res = softmax_exp_on_negative_values(res);
                mm_masked_store_bytes(exp_buff_1, count_left_mask_32, (int)(count_left * 4), res);

                res = softmax_divide_by_power_of_two(res, SOFTMAX_ACCUM_BITS);
                for (uint32_t i = 0; i < count_left; i++) {
                    sum += ((int32_t *)&res)[i];
                }
            } else {
                mm_masked_store_bytes(exp_buff_1, count_left_mask_32, (int)(count_left * 4), _mm_setzero_si128());
            }
        }

        /* END 指数求和 */

        /* START 求倒数 */
        const int32_t headroom = 31 - bsr_u32((uint32_t)sum);
        const int32_t bits_over_unit = SOFTMAX_ACCUM_BITS - headroom + 23;
        const int32_t shifted_scale =
            softmax_one_over_one_plus_x_for_x_in_0_1((sum > 0 ? sum << headroom : 0) - (1 << 31));
        /* END 求倒数 */

        /* START 归一化并量化输出 */
        int32_t *exp_buff_2 = (int32_t *)exp_buff;
        int8_t *out_per = output_data + row_idx * row_size;

        for (uint32_t i = 0; i < count_4x; i++) {
            __m128i exp_data = mm_lddqu_sse2((const void *)exp_buff_2);
            __m128i cmp = _mm_cmpeq_epi32(exp_data, _mm_setzero_si128());
            if (_mm_movemask_epi8(cmp) != 0xFFFF) {
                __m128i result = mm_pmuldfrs_epi32(_mm_set1_epi32(shifted_scale), exp_data);
                result = softmax_divide_by_power_of_two(result, bits_over_unit);
                result = _mm_sub_epi32(result, _mm_set1_epi32(128));
                result = _mm_packs_epi32(result, result);
                result = _mm_packs_epi16(result, result);
                mm_storeu_si32_sse2(out_per, result);
            } else {
                mm_storeu_si32_sse2(out_per, _mm_set1_epi8(-128));
            }
            out_per += 4;
            exp_buff_2 += 4;
        }

        if (count_left) {
            __m128i exp_data = mm_masked_load_bytes(exp_buff_2, count_left_mask_32, (int)(count_left * 4));
            __m128i cmp = _mm_cmpeq_epi32(exp_data, _mm_setzero_si128());
            if (_mm_movemask_epi8(cmp) != 0xFFFF) {
                __m128i result = mm_pmuldfrs_epi32(_mm_set1_epi32(shifted_scale), exp_data);
                result = softmax_divide_by_power_of_two(result, bits_over_unit);
                result = _mm_sub_epi32(result, _mm_set1_epi32(128));
                result = _mm_packs_epi32(result, result);
                result = _mm_packs_epi16(result, result);
                mm_masked_store_bytes(out_per, count_left_mask_8, (int)count_left, result);
            } else {
                mm_masked_store_bytes(out_per, count_left_mask_8, (int)count_left, _mm_set1_epi8(-128));
            }
        }
        /* END 归一化并量化输出 */
    }

    return 0;
}

int nn_softmax_s8_get_scratch_buffer_size(const nn_dims *input_dims, uint32_t *buffer_size) {
    if (input_dims == NULL || buffer_size == NULL) return -1;
    *buffer_size = (((uint32_t)(input_dims->c + 3) >> 2) << 2) * (uint32_t)sizeof(int32_t) + 64u;
    return 0;
}

int main(void) {
    nn_dims input_dims;
    input_dims.n = SOFTMAX1_INPUT_BATCHES;
    input_dims.w = SOFTMAX1_INPUT_W;
    input_dims.h = SOFTMAX1_INPUT_H;
    input_dims.c = SOFTMAX1_IN_CH;

    uint32_t scratch_buf_size = 0;
    if (nn_softmax_s8_get_scratch_buffer_size(&input_dims, &scratch_buf_size) != 0) {
        fprintf(stderr, "scratch size calc failed\n");
        return 1;
    }

    nn_context ctx;
    ctx.scratch_buf_size = scratch_buf_size;
    ctx.scratch_buf = malloc(scratch_buf_size);
    if (!ctx.scratch_buf) {
        fprintf(stderr, "malloc scratch failed\n");
        return 1;
    }

    int8_t *output = (int8_t *)malloc(SOFTMAX1_DST_SIZE);
    if (!output) {
        fprintf(stderr, "malloc output failed\n");
        free(ctx.scratch_buf);
        return 1;
    }

    int rc = nn_softmax_s8(&ctx, &input_dims,
                          softmax1_input,
                          softmax1_input_multiplier,
                          softmax1_input_left_shift,
                          softmax1_diff_min,
                          output);
    if (rc != 0) {
        fprintf(stderr, "nn_softmax_s8 failed: %d\n", rc);
        free(output);
        free(ctx.scratch_buf);
        return 1;
    }

    int ok = 1;
    printf("softmax output: ");
    for (int i = 0; i < SOFTMAX1_DST_SIZE; i++) {
        printf("%d, ", output[i]);
    }
    printf("\n");

    free(output);
    free(ctx.scratch_buf);


    printf("Test Online Softmax\n");
    float x[5] = {1, 2, 3, 4, 5};
    float yref[5] = { 0 };
    float y2pa[5] = { 0 };
    softmax_3pass(x, 5, yref);
    softmax_2pass(x, 5, y2pa);
    printf("Input\t3Pass\t2Pass\n");
    for (int i = 0; i < 5; ++i) {
        printf("%.3f\t%.3f\t%.3f\n", x[i], yref[i], y2pa[i]);
    }
    printf("------------------------\n");
    return 0;
}
