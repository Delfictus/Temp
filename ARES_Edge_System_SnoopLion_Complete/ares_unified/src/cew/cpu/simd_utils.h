/**
 * @file simd_utils.h
 * @brief SIMD utility functions for CPU implementation
 */

#ifndef ARES_CEW_SIMD_UTILS_H
#define ARES_CEW_SIMD_UTILS_H

#include <immintrin.h>
#include <cmath>

namespace ares::cew {

// Fast sine approximation for SIMD
inline __m256 _mm256_sin_ps(__m256 x) {
    // Normalize to [-pi, pi]
    const __m256 inv_2pi = _mm256_set1_ps(0.15915494309189535f);
    const __m256 two_pi = _mm256_set1_ps(6.283185307179586f);
    
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, inv_2pi), _MM_FROUND_TO_NEAREST_INT);
    x = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));
    
    // Taylor series approximation
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c3 = _mm256_set1_ps(-1.0f/6.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f/120.0f);
    const __m256 c7 = _mm256_set1_ps(-1.0f/5040.0f);
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x5 = _mm256_mul_ps(x3, x2);
    __m256 x7 = _mm256_mul_ps(x5, x2);
    
    __m256 result = _mm256_mul_ps(x, c1);
    result = _mm256_add_ps(result, _mm256_mul_ps(x3, c3));
    result = _mm256_add_ps(result, _mm256_mul_ps(x5, c5));
    result = _mm256_add_ps(result, _mm256_mul_ps(x7, c7));
    
    return result;
}

// Fast cosine approximation for SIMD
inline __m256 _mm256_cos_ps(__m256 x) {
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    return _mm256_sin_ps(_mm256_add_ps(x, half_pi));
}

// Fast exponential approximation for SIMD
inline __m256 _mm256_exp_ps(__m256 x) {
    // Clamp input
    const __m256 max_val = _mm256_set1_ps(88.0f);
    const __m256 min_val = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);
    
    // exp(x) = 2^(x * log2(e))
    const __m256 log2e = _mm256_set1_ps(1.44269504088896f);
    __m256 y = _mm256_mul_ps(x, log2e);
    
    // Split into integer and fractional parts
    __m256 yi = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT);
    __m256 yf = _mm256_sub_ps(y, yi);
    
    // Polynomial approximation of 2^yf
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.693147f);
    const __m256 c2 = _mm256_set1_ps(0.240227f);
    const __m256 c3 = _mm256_set1_ps(0.0555041f);
    
    __m256 poly = _mm256_mul_ps(yf, c3);
    poly = _mm256_add_ps(poly, c2);
    poly = _mm256_mul_ps(poly, yf);
    poly = _mm256_add_ps(poly, c1);
    poly = _mm256_mul_ps(poly, yf);
    poly = _mm256_add_ps(poly, c0);
    
    // Convert integer part to exponent
    __m256i exp_int = _mm256_cvtps_epi32(yi);
    exp_int = _mm256_add_epi32(exp_int, _mm256_set1_epi32(127));
    exp_int = _mm256_slli_epi32(exp_int, 23);
    
    // Combine
    __m256 result = _mm256_mul_ps(poly, _mm256_castsi256_ps(exp_int));
    return result;
}

} // namespace ares::cew

#endif // ARES_CEW_SIMD_UTILS_H