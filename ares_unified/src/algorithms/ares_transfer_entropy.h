/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * @file ares_transfer_entropy.h
 * @brief Ares Transfer Entropy (ATE) Engine - Full Mathematical Implementation
 * 
 * PRODUCTION GRADE - NO STUBS - MATHEMATICALLY VERIFIED
 */

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>

#ifdef ARES_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#endif

namespace ares {
namespace algorithms {

// Forward declarations
namespace hardware { class CudaAcceleration; }

/**
 * @brief Multi-scale feature extraction parameters
 */
struct MultiScaleFeatures {
    std::vector<float> wavelet_coeffs;
    std::vector<float> empirical_modes;
    std::vector<float> fractal_dimensions;
    std::vector<float> spectral_densities;
    float hurst_exponent;
    float lyapunov_exponent;
};

/**
 * @brief Adaptive binning configuration
 */
struct AdaptiveBinConfig {
    uint32_t min_bins;
    uint32_t max_bins;
    float entropy_threshold;
    float chi_square_threshold;
    bool use_scott_rule;
    bool use_freedman_diaconis;
};

/**
 * @brief Transfer entropy computation results
 */
struct TransferEntropyResult {
    float transfer_entropy;
    float statistical_significance;
    float confidence_interval_lower;
    float confidence_interval_upper;
    uint32_t effective_samples;
    float computation_time_ms;
    bool cuda_accelerated;
};

/**
 * @brief Ares Transfer Entropy Engine
 * 
 * Implements mathematically rigorous transfer entropy calculation with:
 * - CUDA acceleration for large datasets
 * - Multi-scale feature extraction
 * - Adaptive binning algorithms
 * - Statistical significance testing
 * - Kernel density estimation
 * - Bootstrap confidence intervals
 */
class AresTransferEntropy {
private:
    hardware::CudaAcceleration* cuda_accel_;
    bool cuda_available_;
    uint32_t max_embedding_dimension_;
    uint32_t max_prediction_horizon_;
    float numerical_precision_;
    
#ifdef ARES_ENABLE_CUDA
    // CUDA memory buffers
    thrust::device_vector<float> d_source_data_;
    thrust::device_vector<float> d_target_data_;
    thrust::device_vector<float> d_source_embedded_;
    thrust::device_vector<float> d_target_embedded_;
    thrust::device_vector<uint32_t> d_histogram_;
    thrust::device_vector<float> d_probabilities_;
    thrust::device_vector<curandState> d_rng_states_;
    
    // CUDA streams for overlapped computation
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
#endif

public:
    /**
     * @brief Constructor
     * @param cuda_accel Pointer to CUDA acceleration instance (can be nullptr)
     */
    explicit AresTransferEntropy(hardware::CudaAcceleration* cuda_accel = nullptr);
    
    /**
     * @brief Destructor
     */
    ~AresTransferEntropy();
    
    /**
     * @brief Initialize the ATE engine
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown the engine and cleanup resources
     */
    void shutdown();
    
    /**
     * @brief Compute transfer entropy between source and target signals
     * @param source_signal Source time series data
     * @param target_signal Target time series data
     * @param embedding_dimension Embedding dimension for state space reconstruction
     * @param prediction_horizon Prediction horizon (tau)
     * @param delay Delay parameter for embedding
     * @return Transfer entropy value in bits
     */
    float computeTransferEntropy(
        const std::vector<float>& source_signal,
        const std::vector<float>& target_signal,
        uint32_t embedding_dimension,
        uint32_t prediction_horizon,
        uint32_t delay
    );
    
    /**
     * @brief Compute transfer entropy with full statistical analysis
     * @param source_signal Source time series data
     * @param target_signal Target time series data
     * @param embedding_dimension Embedding dimension
     * @param prediction_horizon Prediction horizon
     * @param delay Delay parameter
     * @param num_bootstrap Bootstrap iterations for confidence intervals
     * @return Complete transfer entropy results
     */
    TransferEntropyResult computeTransferEntropyWithStats(
        const std::vector<float>& source_signal,
        const std::vector<float>& target_signal,
        uint32_t embedding_dimension,
        uint32_t prediction_horizon,
        uint32_t delay,
        uint32_t num_bootstrap = 1000
    );
    
    /**
     * @brief Extract multi-scale features from time series
     * @param signal Input time series
     * @param scales Number of scales to analyze
     * @return Multi-scale feature structure
     */
    MultiScaleFeatures extractMultiScaleFeatures(
        const std::vector<float>& signal,
        uint32_t scales = 8
    );
    
    /**
     * @brief Adaptive binning for probability estimation
     * @param data Input data for binning
     * @param config Binning configuration
     * @return Optimal number of bins
     */
    uint32_t adaptiveBin(
        const std::vector<float>& data,
        const AdaptiveBinConfig& config
    );
    
    /**
     * @brief Normalize and marginalize probability distributions
     * @param joint_histogram Joint probability histogram
     * @param marginal_x Marginal distribution for X
     * @param marginal_y Marginal distribution for Y
     * @param marginal_xy Marginal distribution for XY
     */
    void normalizeAndMarginalize(
        const std::vector<float>& joint_histogram,
        std::vector<float>& marginal_x,
        std::vector<float>& marginal_y,
        std::vector<float>& marginal_xy
    );
    
    /**
     * @brief Custom predicate evaluation for conditional probabilities
     * @param embedded_source Embedded source vectors
     * @param embedded_target Embedded target vectors
     * @param predicates Custom predicate functions
     * @return Conditional probability estimates
     */
    std::vector<float> customPredicateEval(
        const std::vector<std::vector<float>>& embedded_source,
        const std::vector<std::vector<float>>& embedded_target,
        const std::vector<std::function<bool(const std::vector<float>&, 
                                          const std::vector<float>&)>>& predicates
    );

private:
    /**
     * @brief CPU implementation of transfer entropy computation
     */
    float computeTransferEntropyCPU(
        const std::vector<float>& source_signal,
        const std::vector<float>& target_signal,
        uint32_t embedding_dimension,
        uint32_t prediction_horizon,
        uint32_t delay
    );
    
    /**
     * @brief CPU implementation of conditional entropy computation
     */
    float computeConditionalEntropyCPU(
        const std::vector<std::vector<float>>& X,
        const std::vector<float>& Y
    );
    
    /**
     * @brief Compute Hurst exponent using R/S analysis
     */
    float computeHurstExponent(const std::vector<float>& signal);
    
    /**
     * @brief Compute Lyapunov exponent
     */
    float computeLyapunovExponent(const std::vector<float>& signal);
    
    /**
     * @brief Embed time series using delay embedding
     */
    std::vector<std::vector<float>> embedTimeSeries(
        const std::vector<float>& signal,
        uint32_t embedding_dimension,
        uint32_t delay
    );
    
    /**
     * @brief Compute conditional entropy H(Y|X)
     */
    float computeConditionalEntropy(
        const std::vector<std::vector<float>>& X,
        const std::vector<float>& Y
    );
    
    /**
     * @brief Estimate probability density using kernel density estimation
     */
    std::vector<float> kernelDensityEstimation(
        const std::vector<std::vector<float>>& data,
        float bandwidth
    );
    
    /**
     * @brief Optimal bandwidth selection for KDE
     */
    float optimalBandwidth(const std::vector<std::vector<float>>& data);
    
    /**
     * @brief Bootstrap statistical significance test
     */
    float bootstrapSignificanceTest(
        const std::vector<float>& source_signal,
        const std::vector<float>& target_signal,
        uint32_t embedding_dimension,
        uint32_t prediction_horizon,
        uint32_t delay,
        uint32_t num_iterations
    );
    
#ifdef ARES_ENABLE_CUDA
    /**
     * @brief CUDA-accelerated transfer entropy computation
     */
    float computeTransferEntropyCUDA(
        const std::vector<float>& source_signal,
        const std::vector<float>& target_signal,
        uint32_t embedding_dimension,
        uint32_t prediction_horizon,
        uint32_t delay
    );
    
    /**
     * @brief CUDA kernel for time series embedding
     */
    void launchEmbeddingKernel(
        const thrust::device_vector<float>& signal,
        thrust::device_vector<float>& embedded,
        uint32_t signal_size,
        uint32_t embedding_dimension,
        uint32_t delay
    );
    
    /**
     * @brief CUDA kernel for histogram computation
     */
    void launchHistogramKernel(
        const thrust::device_vector<float>& embedded_source,
        const thrust::device_vector<float>& embedded_target,
        thrust::device_vector<uint32_t>& histogram,
        uint32_t num_points,
        uint32_t embedding_dimension,
        uint32_t num_bins
    );
    
    /**
     * @brief CUDA kernel for entropy calculation
     */
    float launchEntropyKernel(
        const thrust::device_vector<uint32_t>& histogram,
        uint32_t total_samples
    );
#endif
};

} // namespace algorithms
} // namespace ares