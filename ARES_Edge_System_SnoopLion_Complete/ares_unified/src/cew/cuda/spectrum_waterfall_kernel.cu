/**
 * @file spectrum_waterfall_kernel.cu
 * @brief CUDA kernels for real-time spectrum waterfall analysis
 * 
 * Optimized for 2 GSPS processing with minimal latency
 */

#include "../include/spectrum_waterfall.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <float.h>

namespace ares::cew {

constexpr uint32_t BLOCK_SIZE = 256;
const float LOG_SCALE = 2.302585f; // log(10);  // For dB conversion

/**
 * @brief Apply window function to IQ samples
 * Reduces spectral leakage for better signal detection
 */
__global__ void apply_window_kernel(
    const float2* __restrict__ iq_samples,
    const float* __restrict__ window,
    cufftComplex* __restrict__ windowed_output,
    uint32_t fft_size
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < fft_size) {
        // Load IQ sample and window coefficient
        float2 sample = iq_samples[tid];
        float w = window[tid];
        
        // Apply window
        windowed_output[tid].x = sample.x * w;
        windowed_output[tid].y = sample.y * w;
    }
}

/**
 * @brief Compute magnitude spectrum and convert to dB
 * Optimized with vectorized operations
 */
__global__ void compute_magnitude_spectrum_kernel(
    const cufftComplex* __restrict__ fft_output,
    float* __restrict__ magnitude,
    float* __restrict__ spectrum_db,
    uint32_t spectrum_bins
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half_bins = spectrum_bins / 2 + 1;
    
    if (tid < half_bins) {
        // Load FFT output
        cufftComplex val = fft_output[tid];
        
        // Compute magnitude squared
        float mag_sq = val.x * val.x + val.y * val.y;
        
        // Normalize by FFT size
        mag_sq /= (float)(spectrum_bins * spectrum_bins);
        
        // Store magnitude
        magnitude[tid] = sqrtf(mag_sq);
        
        // Convert to dB with numerical stability
        float db_val;
        if (mag_sq > FLT_MIN) {
            db_val = ares::cew::LOG_SCALE * logf(mag_sq);
        } else {
            db_val = -140.0f;  // Floor value
        }
        
        spectrum_db[tid] = db_val;
        
        // Mirror for negative frequencies (except DC and Nyquist)
        if (tid > 0 && tid < half_bins - 1) {
            uint32_t mirror_idx = spectrum_bins - tid;
            magnitude[mirror_idx] = magnitude[tid];
            spectrum_db[mirror_idx] = db_val;
        }
    }
}

/**
 * @brief Update waterfall display with circular buffer
 */
__global__ void update_waterfall_kernel(
    const float* __restrict__ spectrum_db,
    float* __restrict__ waterfall,
    uint32_t waterfall_index,
    uint32_t spectrum_bins,
    uint32_t history_depth
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < spectrum_bins) {
        // Calculate position in circular buffer
        uint32_t row = waterfall_index % history_depth;
        uint32_t idx = row * spectrum_bins + tid;
        
        // Update waterfall
        waterfall[idx] = spectrum_db[tid];
    }
}

/**
 * @brief Detect signals above noise floor using parallel reduction
 */
__global__ void detect_signals_kernel(
    const float* __restrict__ spectrum_db,
    const float* __restrict__ noise_floor,
    DetectedSignal* __restrict__ signals,
    uint32_t* __restrict__ signal_count,
    float threshold_db,
    uint32_t min_bins,
    uint32_t spectrum_bins
) {
    // Shared memory for collaborative signal detection
    __shared__ uint32_t shared_signal_start[BLOCK_SIZE];
    __shared__ uint32_t shared_signal_end[BLOCK_SIZE];
    __shared__ float shared_peak_power[BLOCK_SIZE];
    __shared__ uint32_t block_signal_count;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid == 0) {
        block_signal_count = 0;
    }
    __syncthreads();
    
    // Initialize per-thread detection state
    bool in_signal = false;
    uint32_t signal_start = 0;
    uint32_t signal_end = 0;
    float peak_power = -200.0f;
    float total_power = 0.0f;
    
    // Scan spectrum for signals
    for (uint32_t bin = gid; bin < spectrum_bins; bin += gridDim.x * blockDim.x) {
        float power = spectrum_db[bin];
        float noise = noise_floor[bin];
        float snr = power - noise;
        
        if (snr > threshold_db) {
            // Above threshold
            if (!in_signal) {
                // Start of new signal
                in_signal = true;
                signal_start = bin;
                peak_power = power;
                total_power = power;
            } else {
                // Continue signal
                peak_power = fmaxf(peak_power, power);
                total_power += power;
            }
            signal_end = bin;
        } else if (in_signal) {
            // End of signal
            in_signal = false;
            
            // Check if signal is wide enough
            uint32_t width = signal_end - signal_start + 1;
            if (width >= min_bins) {
                // Valid signal detected
                uint32_t local_idx = atomicAdd(&block_signal_count, 1);
                
                if (local_idx < BLOCK_SIZE) {
                    shared_signal_start[local_idx] = signal_start;
                    shared_signal_end[local_idx] = signal_end;
                    shared_peak_power[local_idx] = peak_power;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Block leader writes results
    if (tid == 0 && block_signal_count > 0) {
        uint32_t write_count = min(block_signal_count, (uint32_t)BLOCK_SIZE);
        uint32_t global_idx = atomicAdd(signal_count, write_count);
        
        for (uint32_t i = 0; i < write_count; ++i) {
            if (global_idx + i < 128) {  // Max signals limit
                DetectedSignal& sig = signals[global_idx + i];
                
                uint32_t start = shared_signal_start[i];
                uint32_t end = shared_signal_end[i];
                uint32_t center = (start + end) / 2;
                uint32_t width = end - start + 1;
                
                // Convert bins to frequency
                float bin_width_mhz = SAMPLE_RATE_MSPS / FFT_SIZE;
                sig.center_freq_mhz = center * bin_width_mhz;
                sig.bandwidth_mhz = width * bin_width_mhz;
                sig.power_dbm = shared_peak_power[i];
                sig.snr_db = shared_peak_power[i] - noise_floor[center];
                sig.start_bin = start;
                sig.end_bin = end;
                sig.confidence = min(100, (uint8_t)(sig.snr_db * 2));  // Simple confidence
            }
        }
    }
}

/**
 * @brief Estimate noise floor using statistical methods
 * Uses parallel reduction for efficiency
 */
__global__ void estimate_noise_floor_kernel(
    const float* __restrict__ waterfall,
    float* __restrict__ noise_floor,
    uint32_t spectrum_bins,
    uint32_t history_depth
) {
    extern __shared__ float shared_data[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bin = blockIdx.x;
    
    if (bin >= spectrum_bins) return;
    
    // Each block processes one frequency bin across time
    float sum = 0.0f;
    float sum_sq = 0.0f;
    uint32_t count = 0;
    
    // Load and accumulate statistics
    for (uint32_t t = tid; t < history_depth; t += blockDim.x) {
        float val = waterfall[t * spectrum_bins + bin];
        if (val > -140.0f) {  // Valid value
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    
    // Store in shared memory
    shared_data[tid] = sum;
    shared_data[tid + blockDim.x] = sum_sq;
    shared_data[tid + 2 * blockDim.x] = (float)count;
    __syncthreads();
    
    // Parallel reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
            shared_data[tid + blockDim.x] += shared_data[tid + s + blockDim.x];
            shared_data[tid + 2 * blockDim.x] += shared_data[tid + s + 2 * blockDim.x];
        }
        __syncthreads();
    }
    
    // Thread 0 computes final statistics
    if (tid == 0) {
        float total_sum = shared_data[0];
        float total_sum_sq = shared_data[blockDim.x];
        float total_count = shared_data[2 * blockDim.x];
        
        if (total_count > 0) {
            float mean = total_sum / total_count;
            float variance = (total_sum_sq / total_count) - (mean * mean);
            float std_dev = sqrtf(fmaxf(0.0f, variance));
            
            // Noise floor estimate: mean - 2*sigma (conservative)
            // This captures ~95% of noise samples
            noise_floor[bin] = mean - 2.0f * std_dev;
        } else {
            noise_floor[bin] = NOISE_FLOOR_DBM;
        }
    }
}

/**
 * @brief Generate window function coefficients
 */
__global__ void generate_window_kernel(
    float* __restrict__ window,
    uint32_t size,
    WindowType type
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        float n = (float)tid;
        float N = (float)size;
        float w = 1.0f;
        
        switch (type) {
            case WindowType::RECTANGULAR:
                w = 1.0f;
                break;
                
            case WindowType::HANNING:
                w = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (N - 1.0f)));
                break;
                
            case WindowType::HAMMING:
                w = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (N - 1.0f));
                break;
                
            case WindowType::BLACKMAN:
                w = 0.42f - 0.5f * cosf(2.0f * M_PI * n / (N - 1.0f)) +
                    0.08f * cosf(4.0f * M_PI * n / (N - 1.0f));
                break;
                
            case WindowType::KAISER: {
                // Kaiser window with beta = 8.6 (good sidelobe suppression)
                float beta = 8.6f;
                float alpha = (N - 1.0f) / 2.0f;
                float bessel_arg = beta * sqrtf(1.0f - powf((n - alpha) / alpha, 2.0f));
                
                // Approximate modified Bessel function I0
                float i0_beta = 1.0f;
                float term = 1.0f;
                for (int k = 1; k < 20; ++k) {
                    term *= (bessel_arg / (2.0f * k)) * (bessel_arg / (2.0f * k));
                    i0_beta += term;
                }
                
                float i0_beta_max = 1.0f;
                term = 1.0f;
                for (int k = 1; k < 20; ++k) {
                    term *= (beta / (2.0f * k)) * (beta / (2.0f * k));
                    i0_beta_max += term;
                }
                
                w = i0_beta / i0_beta_max;
                break;
            }
                
            case WindowType::FLAT_TOP:
                // Flat-top window for accurate amplitude measurement
                w = 0.21557895f - 0.41663158f * cosf(2.0f * M_PI * n / (N - 1.0f)) +
                    0.277263158f * cosf(4.0f * M_PI * n / (N - 1.0f)) -
                    0.083578947f * cosf(6.0f * M_PI * n / (N - 1.0f)) +
                    0.006947368f * cosf(8.0f * M_PI * n / (N - 1.0f));
                break;
        }
        
        window[tid] = w;
    }
}

} // namespace ares::cew
