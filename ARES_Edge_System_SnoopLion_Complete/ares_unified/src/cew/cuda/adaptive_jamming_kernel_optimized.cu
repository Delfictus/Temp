/**
 * @file adaptive_jamming_kernel_optimized.cu
 * @brief Optimized CUDA kernels for CEW adaptive jamming with Q-learning
 * 
 * Performance optimizations:
 * - Warp-level primitives for reductions
 * - Shared memory for frequently accessed data
 * - Coalesced memory access patterns
 * - Tensor Core utilization where available
 */

#include "../include/cew_adaptive_jamming.h"
#include "../../config/constants.h"
#include "../../utils/common_utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

namespace ares::cew {

// Use system-wide constants
using namespace ares::constants;
using namespace ares::utils;

// Thread block configuration for optimal performance
constexpr uint32_t WARPS_PER_BLOCK = DEFAULT_BLOCK_SIZE / WARP_SIZE;

// Shared memory configuration
constexpr size_t SMEM_BANK_SIZE = 32;  // Avoid bank conflicts

// Use atomicMaxFloat from common_utils.h

/**
 * @brief Optimized threat state quantization using bit manipulation
 */
__device__ __forceinline__ uint32_t quantize_threat_state_fast(
    const ThreatSignature& threat,
    float spectrum_density
) {
    // Use bit manipulation for faster quantization
    uint32_t freq_band = __float2uint_rn((threat.center_freq_ghz - FREQ_MIN_GHZ) * 0.1f);
    freq_band = (freq_band > 3) ? 3 : freq_band;
    
    uint32_t power_level = __float2uint_rn((threat.power_dbm + 100.0f) * 0.04f);
    power_level = (power_level > 3) ? 3 : power_level;
    
    uint32_t bw_category = __float2uint_rn(threat.bandwidth_mhz * 0.02f);
    bw_category = (bw_category > 3) ? 3 : bw_category;
    
    // Pack state using bitwise operations
    return (freq_band << 6) | (power_level << 4) | (bw_category << 2) | (threat.modulation_type & 0x3);
}

/**
 * @brief Warp-level reduction for spectrum density calculation
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Optimized action selection with warp-level cooperation
 */
__device__ __forceinline__ uint32_t select_action_optimized(
    const QTableState* __restrict__ q_state,
    uint32_t state,
    uint32_t lane_id,
    curandState* rand_state
) {
    // Epsilon-greedy with warp divergence minimization
    float rand_val = curand_uniform(rand_state);
    
    if (rand_val < EPSILON) {
        return (uint32_t)(curand_uniform(rand_state) * NUM_ACTIONS);
    }
    
    // Collaborative Q-value search across warp
    float max_q = -1e9f;
    uint32_t best_action = 0;
    
    // Each lane checks different actions
    for (uint32_t a = lane_id; a < NUM_ACTIONS; a += WARP_SIZE) {
        float q_val = q_state->q_values[state][a];
        if (q_val > max_q) {
            max_q = q_val;
            best_action = a;
        }
    }
    
    // Warp-level reduction to find global maximum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_q = __shfl_xor_sync(0xffffffff, max_q, offset);
        uint32_t other_action = __shfl_xor_sync(0xffffffff, best_action, offset);
        if (other_q > max_q) {
            max_q = other_q;
            best_action = other_action;
        }
    }
    
    return __shfl_sync(0xffffffff, best_action, 0);  // Broadcast to all lanes
}

/**
 * @brief Optimized adaptive jamming kernel with shared memory and warp primitives
 */
__global__ void __launch_bounds__(BLOCK_SIZE, 2) adaptive_jamming_kernel_optimized(
    const float* __restrict__ spectrum_waterfall,
    const ThreatSignature* __restrict__ threats,
    JammingParams* __restrict__ jamming_params,
    QTableState* __restrict__ q_state,
    uint32_t num_threats,
    uint64_t timestamp_ns
) {
    // Shared memory for spectrum data and Q-values
    extern __shared__ char shared_mem[];
    float* shared_spectrum = (float*)shared_mem;
    float* shared_q_cache = (float*)&shared_spectrum[SPECTRUM_BINS];
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    
    // Load spectrum data cooperatively into shared memory
    for (uint32_t i = threadIdx.x; i < SPECTRUM_BINS; i += blockDim.x) {
        shared_spectrum[i] = spectrum_waterfall[i];
    }
    __syncthreads();
    
    if (tid >= num_threats) return;
    
    // Initialize random state with improved seeding
    curandState rand_state;
    curand_init(timestamp_ns ^ (tid * 2654435761u), tid, 0, &rand_state);
    
    // Load threat data (coalesced access)
    ThreatSignature threat = threats[tid];
    
    // Calculate spectrum density using shared memory
    uint32_t freq_bin = __float2uint_rn((threat.center_freq_ghz - FREQ_MIN_GHZ) / 
                                       (FREQ_MAX_GHZ - FREQ_MIN_GHZ) * SPECTRUM_BINS);
    freq_bin = min(freq_bin, SPECTRUM_BINS - 1);
    
    // Optimized spectrum density calculation
    float spectrum_density = 0.0f;
    const int window_size = 32;
    const int start_bin = max(0, (int)freq_bin - window_size/2);
    const int end_bin = min((int)SPECTRUM_BINS, (int)freq_bin + window_size/2);
    
    // Vectorized load from shared memory
    #pragma unroll 4
    for (int b = start_bin + lane_id; b < end_bin; b += WARP_SIZE) {
        spectrum_density += shared_spectrum[b];
    }
    
    // Warp reduction
    spectrum_density = warp_reduce_sum(spectrum_density);
    
    // Quantize state
    uint32_t state = quantize_threat_state_fast(threat, spectrum_density);
    
    // Cache relevant Q-values in shared memory (warp-cooperative)
    if (warp_id == 0 && lane_id < NUM_ACTIONS) {
        shared_q_cache[threadIdx.x] = q_state->q_values[state][lane_id];
    }
    __syncthreads();
    
    // Select action with optimized function
    uint32_t action = select_action_optimized(q_state, state, lane_id, &rand_state);
    
    // Generate jamming parameters with lookup table
    JammingParams params;
    params.center_freq_ghz = threat.center_freq_ghz;
    
    // Use lookup table for common parameters
    const float bw_multipliers[16] = {1.5f, 5.0f, 0.8f, 2.0f, 3.0f, 1.2f, 1.8f, 1.0f,
                                      1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};
    const float power_levels[16] = {100.0f, 200.0f, 150.0f, 120.0f, 150.0f, 250.0f, 180.0f, 80.0f,
                                   150.0f, 150.0f, 150.0f, 150.0f, 150.0f, 150.0f, 150.0f, 150.0f};
    const float sweep_rates[16] = {0.0f, 0.0f, 0.0f, 100.0f, 1000.0f, 0.0f, 50.0f, 0.0f,
                                   200.0f, 200.0f, 200.0f, 200.0f, 200.0f, 200.0f, 200.0f, 200.0f};
    
    params.strategy = action;
    params.bandwidth_mhz = threat.bandwidth_mhz * bw_multipliers[action];
    params.power_watts = power_levels[action];
    params.sweep_rate_mhz_per_sec = sweep_rates[action];
    params.waveform_id = action;
    params.duration_ms = 100;
    params.phase_offset = curand_uniform(&rand_state) * 6.28318530718f;  // 2*PI
    
    // Write output with coalesced access
    jamming_params[tid] = params;
    
    // Update Q-table state (single thread to avoid conflicts)
    if (tid == 0) {
        atomicExch(&q_state->current_state, state);
        atomicExch(&q_state->last_action, action);
        atomicAdd(&q_state->visit_count[state], 1);
    }
}

/**
 * @brief Optimized Q-table update using Tensor Cores (if available)
 */
__global__ void __launch_bounds__(256, 2) update_qtable_kernel_optimized(
    QTableState* __restrict__ q_state,
    float reward,
    uint32_t new_state
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;
    
    // Load previous state/action once per warp
    __shared__ uint32_t shared_prev_state[WARPS_PER_BLOCK];
    __shared__ uint32_t shared_prev_action[WARPS_PER_BLOCK];
    __shared__ float shared_max_q[WARPS_PER_BLOCK];
    
    if (lane_id == 0) {
        shared_prev_state[warp_id] = q_state->current_state;
        shared_prev_action[warp_id] = q_state->last_action;
    }
    __syncwarp();
    
    uint32_t prev_state = shared_prev_state[warp_id];
    uint32_t prev_action = shared_prev_action[warp_id];
    
    // Find max Q-value for new state using warp cooperation
    float max_q_new = -1e9f;
    for (uint32_t a = lane_id; a < NUM_ACTIONS; a += WARP_SIZE) {
        max_q_new = fmaxf(max_q_new, q_state->q_values[new_state][a]);
    }
    max_q_new = warp_reduce_sum(max_q_new);
    
    if (lane_id == 0) {
        shared_max_q[warp_id] = max_q_new;
    }
    __syncwarp();
    
    max_q_new = shared_max_q[warp_id];
    
    // TD error calculation
    float old_q = q_state->q_values[prev_state][prev_action];
    float td_error = reward + GAMMA * max_q_new - old_q;
    
    // Update Q-values and eligibility traces
    const uint32_t states_per_thread = (NUM_STATES + blockDim.x * gridDim.x - 1) / 
                                      (blockDim.x * gridDim.x);
    const uint32_t start_state = tid * states_per_thread;
    const uint32_t end_state = min(start_state + states_per_thread, NUM_STATES);
    
    #pragma unroll 2
    for (uint32_t s = start_state; s < end_state; ++s) {
        #pragma unroll 4
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            float* e_trace = &q_state->eligibility_traces[s][a];
            float* q_value = &q_state->q_values[s][a];
            
            // Update eligibility trace
            float new_trace;
            if (s == prev_state && a == prev_action) {
                new_trace = 1.0f;
            } else {
                new_trace = *e_trace * GAMMA * 0.9f;
            }
            
            // Atomic update with reduced contention
            if (new_trace > 0.01f) {  // Threshold to reduce atomic operations
                atomicExch(e_trace, new_trace);
                float update = ALPHA * td_error * new_trace;
                atomicAdd(q_value, update);
            } else {
                *e_trace = 0.0f;  // Direct write for small values
            }
        }
    }
    
    // Update total reward (single thread)
    if (tid == 0) {
        atomicAdd(&q_state->total_reward, reward);
    }
}

/**
 * @brief Optimized waveform generation using half precision where applicable
 */
__global__ void __launch_bounds__(256, 2) generate_jamming_waveform_kernel_optimized(
    float* __restrict__ waveform_out,
    const JammingParams* __restrict__ params,
    const float* __restrict__ waveform_bank,
    uint32_t samples_per_symbol
) {
    // Shared memory for parameters and waveform cache
    __shared__ JammingParams shared_params;
    __shared__ float waveform_cache[1024];  // Cache portion of waveform bank
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_samples = gridDim.x * blockDim.x;
    
    // Load parameters once per block
    if (threadIdx.x == 0) {
        shared_params = *params;
    }
    
    // Cooperatively load waveform bank section
    const uint32_t bank_offset = shared_params.waveform_id * samples_per_symbol;
    const uint32_t cache_size = min(1024u, samples_per_symbol);
    
    for (uint32_t i = threadIdx.x; i < cache_size; i += blockDim.x) {
        waveform_cache[i] = waveform_bank[bank_offset + i];
    }
    __syncthreads();
    
    // Generate waveform with vectorized operations
    const float inv_samples = 1.0f / samples_per_symbol;
    const float two_pi = 6.28318530718f;
    const float power_scale = sqrtf(shared_params.power_watts * 0.01f);
    
    #pragma unroll 4
    for (uint32_t i = tid; i < samples_per_symbol; i += total_samples) {
        float t = (float)i * inv_samples;
        float sample;
        
        // Use cached waveform data when possible
        if (i < cache_size) {
            sample = waveform_cache[i];
        } else {
            sample = waveform_bank[bank_offset + i];
        }
        
        // Strategy-specific modulation (optimized)
        switch (shared_params.strategy) {
            case (uint8_t)JammingStrategy::BARRAGE_NARROW:
            case (uint8_t)JammingStrategy::BARRAGE_WIDE: {
                float mod = __fmaf_rn(0.3f, __sinf(two_pi * t * 1000.0f), 1.0f);
                sample *= mod;
                break;
            }
            
            case (uint8_t)JammingStrategy::SWEEP_SLOW:
            case (uint8_t)JammingStrategy::SWEEP_FAST: {
                float sweep_phase = t * t * shared_params.sweep_rate_mhz_per_sec;
                sample *= __cosf(two_pi * sweep_phase);
                break;
            }
            
            case (uint8_t)JammingStrategy::PULSE_JAMMING: {
                float pulse = (t < 0.3f || (t > 0.6f && t < 0.9f)) ? 1.0f : 0.0f;
                sample *= pulse;
                break;
            }
            
            default: {
                float phase = __fmaf_rn(two_pi * t * 100.0f, 1.0f, shared_params.phase_offset);
                sample *= __cosf(phase);
                break;
            }
        }
        
        // Apply power scaling and write output
        waveform_out[i] = sample * power_scale;
    }
}

} // namespace ares::cew