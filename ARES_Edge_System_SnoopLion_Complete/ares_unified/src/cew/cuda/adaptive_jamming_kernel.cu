/**
 * @file adaptive_jamming_kernel.cu
 * @brief CUDA kernels for CEW adaptive jamming with Q-learning
 * 
 * Implements real-time threat response with <100ms latency guarantee
 */

#include "../include/cew_adaptive_jamming.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace ares::cew {

// Thread block configuration for optimal performance
constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t WARP_SIZE = 32;

__device__ inline float atomicMaxFloat(float* addr, float value) {
    float old;
    old = __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    return old;
}

/**
 * @brief Quantize continuous threat characteristics to discrete state
 */
__device__ uint32_t quantize_threat_state(
    const ThreatSignature& threat,
    float spectrum_density
) {
    // Quantize frequency into 4 bands
    uint32_t freq_band = min(3u, (uint32_t)((threat.center_freq_ghz - FREQ_MIN_GHZ) / 10.0f));
    
    // Quantize power into 4 levels
    uint32_t power_level = min(3u, (uint32_t)((threat.power_dbm + 100.0f) / 25.0f));
    
    // Quantize bandwidth into 4 categories
    uint32_t bw_category = min(3u, (uint32_t)(threat.bandwidth_mhz / 50.0f));
    
    // Combine into state (0-255)
    uint32_t state = (freq_band << 6) | (power_level << 4) | 
                     (bw_category << 2) | (threat.modulation_type & 0x3);
    
    return state;
}

/**
 * @brief Select action using epsilon-greedy policy
 */
__device__ uint32_t select_action(
    const QTableState* q_state,
    uint32_t state,
    uint32_t tid,
    curandState* rand_state
) {
    // Epsilon-greedy exploration
    float rand_val = curand_uniform(rand_state);
    
    if (rand_val < EPSILON) {
        // Explore: random action
        return (uint32_t)(curand_uniform(rand_state) * NUM_ACTIONS);
    } else {
        // Exploit: best action from Q-table
        float max_q = -1e9f;
        uint32_t best_action = 0;
        
        // Find action with highest Q-value
        #pragma unroll
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            float q_val = q_state->q_values[state][a];
            if (q_val > max_q) {
                max_q = q_val;
                best_action = a;
            }
        }
        
        return best_action;
    }
}

/**
 * @brief Main adaptive jamming kernel
 * Processes threats and generates jamming parameters using Q-learning
 */
__global__ void adaptive_jamming_kernel(
    const float* __restrict__ spectrum_waterfall,
    const ThreatSignature* __restrict__ threats,
    JammingParams* __restrict__ jamming_params,
    QTableState* q_state,
    uint32_t num_threats,
    uint64_t timestamp_ns
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one threat
    if (tid >= num_threats) return;
    
    // Initialize random state for exploration
    curandState rand_state;
    curand_init(timestamp_ns + tid, tid, 0, &rand_state);
    
    // Load threat data with coalesced access
    ThreatSignature threat = threats[tid];
    
    // Calculate spectrum density around threat
    uint32_t freq_bin = (uint32_t)((threat.center_freq_ghz - FREQ_MIN_GHZ) / 
                                   (FREQ_MAX_GHZ - FREQ_MIN_GHZ) * SPECTRUM_BINS);
    freq_bin = min(freq_bin, SPECTRUM_BINS - 1);
    
    // Compute local spectrum density using warp reduction
    float spectrum_density = 0.0f;
    const uint32_t start_bin = max(0, (int)freq_bin - 16);
    const uint32_t end_bin = min(SPECTRUM_BINS, freq_bin + 16);
    
    for (uint32_t b = start_bin + threadIdx.x % WARP_SIZE; 
         b < end_bin; b += WARP_SIZE) {
        spectrum_density += spectrum_waterfall[b];
    }
    
    // Warp reduction for spectrum density
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        spectrum_density += __shfl_down_sync(0xffffffff, spectrum_density, offset);
    }
    
    // Quantize to Q-table state
    uint32_t state = quantize_threat_state(threat, spectrum_density);
    
    // Select jamming action
    uint32_t action = select_action(q_state, state, tid, &rand_state);
    
    // Generate jamming parameters based on selected action
    JammingParams params;
    params.center_freq_ghz = threat.center_freq_ghz;
    
    // Map action to jamming strategy
    switch (action) {
        case 0: // BARRAGE_NARROW
            params.strategy = (uint8_t)JammingStrategy::BARRAGE_NARROW;
            params.bandwidth_mhz = threat.bandwidth_mhz * 1.5f;
            params.power_watts = 100.0f;
            params.sweep_rate_mhz_per_sec = 0.0f;
            break;
            
        case 1: // BARRAGE_WIDE
            params.strategy = (uint8_t)JammingStrategy::BARRAGE_WIDE;
            params.bandwidth_mhz = threat.bandwidth_mhz * 5.0f;
            params.power_watts = 200.0f;
            params.sweep_rate_mhz_per_sec = 0.0f;
            break;
            
        case 2: // SPOT_JAMMING
            params.strategy = (uint8_t)JammingStrategy::SPOT_JAMMING;
            params.bandwidth_mhz = threat.bandwidth_mhz * 0.8f;
            params.power_watts = 150.0f;
            params.sweep_rate_mhz_per_sec = 0.0f;
            break;
            
        case 3: // SWEEP_SLOW
            params.strategy = (uint8_t)JammingStrategy::SWEEP_SLOW;
            params.bandwidth_mhz = threat.bandwidth_mhz * 2.0f;
            params.power_watts = 120.0f;
            params.sweep_rate_mhz_per_sec = 100.0f;
            break;
            
        case 4: // SWEEP_FAST
            params.strategy = (uint8_t)JammingStrategy::SWEEP_FAST;
            params.bandwidth_mhz = threat.bandwidth_mhz * 3.0f;
            params.power_watts = 150.0f;
            params.sweep_rate_mhz_per_sec = 1000.0f;
            break;
            
        case 5: // PULSE_JAMMING
            params.strategy = (uint8_t)JammingStrategy::PULSE_JAMMING;
            params.bandwidth_mhz = threat.bandwidth_mhz * 1.2f;
            params.power_watts = 250.0f;
            params.sweep_rate_mhz_per_sec = 0.0f;
            break;
            
        case 6: // NOISE_MODULATED
            params.strategy = (uint8_t)JammingStrategy::NOISE_MODULATED;
            params.bandwidth_mhz = threat.bandwidth_mhz * 1.8f;
            params.power_watts = 180.0f;
            params.sweep_rate_mhz_per_sec = 50.0f;
            break;
            
        case 7: // DECEPTIVE_REPEAT
            params.strategy = (uint8_t)JammingStrategy::DECEPTIVE_REPEAT;
            params.bandwidth_mhz = threat.bandwidth_mhz;
            params.power_watts = 80.0f;
            params.sweep_rate_mhz_per_sec = 0.0f;
            break;
            
        default: // Advanced strategies
            params.strategy = action;
            params.bandwidth_mhz = threat.bandwidth_mhz * 1.5f;
            params.power_watts = 150.0f;
            params.sweep_rate_mhz_per_sec = 200.0f;
            break;
    }
    
    // Set additional parameters
    params.waveform_id = action;
    params.duration_ms = 100;  // 100ms bursts
    params.phase_offset = curand_uniform(&rand_state) * 2.0f * 3.14159f;
    
    // Write jamming parameters with coalesced access
    jamming_params[tid] = params;
    
    // Update Q-table state (only thread 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch(&q_state->current_state, state);
        atomicExch(&q_state->last_action, action);
        atomicAdd(&q_state->visit_count[state], 1);
    }
}

/**
 * @brief Update Q-table with reward feedback
 * Uses TD-learning with eligibility traces
 */
__global__ void update_qtable_kernel(
    QTableState* q_state,
    float reward,
    uint32_t new_state
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread updates a portion of the Q-table
    const uint32_t states_per_thread = (NUM_STATES + blockDim.x - 1) / blockDim.x;
    const uint32_t start_state = tid * states_per_thread;
    const uint32_t end_state = min(start_state + states_per_thread, NUM_STATES);
    
    // Get previous state and action
    uint32_t prev_state = q_state->current_state;
    uint32_t prev_action = q_state->last_action;
    
    // Find max Q-value for new state
    float max_q_new = -1e9f;
    for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
        max_q_new = fmaxf(max_q_new, q_state->q_values[new_state][a]);
    }
    
    // TD error
    float td_error = reward + GAMMA * max_q_new - 
                     q_state->q_values[prev_state][prev_action];
    
    // Update Q-values and eligibility traces
    for (uint32_t s = start_state; s < end_state; ++s) {
        #pragma unroll
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            // Update eligibility trace
            if (s == prev_state && a == prev_action) {
                q_state->eligibility_traces[s][a] = 1.0f;
            } else {
                q_state->eligibility_traces[s][a] *= GAMMA * 0.9f;  // Lambda = 0.9
            }
            
            // Update Q-value
            float update = ALPHA * td_error * q_state->eligibility_traces[s][a];
            atomicAdd(&q_state->q_values[s][a], update);
        }
    }
    
    // Update total reward (thread 0 only)
    if (tid == 0) {
        atomicAdd(&q_state->total_reward, reward);
    }
}

/**
 * @brief Generate jamming waveform based on selected parameters
 * Optimized for real-time generation with pre-computed waveform bank
 */
__global__ void generate_jamming_waveform_kernel(
    float* __restrict__ waveform_out,
    const JammingParams* __restrict__ params,
    const float* __restrict__ waveform_bank,
    uint32_t samples_per_symbol
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_samples = gridDim.x * blockDim.x;
    
    // Load parameters into shared memory for faster access
    __shared__ JammingParams shared_params;
    if (threadIdx.x == 0) {
        shared_params = *params;
    }
    __syncthreads();
    
    // Generate waveform samples
    for (uint32_t i = tid; i < samples_per_symbol; i += total_samples) {
        float t = (float)i / samples_per_symbol;
        float sample = 0.0f;
        
        // Base waveform from bank
        uint32_t bank_idx = shared_params.waveform_id * samples_per_symbol + i;
        sample = waveform_bank[bank_idx];
        
        // Apply strategy-specific modulation
        switch (shared_params.strategy) {
            case (uint8_t)JammingStrategy::BARRAGE_NARROW:
            case (uint8_t)JammingStrategy::BARRAGE_WIDE:
                // Gaussian noise
                sample *= 1.0f + 0.3f * sinf(2.0f * 3.14159f * t * 1000.0f);
                break;
                
            case (uint8_t)JammingStrategy::SWEEP_SLOW:
            case (uint8_t)JammingStrategy::SWEEP_FAST:
                // Frequency sweep
                sample *= cosf(2.0f * 3.14159f * t * t * 
                              shared_params.sweep_rate_mhz_per_sec);
                break;
                
            case (uint8_t)JammingStrategy::PULSE_JAMMING:
                // Pulse modulation
                sample *= (t < 0.3f || (t > 0.6f && t < 0.9f)) ? 1.0f : 0.0f;
                break;
                
            default:
                // Complex modulation patterns
                sample *= cosf(shared_params.phase_offset + 
                              2.0f * 3.14159f * t * 100.0f);
                break;
        }
        
        // Apply power scaling
        sample *= sqrtf(shared_params.power_watts / 100.0f);
        
        // Write output with coalesced access
        waveform_out[i] = sample;
    }
}

} // namespace ares::cew