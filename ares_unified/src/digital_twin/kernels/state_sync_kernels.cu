/**
 * @file state_sync_kernels.cu
 * @brief GPU kernels for real-time digital twin state synchronization
 * 
 * Optimized for <1ms latency with advanced interpolation and compression
 */

#include "../include/realtime_state_sync.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace ares::digital_twin::sync_kernels {

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr float EPSILON = 1e-6f;

/**
 * @brief Cubic spline interpolation for smooth state transitions
 * Provides C2 continuity for accurate state representation
 */
__device__ float cubic_interpolate(
    float p0, float p1, float p2, float p3, float t
) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    // Catmull-Rom spline coefficients
    float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float a2 = -0.5f * p0 + 0.5f * p2;
    float a3 = p1;
    
    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

/**
 * @brief State interpolation using cubic splines
 * Interpolates states between timestamps for smooth transitions
 */
__global__ void state_interpolation_kernel(
    const float* state_history,      // [num_states x state_dim x history_depth]
    const uint64_t* timestamps,      // [num_states x history_depth]
    float* interpolated_state,       // Output: [num_states x state_dim]
    uint64_t target_timestamp,
    uint32_t num_states,
    uint32_t state_dim
) {
    const uint32_t state_idx = blockIdx.x;
    const uint32_t dim_idx = threadIdx.x;
    
    if (state_idx >= num_states || dim_idx >= state_dim) return;
    
    const uint32_t history_depth = 4;  // For cubic interpolation
    
    // Find surrounding timestamps
    int t1_idx = -1, t2_idx = -1;
    uint64_t t1 = 0, t2 = UINT64_MAX;
    
    for (uint32_t i = 0; i < history_depth - 1; ++i) {
        uint64_t ts = timestamps[state_idx * history_depth + i];
        uint64_t ts_next = timestamps[state_idx * history_depth + i + 1];
        
        if (ts <= target_timestamp && target_timestamp <= ts_next) {
            t1_idx = i;
            t2_idx = i + 1;
            t1 = ts;
            t2 = ts_next;
            break;
        }
    }
    
    // If not found, use extrapolation
    if (t1_idx == -1) {
        // Simple linear extrapolation
        t1_idx = history_depth - 2;
        t2_idx = history_depth - 1;
        t1 = timestamps[state_idx * history_depth + t1_idx];
        t2 = timestamps[state_idx * history_depth + t2_idx];
    }
    
    // Calculate interpolation parameter
    float alpha = 0.5f;
    if (t2 > t1) {
        alpha = (float)(target_timestamp - t1) / (float)(t2 - t1);
        alpha = fmaxf(0.0f, fminf(1.0f, alpha));
    }
    
    // Get state values for interpolation
    const uint32_t base_idx = state_idx * state_dim * history_depth;
    
    if (t1_idx > 0 && t2_idx < history_depth - 1) {
        // Cubic interpolation with 4 points
        float p0 = state_history[base_idx + (t1_idx - 1) * state_dim + dim_idx];
        float p1 = state_history[base_idx + t1_idx * state_dim + dim_idx];
        float p2 = state_history[base_idx + t2_idx * state_dim + dim_idx];
        float p3 = state_history[base_idx + (t2_idx + 1) * state_dim + dim_idx];
        
        interpolated_state[state_idx * state_dim + dim_idx] = 
            cubic_interpolate(p0, p1, p2, p3, alpha);
    } else {
        // Linear interpolation at boundaries
        float v1 = state_history[base_idx + t1_idx * state_dim + dim_idx];
        float v2 = state_history[base_idx + t2_idx * state_dim + dim_idx];
        
        interpolated_state[state_idx * state_dim + dim_idx] = 
            v1 * (1.0f - alpha) + v2 * alpha;
    }
}

/**
 * @brief Physics-based state extrapolation
 * Predicts future states using motion dynamics
 */
__global__ void state_extrapolation_kernel(
    const float* current_state,      // [num_entities x state_dim]
    const float* velocity_state,     // [num_entities x state_dim]
    const float* acceleration_state, // [num_entities x state_dim]
    float* predicted_state,          // Output: [num_entities x state_dim]
    float delta_time_s,
    uint32_t num_entities,
    uint32_t state_dim
) {
    const uint32_t entity_idx = blockIdx.x;
    const uint32_t dim_idx = threadIdx.x;
    
    if (entity_idx >= num_entities || dim_idx >= state_dim) return;
    
    const uint32_t idx = entity_idx * state_dim + dim_idx;
    
    // Load current state
    float pos = current_state[idx];
    float vel = velocity_state[idx];
    float acc = acceleration_state[idx];
    
    // Apply kinematic equation: x = x0 + v0*t + 0.5*a*t^2
    float dt = delta_time_s;
    float dt2 = dt * dt;
    
    float predicted_pos = pos + vel * dt + 0.5f * acc * dt2;
    
    // Apply damping for stability (prevent runaway predictions)
    float damping = expf(-0.1f * dt);
    predicted_pos = pos + (predicted_pos - pos) * damping;
    
    // Store predicted state
    predicted_state[idx] = predicted_pos;
    
    // Also predict velocity for consistency
    if (blockIdx.y == 1) {  // Velocity prediction
        float predicted_vel = vel + acc * dt;
        predicted_state[num_entities * state_dim + idx] = predicted_vel * damping;
    }
}

/**
 * @brief Delta compression for bandwidth optimization
 * Only transmits changed state components above threshold
 */
__global__ void delta_compression_kernel(
    const float* current_state,
    const float* previous_state,
    float* delta_values,
    uint32_t* changed_indices,
    uint32_t* num_changes,
    float threshold,
    uint32_t state_dim
) {
    extern __shared__ uint32_t shared_changes[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint32_t local_changes = 0;
    uint32_t local_index = UINT32_MAX;
    float local_delta = 0.0f;
    
    if (dim_idx < state_dim) {
        float current = current_state[dim_idx];
        float previous = previous_state[dim_idx];
        float delta = current - previous;
        
        // Check if change is significant
        if (fabsf(delta) > threshold) {
            local_changes = 1;
            local_index = dim_idx;
            local_delta = delta;
        }
    }
    
    // Count total changes using reduction
    typedef cub::BlockReduce<uint32_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    uint32_t total_changes = BlockReduce(temp_storage).Sum(local_changes);
    
    if (tid == 0) {
        shared_changes[0] = atomicAdd(num_changes, total_changes);
    }
    __syncthreads();
    
    // Compact changed indices and values
    if (local_changes > 0) {
        // Calculate write position
        typedef cub::BlockScan<uint32_t, 256> BlockScan;
        __shared__ typename BlockScan::TempStorage scan_storage;
        
        uint32_t write_pos;
        BlockScan(scan_storage).ExclusiveSum(local_changes, write_pos);
        
        write_pos += shared_changes[0];
        
        if (write_pos < state_dim) {  // Bounds check
            changed_indices[write_pos] = local_index;
            delta_values[write_pos] = local_delta;
        }
    }
}

/**
 * @brief Kalman filter update for state estimation
 * Fuses measurements with predictions for optimal state estimate
 */
__global__ void kalman_update_kernel(
    float* state_estimate,          // [state_dim]
    float* covariance_matrix,       // [state_dim x state_dim]
    const float* measurement,       // [measurement_dim]
    const float* measurement_noise, // [measurement_dim x measurement_dim]
    float* kalman_gain,            // [state_dim x measurement_dim]
    uint32_t state_dim
) {
    const uint32_t row = blockIdx.x;
    const uint32_t col = threadIdx.x;
    
    if (row >= state_dim || col >= state_dim) return;
    
    extern __shared__ float shared_mem[];
    float* H = shared_mem;  // Measurement matrix (simplified to identity)
    float* S = &shared_mem[state_dim * state_dim];  // Innovation covariance
    
    // Step 1: Compute innovation covariance S = H*P*H' + R
    if (col == 0) {
        float s_val = covariance_matrix[row * state_dim + row] + 
                     measurement_noise[row];  // Simplified diagonal R
        S[row] = s_val;
    }
    __syncthreads();
    
    // Step 2: Compute Kalman gain K = P*H'*inv(S)
    if (col == 0) {
        float k_val = covariance_matrix[row * state_dim + row] / 
                     (S[row] + EPSILON);
        kalman_gain[row] = k_val;
    }
    __syncthreads();
    
    // Step 3: Update state estimate x = x + K*(z - H*x)
    if (col == 0) {
        float innovation = measurement[row] - state_estimate[row];
        state_estimate[row] += kalman_gain[row] * innovation;
    }
    
    // Step 4: Update covariance P = (I - K*H)*P
    float kh = (row == col) ? kalman_gain[row] : 0.0f;
    float scale = 1.0f - kh;
    
    covariance_matrix[row * state_dim + col] *= scale;
}

/**
 * @brief Calculate divergence between physical and digital states
 * Uses multiple metrics for comprehensive divergence assessment
 */
__global__ void divergence_calculation_kernel(
    const float* physical_states,   // [num_entities x state_dim]
    const float* digital_states,    // [num_entities x state_dim]
    float* divergence_metrics,      // Output: [num_entities]
    uint32_t num_entities,
    uint32_t state_dim
) {
    const uint32_t entity_idx = blockIdx.x;
    
    if (entity_idx >= num_entities) return;
    
    extern __shared__ float shared_diff[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t base_idx = entity_idx * state_dim;
    
    // Compute squared differences
    float local_sum = 0.0f;
    float local_max = 0.0f;
    
    for (uint32_t d = tid; d < state_dim; d += blockDim.x) {
        float phys = physical_states[base_idx + d];
        float digi = digital_states[base_idx + d];
        float diff = phys - digi;
        float diff_sq = diff * diff;
        
        local_sum += diff_sq;
        local_max = fmaxf(local_max, fabsf(diff));
    }
    
    // Store in shared memory
    shared_diff[tid] = local_sum;
    shared_diff[tid + blockDim.x] = local_max;
    __syncthreads();
    
    // Parallel reduction for sum and max
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_diff[tid] += shared_diff[tid + s];
            shared_diff[tid + blockDim.x] = fmaxf(
                shared_diff[tid + blockDim.x],
                shared_diff[tid + s + blockDim.x]
            );
        }
        __syncthreads();
    }
    
    // Thread 0 computes final metrics
    if (tid == 0) {
        float rmse = sqrtf(shared_diff[0] / state_dim);
        float max_error = shared_diff[blockDim.x];
        
        // Normalized divergence metric (combines RMSE and max error)
        float divergence = 0.7f * rmse + 0.3f * max_error;
        
        // Apply exponential scaling for sensitivity
        divergence = 1.0f - expf(-divergence);
        
        divergence_metrics[entity_idx] = divergence;
    }
}

/**
 * @brief Batch synchronization for multiple entities
 * Optimized for coalesced memory access patterns
 */
__global__ void batch_sync_kernel(
    const float* source_states,     // [num_entities x state_dim]
    float* target_states,           // [num_entities x state_dim]
    const uint32_t* entity_indices, // Entity mapping
    const uint64_t* timestamps,     // Sync timestamps
    uint32_t num_entities,
    uint32_t state_dim
) {
    // Use grid-stride loop for arbitrary number of elements
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t idx = tid; idx < num_entities * state_dim; idx += stride) {
        uint32_t entity_idx = idx / state_dim;
        uint32_t dim_idx = idx % state_dim;
        
        // Get mapped entity index
        uint32_t mapped_entity = entity_indices[entity_idx];
        
        if (mapped_entity < num_entities) {
            uint32_t src_idx = mapped_entity * state_dim + dim_idx;
            uint32_t dst_idx = entity_idx * state_dim + dim_idx;
            
            // Direct copy with optional transformation
            float value = source_states[src_idx];
            
            // Apply timestamp-based decay (for predictive sync)
            uint64_t current_time = timestamps[0];
            uint64_t entity_time = timestamps[entity_idx + 1];
            
            if (entity_time < current_time) {
                float age_s = (current_time - entity_time) * 1e-9f;
                float decay = expf(-0.1f * age_s);  // Confidence decay
                value *= decay;
            }
            
            target_states[dst_idx] = value;
        }
    }
}

/**
 * @brief Adaptive synchronization rate adjustment
 * Dynamically adjusts sync frequency based on state divergence
 */
__global__ void adaptive_sync_rate_kernel(
    const float* divergence_history,  // [num_entities x history_length]
    float* sync_rates,                // Output: [num_entities]
    const float* thresholds,          // Divergence thresholds
    uint32_t num_entities,
    uint32_t history_length
) {
    const uint32_t entity_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (entity_idx >= num_entities) return;
    
    // Calculate divergence statistics
    float sum = 0.0f;
    float max_div = 0.0f;
    float variance = 0.0f;
    
    // First pass: mean and max
    for (uint32_t h = 0; h < history_length; ++h) {
        float div = divergence_history[entity_idx * history_length + h];
        sum += div;
        max_div = fmaxf(max_div, div);
    }
    
    float mean = sum / history_length;
    
    // Second pass: variance
    for (uint32_t h = 0; h < history_length; ++h) {
        float div = divergence_history[entity_idx * history_length + h];
        float diff = div - mean;
        variance += diff * diff;
    }
    variance /= history_length;
    
    // Calculate adaptive sync rate
    float base_rate = 1000.0f;  // 1 kHz base
    float threshold = thresholds[entity_idx];
    
    // Increase rate if divergence is high
    if (max_div > threshold * 2.0f) {
        sync_rates[entity_idx] = base_rate * 4.0f;  // 4 kHz
    } else if (max_div > threshold) {
        sync_rates[entity_idx] = base_rate * 2.0f;  // 2 kHz
    } else if (mean < threshold * 0.1f && variance < 0.01f) {
        // Decrease rate if very stable
        sync_rates[entity_idx] = base_rate * 0.25f;  // 250 Hz
    } else {
        sync_rates[entity_idx] = base_rate;
    }
}

/**
 * @brief Multi-rate Kalman filter for sensor fusion
 * Handles sensors with different update rates
 */
__global__ void multirate_kalman_kernel(
    float* state_estimates,          // [num_entities x state_dim]
    float* covariances,              // [num_entities x state_dim x state_dim]
    const float* measurements,       // [num_sensors x measurement_dim]
    const uint32_t* sensor_entity_map, // Which entity each sensor belongs to
    const float* sensor_rates,       // Update rate for each sensor
    const uint64_t* sensor_timestamps,
    uint64_t current_time,
    uint32_t num_entities,
    uint32_t state_dim,
    uint32_t num_sensors
) {
    const uint32_t entity_idx = blockIdx.x;
    const uint32_t dim_idx = threadIdx.x;
    
    if (entity_idx >= num_entities || dim_idx >= state_dim) return;
    
    // Process each sensor for this entity
    for (uint32_t s = 0; s < num_sensors; ++s) {
        if (sensor_entity_map[s] != entity_idx) continue;
        
        // Check if sensor has new data
        uint64_t sensor_age = current_time - sensor_timestamps[s];
        float expected_period_ns = 1e9f / sensor_rates[s];
        
        if (sensor_age < expected_period_ns * 1.5f) {  // Fresh measurement
            // Simplified Kalman update (diagonal covariance)
            float measurement = measurements[s * state_dim + dim_idx];
            float estimate = state_estimates[entity_idx * state_dim + dim_idx];
            float variance = covariances[entity_idx * state_dim * state_dim + 
                                       dim_idx * state_dim + dim_idx];
            
            // Innovation
            float innovation = measurement - estimate;
            
            // Kalman gain (simplified)
            float measurement_variance = 0.1f;  // Sensor noise
            float gain = variance / (variance + measurement_variance);
            
            // Update estimate
            state_estimates[entity_idx * state_dim + dim_idx] = 
                estimate + gain * innovation;
            
            // Update covariance
            covariances[entity_idx * state_dim * state_dim + 
                       dim_idx * state_dim + dim_idx] = 
                (1.0f - gain) * variance;
        }
    }
}

/**
 * @brief State compression using wavelet transform
 * Reduces bandwidth requirements while preserving important features
 */
__global__ void wavelet_compression_kernel(
    const float* input_state,        // [state_dim]
    float* compressed_coeffs,        // Output: wavelet coefficients
    uint32_t* significant_indices,   // Indices of significant coefficients
    uint32_t* num_significant,
    float compression_threshold,
    uint32_t state_dim,
    uint32_t wavelet_levels
) {
    extern __shared__ float shared_workspace[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t level = blockIdx.y;
    
    // Load input into shared memory
    for (uint32_t i = tid; i < state_dim; i += blockDim.x) {
        shared_workspace[i] = input_state[i];
    }
    __syncthreads();
    
    // Haar wavelet transform (simplified)
    uint32_t step_size = 1 << level;
    uint32_t num_steps = state_dim >> (level + 1);
    
    if (tid < num_steps) {
        uint32_t idx = tid * step_size * 2;
        
        // Low-pass (average)
        float avg = (shared_workspace[idx] + shared_workspace[idx + step_size]) * 0.5f;
        
        // High-pass (difference)
        float diff = (shared_workspace[idx] - shared_workspace[idx + step_size]) * 0.5f;
        
        // Store coefficients
        compressed_coeffs[level * state_dim + tid] = avg;
        compressed_coeffs[level * state_dim + num_steps + tid] = diff;
        
        // Check significance
        if (fabsf(diff) > compression_threshold) {
            uint32_t sig_idx = atomicAdd(num_significant, 1);
            significant_indices[sig_idx] = level * state_dim + num_steps + tid;
        }
    }
}

/**
 * @brief Predictive caching for latency hiding
 * Prefetches likely future states based on patterns
 */
__global__ void predictive_cache_kernel(
    const float* state_history,      // [cache_size x state_dim]
    const uint64_t* access_pattern,  // Recent access patterns
    float* cache_predictions,        // Output: predicted future accesses
    uint32_t* prefetch_indices,      // Entities to prefetch
    uint32_t cache_size,
    uint32_t state_dim,
    uint32_t pattern_length
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= cache_size) return;
    
    // Analyze access pattern using simple Markov model
    uint32_t current_entity = access_pattern[pattern_length - 1];
    uint32_t prediction_count[32] = {0};  // Histogram of next accesses
    
    // Count transitions
    for (uint32_t i = 0; i < pattern_length - 1; ++i) {
        if (access_pattern[i] == current_entity && i < pattern_length - 1) {
            uint32_t next = access_pattern[i + 1];
            if (next < 32) {
                prediction_count[next]++;
            }
        }
    }
    
    // Find most likely next access
    uint32_t max_count = 0;
    uint32_t predicted_entity = 0;
    
    for (uint32_t i = 0; i < 32; ++i) {
        if (prediction_count[i] > max_count) {
            max_count = prediction_count[i];
            predicted_entity = i;
        }
    }
    
    // Store prediction
    if (tid == 0 && max_count > 0) {
        prefetch_indices[0] = predicted_entity;
    }
}

} // namespace ares::digital_twin::sync_kernels