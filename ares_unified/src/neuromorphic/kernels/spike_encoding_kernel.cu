/**
 * @file spike_encoding_kernel.cu
 * @brief CUDA kernels for Loihi 2 spike encoding
 * 
 * Optimized for 0.1-1ms encoding latency with biologically-inspired algorithms
 */

#include "../include/loihi2_spike_encoder.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace ares::neuromorphic {

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t MAX_SPIKES_PER_NEURON = 100;
constexpr float SPIKE_THRESHOLD = 1.0f;

/**
 * @brief Population coding using Gaussian receptive fields
 * Each neuron responds maximally to a preferred stimulus value
 */
__global__ void population_coding_kernel(
    const float* __restrict__ input,
    const float* __restrict__ receptive_fields,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ spike_counts,
    const PopulationCodingParams params,
    uint32_t time_step,
    uint32_t input_size
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= params.num_neurons) return;
    
    // Initialize random state for Poisson spiking
    curandState rand_state;
    curand_init(time_step + neuron_id, neuron_id, 0, &rand_state);
    
    float activation = 0.0f;
    
    // Compute activation based on receptive field overlap
    for (uint32_t i = 0; i < input_size; ++i) {
        float rf_center = receptive_fields[neuron_id * input_size + i];
        float input_val = input[i];
        
        // Gaussian activation
        float diff = input_val - rf_center;
        float gaussian = expf(-0.5f * diff * diff / (params.sigma * params.sigma));
        
        activation += gaussian;
    }
    
    // Normalize activation
    activation /= input_size;
    
    // Convert to firing rate
    float firing_rate = activation * params.max_rate;
    
    // Generate Poisson spike train
    uint32_t spike_count = 0;
    uint32_t* neuron_spike_times = spike_times + neuron_id * MAX_SPIKES_PER_NEURON;
    
    // Time window of 1ms = 1000 microseconds
    for (uint32_t t = 0; t < 1000; t += TIME_STEP_US) {
        float spike_prob = firing_rate * TIME_STEP_US / 1e6f;  // Convert to probability
        
        if (curand_uniform(&rand_state) < spike_prob) {
            if (spike_count < MAX_SPIKES_PER_NEURON) {
                neuron_spike_times[spike_count] = time_step * 1000 + t;
                spike_count++;
            }
        }
    }
    
    spike_counts[neuron_id] = spike_count;
}

/**
 * @brief Temporal contrast encoding for change detection
 * Implements ON/OFF cells with adaptation
 */
__global__ void temporal_contrast_kernel(
    const float* __restrict__ input,
    float* __restrict__ membrane_potentials,
    float* __restrict__ adaptation_fast,
    float* __restrict__ adaptation_slow,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ spike_counts,
    const TemporalContrastParams params,
    uint32_t time_step,
    uint32_t num_neurons
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t input_idx = tid / 2;  // Two neurons (ON/OFF) per input
    const bool is_on_cell = (tid % 2) == 0;
    
    if (tid >= num_neurons || input_idx >= num_neurons/2) return;
    
    // Load current input and adaptation states
    float current_input = input[input_idx];
    float adapt_fast = adaptation_fast[tid];
    float adapt_slow = adaptation_slow[tid];
    float membrane = membrane_potentials[tid];
    
    // Update adaptation with different time constants
    float tau_fast = params.tau_fast_ms * 1000.0f;  // Convert to microseconds
    float tau_slow = params.tau_slow_ms * 1000.0f;
    
    adapt_fast += (current_input - adapt_fast) * TIME_STEP_US / tau_fast;
    adapt_slow += (current_input - adapt_slow) * TIME_STEP_US / tau_slow;
    
    // Compute temporal contrast
    float contrast = adapt_fast - adapt_slow;
    
    // Apply ON/OFF selectivity
    if (params.on_off_cells) {
        contrast = is_on_cell ? fmaxf(contrast, 0.0f) : -fminf(contrast, 0.0f);
    }
    
    // Update membrane potential
    membrane += contrast * 0.1f;  // Scaling factor
    
    // Check for spike
    uint32_t spike_count = spike_counts[tid];
    if (membrane > params.threshold && spike_count < MAX_SPIKES_PER_NEURON) {
        uint32_t* neuron_spike_times = spike_times + tid * MAX_SPIKES_PER_NEURON;
        neuron_spike_times[spike_count] = time_step;
        spike_counts[tid] = spike_count + 1;
        
        // Reset membrane potential
        membrane = 0.0f;
    }
    
    // Decay membrane potential
    membrane *= expf(-TIME_STEP_US / (10.0f * 1000.0f));  // 10ms decay
    
    // Write back state
    membrane_potentials[tid] = membrane;
    adaptation_fast[tid] = adapt_fast;
    adaptation_slow[tid] = adapt_slow;
}

/**
 * @brief Phase-of-firing encoding for precise timing relationships
 * Encodes information in the phase of spikes relative to oscillations
 */
__global__ void phase_encoding_kernel(
    const float* __restrict__ input,
    float* __restrict__ phase_oscillators,
    uint32_t* __restrict__ spike_times,
    float base_frequency,
    uint32_t time_window_us,
    uint32_t num_channels
) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (channel >= num_channels) return;
    
    // Load input value and current phase
    float input_val = input[channel];
    float phase = phase_oscillators[channel];
    
    // Map input to phase delay (0 to 2π)
    float phase_delay = input_val * 2.0f * M_PI;
    
    // Update oscillator phase
    float omega = 2.0f * M_PI * base_frequency / 1e6f;  // rad/μs
    phase += omega * TIME_STEP_US;
    
    // Wrap phase to [0, 2π]
    phase = fmodf(phase, 2.0f * M_PI);
    
    // Generate spike when phase crosses threshold + delay
    float spike_phase = fmodf(phase_delay, 2.0f * M_PI);
    
    if (fabsf(phase - spike_phase) < omega * TIME_STEP_US) {
        // Spike detected
        uint32_t spike_idx = atomicAdd(&spike_times[num_channels], 1);
        if (spike_idx < MAX_SPIKES_PER_NEURON * num_channels) {
            spike_times[spike_idx] = time_window_us;
        }
    }
    
    // Store updated phase
    phase_oscillators[channel] = phase;
}

/**
 * @brief Burst coding for urgency and salience signaling
 * Important signals trigger rapid bursts of spikes
 */
__global__ void burst_coding_kernel(
    const float* __restrict__ input,
    float* __restrict__ burst_accumulators,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ burst_lengths,
    float urgency_threshold,
    uint32_t num_inputs
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_inputs) return;
    
    float signal_strength = fabsf(input[idx]);
    float accumulator = burst_accumulators[idx];
    
    // Update burst accumulator with leaky integration
    accumulator = accumulator * 0.95f + signal_strength * 0.05f;
    
    // Check for burst condition
    if (accumulator > urgency_threshold) {
        // Calculate burst length based on urgency
        uint32_t burst_len = min(10u, (uint32_t)(accumulator / urgency_threshold * 5));
        
        // Generate burst
        uint32_t base_spike_idx = idx * MAX_SPIKES_PER_NEURON;
        for (uint32_t b = 0; b < burst_len; ++b) {
            spike_times[base_spike_idx + b] = b * 10;  // 10μs inter-spike interval
        }
        
        burst_lengths[idx] = burst_len;
        
        // Reset accumulator after burst
        accumulator = 0.0f;
    }
    
    burst_accumulators[idx] = accumulator;
}

/**
 * @brief Generate Gaussian receptive fields for population coding
 * Creates overlapping receptive fields that tile the input space
 */
__global__ void generate_gaussian_receptive_fields(
    float* __restrict__ receptive_fields,
    uint32_t num_neurons,
    uint32_t input_dim,
    float min_val,
    float max_val,
    float sigma
) {
    const uint32_t neuron = blockIdx.x;
    const uint32_t dim = threadIdx.x;
    
    if (neuron >= num_neurons || dim >= input_dim) return;
    
    // Calculate preferred value for this neuron
    float range = max_val - min_val;
    float preferred_val = min_val + (neuron * range) / (num_neurons - 1);
    
    // Store receptive field center
    receptive_fields[neuron * input_dim + dim] = preferred_val;
}

/**
 * @brief Compute spike train statistics for analysis
 * Calculates firing rates and variability measures
 */
__global__ void spike_train_statistics_kernel(
    const uint32_t* __restrict__ spike_times,
    const uint32_t* __restrict__ spike_counts,
    float* __restrict__ firing_rates,
    float* __restrict__ isi_cv,
    uint32_t num_neurons,
    uint32_t time_window_us
) {
    const uint32_t neuron = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron >= num_neurons) return;
    
    uint32_t num_spikes = spike_counts[neuron];
    const uint32_t* neuron_spikes = spike_times + neuron * MAX_SPIKES_PER_NEURON;
    
    // Calculate firing rate
    firing_rates[neuron] = (float)num_spikes * 1e6f / time_window_us;
    
    // Calculate inter-spike interval statistics
    if (num_spikes > 1) {
        float mean_isi = 0.0f;
        float var_isi = 0.0f;
        
        // First pass: compute mean ISI
        for (uint32_t i = 1; i < num_spikes; ++i) {
            float isi = (float)(neuron_spikes[i] - neuron_spikes[i-1]);
            mean_isi += isi;
        }
        mean_isi /= (num_spikes - 1);
        
        // Second pass: compute variance
        for (uint32_t i = 1; i < num_spikes; ++i) {
            float isi = (float)(neuron_spikes[i] - neuron_spikes[i-1]);
            float diff = isi - mean_isi;
            var_isi += diff * diff;
        }
        var_isi /= (num_spikes - 1);
        
        // Coefficient of variation
        float std_isi = sqrtf(var_isi);
        isi_cv[neuron] = (mean_isi > 0) ? std_isi / mean_isi : 0.0f;
    } else {
        isi_cv[neuron] = 0.0f;
    }
}

/**
 * @brief Event-based visual encoding for DVS-like processing
 * Generates spikes from temporal contrast in visual input
 */
__global__ void event_camera_encoding_kernel(
    const float* __restrict__ current_frame,
    const float* __restrict__ previous_frame,
    float* __restrict__ pixel_states,
    uint32_t* __restrict__ event_buffer,
    uint32_t* __restrict__ event_count,
    uint32_t width,
    uint32_t height,
    float contrast_threshold,
    uint32_t timestamp_us
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t pixel_idx = y * width + x;
    
    // Load pixel values
    float current = current_frame[pixel_idx];
    float previous = previous_frame[pixel_idx];
    float state = pixel_states[pixel_idx];
    
    // Compute log-intensity change
    float log_change = logf(current + 1e-5f) - logf(previous + 1e-5f);
    
    // Update pixel state with temporal filtering
    state = 0.7f * state + 0.3f * log_change;
    
    // Check for ON/OFF events
    bool on_event = state > contrast_threshold;
    bool off_event = state < -contrast_threshold;
    
    if (on_event || off_event) {
        // Generate event
        uint32_t event_idx = atomicAdd(event_count, 1);
        
        if (event_idx < width * height) {  // Buffer size check
            // Pack event: [x:10][y:10][polarity:1][timestamp:11]
            uint32_t event = (x & 0x3FF) << 22 |
                            (y & 0x3FF) << 12 |
                            (on_event ? 1 : 0) << 11 |
                            (timestamp_us & 0x7FF);
            
            event_buffer[event_idx] = event;
        }
        
        // Reset pixel state
        state = 0.0f;
    }
    
    // Write back state
    pixel_states[pixel_idx] = state;
}

} // namespace ares::neuromorphic