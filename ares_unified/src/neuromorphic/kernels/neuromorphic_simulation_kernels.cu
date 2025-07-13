/**
 * @file neuromorphic_simulation_kernels.cu
 * @brief GPU kernels for Loihi 2 neuromorphic simulation
 * 
 * Implements biologically-inspired neuron models and learning rules
 * optimized for <1ms simulation timesteps
 */

#include "../include/loihi2_hardware_abstraction.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;

namespace ares::neuromorphic::kernels {

// Constants for neuron dynamics
constexpr float VOLTAGE_SCALE = 1000.0f;      // mV scale
constexpr float CURRENT_SCALE = 1.0f;         // nA scale
constexpr float TIME_SCALE = 0.001f;          // ms to seconds
constexpr int32_t VOLTAGE_PRECISION = 16384;  // Fixed-point precision

/**
 * @brief Leaky Integrate-and-Fire neuron update with adaptation
 * Implements sub-threshold dynamics and spike generation
 */
__global__ void lif_neuron_update_kernel(
    float* membrane_voltages,
    float* adaptation_variables,
    uint32_t* refractory_counters,
    const uint8_t* neuron_configs,
    const float* input_currents,
    uint32_t* spike_output,
    uint32_t num_neurons,
    float dt_ms
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= num_neurons) return;
    
    // Load neuron configuration
    const NeuronConfig* config = reinterpret_cast<const NeuronConfig*>(neuron_configs);
    const NeuronConfig& n_config = config[neuron_id];
    
    // Load state variables
    float v = membrane_voltages[neuron_id];
    float w = adaptation_variables[neuron_id];
    uint32_t refrac = refractory_counters[neuron_id];
    float i_syn = input_currents[neuron_id];
    
    // Check if in refractory period
    if (refrac > 0) {
        refrac--;
        v = n_config.reset_voltage / VOLTAGE_SCALE;
    } else {
        // Membrane dynamics
        float tau_m = n_config.decay_constant * 0.1f;  // Convert to ms
        float dv = (-v + n_config.resting_voltage / VOLTAGE_SCALE + i_syn - w) / tau_m;
        v += dv * dt_ms;
        
        // Adaptation dynamics
        float tau_w = n_config.adaptation_rate * 1.0f;  // Convert to ms
        float dw = (-w + 0.01f * v) / tau_w;
        w += dw * dt_ms;
        
        // Add noise if enabled
        if (n_config.enable_noise) {
            curandState rand_state;
            curand_init(neuron_id + blockIdx.x, neuron_id, 0, &rand_state);
            v += curand_normal(&rand_state) * 0.1f;
        }
        
        // Check for spike
        if (v >= n_config.threshold / VOLTAGE_SCALE) {
            // Record spike
            uint32_t spike_idx = atomicAdd(&spike_output[num_neurons], 1);
            if (spike_idx < num_neurons) {
                spike_output[spike_idx] = neuron_id;
            }
            
            // Reset state
            v = n_config.reset_voltage / VOLTAGE_SCALE;
            w += 0.1f;  // Spike-triggered adaptation
            refrac = n_config.refractory_period;
        }
    }
    
    // Bound voltage to prevent overflow
    v = fmaxf(-2.0f, fminf(2.0f, v));
    
    // Store updated state
    membrane_voltages[neuron_id] = v;
    adaptation_variables[neuron_id] = w;
    refractory_counters[neuron_id] = refrac;
}

/**
 * @brief Izhikevich neuron model for diverse spiking patterns
 * Supports regular spiking, fast spiking, bursting, etc.
 */
__global__ void izhikevich_neuron_kernel(
    float* membrane_voltages,
    float* recovery_variables,
    const float* neuron_params,  // [a, b, c, d] per neuron
    const float* input_currents,
    uint32_t* spike_output,
    uint32_t num_neurons,
    float dt_ms
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= num_neurons) return;
    
    // Load parameters (a=recovery rate, b=sensitivity, c=reset, d=adaptation)
    const float a = neuron_params[neuron_id * 4 + 0];
    const float b = neuron_params[neuron_id * 4 + 1];
    const float c = neuron_params[neuron_id * 4 + 2];
    const float d = neuron_params[neuron_id * 4 + 3];
    
    // Load state
    float v = membrane_voltages[neuron_id];
    float u = recovery_variables[neuron_id];
    float I = input_currents[neuron_id] * 10.0f;  // Scale input
    
    // Izhikevich dynamics
    float dv = 0.04f * v * v + 5.0f * v + 140.0f - u + I;
    float du = a * (b * v - u);
    
    v += dv * dt_ms;
    u += du * dt_ms;
    
    // Spike detection
    if (v >= 30.0f) {
        // Record spike
        uint32_t spike_idx = atomicAdd(&spike_output[num_neurons], 1);
        if (spike_idx < num_neurons) {
            spike_output[spike_idx] = neuron_id;
        }
        
        // Reset
        v = c;
        u += d;
    }
    
    // Store state
    membrane_voltages[neuron_id] = v;
    recovery_variables[neuron_id] = u;
}

/**
 * @brief Synapse propagation with axonal delays
 * Efficiently propagates spikes through sparse connectivity
 */
__global__ void synapse_propagation_kernel(
    const uint32_t* spike_indices,
    const uint32_t* spike_counts,
    const int8_t* weight_matrix,
    const uint8_t* delay_matrix,
    float* current_buffer,
    uint32_t num_pre,
    uint32_t num_post,
    uint32_t current_timestep
) {
    // Use cooperative groups for efficient reduction
    cg::thread_block block = cg::this_thread_block();
    
    const uint32_t post_neuron = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (post_neuron >= num_post) return;
    
    extern __shared__ float shared_current[];
    shared_current[tid] = 0.0f;
    
    // Each thread processes a subset of presynaptic spikes
    uint32_t num_spikes = spike_counts[0];
    
    for (uint32_t s = tid; s < num_spikes; s += blockDim.x) {
        uint32_t pre_neuron = spike_indices[s];
        
        // Sparse matrix index calculation
        uint32_t synapse_idx = pre_neuron * num_post + post_neuron;
        
        // Check delay
        uint8_t delay = delay_matrix[synapse_idx];
        if ((current_timestep % 64) == delay) {  // Simple delay buffer
            // Get weight and convert from int8 to float
            int8_t weight = weight_matrix[synapse_idx];
            float w = weight / 127.0f * 10.0f;  // Scale to nA
            
            shared_current[tid] += w;
        }
    }
    
    block.sync();
    
    // Parallel reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_current[tid] += shared_current[tid + s];
        }
        block.sync();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        atomicAdd(&current_buffer[post_neuron], shared_current[0]);
    }
}

/**
 * @brief Spike-Timing Dependent Plasticity (STDP) learning
 * Updates weights based on precise spike timing relationships
 */
__global__ void stdp_learning_kernel(
    int8_t* weight_matrix,
    float* eligibility_traces,
    const uint32_t* pre_spikes,
    const uint32_t* post_spikes,
    float learning_rate,
    float tau_plus,
    float tau_minus,
    uint32_t num_synapses
) {
    const uint32_t synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_id >= num_synapses) return;
    
    // Decode pre and post neuron IDs from synapse index
    uint32_t num_post = gridDim.x;  // Simplified
    uint32_t pre_id = synapse_id / num_post;
    uint32_t post_id = synapse_id % num_post;
    
    // Load spike times (simplified - would use circular buffer)
    float pre_time = pre_spikes[pre_id] * 0.001f;   // Convert to ms
    float post_time = post_spikes[post_id] * 0.001f;
    
    // Skip if no recent spikes
    if (pre_time == 0 || post_time == 0) return;
    
    float dt = post_time - pre_time;
    float dw = 0.0f;
    
    // STDP window
    if (dt > 0 && dt < 50.0f) {
        // LTP: pre before post
        dw = learning_rate * expf(-dt / tau_plus);
    } else if (dt < 0 && dt > -50.0f) {
        // LTD: post before pre
        dw = -learning_rate * expf(dt / tau_minus);
    }
    
    // Update eligibility trace
    eligibility_traces[synapse_id] = 0.9f * eligibility_traces[synapse_id] + dw;
    
    // Apply weight change with bounds
    int8_t current_weight = weight_matrix[synapse_id];
    int32_t new_weight = current_weight + (int32_t)(eligibility_traces[synapse_id] * 127.0f);
    
    // Clip to int8 range
    new_weight = max(-128, min(127, new_weight));
    weight_matrix[synapse_id] = (int8_t)new_weight;
}

/**
 * @brief Reward-modulated STDP for reinforcement learning
 * Combines STDP with global reward signal
 */
__global__ void reward_modulated_learning_kernel(
    int8_t* weight_matrix,
    const float* eligibility_traces,
    float reward_signal,
    float learning_rate,
    uint32_t num_synapses
) {
    const uint32_t synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_id >= num_synapses) return;
    
    // Load eligibility trace
    float eligibility = eligibility_traces[synapse_id];
    
    // Compute weight update: eligibility * reward * learning_rate
    float dw = eligibility * reward_signal * learning_rate;
    
    // Apply update with saturation
    int8_t current_weight = weight_matrix[synapse_id];
    int32_t new_weight = current_weight + (int32_t)(dw * 127.0f);
    
    // Soft bounds to prevent saturation
    if (new_weight > 100) {
        new_weight = 100 + (new_weight - 100) / 2;
    } else if (new_weight < -100) {
        new_weight = -100 + (new_weight + 100) / 2;
    }
    
    new_weight = max(-128, min(127, new_weight));
    weight_matrix[synapse_id] = (int8_t)new_weight;
}

/**
 * @brief Multi-compartment dendrite computation
 * Models dendritic integration and nonlinearities
 */
__global__ void dendrite_computation_kernel(
    float* dendrite_voltages,
    const float* synapse_inputs,
    const uint8_t* dendrite_configs,
    float* soma_current,
    uint32_t num_neurons,
    uint32_t dendrites_per_neuron
) {
    const uint32_t neuron_id = blockIdx.x;
    const uint32_t dendrite_id = threadIdx.x;
    
    if (neuron_id >= num_neurons || dendrite_id >= dendrites_per_neuron) return;
    
    const uint32_t global_dendrite_id = neuron_id * dendrites_per_neuron + dendrite_id;
    
    // Load dendrite voltage
    float v_dend = dendrite_voltages[global_dendrite_id];
    
    // Aggregate synaptic input for this dendrite
    float i_syn = 0.0f;
    const uint32_t synapses_per_dendrite = 16;  // Example
    
    for (uint32_t s = 0; s < synapses_per_dendrite; ++s) {
        uint32_t syn_idx = global_dendrite_id * synapses_per_dendrite + s;
        i_syn += synapse_inputs[syn_idx];
    }
    
    // Dendritic dynamics with nonlinearity
    float tau_dend = 5.0f;  // ms
    float dv = (-v_dend + i_syn) / tau_dend;
    v_dend += dv * 0.001f;  // dt = 1us
    
    // Dendritic spike (NMDA-like nonlinearity)
    float dendrite_output = v_dend;
    if (v_dend > 0.5f) {
        dendrite_output = v_dend * v_dend;  // Supralinear integration
    }
    
    // Store updated voltage
    dendrite_voltages[global_dendrite_id] = v_dend;
    
    // Contribute to soma current (using atomic for thread safety)
    atomicAdd(&soma_current[neuron_id], dendrite_output * 0.1f);
}

/**
 * @brief Homeostatic plasticity for stable network dynamics
 * Adjusts neuron excitability to maintain target firing rate
 */
__global__ void homeostatic_plasticity_kernel(
    float* neuron_thresholds,
    const float* firing_rates,
    float target_rate,
    float adaptation_rate,
    uint32_t num_neurons
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= num_neurons) return;
    
    // Load current firing rate and threshold
    float current_rate = firing_rates[neuron_id];
    float threshold = neuron_thresholds[neuron_id];
    
    // Compute error from target rate
    float error = target_rate - current_rate;
    
    // Update threshold (inverse relationship)
    float dthreshold = -adaptation_rate * error;
    threshold += dthreshold;
    
    // Bounds to prevent runaway
    threshold = fmaxf(0.5f, fminf(2.0f, threshold));
    
    // Store updated threshold
    neuron_thresholds[neuron_id] = threshold;
}

/**
 * @brief Structural plasticity - dynamic synapse creation/removal
 * Implements activity-dependent rewiring
 */
__global__ void structural_plasticity_kernel(
    int8_t* weight_matrix,
    uint8_t* synapse_active_flags,
    const float* pre_activity,
    const float* post_activity,
    float creation_threshold,
    float removal_threshold,
    uint32_t num_pre,
    uint32_t num_post
) {
    const uint32_t pre_id = blockIdx.x;
    const uint32_t post_id = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (pre_id >= num_pre || post_id >= num_post) return;
    
    const uint32_t synapse_id = pre_id * num_post + post_id;
    
    // Load activity levels
    float pre_act = pre_activity[pre_id];
    float post_act = post_activity[post_id];
    
    // Hebbian structural plasticity
    float correlation = pre_act * post_act;
    
    // Current synapse state
    bool is_active = synapse_active_flags[synapse_id] > 0;
    int8_t weight = weight_matrix[synapse_id];
    
    if (!is_active && correlation > creation_threshold) {
        // Create new synapse
        synapse_active_flags[synapse_id] = 1;
        weight_matrix[synapse_id] = 10;  // Initial weight
        
    } else if (is_active && correlation < removal_threshold) {
        // Remove weak synapse
        synapse_active_flags[synapse_id] = 0;
        weight_matrix[synapse_id] = 0;
    }
}

/**
 * @brief Neuromodulation kernel for dopamine/serotonin effects
 * Modulates plasticity and excitability based on global signals
 */
__global__ void neuromodulation_kernel(
    float* neuron_gains,
    float* learning_rates,
    const float* neuromodulator_levels,
    uint32_t num_neurons,
    uint32_t num_modulators
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= num_neurons) return;
    
    // Dopamine effect (modulator 0)
    float dopamine = neuromodulator_levels[0];
    float gain = 1.0f + 0.5f * dopamine;  // Increase excitability
    float lr_mult = 1.0f + 2.0f * dopamine;  // Enhance learning
    
    // Serotonin effect (modulator 1)
    if (num_modulators > 1) {
        float serotonin = neuromodulator_levels[1];
        gain *= (1.0f - 0.3f * serotonin);  // Decrease excitability
    }
    
    // Apply modulation
    neuron_gains[neuron_id] = gain;
    learning_rates[neuron_id] = lr_mult;
}

/**
 * @brief Spike compression for efficient storage/transmission
 * Compresses spike events into bit-packed format
 */
__global__ void spike_compression_kernel(
    const SpikeEvent* raw_spikes,
    uint32_t* compressed_spikes,
    uint32_t num_spikes
) {
    const uint32_t spike_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (spike_id >= num_spikes) return;
    
    const SpikeEvent& spike = raw_spikes[spike_id];
    
    // Pack spike into 32 bits:
    // [neuron_id:20][timestamp_low:10][payload:2]
    uint32_t packed = (spike.neuron_id & 0xFFFFF) << 12 |
                     ((spike.timestamp & 0x3FF) << 2) |
                     (spike.payload & 0x3);
    
    compressed_spikes[spike_id] = packed;
}

/**
 * @brief Power monitoring for neuromorphic efficiency
 * Estimates power consumption based on activity
 */
__global__ void power_monitoring_kernel(
    const uint32_t* spike_counts,
    const uint32_t* synapse_activations,
    float* power_estimates,
    uint32_t num_cores
) {
    const uint32_t core_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (core_id >= num_cores) return;
    
    // Power model based on Loihi measurements
    const float static_power_per_core = 0.1f;  // mW
    const float energy_per_spike = 23.6e-9f;    // 23.6 pJ
    const float energy_per_synop = 120e-12f;   // 120 fJ
    
    uint32_t spikes = spike_counts[core_id];
    uint32_t synops = synapse_activations[core_id];
    
    // Calculate dynamic power (energy * frequency)
    float spike_power = spikes * energy_per_spike * 1e6f;     // Convert to mW
    float synapse_power = synops * energy_per_synop * 1e9f;   // Convert to mW
    
    power_estimates[core_id] = static_power_per_core + spike_power + synapse_power;
}

/**
 * @brief Liquid State Machine reservoir dynamics
 * Implements recurrent spiking network for temporal processing
 */
__global__ void liquid_state_machine_kernel(
    float* reservoir_states,
    const float* input_signals,
    const float* recurrent_weights,
    float* readout_states,
    uint32_t reservoir_size,
    uint32_t input_size,
    uint32_t readout_size,
    float spectral_radius
) {
    const uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= reservoir_size) return;
    
    // Compute recurrent input
    float recurrent_input = 0.0f;
    
    // Sparse random connectivity (simplified)
    const uint32_t num_connections = 10;
    uint32_t seed = neuron_id;
    
    for (uint32_t c = 0; c < num_connections; ++c) {
        // Simple linear congruential generator for connectivity
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        uint32_t source = seed % reservoir_size;
        
        float weight = recurrent_weights[neuron_id * num_connections + c];
        recurrent_input += weight * tanhf(reservoir_states[source]);
    }
    
    // Scale by spectral radius
    recurrent_input *= spectral_radius;
    
    // Add input drive
    float input_drive = 0.0f;
    if (neuron_id < input_size) {
        input_drive = input_signals[neuron_id];
    }
    
    // Update state with leak
    float leak_rate = 0.1f;
    float new_state = (1.0f - leak_rate) * reservoir_states[neuron_id] + 
                     leak_rate * (recurrent_input + input_drive);
    
    // Nonlinear activation
    new_state = tanhf(new_state);
    
    // Store updated state
    reservoir_states[neuron_id] = new_state;
    
    // Linear readout (simplified)
    if (neuron_id < readout_size) {
        atomicAdd(&readout_states[neuron_id], new_state * 0.01f);
    }
}

} // namespace ares::neuromorphic::kernels