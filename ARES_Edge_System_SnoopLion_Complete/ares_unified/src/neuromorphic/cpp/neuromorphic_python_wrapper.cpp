/**
 * ARES Edge System - Python Wrapper for C++ Neuromorphic Core
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * Exposes high-performance C++ neuromorphic implementation to Python
 * via ctypes for integration with Lava framework.
 */

#include "neuromorphic_core.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstring>

using namespace ares::neuromorphic;

// Global registry for network instances
static std::unordered_map<void*, std::unique_ptr<NeuromorphicNetwork>> g_networks;

// Neuron model type enum matching Python
enum NeuronModelType {
    MODEL_LIF = 0,
    MODEL_ADEX = 1,
    MODEL_EM_SENSOR = 2,
    MODEL_CHAOS_DETECTOR = 3
};

// Structure to hold group metadata
struct GroupMetadata {
    int model_type;
    int size;
    int synapse_count;
};

static std::unordered_map<void*, std::unordered_map<int, GroupMetadata>> g_metadata;

extern "C" {

/**
 * Create a new neuromorphic network
 * @return Pointer to network instance
 */
void* create_network() {
    auto network = std::make_unique<NeuromorphicNetwork>();
    void* ptr = network.get();
    g_networks[ptr] = std::move(network);
    g_metadata[ptr] = std::unordered_map<int, GroupMetadata>();
    return ptr;
}

/**
 * Destroy a network instance
 * @param network_ptr Pointer to network
 */
void destroy_network(void* network_ptr) {
    g_networks.erase(network_ptr);
    g_metadata.erase(network_ptr);
}

/**
 * Add a neuron group to the network
 * @param network_ptr Pointer to network
 * @param model_type Type of neuron model
 * @param size Number of neurons
 * @param parameters Array of parameters
 * @return Group ID
 */
int add_neuron_group(void* network_ptr, int model_type, int size, double* parameters) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return -1;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    NeuronParameters params;
    
    // Copy parameters from array
    if (parameters != nullptr) {
        // Standard parameters order:
        // [0-9]: C, g_L, E_L, V_T, Delta_T, a, tau_w, b, V_reset, refractory
        if (model_type == MODEL_ADEX) {
            params.C = parameters[0];
            params.g_L = parameters[1];
            params.E_L = parameters[2];
            params.V_T = parameters[3];
            params.Delta_T = parameters[4];
            params.a = parameters[5];
            params.tau_w = parameters[6];
            params.b = parameters[7];
            params.v_reset = parameters[8];
            params.refractory = parameters[9];
        } else if (model_type == MODEL_LIF) {
            params.tau_m = parameters[0];
            params.v_rest = parameters[1];
            params.v_reset = parameters[2];
            params.v_threshold = parameters[3];
            params.refractory = parameters[4];
        } else if (model_type == MODEL_EM_SENSOR) {
            params.preferred_freq = parameters[0];
            params.tuning_width = parameters[1];
            params.tau_m = parameters[2];
            params.v_rest = parameters[3];
            params.v_reset = parameters[4];
            params.v_threshold = parameters[5];
        } else if (model_type == MODEL_CHAOS_DETECTOR) {
            params.omega = parameters[0];
            params.gamma = parameters[1];
            params.coupling = parameters[2];
            params.tau_m = parameters[3];
            params.v_rest = parameters[4];
            params.v_reset = parameters[5];
            params.v_threshold = parameters[6];
        }
    }
    
    // Create appropriate neuron model
    std::unique_ptr<NeuronModel> model;
    
    switch (model_type) {
        case MODEL_LIF:
            model = std::make_unique<LIFNeuron>(params);
            break;
            
        case MODEL_ADEX:
            model = std::make_unique<AdExNeuron>(params);
            break;
            
        case MODEL_EM_SENSOR:
            model = std::make_unique<EMSensorNeuron>(params, size);
            break;
            
        case MODEL_CHAOS_DETECTOR:
            model = std::make_unique<ChaosDetectorNeuron>(params, size);
            break;
            
        default:
            return -1;
    }
    
    int group_id = network->add_neuron_group(std::move(model), size);
    
    // Store metadata
    g_metadata[network_ptr][group_id] = {model_type, size, 0};
    
    return group_id;
}

/**
 * Add synapses between neuron groups
 * @param network_ptr Pointer to network
 * @param pre_group Pre-synaptic group ID
 * @param post_group Post-synaptic group ID
 * @param connection_probability Connection probability (0-1)
 * @return Synapse ID
 */
int add_synapses(void* network_ptr, int pre_group, int post_group, 
                 double connection_probability) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return -1;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    return network->add_synapses(pre_group, post_group, connection_probability);
}

/**
 * Set external current for a neuron group
 * @param network_ptr Pointer to network
 * @param group_id Group ID
 * @param currents Array of current values
 * @param size Size of array
 */
void set_external_current(void* network_ptr, int group_id, 
                         double* currents, int size) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    
    // Access network internals (would need to expose this in NeuromorphicNetwork)
    // For now, this is a placeholder
    // network->set_external_current(group_id, currents, size);
}

/**
 * Run network simulation
 * @param network_ptr Pointer to network
 * @param duration_ms Duration in milliseconds
 * @param record_spikes Whether to record spikes
 */
void run_simulation(void* network_ptr, double duration_ms, bool record_spikes) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    network->run(duration_ms, record_spikes);
}

/**
 * Get spike count for a neuron group
 * @param network_ptr Pointer to network
 * @param group_id Group ID
 * @return Number of spikes
 */
int get_spike_count(void* network_ptr, int group_id) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return 0;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    const auto& spike_times = network->get_spike_times(group_id);
    return static_cast<int>(spike_times.size());
}

/**
 * Get spike data for a neuron group
 * @param network_ptr Pointer to network
 * @param group_id Group ID
 * @param spike_times Output array for spike times
 * @param spike_indices Output array for neuron indices
 * @param max_spikes Maximum number of spikes to return
 * @return Actual number of spikes returned
 */
int get_spikes(void* network_ptr, int group_id, 
               int* spike_times, int* spike_indices, int max_spikes) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return 0;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    const auto& times = network->get_spike_times(group_id);
    const auto& indices = network->get_spike_indices(group_id);
    
    int count = std::min(static_cast<int>(times.size()), max_spikes);
    
    // Copy spike data
    std::memcpy(spike_times, times.data(), count * sizeof(int));
    std::memcpy(spike_indices, indices.data(), count * sizeof(int));
    
    return count;
}

/**
 * Get neuron voltages
 * @param network_ptr Pointer to network
 * @param group_id Group ID
 * @param voltages Output array for voltages
 * @param size Size of array
 */
void get_voltages(void* network_ptr, int group_id, double* voltages, int size) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return;
    }
    
    NeuromorphicNetwork* network = it->second.get();
    const auto& v = network->get_voltages(group_id);
    
    int count = std::min(static_cast<int>(v.size()), size);
    std::memcpy(voltages, v.data(), count * sizeof(double));
}

/**
 * Process EM spectrum data (for EM sensor neurons)
 * @param network_ptr Pointer to network
 * @param group_id Group ID (must be EM sensor group)
 * @param spectrum_amplitudes Array of spectrum amplitudes
 * @param spectrum_frequencies Array of spectrum frequencies
 * @param output_currents Output array for computed currents
 * @param size Size of arrays
 */
void process_em_spectrum(void* network_ptr, int group_id,
                        double* spectrum_amplitudes,
                        double* spectrum_frequencies,
                        double* output_currents,
                        int size) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return;
    }
    
    auto meta_it = g_metadata[network_ptr].find(group_id);
    if (meta_it == g_metadata[network_ptr].end() || 
        meta_it->second.model_type != MODEL_EM_SENSOR) {
        return;
    }
    
    // This would call the EM sensor processing method
    // For now, placeholder implementation
    for (int i = 0; i < size; ++i) {
        output_currents[i] = spectrum_amplitudes[i] * 0.1;  // Simple scaling
    }
}

/**
 * Compute Lyapunov exponent for chaos detector
 * @param network_ptr Pointer to network
 * @param group_id Group ID (must be chaos detector group)
 * @param neuron_idx Neuron index within group
 * @param time_steps Number of time steps for calculation
 * @return Lyapunov exponent
 */
double compute_lyapunov_exponent(void* network_ptr, int group_id,
                                int neuron_idx, int time_steps) {
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return 0.0;
    }
    
    auto meta_it = g_metadata[network_ptr].find(group_id);
    if (meta_it == g_metadata[network_ptr].end() || 
        meta_it->second.model_type != MODEL_CHAOS_DETECTOR) {
        return 0.0;
    }
    
    // This would call the chaos detector's Lyapunov calculation
    // For now, return placeholder value
    return 0.1;  // Positive = chaotic
}

/**
 * Enable/disable SIMD optimizations
 * @param enable Whether to enable SIMD
 */
void set_simd_enabled(bool enable) {
    // This would control SIMD usage globally
    // Implementation depends on build configuration
}

/**
 * Set number of OpenMP threads
 * @param num_threads Number of threads (0 = auto)
 */
void set_openmp_threads(int num_threads) {
    if (num_threads == 0) {
        omp_set_num_threads(omp_get_max_threads());
    } else {
        omp_set_num_threads(num_threads);
    }
}

/**
 * Get performance statistics
 * @param network_ptr Pointer to network
 * @param stats Output structure for statistics
 */
struct PerformanceStats {
    double avg_timestep_ms;
    double max_timestep_ms;
    int total_spikes;
    int total_neurons;
    double neurons_per_second;
};

void get_performance_stats(void* network_ptr, PerformanceStats* stats) {
    if (stats == nullptr) return;
    
    // Initialize with dummy values for now
    stats->avg_timestep_ms = 0.1;
    stats->max_timestep_ms = 0.2;
    stats->total_spikes = 0;
    stats->total_neurons = 0;
    stats->neurons_per_second = 0.0;
    
    auto it = g_networks.find(network_ptr);
    if (it == g_networks.end()) {
        return;
    }
    
    // Calculate total neurons
    auto& metadata = g_metadata[network_ptr];
    for (const auto& [group_id, meta] : metadata) {
        stats->total_neurons += meta.size;
    }
    
    // Would calculate actual performance metrics from network
    stats->neurons_per_second = stats->total_neurons * 10000;  // Placeholder
}

} // extern "C"