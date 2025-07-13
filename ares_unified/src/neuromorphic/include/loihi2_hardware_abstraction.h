/**
 * @file loihi2_hardware_abstraction.h
 * @brief Hardware abstraction layer for Intel Loihi 2 with GPU fallback
 * 
 * Provides seamless switching between Loihi 2 hardware and GPU simulation
 * for neuromorphic processing with guaranteed <1ms inference latency
 */

#ifndef ARES_NEUROMORPHIC_LOIHI2_HARDWARE_ABSTRACTION_H
#define ARES_NEUROMORPHIC_LOIHI2_HARDWARE_ABSTRACTION_H

#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>
#include <thread>
#include <queue>
#include <condition_variable>

namespace ares::neuromorphic {

// Forward declarations
class SpikeRouter;
class NeuronCore;
class SynapseArray;
class AxonProcessor;

// Loihi 2 Architecture Constants
namespace loihi2 {
    constexpr uint32_t MAX_NEUROMORPHIC_CORES = 128;
    constexpr uint32_t NEURONS_PER_CORE = 8192;
    constexpr uint32_t COMPARTMENTS_PER_NEURON = 4;
    constexpr uint32_t SYNAPSES_PER_CORE = 131072;
    constexpr uint32_t DENDRITES_PER_NEURON = 64;
    constexpr uint32_t MAX_FANOUT = 4096;
    constexpr uint32_t SPIKE_PAYLOAD_BITS = 32;
    constexpr uint32_t MAX_DELAY_TIMESTEPS = 63;
    constexpr uint32_t LEARNING_EPOCH_SIZE = 128;
}

// Neuron Models
enum class NeuronModel : uint8_t {
    LIF = 0,                    // Leaky Integrate-and-Fire
    ADAPTIVE_LIF = 1,           // Adaptive LIF with homeostasis
    IZHIKEVICH = 2,            // Izhikevich neuron model
    COMPARTMENTAL = 3,          // Multi-compartment with dendrites
    STOCHASTIC = 4,            // Stochastic spiking
    GRADED = 5                 // Graded/analog neuron
};

// Synapse Types
enum class SynapseType : uint8_t {
    STATIC = 0,                 // Fixed weight
    STDP = 1,                   // Spike-timing dependent plasticity
    REWARD_MODULATED = 2,       // Reward-modulated STDP
    HOMEOSTATIC = 3,           // Homeostatic plasticity
    STRUCTURAL = 4,            // Structural plasticity
    NEUROMODULATED = 5         // Dopamine/serotonin modulated
};

// Hardware Execution Mode
enum class ExecutionMode : uint8_t {
    LOIHI2_HARDWARE = 0,       // Native Loihi 2 execution
    GPU_SIMULATION = 1,        // CUDA-based simulation
    HYBRID = 2,                // Mix of hardware and GPU
    CPU_REFERENCE = 3          // CPU reference implementation
};

// Neuron Configuration
struct NeuronConfig {
    NeuronModel model;
    int16_t threshold;          // Spike threshold (fixed-point)
    int16_t reset_voltage;      // Reset potential
    int16_t resting_voltage;    // Resting potential
    uint16_t refractory_period; // In time steps
    uint8_t decay_constant;     // Membrane time constant
    uint8_t adaptation_rate;    // Adaptation time constant
    bool enable_noise;          // Stochastic behavior
    uint8_t compartment_count;  // For multi-compartment models
};

// Synapse Configuration
struct SynapseConfig {
    SynapseType type;
    int8_t weight;              // Synaptic weight (4-bit to 8-bit)
    uint8_t delay;              // Axonal delay (0-63 timesteps)
    uint8_t tag;                // Learning tag
    bool plastic;               // Enable plasticity
    uint8_t learning_rate;      // For plastic synapses
};

// Spike Event
struct SpikeEvent {
    uint32_t neuron_id;         // Global neuron ID
    uint32_t timestamp;         // Time in microseconds
    uint16_t axon_id;           // Axon identifier
    uint8_t payload;            // Additional spike data
    uint8_t core_id;            // Source core
};

// Learning Event
struct LearningEvent {
    uint32_t pre_neuron_id;
    uint32_t post_neuron_id;
    int32_t weight_delta;       // Weight change
    uint32_t timestamp;
    uint8_t learning_rule;      // STDP, reward, etc.
};

// Core State (for monitoring)
struct CoreState {
    uint32_t active_neurons;
    uint32_t total_spikes;
    float average_firing_rate;
    float power_consumption_mw;
    uint32_t synaptic_operations;
    uint64_t timestamp_us;
};

// Hardware Abstraction Layer
class Loihi2HardwareAbstraction {
public:
    Loihi2HardwareAbstraction();
    ~Loihi2HardwareAbstraction();
    
    // Initialize with execution mode
    cudaError_t initialize(
        ExecutionMode mode = ExecutionMode::GPU_SIMULATION,
        int gpu_device_id = 0,
        const char* loihi_config_file = nullptr
    );
    
    // Configure network topology
    cudaError_t configure_network(
        uint32_t num_cores,
        uint32_t neurons_per_core,
        uint32_t synapses_per_core
    );
    
    // Neuron management
    uint32_t create_neuron_group(
        uint32_t count,
        const NeuronConfig& config,
        uint32_t preferred_core = UINT32_MAX
    );
    
    cudaError_t configure_neuron(
        uint32_t neuron_id,
        const NeuronConfig& config
    );
    
    // Synapse management
    cudaError_t create_synapse(
        uint32_t pre_neuron,
        uint32_t post_neuron,
        const SynapseConfig& config
    );
    
    cudaError_t create_synapse_group(
        const uint32_t* pre_neurons,
        const uint32_t* post_neurons,
        const SynapseConfig* configs,
        uint32_t count
    );
    
    // Spike injection and readout
    cudaError_t inject_spikes(
        const SpikeEvent* spikes,
        uint32_t count
    );
    
    cudaError_t read_spikes(
        SpikeEvent* spike_buffer,
        uint32_t* spike_count,
        uint32_t max_spikes
    );
    
    // Network execution
    cudaError_t run_timesteps(
        uint32_t num_steps,
        float timestep_us = 1.0f
    );
    
    cudaError_t run_until_quiet(
        uint32_t max_steps,
        float activity_threshold = 0.01f
    );
    
    // State management
    cudaError_t save_network_state(const char* filename);
    cudaError_t load_network_state(const char* filename);
    cudaError_t reset_network();
    
    // Monitoring and debugging
    cudaError_t get_core_states(
        CoreState* states,
        uint32_t* num_cores
    );
    
    cudaError_t get_neuron_voltages(
        uint32_t* neuron_ids,
        float* voltages,
        uint32_t count
    );
    
    cudaError_t enable_spike_recording(
        uint32_t* neuron_ids,
        uint32_t count
    );
    
    // Learning control
    cudaError_t enable_learning(
        bool enable,
        float learning_rate = 0.01f
    );
    
    cudaError_t inject_reward(
        float reward,
        uint32_t delay_ms = 0
    );
    
    cudaError_t get_weight_matrix(
        float* weights,
        uint32_t pre_start,
        uint32_t pre_end,
        uint32_t post_start,
        uint32_t post_end
    );
    
    // Performance metrics
    float get_average_latency_us() const { return avg_latency_us_; }
    float get_power_consumption_mw() const { return power_consumption_mw_; }
    uint64_t get_total_spikes() const { return total_spikes_; }
    
    // Mode switching
    cudaError_t switch_execution_mode(ExecutionMode new_mode);
    ExecutionMode get_execution_mode() const { return execution_mode_; }
    
private:
    // Execution mode and state
    ExecutionMode execution_mode_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> current_timestep_;
    
    // Hardware interface (when available)
    void* loihi_context_;
    void* nxsdk_graph_;
    
    // GPU simulation components
    struct GPUSimulation {
        // Neuron states
        float* d_membrane_voltages;
        float* d_adaptation_variables;
        uint32_t* d_refractory_counters;
        uint8_t* d_neuron_configs;
        
        // Synapse arrays
        int8_t* d_weight_matrix;
        uint8_t* d_delay_matrix;
        uint32_t* d_synapse_indices;
        
        // Spike buffers
        uint32_t* d_spike_queue;
        uint32_t* d_spike_counts;
        SpikeEvent* d_spike_events;
        
        // Learning state
        float* d_eligibility_traces;
        float* d_reward_signal;
        int32_t* d_weight_updates;
        
        // CUDA resources
        cudaStream_t compute_stream;
        cudaStream_t transfer_stream;
        cudaEvent_t sync_event;
    } gpu_sim_;
    
    // Network topology
    struct NetworkTopology {
        uint32_t num_cores;
        uint32_t total_neurons;
        uint32_t total_synapses;
        std::vector<uint32_t> core_neuron_offsets;
        std::vector<uint32_t> core_synapse_offsets;
    } topology_;
    
    // Spike routing
    std::unique_ptr<SpikeRouter> spike_router_;
    std::queue<SpikeEvent> spike_input_queue_;
    std::queue<SpikeEvent> spike_output_queue_;
    std::mutex spike_queue_mutex_;
    std::condition_variable spike_cv_;
    
    // Performance tracking
    float avg_latency_us_;
    float power_consumption_mw_;
    std::atomic<uint64_t> total_spikes_;
    std::atomic<uint64_t> synaptic_ops_;
    
    // Worker threads for async operations
    std::thread spike_router_thread_;
    std::thread learning_thread_;
    std::thread monitor_thread_;
    
    // Internal methods
    cudaError_t init_gpu_simulation();
    cudaError_t init_loihi_hardware();
    cudaError_t allocate_gpu_memory();
    cudaError_t copy_config_to_device();
    
    void spike_routing_worker();
    void learning_update_worker();
    void performance_monitor_worker();
    
    cudaError_t simulate_timestep_gpu(uint32_t timestep);
    cudaError_t execute_timestep_loihi(uint32_t timestep);
};

// Spike Router for efficient spike propagation
class SpikeRouter {
public:
    SpikeRouter(uint32_t num_cores, uint32_t max_fanout);
    ~SpikeRouter();
    
    void add_connection(uint32_t src_neuron, uint32_t dst_neuron, uint8_t delay);
    void route_spike(const SpikeEvent& spike, std::vector<SpikeEvent>& output);
    void optimize_routing_tables();
    
private:
    struct RoutingEntry {
        uint32_t target_neuron;
        uint8_t target_core;
        uint8_t delay;
    };
    
    std::unordered_map<uint32_t, std::vector<RoutingEntry>> routing_table_;
    uint32_t num_cores_;
    uint32_t max_fanout_;
};

// GPU Kernels for neuromorphic simulation
namespace kernels {

__global__ void lif_neuron_update_kernel(
    float* membrane_voltages,
    float* adaptation_variables,
    uint32_t* refractory_counters,
    const uint8_t* neuron_configs,
    const float* input_currents,
    uint32_t* spike_output,
    uint32_t num_neurons,
    float dt_ms
);

__global__ void synapse_propagation_kernel(
    const uint32_t* spike_indices,
    const uint32_t* spike_counts,
    const int8_t* weight_matrix,
    const uint8_t* delay_matrix,
    float* current_buffer,
    uint32_t num_pre,
    uint32_t num_post,
    uint32_t current_timestep
);

__global__ void stdp_learning_kernel(
    int8_t* weight_matrix,
    float* eligibility_traces,
    const uint32_t* pre_spikes,
    const uint32_t* post_spikes,
    float learning_rate,
    float tau_plus,
    float tau_minus,
    uint32_t num_synapses
);

__global__ void reward_modulated_learning_kernel(
    int8_t* weight_matrix,
    const float* eligibility_traces,
    float reward_signal,
    float learning_rate,
    uint32_t num_synapses
);

__global__ void dendrite_computation_kernel(
    float* dendrite_voltages,
    const float* synapse_inputs,
    const uint8_t* dendrite_configs,
    float* soma_current,
    uint32_t num_neurons,
    uint32_t dendrites_per_neuron
);

__global__ void homeostatic_plasticity_kernel(
    float* neuron_thresholds,
    const float* firing_rates,
    float target_rate,
    float adaptation_rate,
    uint32_t num_neurons
);

__global__ void spike_compression_kernel(
    const SpikeEvent* raw_spikes,
    uint32_t* compressed_spikes,
    uint32_t num_spikes
);

__global__ void power_monitoring_kernel(
    const uint32_t* spike_counts,
    const uint32_t* synapse_activations,
    float* power_estimates,
    uint32_t num_cores
);

} // namespace kernels

} // namespace ares::neuromorphic

#endif // ARES_NEUROMORPHIC_LOIHI2_HARDWARE_ABSTRACTION_H