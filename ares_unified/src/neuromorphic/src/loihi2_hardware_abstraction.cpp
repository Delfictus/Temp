/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Company: DELFICTUS I/O LLC
 * CAGE Code: 13H70
 * UEI: LXT3B9GMY4N8
 * Active DoD Contractor
 * 
 * Location: Los Angeles, California 90013 United States
 * 
 * This software contains trade secrets and proprietary information
 * of DELFICTUS I/O LLC. Unauthorized use, reproduction, or distribution
 * is strictly prohibited. This technology is subject to export controls
 * under ITAR and EAR regulations.
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * WARNING: This system is designed for authorized U.S. Department of Defense
 * use only. Misuse may result in severe criminal and civil penalties.
 */

/**
 * @file loihi2_hardware_abstraction.cpp
 * @brief Implementation of Loihi 2 hardware abstraction layer
 */

#include "../include/loihi2_hardware_abstraction.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstring>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace ares::neuromorphic {

using namespace std::chrono;

// Spike Router Implementation
SpikeRouter::SpikeRouter(uint32_t num_cores, uint32_t max_fanout) 
    : num_cores_(num_cores), max_fanout_(max_fanout) {
    routing_table_.reserve(num_cores * loihi2::NEURONS_PER_CORE);
}

SpikeRouter::~SpikeRouter() = default;

void SpikeRouter::add_connection(uint32_t src_neuron, uint32_t dst_neuron, uint8_t delay) {
    uint32_t dst_core = dst_neuron / loihi2::NEURONS_PER_CORE;
    
    RoutingEntry entry{dst_neuron, static_cast<uint8_t>(dst_core), delay};
    routing_table_[src_neuron].push_back(entry);
    
    // Limit fanout
    if (routing_table_[src_neuron].size() > max_fanout_) {
        routing_table_[src_neuron].resize(max_fanout_);
    }
}

void SpikeRouter::route_spike(const SpikeEvent& spike, std::vector<SpikeEvent>& output) {
    auto it = routing_table_.find(spike.neuron_id);
    if (it == routing_table_.end()) return;
    
    for (const auto& entry : it->second) {
        SpikeEvent routed_spike = spike;
        routed_spike.neuron_id = entry.target_neuron;
        routed_spike.core_id = entry.target_core;
        routed_spike.timestamp += entry.delay;  // Add axonal delay
        output.push_back(routed_spike);
    }
}

void SpikeRouter::optimize_routing_tables() {
    // Sort routing entries by target core for better cache locality
    for (auto& [src, entries] : routing_table_) {
        std::sort(entries.begin(), entries.end(), 
            [](const RoutingEntry& a, const RoutingEntry& b) {
                return a.target_core < b.target_core;
            });
    }
}

// Loihi2HardwareAbstraction Implementation
Loihi2HardwareAbstraction::Loihi2HardwareAbstraction()
    : execution_mode_(ExecutionMode::GPU_SIMULATION)
    , running_(false)
    , current_timestep_(0)
    , loihi_context_(nullptr)
    , nxsdk_graph_(nullptr)
    , avg_latency_us_(0.0f)
    , power_consumption_mw_(0.0f)
    , total_spikes_(0)
    , synaptic_ops_(0) {
    
    std::memset(&gpu_sim_, 0, sizeof(gpu_sim_));
    std::memset(&topology_, 0, sizeof(topology_));
}

Loihi2HardwareAbstraction::~Loihi2HardwareAbstraction() {
    running_ = false;
    
    // Stop worker threads
    spike_cv_.notify_all();
    if (spike_router_thread_.joinable()) spike_router_thread_.join();
    if (learning_thread_.joinable()) learning_thread_.join();
    if (monitor_thread_.joinable()) monitor_thread_.join();
    
    // Free GPU resources
    if (gpu_sim_.d_membrane_voltages) cudaFree(gpu_sim_.d_membrane_voltages);
    if (gpu_sim_.d_adaptation_variables) cudaFree(gpu_sim_.d_adaptation_variables);
    if (gpu_sim_.d_refractory_counters) cudaFree(gpu_sim_.d_refractory_counters);
    if (gpu_sim_.d_neuron_configs) cudaFree(gpu_sim_.d_neuron_configs);
    if (gpu_sim_.d_weight_matrix) cudaFree(gpu_sim_.d_weight_matrix);
    if (gpu_sim_.d_delay_matrix) cudaFree(gpu_sim_.d_delay_matrix);
    if (gpu_sim_.d_synapse_indices) cudaFree(gpu_sim_.d_synapse_indices);
    if (gpu_sim_.d_spike_queue) cudaFree(gpu_sim_.d_spike_queue);
    if (gpu_sim_.d_spike_counts) cudaFree(gpu_sim_.d_spike_counts);
    if (gpu_sim_.d_spike_events) cudaFree(gpu_sim_.d_spike_events);
    if (gpu_sim_.d_eligibility_traces) cudaFree(gpu_sim_.d_eligibility_traces);
    if (gpu_sim_.d_reward_signal) cudaFree(gpu_sim_.d_reward_signal);
    if (gpu_sim_.d_weight_updates) cudaFree(gpu_sim_.d_weight_updates);
    
    if (gpu_sim_.compute_stream) cudaStreamDestroy(gpu_sim_.compute_stream);
    if (gpu_sim_.transfer_stream) cudaStreamDestroy(gpu_sim_.transfer_stream);
    if (gpu_sim_.sync_event) cudaEventDestroy(gpu_sim_.sync_event);
}

cudaError_t Loihi2HardwareAbstraction::initialize(
    ExecutionMode mode,
    int gpu_device_id,
    const char* loihi_config_file
) {
    execution_mode_ = mode;
    cudaError_t err = cudaSuccess;
    
    switch (mode) {
        case ExecutionMode::GPU_SIMULATION:
            err = init_gpu_simulation();
            break;
            
        case ExecutionMode::LOIHI2_HARDWARE:
            err = init_loihi_hardware();
            if (err != cudaSuccess) {
                // Fallback to GPU if hardware not available
                execution_mode_ = ExecutionMode::GPU_SIMULATION;
                err = init_gpu_simulation();
            }
            break;
            
        case ExecutionMode::HYBRID:
            err = init_gpu_simulation();
            if (err == cudaSuccess) {
                init_loihi_hardware();  // Try to init hardware, ignore failure
            }
            break;
            
        default:
            return cudaErrorInvalidValue;
    }
    
    if (err != cudaSuccess) return err;
    
    // Start worker threads
    running_ = true;
    spike_router_thread_ = std::thread(&Loihi2HardwareAbstraction::spike_routing_worker, this);
    learning_thread_ = std::thread(&Loihi2HardwareAbstraction::learning_update_worker, this);
    monitor_thread_ = std::thread(&Loihi2HardwareAbstraction::performance_monitor_worker, this);
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::init_gpu_simulation() {
    cudaError_t err;
    
    // Create CUDA streams with priorities
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    
    err = cudaStreamCreateWithPriority(&gpu_sim_.compute_stream, 
                                      cudaStreamNonBlocking, priority_high);
    if (err != cudaSuccess) return err;
    
    err = cudaStreamCreateWithPriority(&gpu_sim_.transfer_stream, 
                                      cudaStreamNonBlocking, priority_low);
    if (err != cudaSuccess) return err;
    
    err = cudaEventCreate(&gpu_sim_.sync_event);
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::init_loihi_hardware() {
    // This would contain actual Loihi 2 initialization code
    // For now, return error to trigger GPU fallback
    return cudaErrorNotSupported;
}

cudaError_t Loihi2HardwareAbstraction::configure_network(
    uint32_t num_cores,
    uint32_t neurons_per_core,
    uint32_t synapses_per_core
) {
    topology_.num_cores = num_cores;
    topology_.total_neurons = num_cores * neurons_per_core;
    topology_.total_synapses = num_cores * synapses_per_core;
    
    // Initialize core offsets
    topology_.core_neuron_offsets.resize(num_cores + 1);
    topology_.core_synapse_offsets.resize(num_cores + 1);
    
    for (uint32_t i = 0; i <= num_cores; ++i) {
        topology_.core_neuron_offsets[i] = i * neurons_per_core;
        topology_.core_synapse_offsets[i] = i * synapses_per_core;
    }
    
    // Initialize spike router
    spike_router_ = std::make_unique<SpikeRouter>(num_cores, loihi2::MAX_FANOUT);
    
    // Allocate GPU memory
    return allocate_gpu_memory();
}

cudaError_t Loihi2HardwareAbstraction::allocate_gpu_memory() {
    cudaError_t err;
    
    const size_t num_neurons = topology_.total_neurons;
    const size_t num_synapses = topology_.total_synapses;
    
    // Neuron state arrays
    err = cudaMalloc(&gpu_sim_.d_membrane_voltages, num_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_adaptation_variables, num_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_refractory_counters, num_neurons * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_neuron_configs, num_neurons * sizeof(NeuronConfig));
    if (err != cudaSuccess) return err;
    
    // Synapse arrays (sparse representation)
    err = cudaMalloc(&gpu_sim_.d_weight_matrix, num_synapses * sizeof(int8_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_delay_matrix, num_synapses * sizeof(uint8_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_synapse_indices, num_synapses * 2 * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    // Spike buffers
    const size_t spike_buffer_size = num_neurons * 100;  // Assume max 100 spikes per timestep per neuron
    err = cudaMalloc(&gpu_sim_.d_spike_queue, spike_buffer_size * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_spike_counts, topology_.num_cores * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_spike_events, spike_buffer_size * sizeof(SpikeEvent));
    if (err != cudaSuccess) return err;
    
    // Learning arrays
    err = cudaMalloc(&gpu_sim_.d_eligibility_traces, num_synapses * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_reward_signal, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&gpu_sim_.d_weight_updates, num_synapses * sizeof(int32_t));
    if (err != cudaSuccess) return err;
    
    // Initialize arrays to zero
    err = cudaMemset(gpu_sim_.d_membrane_voltages, 0, num_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(gpu_sim_.d_adaptation_variables, 0, num_neurons * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(gpu_sim_.d_refractory_counters, 0, num_neurons * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(gpu_sim_.d_spike_counts, 0, topology_.num_cores * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMemset(gpu_sim_.d_eligibility_traces, 0, num_synapses * sizeof(float));
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

uint32_t Loihi2HardwareAbstraction::create_neuron_group(
    uint32_t count,
    const NeuronConfig& config,
    uint32_t preferred_core
) {
    // Find available core with enough space
    uint32_t selected_core = preferred_core;
    if (selected_core == UINT32_MAX) {
        // Auto-select core with most available neurons
        // (simplified - would track actual usage)
        selected_core = current_timestep_ % topology_.num_cores;
    }
    
    uint32_t base_neuron_id = topology_.core_neuron_offsets[selected_core];
    
    // Configure neurons
    NeuronConfig* h_configs = new NeuronConfig[count];
    for (uint32_t i = 0; i < count; ++i) {
        h_configs[i] = config;
    }
    
    // Copy configuration to device
    cudaMemcpyAsync(gpu_sim_.d_neuron_configs + base_neuron_id, 
                    h_configs, 
                    count * sizeof(NeuronConfig),
                    cudaMemcpyHostToDevice,
                    gpu_sim_.transfer_stream);
    
    delete[] h_configs;
    
    return base_neuron_id;
}

cudaError_t Loihi2HardwareAbstraction::create_synapse(
    uint32_t pre_neuron,
    uint32_t post_neuron,
    const SynapseConfig& config
) {
    // Add to spike router
    spike_router_->add_connection(pre_neuron, post_neuron, config.delay);
    
    // Add to GPU synapse arrays (simplified - would use sparse format)
    // This is a placeholder for actual sparse matrix update
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::inject_spikes(
    const SpikeEvent* spikes,
    uint32_t count
) {
    // Add spikes to input queue
    std::lock_guard<std::mutex> lock(spike_queue_mutex_);
    
    for (uint32_t i = 0; i < count; ++i) {
        spike_input_queue_.push(spikes[i]);
    }
    
    spike_cv_.notify_one();
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::run_timesteps(
    uint32_t num_steps,
    float timestep_us
) {
    cudaError_t err = cudaSuccess;
    
    auto start_time = high_resolution_clock::now();
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t timestep = current_timestep_.fetch_add(1);
        
        switch (execution_mode_) {
            case ExecutionMode::GPU_SIMULATION:
                err = simulate_timestep_gpu(timestep);
                break;
                
            case ExecutionMode::LOIHI2_HARDWARE:
                err = execute_timestep_loihi(timestep);
                break;
                
            case ExecutionMode::HYBRID:
                // Distribute work between GPU and Loihi
                if (timestep % 2 == 0) {
                    err = simulate_timestep_gpu(timestep);
                } else {
                    err = execute_timestep_loihi(timestep);
                    if (err != cudaSuccess) {
                        err = simulate_timestep_gpu(timestep);  // Fallback
                    }
                }
                break;
                
            default:
                return cudaErrorInvalidValue;
        }
        
        if (err != cudaSuccess) return err;
    }
    
    // Wait for all operations to complete
    cudaStreamSynchronize(gpu_sim_.compute_stream);
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    
    // Update latency statistics
    float step_latency = duration.count() / (float)num_steps;
    avg_latency_us_ = 0.95f * avg_latency_us_ + 0.05f * step_latency;
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::simulate_timestep_gpu(uint32_t timestep) {
    // Launch neuron update kernel
    const uint32_t block_size = 256;
    const uint32_t grid_size = (topology_.total_neurons + block_size - 1) / block_size;
    
    // Placeholder for actual kernel launches
    // These would call the kernels defined in the header
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::execute_timestep_loihi(uint32_t timestep) {
    // Would contain actual Loihi execution code
    return cudaErrorNotSupported;
}

void Loihi2HardwareAbstraction::spike_routing_worker() {
    std::vector<SpikeEvent> routed_spikes;
    routed_spikes.reserve(1000);
    
    while (running_) {
        std::unique_lock<std::mutex> lock(spike_queue_mutex_);
        spike_cv_.wait(lock, [this] { 
            return !spike_input_queue_.empty() || !running_; 
        });
        
        while (!spike_input_queue_.empty()) {
            SpikeEvent spike = spike_input_queue_.front();
            spike_input_queue_.pop();
            
            // Route spike to target neurons
            routed_spikes.clear();
            spike_router_->route_spike(spike, routed_spikes);
            
            // Add routed spikes to output queue
            for (const auto& routed : routed_spikes) {
                spike_output_queue_.push(routed);
            }
            
            total_spikes_++;
        }
    }
}

void Loihi2HardwareAbstraction::learning_update_worker() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Periodically update weights based on eligibility traces
        if (current_timestep_ % loihi2::LEARNING_EPOCH_SIZE == 0) {
            // Launch learning kernel
            // This would call the STDP or reward-modulated learning kernels
        }
    }
}

void Loihi2HardwareAbstraction::performance_monitor_worker() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Calculate power consumption estimate
        uint64_t recent_spikes = total_spikes_.exchange(0);
        uint64_t recent_ops = synaptic_ops_.exchange(0);
        
        // Simple power model: static + dynamic based on activity
        const float static_power_mw = 10.0f;  // Base power
        const float energy_per_spike_nj = 23.6f;  // From Loihi papers
        const float energy_per_op_pj = 120.0f;
        
        float dynamic_power = (recent_spikes * energy_per_spike_nj * 1e-6f + 
                              recent_ops * energy_per_op_pj * 1e-9f) * 10.0f;  // 10Hz update
        
        power_consumption_mw_ = static_power_mw + dynamic_power;
    }
}

cudaError_t Loihi2HardwareAbstraction::get_neuron_voltages(
    uint32_t* neuron_ids,
    float* voltages,
    uint32_t count
) {
    // Create temporary device array for neuron IDs
    uint32_t* d_neuron_ids;
    cudaMalloc(&d_neuron_ids, count * sizeof(uint32_t));
    cudaMemcpy(d_neuron_ids, neuron_ids, count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel to gather voltages (simplified)
    // Would use a gather kernel to collect specific neuron voltages
    
    cudaFree(d_neuron_ids);
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::enable_learning(
    bool enable,
    float learning_rate
) {
    // Set learning rate on device
    // This would update device constants used by learning kernels
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::inject_reward(
    float reward,
    uint32_t delay_ms
) {
    // Schedule reward signal
    cudaMemcpyAsync(gpu_sim_.d_reward_signal, &reward, sizeof(float), 
                    cudaMemcpyHostToDevice, gpu_sim_.transfer_stream);
    
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::save_network_state(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return cudaErrorInvalidValue;
    
    // Save topology
    file.write(reinterpret_cast<const char*>(&topology_), sizeof(topology_));
    
    // Save neuron states
    float* h_voltages = new float[topology_.total_neurons];
    cudaMemcpy(h_voltages, gpu_sim_.d_membrane_voltages, 
               topology_.total_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(h_voltages), 
               topology_.total_neurons * sizeof(float));
    delete[] h_voltages;
    
    // Save weights (simplified - would save sparse format)
    
    file.close();
    return cudaSuccess;
}

cudaError_t Loihi2HardwareAbstraction::load_network_state(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return cudaErrorInvalidValue;
    
    // Load topology
    file.read(reinterpret_cast<char*>(&topology_), sizeof(topology_));
    
    // Reallocate if needed
    allocate_gpu_memory();
    
    // Load neuron states
    float* h_voltages = new float[topology_.total_neurons];
    file.read(reinterpret_cast<char*>(h_voltages), 
              topology_.total_neurons * sizeof(float));
    cudaMemcpy(gpu_sim_.d_membrane_voltages, h_voltages, 
               topology_.total_neurons * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_voltages;
    
    file.close();
    return cudaSuccess;
}

} // namespace ares::neuromorphic