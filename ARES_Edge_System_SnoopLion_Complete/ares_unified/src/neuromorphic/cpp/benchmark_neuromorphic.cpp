/**
 * ARES Edge System - Neuromorphic Performance Benchmarks
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * Comprehensive benchmarking suite for neuromorphic algorithms
 */

#include "neuromorphic_core.h"
#include "custom_neuron_models.cpp"
#include "synaptic_models.cpp"
#include <benchmark/benchmark.h>
#include <random>
#include <numeric>
#include <fstream>

namespace ares {
namespace neuromorphic {
namespace benchmarks {

// Global random number generator
std::random_device g_rd;
std::mt19937 g_gen(g_rd());

/**
 * Benchmark LIF neuron model performance
 */
static void BM_LIFNeuron(benchmark::State& state) {
    const int num_neurons = state.range(0);
    const double dt = 0.1;  // ms
    
    // Initialize data
    NeuronParameters params;
    LIFNeuron model(params);
    
    std::vector<double> voltages(num_neurons, -65.0);
    std::vector<double> adaptations(num_neurons, 0.0);
    std::vector<double> currents(num_neurons);
    std::vector<bool> spiked(num_neurons);
    
    // Random input currents
    std::normal_distribution<> current_dist(0.0, 2.0);
    for (auto& I : currents) {
        I = current_dist(g_gen);
    }
    
    // Benchmark loop
    for (auto _ : state) {
        model.update_state(voltages.data(), adaptations.data(), 
                          currents.data(), num_neurons, dt);
        model.check_threshold(voltages.data(), spiked.data(), num_neurons);
        model.reset(voltages.data(), adaptations.data(), 
                   spiked.data(), num_neurons);
    }
    
    // Report neurons processed per second
    state.SetItemsProcessed(state.iterations() * num_neurons);
    state.SetBytesProcessed(state.iterations() * num_neurons * sizeof(double) * 3);
}
BENCHMARK(BM_LIFNeuron)->Range(1000, 1000000)->UseRealTime();

/**
 * Benchmark AdEx neuron model performance
 */
static void BM_AdExNeuron(benchmark::State& state) {
    const int num_neurons = state.range(0);
    const double dt = 0.1;
    
    NeuronParameters params;
    AdExNeuron model(params);
    
    std::vector<double> voltages(num_neurons, -70.6);
    std::vector<double> adaptations(num_neurons, 0.0);
    std::vector<double> currents(num_neurons);
    std::vector<bool> spiked(num_neurons);
    
    std::normal_distribution<> current_dist(0.0, 0.1);
    for (auto& I : currents) {
        I = current_dist(g_gen);
    }
    
    for (auto _ : state) {
        model.update_state(voltages.data(), adaptations.data(), 
                          currents.data(), num_neurons, dt);
        model.check_threshold(voltages.data(), spiked.data(), num_neurons);
        model.reset(voltages.data(), adaptations.data(), 
                   spiked.data(), num_neurons);
    }
    
    state.SetItemsProcessed(state.iterations() * num_neurons);
}
BENCHMARK(BM_AdExNeuron)->Range(1000, 1000000)->UseRealTime();

/**
 * Benchmark sparse synaptic propagation
 */
static void BM_SparseSynapticPropagation(benchmark::State& state) {
    const int num_pre = state.range(0);
    const int num_post = state.range(0);
    const double connection_prob = 0.1;
    
    // Create sparse connectivity
    std::vector<std::pair<int, int>> connections;
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    
    for (int i = 0; i < num_pre; ++i) {
        for (int j = 0; j < num_post; ++j) {
            if (prob_dist(g_gen) < connection_prob) {
                connections.push_back({i, j});
            }
        }
    }
    
    SparseConnectivity connectivity(num_pre, num_post);
    connectivity.build_from_pairs(connections);
    
    // Create synaptic model
    SynapticModel synapses(connections.size());
    
    // Test data
    std::vector<bool> pre_spikes(num_pre);
    std::vector<double> post_currents(num_post);
    
    // Random spikes (10% firing rate)
    for (auto& spike : pre_spikes) {
        spike = prob_dist(g_gen) < 0.1;
    }
    
    for (auto _ : state) {
        std::fill(post_currents.begin(), post_currents.end(), 0.0);
        synapses.propagate_spikes(pre_spikes.data(), post_currents.data());
    }
    
    state.SetItemsProcessed(state.iterations() * connections.size());
    state.counters["Synapses"] = connections.size();
    state.counters["SparsityPct"] = 100.0 * connections.size() / (num_pre * num_post);
}
BENCHMARK(BM_SparseSynapticPropagation)->Range(100, 10000)->UseRealTime();

/**
 * Benchmark STDP learning
 */
static void BM_STDPLearning(benchmark::State& state) {
    const int num_synapses = state.range(0);
    const double dt = 0.1;
    
    // Create connectivity
    int num_neurons = sqrt(num_synapses / 0.1);  // Assuming 10% connectivity
    std::vector<std::pair<int, int>> connections;
    
    for (int i = 0; i < num_synapses; ++i) {
        connections.push_back({rand() % num_neurons, rand() % num_neurons});
    }
    
    SparseConnectivity connectivity(num_neurons, num_neurons);
    connectivity.build_from_pairs(connections);
    
    TripletSTDP stdp_model(connectivity);
    
    // Random spike trains
    std::vector<bool> spikes(num_neurons);
    std::vector<double> currents(num_neurons, 0.0);
    
    for (auto _ : state) {
        // Generate random spikes
        for (auto& spike : spikes) {
            spike = (rand() % 100) < 5;  // 5% spike probability
        }
        
        // Update traces and propagate
        stdp_model.update_traces(dt);
        stdp_model.propagate_spikes(spikes.data(), currents.data());
    }
    
    state.SetItemsProcessed(state.iterations() * num_synapses);
}
BENCHMARK(BM_STDPLearning)->Range(1000, 1000000)->UseRealTime();

/**
 * Benchmark full network simulation
 */
static void BM_FullNetworkSimulation(benchmark::State& state) {
    const int num_neurons = state.range(0);
    const double sim_time = 10.0;  // ms
    const double dt = 0.1;
    const int steps = sim_time / dt;
    
    // Create network
    NeuromorphicNetwork network;
    
    // Add layers
    NeuronParameters params;
    auto input_layer = network.add_neuron_group(
        std::make_unique<LIFNeuron>(params), num_neurons);
    auto hidden_layer = network.add_neuron_group(
        std::make_unique<AdExNeuron>(params), num_neurons / 2);
    auto output_layer = network.add_neuron_group(
        std::make_unique<LIFNeuron>(params), num_neurons / 10);
    
    // Add connections
    network.add_synapses(input_layer, hidden_layer, 0.1);
    network.add_synapses(hidden_layer, output_layer, 0.2);
    
    for (auto _ : state) {
        network.run(sim_time, false);  // Don't record spikes for performance
    }
    
    state.SetItemsProcessed(state.iterations() * num_neurons * steps);
    state.counters["SimSteps"] = steps;
}
BENCHMARK(BM_FullNetworkSimulation)->Range(100, 10000)->UseRealTime();

/**
 * Benchmark EM spectrum processing
 */
static void BM_EMSpectrumProcessing(benchmark::State& state) {
    const int num_sensors = state.range(0);
    const int spectrum_size = 1024;
    
    NeuronParameters params;
    EMSensorNeuron sensors(params, num_sensors);
    
    // Generate test spectrum
    std::vector<double> spectrum_amplitudes(spectrum_size);
    std::vector<double> spectrum_frequencies(spectrum_size);
    std::vector<double> sensor_outputs(num_sensors);
    
    // Initialize spectrum
    for (int i = 0; i < spectrum_size; ++i) {
        spectrum_frequencies[i] = 1e9 + i * 5e6;  // 1-6 GHz range
        spectrum_amplitudes[i] = rand() / (double)RAND_MAX;
    }
    
    for (auto _ : state) {
        sensors.process_em_spectrum(spectrum_amplitudes.data(),
                                  spectrum_frequencies.data(),
                                  sensor_outputs.data(),
                                  num_sensors);
    }
    
    state.SetItemsProcessed(state.iterations() * num_sensors * spectrum_size);
}
BENCHMARK(BM_EMSpectrumProcessing)->Range(100, 10000)->UseRealTime();

/**
 * Benchmark chaos detection
 */
static void BM_ChaosDetection(benchmark::State& state) {
    const int num_detectors = state.range(0);
    const double dt = 0.1;
    
    NeuronParameters params;
    ChaosDetectorNeuron detectors(params, num_detectors);
    
    std::vector<double> voltages(num_detectors, -65.0);
    std::vector<double> adaptations(num_detectors, 0.0);
    std::vector<double> inputs(num_detectors);
    std::vector<bool> spiked(num_detectors);
    
    // Chaotic input signal
    for (int i = 0; i < num_detectors; ++i) {
        inputs[i] = sin(i * 0.1) + 0.5 * sin(i * 0.21);
    }
    
    for (auto _ : state) {
        detectors.update_state(voltages.data(), adaptations.data(),
                             inputs.data(), num_detectors, dt);
        detectors.check_threshold(voltages.data(), spiked.data(), num_detectors);
    }
    
    state.SetItemsProcessed(state.iterations() * num_detectors);
}
BENCHMARK(BM_ChaosDetection)->Range(100, 10000)->UseRealTime();

/**
 * Benchmark memory bandwidth
 */
static void BM_MemoryBandwidth(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<double> src(size), dst(size);
    
    // Initialize with random data
    std::generate(src.begin(), src.end(), []() { return rand() / (double)RAND_MAX; });
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
        std::copy(src.begin(), src.end(), dst.begin());
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(double));
}
BENCHMARK(BM_MemoryBandwidth)->Range(1<<10, 1<<24)->UseRealTime();

/**
 * Benchmark SIMD operations
 */
static void BM_SIMDOperations(benchmark::State& state) {
    const int size = state.range(0);
    const int aligned_size = (size + 3) & ~3;  // Align to 4 elements
    
    std::vector<double, aligned_allocator<double, 32>> a(aligned_size);
    std::vector<double, aligned_allocator<double, 32>> b(aligned_size);
    std::vector<double, aligned_allocator<double, 32>> c(aligned_size);
    
    // Initialize
    for (int i = 0; i < size; ++i) {
        a[i] = rand() / (double)RAND_MAX;
        b[i] = rand() / (double)RAND_MAX;
    }
    
    for (auto _ : state) {
        // Vectorized multiply-add
        for (int i = 0; i < aligned_size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vb = _mm256_load_pd(&b[i]);
            __m256d vc = _mm256_mul_pd(va, vb);
            _mm256_store_pd(&c[i], vc);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_SIMDOperations)->Range(1000, 1000000)->UseRealTime();

/**
 * Custom reporter for neuromorphic metrics
 */
class NeuromorphicReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        PrintBasicContext(&GetErrorStream(), context);
        
        GetOutputStream() << "\n"
                         << "ARES Neuromorphic Performance Benchmarks\n"
                         << "========================================\n\n";
        
        return true;
    }
    
    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);
        
        // Calculate and print summary statistics
        double total_neurons_per_sec = 0;
        int neuron_benchmarks = 0;
        
        for (const auto& run : reports) {
            if (run.run_name.find("Neuron") != std::string::npos) {
                total_neurons_per_sec += run.items_per_second;
                neuron_benchmarks++;
            }
        }
        
        if (neuron_benchmarks > 0) {
            GetOutputStream() << "\n"
                            << "Average neuron processing rate: "
                            << (total_neurons_per_sec / neuron_benchmarks / 1e6)
                            << " M neurons/sec\n";
        }
    }
};

/**
 * Write detailed benchmark results to CSV
 */
void write_results_to_csv(const std::string& filename) {
    std::ofstream file(filename);
    file << "Benchmark,Size,Time_ns,Neurons_per_sec,Bytes_per_sec\n";
    
    // Run benchmarks programmatically and collect results
    // This would require custom benchmark registration
    
    file.close();
}

} // namespace benchmarks
} // namespace neuromorphic
} // namespace ares

// Main benchmark runner
int main(int argc, char** argv) {
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Set custom reporter
    ares::neuromorphic::benchmarks::NeuromorphicReporter reporter;
    ::benchmark::RunSpecifiedBenchmarks(&reporter);
    
    // Write detailed results
    ares::neuromorphic::benchmarks::write_results_to_csv("neuromorphic_benchmarks.csv");
    
    return 0;
}