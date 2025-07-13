//===- Brian2Integration.h - Brian2 SNN integration -------*- C++ -*-===//
//
// ARES Edge System - Brian2 Integration Header
// Copyright (c) 2024 DELFICTUS I/O LLC
//
// Declares interfaces for Brian2 simulator integration with MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef BRIAN2_INTEGRATION_H
#define BRIAN2_INTEGRATION_H

#include <vector>
#include <memory>
#include <string>

namespace ares {
namespace mlir {
namespace neuromorphic {

/**
 * Interface for Brian2-compatible neuron models
 */
class IBrian2NeuronModel {
public:
    virtual ~IBrian2NeuronModel() = default;
    
    // Get Brian2 equation string
    virtual std::string getBrian2Equations() const = 0;
    
    // Get MLIR representation
    virtual std::string getMLIRDefinition() const = 0;
    
    // Validate parameters
    virtual bool validateParameters() const = 0;
};

/**
 * Interface for Brian2-compatible synaptic models
 */
class IBrian2SynapseModel {
public:
    virtual ~IBrian2SynapseModel() = default;
    
    // Get Brian2 synaptic equations
    virtual std::string getBrian2Equations() const = 0;
    
    // Get plasticity rule
    virtual std::string getPlasticityRule() const = 0;
};

/**
 * Brian2 benchmark configuration
 */
struct Brian2BenchmarkConfig {
    // Network parameters
    size_t num_neurons = 1000;
    double connection_probability = 0.1;
    double simulation_duration_ms = 1000.0;
    
    // Hardware targets to benchmark
    std::vector<std::string> targets = {"cpu", "gpu", "tpu"};
    
    // Benchmark iterations
    int iterations = 10;
    
    // Output options
    bool generate_mlir = true;
    bool save_spike_data = true;
    bool profile_memory = true;
};

/**
 * Brian2 benchmark results
 */
struct Brian2BenchmarkResults {
    struct TargetResult {
        std::string target;
        double mean_wall_time_ms;
        double std_wall_time_ms;
        double speedup;
        double power_watts;
        double efficiency_gflops_per_watt;
        size_t memory_bytes;
    };
    
    std::vector<TargetResult> target_results;
    std::string mlir_code;
    
    // Validation metrics
    double spike_rate_hz;
    double cv_isi;  // Coefficient of variation of ISI
    double correlation_coefficient;
};

/**
 * Run Brian2-based benchmarks with MLIR optimization
 */
Brian2BenchmarkResults runBrian2Benchmark(const Brian2BenchmarkConfig& config);

/**
 * Verify equivalence between Brian2 and MLIR implementations
 */
bool verifyBrian2MLIREquivalence(const std::string& brian2_code,
                                 const std::string& mlir_code,
                                 double tolerance = 1e-6);

} // namespace neuromorphic
} // namespace mlir
} // namespace ares

#endif // BRIAN2_INTEGRATION_H
