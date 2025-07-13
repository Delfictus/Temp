/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file implements a simplified version of ChaosInductionEngine
 * to fix compilation errors in last_man_standing_coordinator.cu
 */

#include "chaos_induction_engine.cuh"
#include <chrono>
#include <stdexcept>

namespace ares::countermeasures {

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
    } \
} while(0)

ChaosInductionEngine::ChaosInductionEngine() {
    CUDA_CHECK(cudaStreamCreate(&chaos_stream));
}

ChaosInductionEngine::~ChaosInductionEngine() {
    cudaStreamDestroy(chaos_stream);
}

void ChaosInductionEngine::initialize(uint32_t num_states, uint64_t seed) {
    if (seed == 0) {
        // Use current time as seed if not provided
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    d_random_states.resize(num_states);
    
    // Calculate grid and block dimensions
    dim3 block_size(256);
    dim3 grid_size((num_states + block_size.x - 1) / block_size.x);
    
    // Initialize random states
    initialize_chaos_random_states<<<grid_size, block_size, 0, chaos_stream>>>(
        thrust::raw_pointer_cast(d_random_states.data()),
        num_states,
        seed
    );
    
    CUDA_CHECK(cudaStreamSynchronize(chaos_stream));
    initialized = true;
}

void ChaosInductionEngine::set_chaos_mode(ChaosMode mode) {
    if (!initialized) {
        throw std::runtime_error("ChaosInductionEngine not initialized");
    }
    chaos_mode.store(mode);
}

void ChaosInductionEngine::set_chaos_intensity(float intensity) {
    if (intensity < 0.0f || intensity > 1.0f) {
        throw std::runtime_error("Chaos intensity must be between 0 and 1");
    }
    chaos_intensity.store(intensity);
}

void ChaosInductionEngine::emergency_maximum_chaos() {
    set_chaos_mode(ChaosMode::MAXIMUM_ENTROPY);
    set_chaos_intensity(1.0f);
}

ChaosMode ChaosInductionEngine::get_chaos_mode() const {
    return chaos_mode.load();
}

float ChaosInductionEngine::get_chaos_intensity() const {
    return chaos_intensity.load();
}

} // namespace ares::countermeasures
