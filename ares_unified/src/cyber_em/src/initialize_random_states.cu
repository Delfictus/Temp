/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file implements the initialize_random_states function for the ARES Edge System
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

namespace ares::cyber_em {

// Implementation of the CUDA kernel for initializing random states
__global__ void initialize_random_states_kernel(curandState* states, uint32_t num_states, uint64_t seed) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        // Each thread gets a different seed
        curand_init(seed + tid, 0, 0, &states[tid]);
    }
}

// Host function to call the kernel
void initialize_random_states(curandState* states, uint32_t num_states, cudaStream_t stream = 0) {
    // Use current time as seed if not provided
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Calculate grid and block dimensions
    dim3 block_size(256);
    dim3 grid_size((num_states + block_size.x - 1) / block_size.x);
    
    // Initialize random states
    initialize_random_states_kernel<<<grid_size, block_size, 0, stream>>>(states, num_states, seed);
}

} // namespace ares::cyber_em
