/**
 * Implementation of kernel functions for self_destruct_protocol.cu
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

// Implementation of the CUDA kernel for initializing random states
__global__ void initialize_chaos_random_states(curandState* states, uint32_t num_states, uint64_t seed) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        // Each thread gets a different seed
        curand_init(seed, tid, 0, &states[tid]);
    }
}
