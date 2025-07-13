/**
 * ARES Edge System - Neuromorphic CUDA Bridge
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * CUDA kernels for hybrid neuromorphic/GPU processing
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_SHARED_MEMORY 48000  // 48KB shared memory

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

namespace ares {
namespace neuromorphic {
namespace cuda {

/**
 * CUDA kernel for LIF neuron updates with coalesced memory access
 */
__global__ void lif_neuron_update_kernel(
    float* __restrict__ v,          // Membrane potentials
    const float* __restrict__ I_ext, // External currents
    const float* __restrict__ I_syn, // Synaptic currents
    bool* __restrict__ spiked,       // Spike flags
    const float v_rest,
    const float v_threshold,
    const float v_reset,
    const float tau,
    const float dt,
    const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < N; i += stride) {
        // Load voltage
        float v_i = v[i];
        
        // Skip if in refractory period (indicated by v = v_reset)
        if (v_i > v_reset + 0.1f) {
            // Update voltage: dv/dt = (v_rest - v + I) / tau
            float I_total = I_ext[i] + I_syn[i];
            float dv = (v_rest - v_i + I_total) / tau;
            v_i += dv * dt;
            
            // Check threshold
            if (v_i > v_threshold) {
                spiked[i] = true;
                v_i = v_reset;
            } else {
                spiked[i] = false;
            }
        } else {
            spiked[i] = false;
            // Refractory period recovery
            v_i += 0.5f;  // Slowly recover from reset
        }
        
        // Store updated voltage
        v[i] = v_i;
    }
}

/**
 * CUDA kernel for AdEx neuron updates with shared memory optimization
 */
__global__ void adex_neuron_update_kernel(
    float* __restrict__ v,
    float* __restrict__ w,
    const float* __restrict__ I_ext,
    bool* __restrict__ spiked,
    const float C,
    const float g_L,
    const float E_L,
    const float V_T,
    const float Delta_T,
    const float a,
    const float tau_w,
    const float b,
    const float v_reset,
    const float dt,
    const int N)
{
    // Shared memory for caching
    extern __shared__ float shared_mem[];
    float* shared_v = shared_mem;
    float* shared_w = &shared_mem[blockDim.x];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    
    if (tid < N) {
        // Load to shared memory
        shared_v[local_tid] = v[tid];
        shared_w[local_tid] = w[tid];
        __syncthreads();
        
        float v_i = shared_v[local_tid];
        float w_i = shared_w[local_tid];
        float I_i = I_ext[tid];
        
        // Exponential term with fast approximation
        float exp_term = __expf((v_i - V_T) / Delta_T);
        exp_term = fminf(exp_term, 1000.0f);  // Prevent overflow
        
        // Update equations
        float dv = (g_L * (E_L - v_i) + g_L * Delta_T * exp_term - w_i + I_i) / C;
        float dw = (a * (v_i - E_L) - w_i) / tau_w;
        
        v_i += dv * dt;
        w_i += dw * dt;
        
        // Check for spike
        if (v_i > 0.0f) {
            spiked[tid] = true;
            v_i = v_reset;
            w_i += b;
        } else {
            spiked[tid] = false;
        }
        
        // Write back
        v[tid] = v_i;
        w[tid] = w_i;
    }
}

/**
 * CUDA kernel for sparse matrix-vector multiplication (synaptic propagation)
 */
__global__ void sparse_synapse_propagation_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ weights,
    const bool* __restrict__ pre_spikes,
    float* __restrict__ post_currents,
    const int num_pre)
{
    const int pre_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pre_idx < num_pre && pre_spikes[pre_idx]) {
        // Get range of post-synaptic neurons
        int row_start = row_ptr[pre_idx];
        int row_end = row_ptr[pre_idx + 1];
        
        // Propagate spike to all connected post-synaptic neurons
        for (int j = row_start; j < row_end; ++j) {
            int post_idx = col_idx[j];
            float weight = weights[j];
            
            // Atomic add for thread safety
            atomicAdd(&post_currents[post_idx], weight);
        }
    }
}

/**
 * CUDA kernel for STDP weight updates with warp-level primitives
 */
__global__ void stdp_weight_update_kernel(
    float* __restrict__ weights,
    float* __restrict__ pre_trace,
    float* __restrict__ post_trace,
    const bool* __restrict__ pre_spikes,
    const bool* __restrict__ post_spikes,
    const int* __restrict__ pre_indices,
    const int* __restrict__ post_indices,
    const float A_plus,
    const float A_minus,
    const float tau_pre,
    const float tau_post,
    const float w_max,
    const float dt,
    const int num_synapses)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (tid < num_synapses) {
        int pre_idx = pre_indices[tid];
        int post_idx = post_indices[tid];
        
        // Decay traces
        float decay_pre = __expf(-dt / tau_pre);
        float decay_post = __expf(-dt / tau_post);
        
        pre_trace[tid] *= decay_pre;
        post_trace[tid] *= decay_post;
        
        // Check for spikes using warp vote functions
        unsigned pre_spike_mask = __ballot_sync(0xFFFFFFFF, pre_spikes[pre_idx]);
        unsigned post_spike_mask = __ballot_sync(0xFFFFFFFF, post_spikes[post_idx]);
        
        float weight = weights[tid];
        
        // Pre-synaptic spike: LTD
        if (pre_spike_mask & (1 << lane_id)) {
            pre_trace[tid] += 1.0f;
            weight += A_minus * post_trace[tid];
        }
        
        // Post-synaptic spike: LTP
        if (post_spike_mask & (1 << lane_id)) {
            post_trace[tid] += 1.0f;
            weight += A_plus * pre_trace[tid];
        }
        
        // Clamp weight
        weights[tid] = fmaxf(0.0f, fminf(weight, w_max));
    }
}

/**
 * CUDA kernel for EM spectrum processing with FFT preparation
 */
__global__ void em_spectrum_to_neuron_input_kernel(
    const float* __restrict__ spectrum_amplitude,
    const float* __restrict__ spectrum_freq,
    float* __restrict__ neuron_input,
    const float* __restrict__ preferred_freq,
    const float tuning_width,
    const int num_freq_bins,
    const int num_neurons)
{
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < num_neurons) {
        float pref_freq = preferred_freq[neuron_idx];
        float input_sum = 0.0f;
        
        // Compute Gaussian tuning over frequency bins
        for (int freq_idx = 0; freq_idx < num_freq_bins; ++freq_idx) {
            float freq = spectrum_freq[freq_idx];
            float amplitude = spectrum_amplitude[freq_idx];
            
            float freq_diff = freq - pref_freq;
            float tuning = __expf(-(freq_diff * freq_diff) / (2.0f * tuning_width * tuning_width));
            
            input_sum += amplitude * tuning;
        }
        
        neuron_input[neuron_idx] = input_sum;
    }
}

/**
 * CUDA kernel for chaos detection using Lorenz dynamics
 */
__global__ void chaos_detector_kernel(
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ z,
    float* __restrict__ chaos_metric,
    const float* __restrict__ input_signal,
    const float sigma,
    const float rho,
    const float beta,
    const float coupling,
    const float dt,
    const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // Load state
        float x_i = x[tid];
        float y_i = y[tid];
        float z_i = z[tid];
        float input = input_signal[tid];
        
        // Lorenz equations with input coupling
        float dx = sigma * (y_i - x_i) + coupling * input;
        float dy = x_i * (rho - z_i) - y_i;
        float dz = x_i * y_i - beta * z_i;
        
        // Update state
        x_i += dx * dt;
        y_i += dy * dt;
        z_i += dz * dt;
        
        // Compute chaos metric (local Lyapunov exponent approximation)
        float divergence = sqrtf(dx * dx + dy * dy + dz * dz);
        chaos_metric[tid] = divergence;
        
        // Store updated state
        x[tid] = x_i;
        y[tid] = y_i;
        z[tid] = z_i;
    }
}

/**
 * High-level C++ wrapper class for CUDA neuromorphic operations
 */
class CUDANeuromorphicProcessor {
private:
    // Device memory pointers
    float *d_voltages, *d_adaptations, *d_currents_ext, *d_currents_syn;
    bool *d_spikes;
    float *d_weights, *d_pre_trace, *d_post_trace;
    int *d_row_ptr, *d_col_idx, *d_pre_indices, *d_post_indices;
    
    // Neuron parameters
    int num_neurons;
    int num_synapses;
    
    // CUDA resources
    cublasHandle_t cublas_handle;
    cufftHandle fft_plan;
    cudaStream_t stream;
    
public:
    CUDANeuromorphicProcessor(int n_neurons, int n_synapses) 
        : num_neurons(n_neurons), num_synapses(n_synapses) {
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_voltages, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_adaptations, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_currents_ext, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_currents_syn, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spikes, num_neurons * sizeof(bool)));
        
        CUDA_CHECK(cudaMalloc(&d_weights, num_synapses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_trace, num_synapses * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_post_trace, num_synapses * sizeof(float)));
        
        // Initialize CUDA resources
        cublasCreate(&cublas_handle);
        cudaStreamCreate(&stream);
        
        // Set stream for cuBLAS
        cublasSetStream(cublas_handle, stream);
        
        // Initialize values
        initialize_neurons();
        initialize_synapses();
    }
    
    ~CUDANeuromorphicProcessor() {
        // Free device memory
        cudaFree(d_voltages);
        cudaFree(d_adaptations);
        cudaFree(d_currents_ext);
        cudaFree(d_currents_syn);
        cudaFree(d_spikes);
        cudaFree(d_weights);
        cudaFree(d_pre_trace);
        cudaFree(d_post_trace);
        
        // Destroy CUDA resources
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(stream);
    }
    
    void initialize_neurons() {
        // Initialize voltages to resting potential
        thrust::device_ptr<float> v_ptr(d_voltages);
        thrust::fill(v_ptr, v_ptr + num_neurons, -65.0f);
        
        // Zero out other arrays
        CUDA_CHECK(cudaMemset(d_adaptations, 0, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_currents_ext, 0, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_currents_syn, 0, num_neurons * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_spikes, 0, num_neurons * sizeof(bool)));
    }
    
    void initialize_synapses() {
        // Initialize random weights
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_weights, num_synapses);
        curandDestroyGenerator(gen);
        
        // Zero traces
        CUDA_CHECK(cudaMemset(d_pre_trace, 0, num_synapses * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_post_trace, 0, num_synapses * sizeof(float)));
    }
    
    void run_lif_neurons(float dt, float tau = 10.0f, float v_rest = -65.0f,
                        float v_threshold = -50.0f, float v_reset = -70.0f) {
        int blocks = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        lif_neuron_update_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_voltages, d_currents_ext, d_currents_syn, d_spikes,
            v_rest, v_threshold, v_reset, tau, dt, num_neurons
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void run_adex_neurons(float dt, float C = 281.0f, float g_L = 30.0f,
                         float E_L = -70.6f, float V_T = -50.4f,
                         float Delta_T = 2.0f, float a = 4.0f,
                         float tau_w = 144.0f, float b = 0.0805f,
                         float v_reset = -70.6f) {
        int blocks = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);
        
        adex_neuron_update_kernel<<<blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
            d_voltages, d_adaptations, d_currents_ext, d_spikes,
            C, g_L, E_L, V_T, Delta_T, a, tau_w, b, v_reset, dt, num_neurons
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void propagate_spikes(int num_pre_neurons) {
        // Clear synaptic currents
        CUDA_CHECK(cudaMemsetAsync(d_currents_syn, 0, 
                                  num_neurons * sizeof(float), stream));
        
        int blocks = (num_pre_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        sparse_synapse_propagation_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_row_ptr, d_col_idx, d_weights, d_spikes, d_currents_syn, num_pre_neurons
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void update_stdp_weights(float dt, float A_plus = 0.01f, float A_minus = -0.0105f,
                           float tau_pre = 20.0f, float tau_post = 20.0f,
                           float w_max = 1.0f) {
        int blocks = (num_synapses + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        stdp_weight_update_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_weights, d_pre_trace, d_post_trace, d_spikes, d_spikes,
            d_pre_indices, d_post_indices, A_plus, A_minus,
            tau_pre, tau_post, w_max, dt, num_synapses
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void set_sparse_connectivity(const int* h_row_ptr, const int* h_col_idx,
                               const int* h_pre_indices, const int* h_post_indices,
                               int nnz) {
        // Allocate and copy connectivity data
        CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_neurons + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pre_indices, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_post_indices, nnz * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, h_row_ptr, 
                                  (num_neurons + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_col_idx, h_col_idx, 
                                  nnz * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pre_indices, h_pre_indices,
                                  nnz * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_post_indices, h_post_indices,
                                  nnz * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
    }
    
    void get_spike_data(bool* h_spikes) {
        CUDA_CHECK(cudaMemcpyAsync(h_spikes, d_spikes, 
                                  num_neurons * sizeof(bool),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    void set_external_current(const float* h_currents) {
        CUDA_CHECK(cudaMemcpyAsync(d_currents_ext, h_currents,
                                  num_neurons * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));
    }
    
    float get_average_voltage() {
        float sum;
        cublasSasum(cublas_handle, num_neurons, d_voltages, 1, &sum);
        return sum / num_neurons;
    }
};

} // namespace cuda
} // namespace neuromorphic
} // namespace ares