/**
 * @file cew_cuda_module.cpp
 * @brief CUDA implementation of the CEW module
 */

#include "cew_cuda_module.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <cmath>

namespace ares::cew {

CEWCudaModule::CEWCudaModule() 
    : initialized_(false)
    , device_id_(-1)
    , d_spectrum_buffer_(nullptr)
    , d_qtable_(nullptr)
    , d_waveform_bank_(nullptr)
    , fft_plan_(0)
    , compute_stream_(nullptr)
    , transfer_stream_(nullptr)
    , start_event_(nullptr)
    , stop_event_(nullptr) {
    std::memset(&metrics_, 0, sizeof(metrics_));
}

CEWCudaModule::~CEWCudaModule() {
    cleanup();
}

void CEWCudaModule::cleanup() {
    if (d_spectrum_buffer_) {
        cudaFree(d_spectrum_buffer_);
        d_spectrum_buffer_ = nullptr;
    }
    if (d_qtable_) {
        cudaFree(d_qtable_);
        d_qtable_ = nullptr;
    }
    if (d_waveform_bank_) {
        cudaFree(d_waveform_bank_);
        d_waveform_bank_ = nullptr;
    }
    if (fft_plan_) {
        cufftDestroy(fft_plan_);
        fft_plan_ = 0;
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }
    if (transfer_stream_) {
        cudaStreamDestroy(transfer_stream_);
        transfer_stream_ = nullptr;
    }
    if (start_event_) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
    }
    if (stop_event_) {
        cudaEventDestroy(stop_event_);
        stop_event_ = nullptr;
    }
    initialized_ = false;
}

bool CEWCudaModule::check_cuda_error(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

bool CEWCudaModule::is_cuda_available() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

bool CEWCudaModule::initialize(int device_id) {
    if (initialized_) {
        return true;
    }
    
    device_id_ = device_id;
    
    // Set device
    if (!check_cuda_error(cudaSetDevice(device_id), "cudaSetDevice")) {
        return false;
    }
    
    // Enable fastest GPU clocks
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    // Create high-priority streams
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    
    if (!check_cuda_error(
        cudaStreamCreateWithPriority(&compute_stream_, cudaStreamNonBlocking, priority_high),
        "create compute stream")) {
        cleanup();
        return false;
    }
    
    if (!check_cuda_error(
        cudaStreamCreateWithPriority(&transfer_stream_, cudaStreamNonBlocking, priority_low),
        "create transfer stream")) {
        cleanup();
        return false;
    }
    
    // Create timing events
    if (!check_cuda_error(cudaEventCreate(&start_event_), "create start event") ||
        !check_cuda_error(cudaEventCreate(&stop_event_), "create stop event")) {
        cleanup();
        return false;
    }
    
    // Allocate spectrum buffer
    size_t spectrum_size = SPECTRUM_BINS * WATERFALL_HISTORY * sizeof(float);
    if (!check_cuda_error(
        cudaMallocManaged(&d_spectrum_buffer_, spectrum_size),
        "allocate spectrum buffer")) {
        cleanup();
        return false;
    }
    
    // Advise GPU preferred location
    cudaMemAdvise(d_spectrum_buffer_, spectrum_size, 
                  cudaMemAdviseSetPreferredLocation, device_id);
    
    // Allocate and initialize Q-table
    if (!check_cuda_error(
        cudaMallocManaged(&d_qtable_, sizeof(QTableState)),
        "allocate Q-table")) {
        cleanup();
        return false;
    }
    
    // Initialize Q-table
    auto* h_qtable = new QTableState;
    for (uint32_t s = 0; s < NUM_STATES; ++s) {
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            h_qtable->q_values[s][a] = 0.01f * ((float)rand() / RAND_MAX);
            h_qtable->eligibility_traces[s][a] = 0.0f;
        }
        h_qtable->visit_count[s] = 0;
    }
    h_qtable->current_state = 0;
    h_qtable->last_action = 0;
    h_qtable->total_reward = 0.0f;
    
    if (!check_cuda_error(
        cudaMemcpy(d_qtable_, h_qtable, sizeof(QTableState), cudaMemcpyHostToDevice),
        "copy Q-table")) {
        delete h_qtable;
        cleanup();
        return false;
    }
    delete h_qtable;
    
    // Generate waveform bank
    if (generate_waveform_bank() != cudaSuccess) {
        cleanup();
        return false;
    }
    
    // Create FFT plan
    cufftResult fft_result = cufftPlan1d(&fft_plan_, SPECTRUM_BINS, CUFFT_R2C, 1);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "Failed to create FFT plan" << std::endl;
        cleanup();
        return false;
    }
    
    cufftSetStream(fft_plan_, compute_stream_);
    
    initialized_ = true;
    return true;
}

bool CEWCudaModule::process_spectrum(
    const float* spectrum_waterfall,
    ThreatSignature* threats,
    uint32_t num_threats,
    JammingParams* jamming_params,
    uint64_t timestamp_ns
) {
    if (!initialized_) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Record GPU start time
    cudaEventRecord(start_event_, compute_stream_);
    
    // Limit threats to maximum
    if (num_threats > MAX_THREATS) {
        num_threats = MAX_THREATS;
    }
    
    // Configure kernel launch parameters
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_threats + block_size - 1) / block_size;
    
    // Launch adaptive jamming kernel
    adaptive_jamming_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
        spectrum_waterfall,
        threats,
        jamming_params,
        d_qtable_,
        num_threats,
        timestamp_ns
    );
    
    if (!check_cuda_error(cudaGetLastError(), "launch adaptive jamming kernel")) {
        return false;
    }
    
    // Record GPU stop time
    cudaEventRecord(stop_event_, compute_stream_);
    
    // Wait for completion
    if (!check_cuda_error(cudaEventSynchronize(stop_event_), "synchronize")) {
        return false;
    }
    
    // Measure GPU time
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start_event_, stop_event_);
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    metrics_.total_processing_time_us += total_time_us;
    metrics_.gpu_processing_time_us += static_cast<uint64_t>(gpu_time_ms * 1000);
    
    float response_time_us = gpu_time_ms * 1000.0f;
    if (response_time_us > MAX_LATENCY_US) {
        metrics_.deadline_misses++;
    }
    
    metrics_.average_response_time_us = 
        0.95f * metrics_.average_response_time_us + 0.05f * response_time_us;
    
    metrics_.threats_detected += num_threats;
    metrics_.jamming_activated += num_threats;
    
    return true;
}

bool CEWCudaModule::update_qlearning(float reward) {
    if (!initialized_) {
        return false;
    }
    
    // Get current state from threats (simplified)
    uint32_t new_state = 0;
    
    // Update Q-table
    const uint32_t block_size = 256;
    const uint32_t grid_size = (NUM_STATES + block_size - 1) / block_size;
    
    update_qtable_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
        d_qtable_,
        reward,
        new_state
    );
    
    if (!check_cuda_error(cudaGetLastError(), "launch Q-table update kernel")) {
        return false;
    }
    
    // Update effectiveness metric
    metrics_.jamming_effectiveness = 
        0.95f * metrics_.jamming_effectiveness + 0.05f * reward;
    
    return true;
}

cudaError_t CEWCudaModule::generate_waveform_bank() {
    // Allocate waveform bank
    const uint32_t samples_per_waveform = 4096;
    const size_t bank_size = NUM_ACTIONS * samples_per_waveform * sizeof(float);
    
    cudaError_t err = cudaMalloc(&d_waveform_bank_, bank_size);
    if (err != cudaSuccess) return err;
    
    // Generate waveforms on CPU and transfer
    auto* h_waveforms = new float[NUM_ACTIONS * samples_per_waveform];
    
    for (uint32_t w = 0; w < NUM_ACTIONS; ++w) {
        for (uint32_t i = 0; i < samples_per_waveform; ++i) {
            float t = static_cast<float>(i) / samples_per_waveform;
            float phase = 2.0f * M_PI * t;
            
            // Different waveforms for each strategy
            switch (static_cast<JammingStrategy>(w)) {
                case JammingStrategy::BARRAGE_NARROW:
                case JammingStrategy::BARRAGE_WIDE:
                    // Noise
                    h_waveforms[w * samples_per_waveform + i] = 
                        2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
                    break;
                    
                case JammingStrategy::SPOT_JAMMING:
                case JammingStrategy::PHASE_ALIGNED:
                    // Sine wave
                    h_waveforms[w * samples_per_waveform + i] = sinf(phase * (w + 1));
                    break;
                    
                case JammingStrategy::SWEEP_SLOW:
                case JammingStrategy::SWEEP_FAST:
                    // Chirp
                    h_waveforms[w * samples_per_waveform + i] = 
                        sinf(phase * (1.0f + 0.5f * t) * (w + 1));
                    break;
                    
                case JammingStrategy::PULSE_JAMMING:
                case JammingStrategy::TIME_SLICED:
                    // Pulsed
                    h_waveforms[w * samples_per_waveform + i] = 
                        (sinf(phase * 10) > 0.5f) ? sinf(phase * (w + 1)) : 0.0f;
                    break;
                    
                default:
                    // Complex modulation
                    h_waveforms[w * samples_per_waveform + i] = 
                        sinf(phase * (w + 1)) * (1.0f + 0.5f * sinf(phase * 0.1f));
                    break;
            }
        }
    }
    
    err = cudaMemcpy(d_waveform_bank_, h_waveforms, bank_size, cudaMemcpyHostToDevice);
    delete[] h_waveforms;
    
    return err;
}

} // namespace ares::cew