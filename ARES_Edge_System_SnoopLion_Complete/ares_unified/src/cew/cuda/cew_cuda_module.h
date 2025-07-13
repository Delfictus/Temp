/**
 * @file cew_cuda_module.h
 * @brief CUDA implementation of the CEW module
 */

#ifndef ARES_CEW_CUDA_MODULE_H
#define ARES_CEW_CUDA_MODULE_H

#include "../include/cew_unified_interface.h"
#include "../include/cew_adaptive_jamming.h"
#include <cuda_runtime.h>
#include <cufft.h>

namespace ares::cew {

class CEWCudaModule : public ICEWModule {
public:
    CEWCudaModule();
    ~CEWCudaModule() override;
    
    // ICEWModule interface implementation
    bool initialize(int device_id = 0) override;
    
    bool process_spectrum(
        const float* spectrum_waterfall,
        ThreatSignature* threats,
        uint32_t num_threats,
        JammingParams* jamming_params,
        uint64_t timestamp_ns
    ) override;
    
    bool update_qlearning(float reward) override;
    
    CEWMetrics get_metrics() const override { return metrics_; }
    
    CEWBackend get_backend() const override { return CEWBackend::CUDA; }
    
    bool is_cuda_available() const override;
    
private:
    // CUDA resources
    bool initialized_;
    int device_id_;
    
    // Device pointers
    float* d_spectrum_buffer_;
    QTableState* d_qtable_;
    float* d_waveform_bank_;
    cufftHandle fft_plan_;
    
    // CUDA streams for concurrent execution
    cudaStream_t compute_stream_;
    cudaStream_t transfer_stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Metrics
    CEWMetrics metrics_;
    
    // Internal methods
    void cleanup();
    cudaError_t generate_waveform_bank();
    cudaError_t analyze_threats(const float* spectrum, ThreatSignature* threats);
    
    // Convert CUDA errors to bool
    bool check_cuda_error(cudaError_t err, const char* operation);
};

// CUDA Kernels (defined in separate .cu files)
extern "C" {
    
__global__ void adaptive_jamming_kernel(
    const float* __restrict__ spectrum_waterfall,
    const ThreatSignature* __restrict__ threats,
    JammingParams* __restrict__ jamming_params,
    QTableState* q_state,
    uint32_t num_threats,
    uint64_t timestamp_ns
);

__global__ void update_qtable_kernel(
    QTableState* q_state,
    float reward,
    uint32_t new_state
);

__global__ void generate_jamming_waveform_kernel(
    float* __restrict__ waveform_out,
    const JammingParams* __restrict__ params,
    const float* __restrict__ waveform_bank,
    uint32_t samples_per_symbol
);

} // extern "C"

} // namespace ares::cew

#endif // ARES_CEW_CUDA_MODULE_H