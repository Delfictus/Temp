/**
 * @file cew_cpu_module.h
 * @brief CPU implementation of the CEW module
 */

#ifndef ARES_CEW_CPU_MODULE_H
#define ARES_CEW_CPU_MODULE_H

#include "../include/cew_unified_interface.h"
#include "../include/cew_adaptive_jamming.h"
#include <vector>
#include <mutex>
#include <random>
#include <chrono>
#include <thread>

namespace ares::cew {

class CEWCpuModule : public ICEWModule {
public:
    CEWCpuModule();
    ~CEWCpuModule() override;
    
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
    
    CEWMetrics get_metrics() const override;
    
    CEWBackend get_backend() const override { return CEWBackend::CPU; }
    
    bool is_cuda_available() const override { return false; }
    
private:
    // Thread pool for parallel processing
    class ThreadPool;
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // CPU memory buffers
    std::vector<float> spectrum_buffer_;
    std::unique_ptr<QTableState> qtable_;
    std::vector<float> waveform_bank_;
    
    // Synchronization
    mutable std::mutex metrics_mutex_;
    std::mutex qtable_mutex_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
    
    // Metrics
    CEWMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Internal methods
    void generate_waveform_bank();
    uint32_t quantize_threat_state(const ThreatSignature& threat);
    uint32_t select_action(uint32_t state);
    void process_threat_batch(
        const float* spectrum,
        const ThreatSignature* threats,
        JammingParams* jamming_params,
        size_t start_idx,
        size_t end_idx,
        uint64_t timestamp_ns
    );
    
    // Signal processing functions
    void apply_fft(const float* input, float* output, size_t size);
    void generate_jamming_waveform(
        float* waveform_out,
        const JammingParams& params,
        size_t num_samples
    );
    
    // Performance optimization
    void optimize_thread_affinity();
    size_t calculate_optimal_batch_size(uint32_t num_threats);
};

// Thread pool implementation for parallel processing
class CEWCpuModule::ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    template<typename F>
    void enqueue(F&& f);
    
    void wait_all();
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    bool stop_;
    std::atomic<size_t> active_tasks_;
};

} // namespace ares::cew

#endif // ARES_CEW_CPU_MODULE_H