/**
 * @file cew_cpu_module.cpp
 * @brief CPU implementation of the CEW module with SIMD optimizations
 */

#include "cew_cpu_module.h"
#include "simd_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <immintrin.h>  // For SIMD intrinsics

namespace ares::cew {

// Thread pool implementation
CEWCpuModule::ThreadPool::ThreadPool(size_t num_threads) 
    : stop_(false), active_tasks_(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                    active_tasks_++;
                }
                task();
                active_tasks_--;
                finished_.notify_all();
            }
        });
    }
}

CEWCpuModule::ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
        worker.join();
    }
}

template<typename F>
void CEWCpuModule::ThreadPool::enqueue(F&& f) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
}

void CEWCpuModule::ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finished_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

// CEWCpuModule implementation
CEWCpuModule::CEWCpuModule()
    : rng_(std::chrono::steady_clock::now().time_since_epoch().count())
    , uniform_dist_(0.0f, 1.0f) {
    std::memset(&metrics_, 0, sizeof(metrics_));
    start_time_ = std::chrono::high_resolution_clock::now();
}

CEWCpuModule::~CEWCpuModule() = default;

bool CEWCpuModule::initialize(int device_id) {
    // Create thread pool with optimal number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    thread_pool_ = std::make_unique<ThreadPool>(num_threads);
    
    // Allocate spectrum buffer
    spectrum_buffer_.resize(SPECTRUM_BINS * WATERFALL_HISTORY);
    
    // Initialize Q-table
    qtable_ = std::make_unique<QTableState>();
    for (uint32_t s = 0; s < NUM_STATES; ++s) {
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            qtable_->q_values[s][a] = 0.01f * uniform_dist_(rng_);
            qtable_->eligibility_traces[s][a] = 0.0f;
        }
        qtable_->visit_count[s] = 0;
    }
    qtable_->current_state = 0;
    qtable_->last_action = 0;
    qtable_->total_reward = 0.0f;
    
    // Generate waveform bank
    generate_waveform_bank();
    
    // Optimize thread affinity for better cache performance
    optimize_thread_affinity();
    
    return true;
}

void CEWCpuModule::generate_waveform_bank() {
    const uint32_t samples_per_waveform = 4096;
    waveform_bank_.resize(NUM_ACTIONS * samples_per_waveform);
    
    for (uint32_t w = 0; w < NUM_ACTIONS; ++w) {
        float* waveform = &waveform_bank_[w * samples_per_waveform];
        
        // Generate waveforms using SIMD where possible
        for (uint32_t i = 0; i < samples_per_waveform; i += 8) {
            __m256 t = _mm256_set_ps(
                (i+7)/(float)samples_per_waveform, (i+6)/(float)samples_per_waveform,
                (i+5)/(float)samples_per_waveform, (i+4)/(float)samples_per_waveform,
                (i+3)/(float)samples_per_waveform, (i+2)/(float)samples_per_waveform,
                (i+1)/(float)samples_per_waveform, (i+0)/(float)samples_per_waveform
            );
            
            __m256 phase = _mm256_mul_ps(t, _mm256_set1_ps(2.0f * M_PI));
            
            switch (static_cast<JammingStrategy>(w)) {
                case JammingStrategy::BARRAGE_NARROW:
                case JammingStrategy::BARRAGE_WIDE:
                    // Noise - can't vectorize random
                    for (int j = 0; j < 8; ++j) {
                        waveform[i + j] = 2.0f * uniform_dist_(rng_) - 1.0f;
                    }
                    break;
                    
                case JammingStrategy::SPOT_JAMMING:
                case JammingStrategy::PHASE_ALIGNED:
                    // Sine wave - use approximation for SIMD
                    {
                        __m256 result = _mm256_sin_ps(_mm256_mul_ps(phase, _mm256_set1_ps(w + 1)));
                        _mm256_storeu_ps(&waveform[i], result);
                    }
                    break;
                    
                default:
                    // Complex modulation - fall back to scalar
                    for (int j = 0; j < 8; ++j) {
                        float t_scalar = (i + j) / (float)samples_per_waveform;
                        float phase_scalar = 2.0f * M_PI * t_scalar;
                        waveform[i + j] = sinf(phase_scalar * (w + 1)) * 
                                        (1.0f + 0.5f * sinf(phase_scalar * 0.1f));
                    }
                    break;
            }
        }
    }
}

bool CEWCpuModule::process_spectrum(
    const float* spectrum_waterfall,
    ThreatSignature* threats,
    uint32_t num_threats,
    JammingParams* jamming_params,
    uint64_t timestamp_ns
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (num_threats > MAX_THREATS) {
        num_threats = MAX_THREATS;
    }
    
    // Calculate optimal batch size for thread pool
    size_t batch_size = calculate_optimal_batch_size(num_threats);
    size_t num_batches = (num_threats + batch_size - 1) / batch_size;
    
    // Process threats in parallel batches
    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start_idx = batch * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, (size_t)num_threats);
        
        thread_pool_->enqueue([this, spectrum_waterfall, threats, jamming_params, 
                              start_idx, end_idx, timestamp_ns]() {
            process_threat_batch(spectrum_waterfall, threats, jamming_params, 
                               start_idx, end_idx, timestamp_ns);
        });
    }
    
    // Wait for all threads to complete
    thread_pool_->wait_all();
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        metrics_.total_processing_time_us += elapsed_us;
        metrics_.cpu_processing_time_us += elapsed_us;
        
        if (elapsed_us > MAX_LATENCY_US) {
            metrics_.deadline_misses++;
        }
        
        metrics_.average_response_time_us = 
            0.95f * metrics_.average_response_time_us + 0.05f * elapsed_us;
        
        metrics_.threats_detected += num_threats;
        metrics_.jamming_activated += num_threats;
    }
    
    return true;
}

void CEWCpuModule::process_threat_batch(
    const float* spectrum,
    const ThreatSignature* threats,
    JammingParams* jamming_params,
    size_t start_idx,
    size_t end_idx,
    uint64_t timestamp_ns
) {
    for (size_t i = start_idx; i < end_idx; ++i) {
        const ThreatSignature& threat = threats[i];
        JammingParams& params = jamming_params[i];
        
        // Quantize threat to state
        uint32_t state = quantize_threat_state(threat);
        
        // Select action using Q-learning
        uint32_t action = select_action(state);
        
        // Generate jamming parameters based on action
        params.strategy = static_cast<uint8_t>(action);
        params.center_freq_ghz = threat.center_freq_ghz;
        params.bandwidth_mhz = threat.bandwidth_mhz * 1.5f; // Wider jamming bandwidth
        params.power_watts = 10.0f * powf(10.0f, (threat.power_dbm - 30.0f) / 10.0f) * 2.0f;
        params.waveform_id = action;
        params.duration_ms = 100; // Default duration
        params.phase_offset = 0.0f;
        
        // Strategy-specific parameters
        switch (static_cast<JammingStrategy>(action)) {
            case JammingStrategy::SWEEP_SLOW:
                params.sweep_rate_mhz_per_sec = 100.0f;
                break;
            case JammingStrategy::SWEEP_FAST:
                params.sweep_rate_mhz_per_sec = 1000.0f;
                break;
            case JammingStrategy::FREQUENCY_HOPPING:
                params.sweep_rate_mhz_per_sec = 5000.0f;
                break;
            default:
                params.sweep_rate_mhz_per_sec = 0.0f;
                break;
        }
    }
}

uint32_t CEWCpuModule::quantize_threat_state(const ThreatSignature& threat) {
    uint32_t freq_band = std::min(3u, quantize_frequency(threat.center_freq_ghz));
    uint32_t power_level = std::min(3u, quantize_power(threat.power_dbm));
    uint32_t bw_category = std::min(3u, quantize_bandwidth(threat.bandwidth_mhz));
    
    return (freq_band << 6) | (power_level << 4) | 
           (bw_category << 2) | (threat.modulation_type & 0x3);
}

uint32_t CEWCpuModule::select_action(uint32_t state) {
    std::lock_guard<std::mutex> lock(qtable_mutex_);
    
    // Epsilon-greedy policy
    if (uniform_dist_(rng_) < EPSILON) {
        // Explore: random action
        return static_cast<uint32_t>(uniform_dist_(rng_) * NUM_ACTIONS);
    } else {
        // Exploit: best action from Q-table
        float max_q = -1e9f;
        uint32_t best_action = 0;
        
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            if (qtable_->q_values[state][a] > max_q) {
                max_q = qtable_->q_values[state][a];
                best_action = a;
            }
        }
        
        return best_action;
    }
}

bool CEWCpuModule::update_qlearning(float reward) {
    std::lock_guard<std::mutex> lock(qtable_mutex_);
    
    uint32_t current_state = qtable_->current_state;
    uint32_t last_action = qtable_->last_action;
    
    // TD(0) update with eligibility traces
    float td_error = reward - qtable_->q_values[current_state][last_action];
    
    // Update Q-value
    qtable_->q_values[current_state][last_action] += ALPHA * td_error;
    
    // Update eligibility trace
    qtable_->eligibility_traces[current_state][last_action] = 1.0f;
    
    // Decay all eligibility traces
    for (uint32_t s = 0; s < NUM_STATES; ++s) {
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            qtable_->eligibility_traces[s][a] *= GAMMA * 0.9f; // Lambda = 0.9
        }
    }
    
    // Update metrics
    {
        std::lock_guard<std::mutex> metric_lock(metrics_mutex_);
        metrics_.jamming_effectiveness = 
            0.95f * metrics_.jamming_effectiveness + 0.05f * reward;
    }
    
    return true;
}

CEWMetrics CEWCpuModule::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void CEWCpuModule::optimize_thread_affinity() {
    // Platform-specific optimization
#ifdef __linux__
    // Set thread affinity to avoid NUMA issues
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int num_cpus = std::thread::hardware_concurrency();
    for (int i = 0; i < num_cpus; ++i) {
        CPU_SET(i, &cpuset);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

size_t CEWCpuModule::calculate_optimal_batch_size(uint32_t num_threats) {
    // Calculate based on cache size and thread count
    size_t cache_line_size = 64;
    size_t threat_size = sizeof(ThreatSignature) + sizeof(JammingParams);
    size_t threats_per_cache_line = cache_line_size / threat_size;
    
    // Aim for good cache locality
    size_t optimal_batch = std::max(threats_per_cache_line * 4, (size_t)16);
    
    // But don't create too many small batches
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threats / optimal_batch < num_threads) {
        optimal_batch = std::max((size_t)1, num_threats / num_threads);
    }
    
    return optimal_batch;
}

void CEWCpuModule::apply_fft(const float* input, float* output, size_t size) {
    // Simplified FFT implementation - in production, use FFTW or similar
    // For now, just copy the data
    std::copy(input, input + size, output);
}

void CEWCpuModule::generate_jamming_waveform(
    float* waveform_out,
    const JammingParams& params,
    size_t num_samples
) {
    // Generate waveform based on parameters
    size_t waveform_idx = params.waveform_id;
    const float* base_waveform = &waveform_bank_[waveform_idx * 4096];
    
    // Apply frequency and phase modulation
    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / num_samples;
        float phase = 2.0f * M_PI * params.center_freq_ghz * t + params.phase_offset;
        
        // Apply sweep if needed
        if (params.sweep_rate_mhz_per_sec > 0) {
            phase += M_PI * params.sweep_rate_mhz_per_sec * t * t / 1000.0f;
        }
        
        // Modulate base waveform
        size_t base_idx = (i * 4096) / num_samples;
        waveform_out[i] = base_waveform[base_idx] * sinf(phase);
    }
}

} // namespace ares::cew