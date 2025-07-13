/**
 * @file test_cew_unified.cpp
 * @brief Test program for unified CEW interface
 */

#include "../include/cew_unified_interface.h"
#include "../include/cew_adaptive_jamming.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

using namespace ares::cew;

// Generate synthetic spectrum data
void generate_test_spectrum(float* spectrum, size_t size) {
    static std::mt19937 rng(42);
    static std::normal_distribution<float> noise(-100.0f, 5.0f);
    static std::uniform_real_distribution<float> signal(0.0f, 1.0f);
    
    // Fill with noise floor
    for (size_t i = 0; i < size; ++i) {
        spectrum[i] = noise(rng);
    }
    
    // Add some test signals
    for (int sig = 0; sig < 5; ++sig) {
        size_t center = (size / 6) * (sig + 1);
        size_t width = 20 + sig * 10;
        float power = -60.0f + sig * 5.0f;
        
        for (size_t i = center - width/2; i < center + width/2 && i < size; ++i) {
            spectrum[i] = power + signal(rng) * 3.0f;
        }
    }
}

// Generate test threats from spectrum
std::vector<ThreatSignature> detect_threats(const float* spectrum, size_t size) {
    std::vector<ThreatSignature> threats;
    
    float threshold = -80.0f;
    bool in_signal = false;
    size_t signal_start = 0;
    
    for (size_t i = 0; i < size; ++i) {
        if (!in_signal && spectrum[i] > threshold) {
            in_signal = true;
            signal_start = i;
        } else if (in_signal && spectrum[i] <= threshold) {
            in_signal = false;
            
            ThreatSignature threat;
            size_t signal_center = (signal_start + i) / 2;
            size_t signal_width = i - signal_start;
            
            threat.center_freq_ghz = FREQ_MIN_GHZ + 
                (signal_center * (FREQ_MAX_GHZ - FREQ_MIN_GHZ) / size);
            threat.bandwidth_mhz = (signal_width * 1000.0f * 
                (FREQ_MAX_GHZ - FREQ_MIN_GHZ)) / size;
            threat.power_dbm = spectrum[signal_center];
            threat.modulation_type = threats.size() % 4;
            threat.protocol_id = threats.size() % 8;
            threat.priority = 5 - std::min(5, (int)threats.size());
            
            threats.push_back(threat);
        }
    }
    
    return threats;
}

void print_metrics(const CEWMetrics& metrics) {
    std::cout << "\n=== CEW Performance Metrics ===" << std::endl;
    std::cout << "Threats Detected: " << metrics.threats_detected << std::endl;
    std::cout << "Jamming Activated: " << metrics.jamming_activated << std::endl;
    std::cout << "Average Response Time: " << std::fixed << std::setprecision(2) 
              << metrics.average_response_time_us << " µs" << std::endl;
    std::cout << "Jamming Effectiveness: " << std::fixed << std::setprecision(3) 
              << metrics.jamming_effectiveness << std::endl;
    std::cout << "Deadline Misses: " << metrics.deadline_misses << std::endl;
    std::cout << "Backend Switches: " << metrics.backend_switches << std::endl;
    std::cout << "Total Processing Time: " << metrics.total_processing_time_us << " µs" << std::endl;
    std::cout << "CPU Processing Time: " << metrics.cpu_processing_time_us << " µs" << std::endl;
    std::cout << "GPU Processing Time: " << metrics.gpu_processing_time_us << " µs" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== ARES CEW Unified Interface Test ===" << std::endl;
    
    // Check available CUDA devices
    auto cuda_devices = CEWModuleFactory::get_cuda_devices();
    if (!cuda_devices.empty()) {
        std::cout << "\nAvailable CUDA devices:" << std::endl;
        for (size_t i = 0; i < cuda_devices.size(); ++i) {
            std::cout << "  [" << i << "] " << cuda_devices[i] << std::endl;
        }
    } else {
        std::cout << "\nNo CUDA devices available, will use CPU backend" << std::endl;
    }
    
    // Test all backend types
    std::vector<CEWBackend> backends = {CEWBackend::AUTO, CEWBackend::CPU};
    if (CEWModuleFactory::is_cuda_available()) {
        backends.push_back(CEWBackend::CUDA);
    }
    
    for (auto backend : backends) {
        std::cout << "\n\n--- Testing backend: ";
        switch (backend) {
            case CEWBackend::AUTO: std::cout << "AUTO"; break;
            case CEWBackend::CPU: std::cout << "CPU"; break;
            case CEWBackend::CUDA: std::cout << "CUDA"; break;
        }
        std::cout << " ---" << std::endl;
        
        // Create CEW manager
        CEWManager cew_manager(backend);
        
        // Initialize
        if (!cew_manager.initialize(0)) {
            std::cerr << "Failed to initialize CEW manager" << std::endl;
            continue;
        }
        
        std::cout << "Active backend: ";
        switch (cew_manager.get_backend()) {
            case CEWBackend::CPU: std::cout << "CPU"; break;
            case CEWBackend::CUDA: std::cout << "CUDA"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << std::endl;
        
        // Generate test data
        std::vector<float> spectrum(SPECTRUM_BINS * WATERFALL_HISTORY);
        generate_test_spectrum(spectrum.data(), spectrum.size());
        
        // Run multiple iterations
        const int num_iterations = 100;
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Detect threats
            auto threats = detect_threats(spectrum.data(), SPECTRUM_BINS);
            
            // Allocate jamming params
            std::vector<JammingParams> jamming_params(threats.size());
            
            // Process spectrum
            uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            bool success = cew_manager.process_spectrum_threadsafe(
                spectrum.data(),
                threats.data(),
                threats.size(),
                jamming_params.data(),
                timestamp_ns
            );
            
            if (!success) {
                std::cerr << "Failed to process spectrum at iteration " << iter << std::endl;
                break;
            }
            
            // Simulate reward feedback
            float reward = 0.7f + 0.3f * ((float)rand() / RAND_MAX);
            cew_manager.update_qlearning_threadsafe(reward);
            
            // Update spectrum for next iteration
            generate_test_spectrum(spectrum.data(), spectrum.size());
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start).count();
        
        std::cout << "\nCompleted " << num_iterations << " iterations in " 
                  << total_duration << " ms" << std::endl;
        std::cout << "Average time per iteration: " 
                  << (float)total_duration / num_iterations << " ms" << std::endl;
        
        // Print metrics
        print_metrics(cew_manager.get_metrics());
        
        // Memory usage
        std::cout << "\nMemory usage: " << cew_manager.get_memory_usage() / (1024*1024) 
                  << " MB" << std::endl;
    }
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}