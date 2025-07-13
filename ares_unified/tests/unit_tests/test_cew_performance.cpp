/**
 * @file test_cew_performance.cpp
 * @brief Performance and real-time constraint tests for CEW module
 */

#include "cew_unified_interface.h"
#include "cew_adaptive_jamming.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>

using namespace ares::cew;

// Performance test framework
class PerformanceTest {
public:
    static void time_operation(const std::string& name, std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[PERF] " << name << ": " << duration.count() << " μs" << std::endl;
        
        // Record for analysis
        results_[name] = duration.count();
    }
    
    static void benchmark_throughput(const std::string& name, std::function<void()> operation, int iterations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            operation();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double ops_per_sec = (iterations * 1000000.0) / total_time.count();
        double time_per_op = (double)total_time.count() / iterations;
        
        std::cout << "[THROUGHPUT] " << name << ":" << std::endl;
        std::cout << "  Operations: " << iterations << std::endl;
        std::cout << "  Total time: " << total_time.count() << " μs" << std::endl;
        std::cout << "  Time per op: " << time_per_op << " μs" << std::endl;
        std::cout << "  Ops/sec: " << (int)ops_per_sec << std::endl;
        
        throughput_results_[name] = {time_per_op, ops_per_sec};
    }
    
    static void print_summary() {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        
        // Check real-time constraints
        std::cout << "\nReal-time Constraint Analysis:" << std::endl;
        for (const auto& [name, time_us] : results_) {
            bool meets_constraint = time_us < 10000; // 10ms requirement
            std::cout << "  " << name << ": " << time_us << " μs " 
                     << (meets_constraint ? "[PASS]" : "[FAIL - exceeds 10ms]") << std::endl;
        }
        
        std::cout << "\nThroughput Analysis:" << std::endl;
        for (const auto& [name, metrics] : throughput_results_) {
            std::cout << "  " << name << ": " << (int)metrics.second << " ops/sec" << std::endl;
        }
    }
    
private:
    static std::map<std::string, double> results_;
    static std::map<std::string, std::pair<double, double>> throughput_results_;
};

std::map<std::string, double> PerformanceTest::results_;
std::map<std::string, std::pair<double, double>> PerformanceTest::throughput_results_;

// Generate realistic test data
class TestDataGenerator {
public:
    static std::vector<float> generate_spectrum(size_t size, int num_signals = 3) {
        std::vector<float> spectrum(size, -100.0f); // Noise floor
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> noise(-5.0f, 5.0f);
        std::uniform_int_distribution<int> pos(50, size - 100);
        std::uniform_real_distribution<float> power(-80.0f, -40.0f);
        
        // Add noise
        for (size_t i = 0; i < size; ++i) {
            spectrum[i] += noise(rng);
        }
        
        // Add signals
        for (int sig = 0; sig < num_signals; ++sig) {
            int center = pos(rng);
            int width = 10 + sig * 5;
            float signal_power = power(rng);
            
            for (int i = center - width/2; i < center + width/2 && i < (int)size; ++i) {
                if (i >= 0) {
                    spectrum[i] = signal_power + noise(rng) * 0.5f;
                }
            }
        }
        
        return spectrum;
    }
    
    static std::vector<ThreatSignature> generate_threats(size_t count) {
        std::vector<ThreatSignature> threats(count);
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> freq(1.0f, 6.0f);
        std::uniform_real_distribution<float> bw(5.0f, 40.0f);
        std::uniform_real_distribution<float> power(-90.0f, -30.0f);
        std::uniform_int_distribution<int> mod(0, 7);
        std::uniform_int_distribution<int> proto(0, 15);
        std::uniform_int_distribution<int> prio(1, 10);
        
        for (size_t i = 0; i < count; ++i) {
            threats[i].center_freq_ghz = freq(rng);
            threats[i].bandwidth_mhz = bw(rng);
            threats[i].power_dbm = power(rng);
            threats[i].modulation_type = mod(rng);
            threats[i].protocol_id = proto(rng);
            threats[i].priority = prio(rng);
        }
        
        return threats;
    }
};

// Test module initialization performance
void test_initialization_performance() {
    std::cout << "\n--- Testing Initialization Performance ---" << std::endl;
    
    PerformanceTest::time_operation("Module Creation", []() {
        auto module = CEWModuleFactory::create(CEWBackend::CPU);
    });
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (module) {
        PerformanceTest::time_operation("Module Initialization", [&module]() {
            module->initialize(0);
        });
    }
}

// Test spectrum processing performance
void test_spectrum_processing_performance() {
    std::cout << "\n--- Testing Spectrum Processing Performance ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Spectrum processing performance - initialization failed" << std::endl;
        return;
    }
    
    // Test different data sizes
    std::vector<size_t> spectrum_sizes = {SPECTRUM_BINS/4, SPECTRUM_BINS/2, SPECTRUM_BINS, SPECTRUM_BINS*2};
    std::vector<size_t> threat_counts = {1, 5, 10, 20};
    
    for (auto spectrum_size : spectrum_sizes) {
        for (auto threat_count : threat_counts) {
            std::string test_name = "Spectrum Processing (size=" + std::to_string(spectrum_size) + 
                                  ", threats=" + std::to_string(threat_count) + ")";
            
            auto spectrum = TestDataGenerator::generate_spectrum(spectrum_size);
            auto threats = TestDataGenerator::generate_threats(threat_count);
            std::vector<JammingParams> jamming_params(threat_count);
            
            PerformanceTest::time_operation(test_name, [&]() {
                module->process_spectrum_threadsafe(
                    spectrum.data(),
                    threats.data(),
                    threat_count,
                    jamming_params.data(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()
                );
            });
        }
    }
}

// Test throughput under load
void test_throughput_performance() {
    std::cout << "\n--- Testing Throughput Performance ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Throughput performance - initialization failed" << std::endl;
        return;
    }
    
    auto spectrum = TestDataGenerator::generate_spectrum(SPECTRUM_BINS);
    auto threats = TestDataGenerator::generate_threats(5);
    std::vector<JammingParams> jamming_params(5);
    
    // Test sustained throughput
    PerformanceTest::benchmark_throughput("Sustained Processing", [&]() {
        module->process_spectrum_threadsafe(
            spectrum.data(),
            threats.data(),
            5,
            jamming_params.data(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()
        );
    }, 1000);
    
    // Test Q-learning update throughput
    PerformanceTest::benchmark_throughput("Q-learning Updates", [&]() {
        module->update_qlearning_threadsafe(0.7f);
    }, 10000);
    
    // Test metrics collection throughput
    PerformanceTest::benchmark_throughput("Metrics Collection", [&]() {
        auto metrics = module->get_metrics();
    }, 100000);
}

// Test memory allocation performance
void test_memory_performance() {
    std::cout << "\n--- Testing Memory Performance ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Memory performance - initialization failed" << std::endl;
        return;
    }
    
    // Monitor memory usage over time
    size_t initial_memory = module->get_memory_usage();
    std::cout << "[INFO] Initial memory usage: " << initial_memory / 1024 << " KB" << std::endl;
    
    // Run processing operations and monitor memory
    auto spectrum = TestDataGenerator::generate_spectrum(SPECTRUM_BINS);
    auto threats = TestDataGenerator::generate_threats(10);
    std::vector<JammingParams> jamming_params(10);
    
    for (int i = 0; i < 100; ++i) {
        module->process_spectrum_threadsafe(
            spectrum.data(),
            threats.data(),
            10,
            jamming_params.data(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()
        );
        
        if (i % 20 == 0) {
            size_t current_memory = module->get_memory_usage();
            std::cout << "[INFO] Memory after " << i << " operations: " 
                     << current_memory / 1024 << " KB" << std::endl;
        }
    }
    
    size_t final_memory = module->get_memory_usage();
    double memory_growth = ((double)final_memory - initial_memory) / initial_memory * 100.0;
    
    std::cout << "[INFO] Final memory usage: " << final_memory / 1024 << " KB" << std::endl;
    std::cout << "[INFO] Memory growth: " << memory_growth << "%" << std::endl;
    
    if (memory_growth > 50.0) {
        std::cout << "[WARN] Significant memory growth detected - potential leak" << std::endl;
    } else {
        std::cout << "[PASS] Memory usage stable" << std::endl;
    }
}

// Test concurrent performance
void test_concurrent_performance() {
    std::cout << "\n--- Testing Concurrent Performance ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Concurrent performance - initialization failed" << std::endl;
        return;
    }
    
    const int num_threads = 4;
    const int operations_per_thread = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&module, operations_per_thread]() {
            auto spectrum = TestDataGenerator::generate_spectrum(SPECTRUM_BINS / 2);
            auto threats = TestDataGenerator::generate_threats(3);
            std::vector<JammingParams> jamming_params(3);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                module->process_spectrum_threadsafe(
                    spectrum.data(),
                    threats.data(),
                    3,
                    jamming_params.data(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()
                );
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    int total_operations = num_threads * operations_per_thread;
    double ops_per_sec = (total_operations * 1000000.0) / total_time.count();
    
    std::cout << "[CONCURRENT] " << num_threads << " threads, " << total_operations << " operations" << std::endl;
    std::cout << "[CONCURRENT] Total time: " << total_time.count() << " μs" << std::endl;
    std::cout << "[CONCURRENT] Concurrent throughput: " << (int)ops_per_sec << " ops/sec" << std::endl;
}

int main() {
    std::cout << "=== CEW Module Performance Tests ===" << std::endl;
    
    test_initialization_performance();
    test_spectrum_processing_performance();
    test_throughput_performance();
    test_memory_performance();
    test_concurrent_performance();
    
    PerformanceTest::print_summary();
    
    return 0;
}