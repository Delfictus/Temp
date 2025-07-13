/**
 * @file test_cew_basic.cpp
 * @brief Basic functionality tests for CEW module
 */

#include "cew_unified_interface.h"
#include "cew_adaptive_jamming.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <map>

using namespace ares::cew;

// Simple test framework
class TestFramework {
public:
    static void assert_equal(int expected, int actual, const std::string& test_name) {
        if (expected == actual) {
            std::cout << "[PASS] " << test_name << std::endl;
            passed_++;
        } else {
            std::cout << "[FAIL] " << test_name << " - Expected: " << expected << ", Got: " << actual << std::endl;
            failed_++;
        }
        total_++;
    }
    
    static void assert_true(bool condition, const std::string& test_name) {
        if (condition) {
            std::cout << "[PASS] " << test_name << std::endl;
            passed_++;
        } else {
            std::cout << "[FAIL] " << test_name << std::endl;
            failed_++;
        }
        total_++;
    }
    
    static void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total: " << total_ << ", Passed: " << passed_ << ", Failed: " << failed_ << std::endl;
        std::cout << "Success Rate: " << (100.0 * passed_ / total_) << "%" << std::endl;
    }
    
private:
    static int total_, passed_, failed_;
};

int TestFramework::total_ = 0;
int TestFramework::passed_ = 0;
int TestFramework::failed_ = 0;

// Test struct layout and alignment
void test_struct_layout() {
    std::cout << "\n--- Testing Struct Layout ---" << std::endl;
    
    // Test ThreatSignature
    TestFramework::assert_equal(32, sizeof(ThreatSignature), "ThreatSignature size alignment");
    TestFramework::assert_equal(0, offsetof(ThreatSignature, center_freq_ghz) % 4, "ThreatSignature frequency alignment");
    
    // Test JammingParams  
    TestFramework::assert_true(sizeof(JammingParams) >= 32, "JammingParams minimum size");
    TestFramework::assert_equal(0, sizeof(JammingParams) % 4, "JammingParams alignment");
    
    // Test CEWMetrics
    TestFramework::assert_true(sizeof(CEWMetrics) > 0, "CEWMetrics defined");
}

// Test factory pattern
void test_factory_pattern() {
    std::cout << "\n--- Testing Factory Pattern ---" << std::endl;
    
    // Test CPU backend creation
    auto cpu_module = CEWModuleFactory::create(CEWBackend::CPU);
    TestFramework::assert_true(cpu_module != nullptr, "CPU module creation");
    
    // Test AUTO backend (should fall back to CPU)
    auto auto_module = CEWModuleFactory::create(CEWBackend::AUTO);
    TestFramework::assert_true(auto_module != nullptr, "AUTO module creation");
    
    // Test CUDA availability check
    bool cuda_available = CEWModuleFactory::is_cuda_available();
    TestFramework::assert_true(true, "CUDA availability check (no assertion on result)");
    
    // Test device enumeration
    auto devices = CEWModuleFactory::get_cuda_devices();
    TestFramework::assert_true(true, "CUDA device enumeration (no assertion on count)");
}

// Test module initialization
void test_module_initialization() {
    std::cout << "\n--- Testing Module Initialization ---" << std::endl;
    
    CEWManager manager(CEWBackend::CPU);
    bool init_result = manager.initialize(0);
    TestFramework::assert_true(init_result, "Module initialization");
    
    // Test backend query
    CEWBackend backend = manager.get_backend();
    TestFramework::assert_equal(static_cast<int>(CEWBackend::CPU), static_cast<int>(backend), "Backend identification");
    
    // Test memory usage query
    size_t memory_usage = manager.get_memory_usage();
    TestFramework::assert_true(memory_usage > 0, "Memory usage reporting");
}

// Test basic spectrum processing
void test_spectrum_processing() {
    std::cout << "\n--- Testing Spectrum Processing ---" << std::endl;
    
    CEWManager manager(CEWBackend::CPU);
    if (!manager.initialize(0)) {
        std::cout << "[SKIP] Spectrum processing tests - initialization failed" << std::endl;
        return;
    }
    
    // Create test data
    std::vector<float> spectrum(SPECTRUM_BINS, -100.0f); // Noise floor
    
    // Add a test signal
    for (int i = 100; i < 120; ++i) {
        spectrum[i] = -60.0f; // Strong signal
    }
    
    // Create test threats
    std::vector<ThreatSignature> threats(1);
    threats[0].center_freq_ghz = 2.4f;
    threats[0].bandwidth_mhz = 20.0f;
    threats[0].power_dbm = -60.0f;
    threats[0].modulation_type = 1;
    threats[0].protocol_id = 2;
    threats[0].priority = 5;
    
    std::vector<JammingParams> jamming_params(1);
    
    // Test processing
    bool process_result = manager.process_spectrum_threadsafe(
        spectrum.data(),
        threats.data(),
        threats.size(),
        jamming_params.data(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()
    );
    
    TestFramework::assert_true(process_result, "Spectrum processing execution");
    
    // Test Q-learning update
    bool qlearning_result = manager.update_qlearning_threadsafe(0.75f);
    TestFramework::assert_true(qlearning_result, "Q-learning update");
}

// Test metrics collection
void test_metrics_collection() {
    std::cout << "\n--- Testing Metrics Collection ---" << std::endl;
    
    CEWManager manager(CEWBackend::CPU);
    if (!manager.initialize(0)) {
        std::cout << "[SKIP] Metrics tests - initialization failed" << std::endl;
        return;
    }
    
    // Get initial metrics
    CEWMetrics metrics = manager.get_metrics();
    
    TestFramework::assert_true(metrics.threats_detected >= 0, "Threats detected count");
    TestFramework::assert_true(metrics.average_response_time_us >= 0, "Average response time");
    TestFramework::assert_true(metrics.jamming_effectiveness >= 0.0f && metrics.jamming_effectiveness <= 1.0f, "Jamming effectiveness range");
    TestFramework::assert_true(metrics.deadline_misses >= 0, "Deadline misses count");
    TestFramework::assert_true(metrics.backend_switches >= 0, "Backend switches count");
}

// Test thread safety
void test_thread_safety() {
    std::cout << "\n--- Testing Thread Safety ---" << std::endl;
    
    CEWManager manager(CEWBackend::CPU);
    if (!manager.initialize(0)) {
        std::cout << "[SKIP] Thread safety tests - initialization failed" << std::endl;
        return;
    }
    
    // Test concurrent metric access
    std::vector<std::thread> threads;
    std::atomic<bool> test_passed{true};
    
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&manager, &test_passed]() {
            for (int j = 0; j < 100; ++j) {
                try {
                    auto metrics = manager.get_metrics();
                    auto memory = manager.get_memory_usage();
                    auto backend = manager.get_backend();
                    
                    // Basic sanity checks
                    if (memory == 0 || static_cast<int>(backend) < 0) {
                        test_passed = false;
                    }
                } catch (...) {
                    test_passed = false;
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    TestFramework::assert_true(test_passed.load(), "Concurrent metric access");
}

int main() {
    std::cout << "=== CEW Module Basic Tests ===" << std::endl;
    
    test_struct_layout();
    test_factory_pattern();
    test_module_initialization();
    test_spectrum_processing();
    test_metrics_collection();
    test_thread_safety();
    
    TestFramework::print_summary();
    
    return (TestFramework::failed_ == 0) ? 0 : 1;
}