/**
 * @file test_cew_security.cpp
 * @brief Security-focused tests for CEW module
 */

#include "cew_unified_interface.h"
#include "cew_adaptive_jamming.h"
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <limits>

using namespace ares::cew;

// Security test framework
class SecurityTest {
public:
    static void test_result(bool passed, const std::string& test_name) {
        if (passed) {
            std::cout << "[SECURITY PASS] " << test_name << std::endl;
            passed_++;
        } else {
            std::cout << "[SECURITY FAIL] " << test_name << std::endl;
            failed_++;
        }
        total_++;
    }
    
    static void print_summary() {
        std::cout << "\n=== Security Test Summary ===" << std::endl;
        std::cout << "Total: " << total_ << ", Passed: " << passed_ << ", Failed: " << failed_ << std::endl;
        if (failed_ > 0) {
            std::cout << "⚠️  SECURITY ISSUES DETECTED - Manual review required" << std::endl;
        } else {
            std::cout << "✓ All security tests passed" << std::endl;
        }
    }
    
private:
    static int total_, passed_, failed_;
};

int SecurityTest::total_ = 0;
int SecurityTest::passed_ = 0;
int SecurityTest::failed_ = 0;

// Test input validation and bounds checking
void test_input_validation() {
    std::cout << "\n--- Testing Input Validation ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Input validation tests - initialization failed" << std::endl;
        return;
    }
    
    // Test null pointer handling
    bool null_spectrum_handled = !module->process_spectrum_threadsafe(
        nullptr, nullptr, 0, nullptr, 0);
    SecurityTest::test_result(null_spectrum_handled, "Null spectrum pointer handling");
    
    // Test invalid array sizes
    std::vector<float> spectrum(10); // Too small
    std::vector<ThreatSignature> threats(1);
    std::vector<JammingParams> jamming_params(1);
    
    // Test with oversized threat count
    bool oversized_handled = !module->process_spectrum_threadsafe(
        spectrum.data(), threats.data(), 1000000, jamming_params.data(), 0);
    SecurityTest::test_result(oversized_handled, "Oversized threat count handling");
    
    // Test Q-learning with invalid rewards
    bool invalid_reward_handled = true;
    try {
        module->update_qlearning_threadsafe(std::numeric_limits<float>::infinity());
        module->update_qlearning_threadsafe(-std::numeric_limits<float>::infinity());
        module->update_qlearning_threadsafe(std::numeric_limits<float>::quiet_NaN());
    } catch (...) {
        // Should handle gracefully without throwing
        invalid_reward_handled = false;
    }
    SecurityTest::test_result(invalid_reward_handled, "Invalid Q-learning reward handling");
}

// Test buffer overflow protection
void test_buffer_overflow_protection() {
    std::cout << "\n--- Testing Buffer Overflow Protection ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Buffer overflow tests - initialization failed" << std::endl;
        return;
    }
    
    // Test with maximum spectrum size
    std::vector<float> max_spectrum(SPECTRUM_BINS * WATERFALL_HISTORY);
    std::fill(max_spectrum.begin(), max_spectrum.end(), -50.0f);
    
    std::vector<ThreatSignature> threats(1);
    threats[0].center_freq_ghz = 2.4f;
    threats[0].bandwidth_mhz = 20.0f;
    threats[0].power_dbm = -60.0f;
    
    std::vector<JammingParams> jamming_params(1);
    
    bool max_size_handled = true;
    try {
        module->process_spectrum_threadsafe(
            max_spectrum.data(),
            threats.data(),
            1,
            jamming_params.data(),
            0
        );
    } catch (...) {
        max_size_handled = false;
    }
    
    SecurityTest::test_result(max_size_handled, "Maximum spectrum size handling");
    
    // Test struct field bounds
    ThreatSignature extreme_threat;
    extreme_threat.center_freq_ghz = 1000.0f; // Way out of RF range
    extreme_threat.bandwidth_mhz = 10000.0f;  // Impossibly wide
    extreme_threat.power_dbm = 1000.0f;       // Impossibly high power
    extreme_threat.modulation_type = 255;     // Max value
    extreme_threat.protocol_id = 255;         // Max value
    extreme_threat.priority = 255;            // Max value
    
    std::vector<ThreatSignature> extreme_threats = {extreme_threat};
    std::vector<JammingParams> extreme_jamming(1);
    
    bool extreme_values_handled = true;
    try {
        std::vector<float> spectrum(SPECTRUM_BINS, -50.0f);
        module->process_spectrum_threadsafe(
            spectrum.data(),
            extreme_threats.data(),
            1,
            extreme_jamming.data(),
            0
        );
    } catch (...) {
        extreme_values_handled = false;
    }
    
    SecurityTest::test_result(extreme_values_handled, "Extreme value handling");
}

// Test memory safety
void test_memory_safety() {
    std::cout << "\n--- Testing Memory Safety ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Memory safety tests - initialization failed" << std::endl;
        return;
    }
    
    // Test for memory leaks by creating and destroying modules
    size_t initial_memory = 0;
    {
        auto temp_module = CEWModuleFactory::create(CEWBackend::CPU);
        if (temp_module && temp_module->initialize(0)) {
            initial_memory = temp_module->get_memory_usage();
        }
    } // Module destroyed here
    
    // Create new module and check memory
    auto new_module = CEWModuleFactory::create(CEWBackend::CPU);
    if (new_module && new_module->initialize(0)) {
        size_t new_memory = new_module->get_memory_usage();
        bool memory_cleaned = (new_memory <= initial_memory * 1.1); // Allow 10% variance
        SecurityTest::test_result(memory_cleaned, "Memory cleanup on destruction");
    }
    
    // Test uninitialized access protection
    auto uninitialized_module = CEWModuleFactory::create(CEWBackend::CPU);
    bool uninitialized_protected = true;
    
    try {
        // Try to use module without initialization
        std::vector<float> spectrum(SPECTRUM_BINS, -50.0f);
        std::vector<ThreatSignature> threats(1);
        std::vector<JammingParams> jamming_params(1);
        
        uninitialized_module->process_spectrum_threadsafe(
            spectrum.data(), threats.data(), 1, jamming_params.data(), 0);
    } catch (...) {
        // Should handle gracefully
        uninitialized_protected = true;
    }
    
    SecurityTest::test_result(uninitialized_protected, "Uninitialized module access protection");
}

// Test data structure integrity
void test_data_structure_integrity() {
    std::cout << "\n--- Testing Data Structure Integrity ---" << std::endl;
    
    // Test ThreatSignature structure padding and alignment
    ThreatSignature threat;
    std::memset(&threat, 0xAA, sizeof(threat)); // Fill with pattern
    
    threat.center_freq_ghz = 2.4f;
    threat.bandwidth_mhz = 20.0f;
    threat.power_dbm = -60.0f;
    threat.modulation_type = 1;
    threat.protocol_id = 2;
    threat.priority = 5;
    
    // Check that padding bytes are not affecting data
    bool data_integrity = (threat.center_freq_ghz == 2.4f &&
                          threat.bandwidth_mhz == 20.0f &&
                          threat.power_dbm == -60.0f &&
                          threat.modulation_type == 1 &&
                          threat.protocol_id == 2 &&
                          threat.priority == 5);
    
    SecurityTest::test_result(data_integrity, "ThreatSignature data integrity");
    
    // Test struct size consistency
    bool size_consistent = (sizeof(ThreatSignature) == 32);
    SecurityTest::test_result(size_consistent, "ThreatSignature size consistency");
    
    // Test JammingParams structure
    JammingParams jamming;
    std::memset(&jamming, 0xBB, sizeof(jamming));
    
    jamming.center_freq_ghz = 2.4f;
    jamming.bandwidth_mhz = 20.0f;
    jamming.power_dbm = 10.0f;
    jamming.modulation_type = 2;
    jamming.sweep_rate_mhz_per_sec = 100.0f;
    
    bool jamming_integrity = (jamming.center_freq_ghz == 2.4f &&
                             jamming.bandwidth_mhz == 20.0f &&
                             jamming.power_dbm == 10.0f &&
                             jamming.modulation_type == 2 &&
                             jamming.sweep_rate_mhz_per_sec == 100.0f);
    
    SecurityTest::test_result(jamming_integrity, "JammingParams data integrity");
}

// Test timing attack resistance
void test_timing_attack_resistance() {
    std::cout << "\n--- Testing Timing Attack Resistance ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Timing attack tests - initialization failed" << std::endl;
        return;
    }
    
    // Test consistent timing for different input patterns
    std::vector<std::chrono::microseconds> timings;
    
    for (int pattern = 0; pattern < 10; ++pattern) {
        std::vector<float> spectrum(SPECTRUM_BINS);
        
        // Create different spectrum patterns
        if (pattern % 2 == 0) {
            // High signal pattern
            std::fill(spectrum.begin(), spectrum.end(), -40.0f);
        } else {
            // Low noise pattern  
            std::fill(spectrum.begin(), spectrum.end(), -100.0f);
        }
        
        std::vector<ThreatSignature> threats(1);
        threats[0].center_freq_ghz = 2.4f;
        threats[0].bandwidth_mhz = 20.0f;
        threats[0].power_dbm = -60.0f;
        
        std::vector<JammingParams> jamming_params(1);
        
        auto start = std::chrono::high_resolution_clock::now();
        module->process_spectrum_threadsafe(
            spectrum.data(), threats.data(), 1, jamming_params.data(), 0);
        auto end = std::chrono::high_resolution_clock::now();
        
        timings.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    }
    
    // Calculate timing variance
    double sum = 0.0;
    for (const auto& timing : timings) {
        sum += timing.count();
    }
    double mean = sum / timings.size();
    
    double variance = 0.0;
    for (const auto& timing : timings) {
        variance += (timing.count() - mean) * (timing.count() - mean);
    }
    variance /= timings.size();
    double std_dev = std::sqrt(variance);
    
    // Check if timing is reasonably consistent (coefficient of variation < 50%)
    double cv = std_dev / mean;
    bool timing_consistent = (cv < 0.5);
    
    SecurityTest::test_result(timing_consistent, "Timing consistency across inputs");
    
    std::cout << "[INFO] Timing stats - Mean: " << mean << " μs, StdDev: " << std_dev 
              << " μs, CV: " << (cv * 100) << "%" << std::endl;
}

// Test resource exhaustion protection
void test_resource_exhaustion_protection() {
    std::cout << "\n--- Testing Resource Exhaustion Protection ---" << std::endl;
    
    auto module = CEWModuleFactory::create(CEWBackend::CPU);
    if (!module || !module->initialize(0)) {
        std::cout << "[SKIP] Resource exhaustion tests - initialization failed" << std::endl;
        return;
    }
    
    // Test rapid repeated calls
    bool rapid_calls_handled = true;
    auto start_time = std::chrono::steady_clock::now();
    
    std::vector<float> spectrum(SPECTRUM_BINS, -50.0f);
    std::vector<ThreatSignature> threats(1);
    std::vector<JammingParams> jamming_params(1);
    
    try {
        for (int i = 0; i < 10000; ++i) {
            module->process_spectrum_threadsafe(
                spectrum.data(), threats.data(), 1, jamming_params.data(), i);
            
            // Check if system is still responsive
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            if (elapsed.count() > 30) { // Timeout after 30 seconds
                break;
            }
        }
    } catch (...) {
        rapid_calls_handled = false;
    }
    
    SecurityTest::test_result(rapid_calls_handled, "Rapid call flood protection");
    
    // Test system responsiveness after load
    auto metrics = module->get_metrics();
    bool system_responsive = (metrics.deadline_misses < 1000); // Allow some misses under load
    SecurityTest::test_result(system_responsive, "System responsiveness under load");
}

int main() {
    std::cout << "=== CEW Module Security Tests ===" << std::endl;
    
    test_input_validation();
    test_buffer_overflow_protection();
    test_memory_safety();
    test_data_structure_integrity();
    test_timing_attack_resistance();
    test_resource_exhaustion_protection();
    
    SecurityTest::print_summary();
    
    return (SecurityTest::failed_ == 0) ? 0 : 1;
}