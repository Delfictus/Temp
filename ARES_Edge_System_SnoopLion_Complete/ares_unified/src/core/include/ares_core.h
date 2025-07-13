/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Company: DELFICTUS I/O LLC
 * CAGE Code: 13H70
 * UEI: LXT3B9GMY4N8
 * Active DoD Contractor
 * 
 * Location: Los Angeles, California 90013 United States
 * 
 * This software contains trade secrets and proprietary information
 * of DELFICTUS I/O LLC. Unauthorized use, reproduction, or distribution
 * is strictly prohibited. This technology is subject to export controls
 * under ITAR and EAR regulations.
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * WARNING: This system is designed for authorized U.S. Department of Defense
 * use only. Misuse may result in severe criminal and civil penalties.
 */

#ifndef ARES_CORE_H
#define ARES_CORE_H

#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>

// Platform detection
#ifdef __CUDACC__
    #define ARES_CUDA_AVAILABLE
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#endif

// Export macros
#ifdef _WIN32
    #ifdef ARES_EXPORTS
        #define ARES_API __declspec(dllexport)
    #else
        #define ARES_API __declspec(dllimport)
    #endif
#else
    #define ARES_API __attribute__((visibility("default")))
#endif

namespace ares {

// Forward declarations
namespace quantum {
    class QuantumResilientARESCore;
    class QuantumSignature;
    class DeterministicByzantineConsensus;
}

namespace neuromorphic {
    class NeuromorphicNetwork;
    class ThreatDetectionNetwork;
}

// Core version information
constexpr uint32_t ARES_VERSION_MAJOR = 1;
constexpr uint32_t ARES_VERSION_MINOR = 0;
constexpr uint32_t ARES_VERSION_PATCH = 0;

/**
 * @brief Core configuration structure
 */
struct ARESConfig {
    // System configuration
    bool enable_cuda = true;
    bool enable_quantum_resilience = true;
    bool enable_neuromorphic = true;
    bool enable_em_spectrum_analysis = true;
    
    // Performance tuning
    uint32_t gpu_device_id = 0;
    uint32_t num_threads = 0;  // 0 = auto-detect
    size_t gpu_memory_limit = 0;  // 0 = no limit
    
    // Security settings
    bool enable_post_quantum_crypto = true;
    bool enable_secure_erasure = true;
    uint32_t quantum_signature_bits = 256;
    
    // Network settings
    bool enable_auto_network_discovery = true;
    bool enable_opportunistic_connections = true;
    uint32_t max_simultaneous_connections = 10;
    
    // Debug settings
    bool enable_debug_logging = false;
    bool enable_performance_monitoring = false;
    std::string log_file_path;
};

/**
 * @brief Core system status
 */
struct ARESStatus {
    // System state
    enum State {
        UNINITIALIZED = 0,
        INITIALIZING = 1,
        READY = 2,
        ACTIVE = 3,
        ERROR = 4,
        SHUTDOWN = 5
    };
    
    State current_state = UNINITIALIZED;
    std::string status_message;
    
    // Resource utilization
    float cpu_usage_percent = 0.0f;
    float gpu_usage_percent = 0.0f;
    size_t memory_used_bytes = 0;
    size_t gpu_memory_used_bytes = 0;
    
    // Component status
    bool quantum_core_active = false;
    bool neuromorphic_active = false;
    bool em_spectrum_active = false;
    uint32_t active_connections = 0;
    
    // Performance metrics
    uint64_t operations_per_second = 0;
    float average_latency_ms = 0.0f;
    uint64_t total_operations = 0;
};

/**
 * @brief Base interface for ARES core components
 */
class ARES_API IARESComponent {
public:
    virtual ~IARESComponent() = default;
    
    /**
     * @brief Initialize the component
     * @param config System configuration
     * @return true if initialization successful
     */
    virtual bool initialize(const ARESConfig& config) = 0;
    
    /**
     * @brief Shutdown the component
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief Get component status
     * @return Current status
     */
    virtual ARESStatus::State getStatus() const = 0;
    
    /**
     * @brief Get component name
     * @return Component identifier
     */
    virtual std::string getName() const = 0;
};

/**
 * @brief Main ARES Core class - orchestrates all subsystems
 */
class ARES_API ARESCore {
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    /**
     * @brief Construct ARES Core
     */
    ARESCore();
    
    /**
     * @brief Destructor
     */
    ~ARESCore();
    
    // Disable copy/move for singleton-like behavior
    ARESCore(const ARESCore&) = delete;
    ARESCore& operator=(const ARESCore&) = delete;
    ARESCore(ARESCore&&) = delete;
    ARESCore& operator=(ARESCore&&) = delete;
    
    /**
     * @brief Initialize ARES system
     * @param config System configuration
     * @return true if initialization successful
     */
    bool initialize(const ARESConfig& config = ARESConfig{});
    
    /**
     * @brief Shutdown ARES system
     */
    void shutdown();
    
    /**
     * @brief Get current system status
     * @return System status structure
     */
    ARESStatus getStatus() const;
    
    /**
     * @brief Register a custom component
     * @param component Component to register
     * @return true if registration successful
     */
    bool registerComponent(std::shared_ptr<IARESComponent> component);
    
    /**
     * @brief Get component by name
     * @param name Component name
     * @return Component pointer or nullptr if not found
     */
    std::shared_ptr<IARESComponent> getComponent(const std::string& name) const;
    
    /**
     * @brief Process incoming data
     * @param data Input data buffer
     * @param size Data size in bytes
     * @return true if processing successful
     */
    bool processData(const uint8_t* data, size_t size);
    
    /**
     * @brief Execute quantum-resilient operation
     * @param operation_id Operation identifier
     * @param params Operation parameters
     * @return Operation result
     */
    std::vector<uint8_t> executeQuantumOperation(uint32_t operation_id, 
                                                 const std::vector<uint8_t>& params);
    
    /**
     * @brief Run neuromorphic inference
     * @param input_data Input tensor
     * @param input_size Size of input
     * @return Inference results
     */
    std::vector<float> runNeuromorphicInference(const float* input_data, 
                                                size_t input_size);
    
    /**
     * @brief Scan EM spectrum for networks
     * @param start_freq Start frequency in Hz
     * @param end_freq End frequency in Hz
     * @param callback Callback for discovered networks
     */
    void scanEMSpectrum(uint64_t start_freq, uint64_t end_freq,
                       std::function<void(const std::string&)> callback);
    
    /**
     * @brief Get version string
     * @return Version in format "major.minor.patch"
     */
    static std::string getVersion();
    
    /**
     * @brief Check if CUDA is available
     * @return true if CUDA runtime and devices are available
     */
    static bool isCUDAAvailable();
    
    /**
     * @brief Get available GPU memory
     * @param device_id GPU device ID
     * @return Available memory in bytes
     */
    static size_t getAvailableGPUMemory(int device_id = 0);
};

/**
 * @brief Factory function to create ARES Core instance
 * @return Unique pointer to ARESCore
 */
ARES_API std::unique_ptr<ARESCore> createARESCore();

} // namespace ares

#endif // ARES_CORE_H