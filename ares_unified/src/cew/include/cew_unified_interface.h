/**
 * @file cew_unified_interface.h
 * @brief Unified interface for CEW module with runtime CPU/GPU selection
 * 
 * This interface provides a consistent API regardless of whether CUDA is available,
 * automatically falling back to CPU implementation when necessary.
 */

#ifndef ARES_CEW_UNIFIED_INTERFACE_H
#define ARES_CEW_UNIFIED_INTERFACE_H

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

namespace ares::cew {

// Forward declarations of shared structures
struct ThreatSignature;
struct JammingParams;
struct CEWMetrics;

/**
 * @brief Backend types for CEW processing
 */
enum class CEWBackend {
    AUTO,   // Automatically select best available backend
    CPU,    // Force CPU implementation
    CUDA    // Force CUDA implementation (fails if not available)
};

/**
 * @brief Abstract interface for CEW implementations
 */
class ICEWModule {
public:
    virtual ~ICEWModule() = default;
    
    // Initialize the module
    virtual bool initialize(int device_id = 0) = 0;
    
    // Process spectrum and generate jamming response
    virtual bool process_spectrum(
        const float* spectrum_waterfall,
        ThreatSignature* threats,
        uint32_t num_threats,
        JammingParams* jamming_params,
        uint64_t timestamp_ns
    ) = 0;
    
    // Update Q-learning model with reward feedback
    virtual bool update_qlearning(float reward) = 0;
    
    // Get performance metrics
    virtual CEWMetrics get_metrics() const = 0;
    
    // Get backend type
    virtual CEWBackend get_backend() const = 0;
    
    // Check if CUDA is available
    virtual bool is_cuda_available() const = 0;
};

/**
 * @brief Factory for creating CEW module with appropriate backend
 */
class CEWModuleFactory {
public:
    /**
     * @brief Create CEW module with specified backend
     * @param backend Desired backend (AUTO will select best available)
     * @return Unique pointer to CEW module implementation
     */
    static std::unique_ptr<ICEWModule> create(CEWBackend backend = CEWBackend::AUTO);
    
    /**
     * @brief Check if CUDA is available on the system
     */
    static bool is_cuda_available();
    
    /**
     * @brief Get list of available CUDA devices
     */
    static std::vector<std::string> get_cuda_devices();
};

/**
 * @brief Thread-safe CEW manager for concurrent access
 */
class CEWManager {
public:
    CEWManager(CEWBackend backend = CEWBackend::AUTO);
    ~CEWManager();
    
    // Initialize with device selection
    bool initialize(int device_id = 0);
    
    // Thread-safe spectrum processing
    bool process_spectrum_threadsafe(
        const float* spectrum_waterfall,
        ThreatSignature* threats,
        uint32_t num_threats,
        JammingParams* jamming_params,
        uint64_t timestamp_ns
    );
    
    // Thread-safe Q-learning update
    bool update_qlearning_threadsafe(float reward);
    
    // Get current metrics (thread-safe)
    CEWMetrics get_metrics() const;
    
    // Get active backend
    CEWBackend get_backend() const;
    
    // Resource management
    void set_memory_limit(size_t bytes);
    size_t get_memory_usage() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ares::cew

#endif // ARES_CEW_UNIFIED_INTERFACE_H