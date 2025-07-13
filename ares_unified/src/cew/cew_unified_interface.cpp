/**
 * @file cew_unified_interface.cpp
 * @brief Implementation of the unified CEW interface
 */

#include "include/cew_unified_interface.h"
#include "cpu/cew_cpu_module.h"
#ifdef CEW_CUDA_AVAILABLE
#include "cuda/cew_cuda_module.h"
#include <cuda_runtime.h>
#endif
#include <mutex>
#include <iostream>

namespace ares::cew {

// Factory implementation
std::unique_ptr<ICEWModule> CEWModuleFactory::create(CEWBackend backend) {
    switch (backend) {
        case CEWBackend::AUTO:
            // Try CUDA first, fall back to CPU
            if (is_cuda_available()) {
                std::cout << "CEW: Using CUDA backend (auto-detected)" << std::endl;
#ifdef CEW_CUDA_AVAILABLE
                return std::make_unique<CEWCudaModule>();
#else
                std::cout << "CEW: CUDA not compiled in, using CPU backend" << std::endl;
                return std::make_unique<CEWCpuModule>();
#endif
            } else {
                std::cout << "CEW: Using CPU backend (CUDA not available)" << std::endl;
                return std::make_unique<CEWCpuModule>();
            }
            
        case CEWBackend::CUDA:
#ifdef CEW_CUDA_AVAILABLE
            if (!is_cuda_available()) {
                std::cerr << "CEW: CUDA backend requested but not available" << std::endl;
                return nullptr;
            }
            std::cout << "CEW: Using CUDA backend (forced)" << std::endl;
            return std::make_unique<CEWCudaModule>();
#else
            std::cerr << "CEW: CUDA backend requested but not compiled in" << std::endl;
            return nullptr;
#endif
            
        case CEWBackend::CPU:
            std::cout << "CEW: Using CPU backend (forced)" << std::endl;
            return std::make_unique<CEWCpuModule>();
            
        default:
            return nullptr;
    }
}

bool CEWModuleFactory::is_cuda_available() {
#ifdef CEW_CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

std::vector<std::string> CEWModuleFactory::get_cuda_devices() {
    std::vector<std::string> devices;
#ifdef CEW_CUDA_AVAILABLE
    int device_count = 0;
    
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return devices;
    }
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            std::string device_info = std::string(prop.name) + 
                " (SM " + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                ", " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB)";
            devices.push_back(device_info);
        }
    }
#endif
    return devices;
}

// CEWManager implementation
class CEWManager::Impl {
public:
    Impl(CEWBackend backend) : backend_(backend), initialized_(false) {
        module_ = CEWModuleFactory::create(backend);
        if (!module_) {
            throw std::runtime_error("Failed to create CEW module");
        }
    }
    
    bool initialize(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return true;
        }
        
        initialized_ = module_->initialize(device_id);
        return initialized_;
    }
    
    bool process_spectrum_threadsafe(
        const float* spectrum_waterfall,
        ThreatSignature* threats,
        uint32_t num_threats,
        JammingParams* jamming_params,
        uint64_t timestamp_ns
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return false;
        }
        
        return module_->process_spectrum(
            spectrum_waterfall, threats, num_threats, jamming_params, timestamp_ns
        );
    }
    
    bool update_qlearning_threadsafe(float reward) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return false;
        }
        
        return module_->update_qlearning(reward);
    }
    
    CEWMetrics get_metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return module_->get_metrics();
    }
    
    CEWBackend get_backend() const {
        return module_->get_backend();
    }
    
    void set_memory_limit(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        memory_limit_ = bytes;
    }
    
    size_t get_memory_usage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // Estimate based on buffer sizes
        size_t usage = sizeof(float) * SPECTRUM_BINS * WATERFALL_HISTORY;
        usage += sizeof(QTableState);
        usage += sizeof(float) * NUM_ACTIONS * 4096; // Waveform bank
        return usage;
    }
    
private:
    std::unique_ptr<ICEWModule> module_;
    CEWBackend backend_;
    bool initialized_;
    size_t memory_limit_ = 0;
    mutable std::mutex mutex_;
};

// CEWManager public interface
CEWManager::CEWManager(CEWBackend backend) 
    : pImpl(std::make_unique<Impl>(backend)) {
}

CEWManager::~CEWManager() = default;

bool CEWManager::initialize(int device_id) {
    return pImpl->initialize(device_id);
}

bool CEWManager::process_spectrum_threadsafe(
    const float* spectrum_waterfall,
    ThreatSignature* threats,
    uint32_t num_threats,
    JammingParams* jamming_params,
    uint64_t timestamp_ns
) {
    return pImpl->process_spectrum_threadsafe(
        spectrum_waterfall, threats, num_threats, jamming_params, timestamp_ns
    );
}

bool CEWManager::update_qlearning_threadsafe(float reward) {
    return pImpl->update_qlearning_threadsafe(reward);
}

CEWMetrics CEWManager::get_metrics() const {
    return pImpl->get_metrics();
}

CEWBackend CEWManager::get_backend() const {
    return pImpl->get_backend();
}

void CEWManager::set_memory_limit(size_t bytes) {
    pImpl->set_memory_limit(bytes);
}

size_t CEWManager::get_memory_usage() const {
    return pImpl->get_memory_usage();
}

} // namespace ares::cew