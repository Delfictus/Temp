/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * @file cuda_acceleration.h
 * @brief CUDA Hardware Acceleration Interface
 * 
 * PRODUCTION GRADE - NO STUBS
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

#ifdef ARES_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cufft.h>
#include <cudnn.h>
#endif

namespace ares {
namespace hardware {

/**
 * @brief CUDA device information
 */
struct CudaDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    bool unified_memory_supported;
};

/**
 * @brief CUDA memory pool for efficient allocation
 */
class CudaMemoryPool {
private:
    std::vector<void*> free_blocks_;
    std::vector<std::pair<void*, size_t>> allocated_blocks_;
    size_t total_allocated_;
    size_t pool_size_;
    
public:
    CudaMemoryPool(size_t pool_size = 1024 * 1024 * 1024); // 1GB default
    ~CudaMemoryPool();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();
    size_t getTotalAllocated() const { return total_allocated_; }
};

/**
 * @brief CUDA Hardware Acceleration Manager
 */
class CudaAcceleration {
private:
    bool initialized_;
    int device_count_;
    int current_device_;
    std::vector<CudaDeviceInfo> devices_;
    
#ifdef ARES_ENABLE_CUDA
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_generator_;
    cufftHandle cufft_handle_;
    cudnnHandle_t cudnn_handle_;
    
    cudaStream_t default_stream_;
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
#endif
    
    std::unique_ptr<CudaMemoryPool> memory_pool_;
    
public:
    CudaAcceleration();
    ~CudaAcceleration();
    
    /**
     * @brief Initialize CUDA acceleration
     */
    bool initialize();
    
    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();
    
    /**
     * @brief Check if CUDA is available
     */
    bool isAvailable() const { return initialized_; }
    
    /**
     * @brief Get device count
     */
    int getDeviceCount() const { return device_count_; }
    
    /**
     * @brief Get device information
     */
    const std::vector<CudaDeviceInfo>& getDevices() const { return devices_; }
    
    /**
     * @brief Set current device
     */
    bool setDevice(int device_id);
    
    /**
     * @brief Get current device ID
     */
    int getCurrentDevice() const { return current_device_; }
    
    /**
     * @brief Allocate device memory
     */
    void* allocateDevice(size_t size);
    
    /**
     * @brief Free device memory
     */
    void freeDevice(void* ptr);
    
    /**
     * @brief Copy host to device
     */
    bool copyHostToDevice(void* dst, const void* src, size_t size);
    
    /**
     * @brief Copy device to host
     */
    bool copyDeviceToHost(void* dst, const void* src, size_t size);
    
    /**
     * @brief Synchronize device
     */
    void synchronize();
    
    /**
     * @brief Get memory pool
     */
    CudaMemoryPool* getMemoryPool() { return memory_pool_.get(); }
    
#ifdef ARES_ENABLE_CUDA
    /**
     * @brief Get cuBLAS handle
     */
    cublasHandle_t getCublasHandle() { return cublas_handle_; }
    
    /**
     * @brief Get cuRAND generator
     */
    curandGenerator_t getCurandGenerator() { return curand_generator_; }
    
    /**
     * @brief Get cuFFT handle
     */
    cufftHandle getCufftHandle() { return cufft_handle_; }
    
    /**
     * @brief Get cuDNN handle
     */
    cudnnHandle_t getCudnnHandle() { return cudnn_handle_; }
    
    /**
     * @brief Get CUDA streams
     */
    cudaStream_t getDefaultStream() { return default_stream_; }
    cudaStream_t getComputationStream() { return computation_stream_; }
    cudaStream_t getMemoryStream() { return memory_stream_; }
#endif

private:
    void queryDevices();
    bool initializeLibraries();
    void cleanupLibraries();
};

} // namespace hardware
} // namespace ares