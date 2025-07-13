/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Common Utilities
 * 
 * This file consolidates common utility functions used across different modules
 * to avoid code duplication and ensure consistency.
 */

#ifndef ARES_COMMON_UTILS_H
#define ARES_COMMON_UTILS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <thrust/complex.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace ares {
namespace utils {

// =============================================================================
// CUDA Error Checking
// =============================================================================

/**
 * @brief CUDA error checking macro with detailed error reporting
 */
#define ARES_CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * @brief CUDA kernel launch error checking
 */
#define ARES_CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(error)); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA synchronize error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(std::string("CUDA sync error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * @brief cuFFT error checking macro
 */
#define ARES_CUFFT_CHECK(call) \
    do { \
        cufftResult_t error = call; \
        if (error != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                      << " - Error code: " << error << std::endl; \
            throw std::runtime_error("cuFFT error"); \
        } \
    } while(0)

/**
 * @brief cuBLAS error checking macro
 */
#define ARES_CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t error = call; \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - Error code: " << error << std::endl; \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

/**
 * @brief cuDNN error checking macro
 */
#define ARES_CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t error = call; \
        if (error != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudnnGetErrorString(error) << std::endl; \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(error)); \
        } \
    } while(0)

// =============================================================================
// CUDA Helper Functions
// =============================================================================

/**
 * @brief Get optimal grid and block dimensions for kernel launch
 */
inline void getOptimalLaunchConfig(int totalElements, int& gridSize, int& blockSize, 
                                   int maxBlockSize = 256) {
    blockSize = std::min(maxBlockSize, 
                         ((totalElements + 31) / 32) * 32);  // Round up to warp size
    gridSize = (totalElements + blockSize - 1) / blockSize;
}

/**
 * @brief Print CUDA device properties
 */
inline void printCudaDeviceInfo(int device = 0) {
    cudaDeviceProp prop;
    ARES_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "CUDA Device " << device << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
}

/**
 * @brief Initialize CUDA and select best device
 */
inline int initializeCuda(bool verbose = false) {
    int deviceCount = 0;
    ARES_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }
    
    // Select device with highest compute capability and most memory
    int bestDevice = 0;
    int bestScore = 0;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        ARES_CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        int score = prop.major * 1000 + prop.minor * 10 + 
                   (prop.totalGlobalMem / (1024*1024*1024));  // GB of memory
        
        if (score > bestScore) {
            bestScore = score;
            bestDevice = i;
        }
    }
    
    ARES_CUDA_CHECK(cudaSetDevice(bestDevice));
    
    if (verbose) {
        std::cout << "Selected CUDA device " << bestDevice << std::endl;
        printCudaDeviceInfo(bestDevice);
    }
    
    return bestDevice;
}

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief Safe CUDA memory allocation with alignment
 */
template<typename T>
inline T* cudaAllocAligned(size_t count, size_t alignment = 256) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    
    // Ensure size is aligned
    size = ((size + alignment - 1) / alignment) * alignment;
    
    ARES_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

/**
 * @brief Safe CUDA memory deallocation
 */
template<typename T>
inline void cudaSafeFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);  // Ignore error as we're cleaning up
        ptr = nullptr;
    }
}

/**
 * @brief RAII wrapper for CUDA memory
 */
template<typename T>
class CudaBuffer {
private:
    T* data_ = nullptr;
    size_t size_ = 0;
    
public:
    CudaBuffer() = default;
    
    explicit CudaBuffer(size_t count) : size_(count) {
        if (size_ > 0) {
            data_ = cudaAllocAligned<T>(size_);
        }
    }
    
    ~CudaBuffer() {
        cudaSafeFree(data_);
    }
    
    // Delete copy constructor and assignment
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Move constructor and assignment
    CudaBuffer(CudaBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            cudaSafeFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return size_; }
    
    void resize(size_t newSize) {
        if (newSize != size_) {
            cudaSafeFree(data_);
            size_ = newSize;
            if (size_ > 0) {
                data_ = cudaAllocAligned<T>(size_);
            }
        }
    }
};

// =============================================================================
// Complex Number Operations (Device Functions)
// =============================================================================

#ifdef __CUDACC__

/**
 * @brief Create complex number from real and imaginary parts
 */
__device__ inline thrust::complex<float> makeComplex(float real, float imag) {
    return thrust::complex<float>(real, imag);
}

/**
 * @brief Atomic add for float values
 */
__device__ inline float atomicAddFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
                       __float_as_uint(__uint_as_float(assumed) + val));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

/**
 * @brief Atomic max for float values
 */
__device__ inline float atomicMaxFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
                       __float_as_uint(fmaxf(__uint_as_float(assumed), val)));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

/**
 * @brief Initialize cuRAND state
 */
__device__ inline void initCurandState(curandState_t* state, uint64_t seed, 
                                       uint32_t tid, uint64_t offset = 0) {
    curand_init(seed, tid, offset, state);
}

#endif // __CUDACC__

// =============================================================================
// Time and Performance Utilities
// =============================================================================

/**
 * @brief High-resolution timer
 */
class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
    TimePoint start_;
    std::string name_;
    
public:
    explicit Timer(const std::string& name = "") : name_(name) {
        start_ = Clock::now();
    }
    
    double elapsed() const {
        auto end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;  // Return milliseconds
    }
    
    void reset() {
        start_ = Clock::now();
    }
    
    ~Timer() {
        if (!name_.empty()) {
            std::cout << name_ << " took " << elapsed() << " ms" << std::endl;
        }
    }
};

/**
 * @brief CUDA event-based timer
 */
class CudaTimer {
private:
    cudaEvent_t start_, stop_;
    std::string name_;
    bool started_ = false;
    
public:
    explicit CudaTimer(const std::string& name = "") : name_(name) {
        ARES_CUDA_CHECK(cudaEventCreate(&start_));
        ARES_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        ARES_CUDA_CHECK(cudaEventRecord(start_));
        started_ = true;
    }
    
    float stop() {
        if (!started_) {
            throw std::runtime_error("Timer not started");
        }
        
        ARES_CUDA_CHECK(cudaEventRecord(stop_));
        ARES_CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float milliseconds = 0;
        ARES_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        
        if (!name_.empty()) {
            std::cout << name_ << " took " << milliseconds << " ms" << std::endl;
        }
        
        started_ = false;
        return milliseconds;
    }
};

// =============================================================================
// String and Formatting Utilities
// =============================================================================

/**
 * @brief Format bytes into human-readable string
 */
inline std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && unit < 4) {
        size /= 1024;
        unit++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

/**
 * @brief Format frequency into human-readable string
 */
inline std::string formatFrequency(double hz) {
    const char* units[] = {"Hz", "kHz", "MHz", "GHz", "THz"};
    int unit = 0;
    double freq = hz;
    
    while (freq >= 1000 && unit < 4) {
        freq /= 1000;
        unit++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << freq << " " << units[unit];
    return oss.str();
}

/**
 * @brief Generate timestamp string
 */
inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// =============================================================================
// Math Utilities
// =============================================================================

/**
 * @brief Clamp value between min and max
 */
template<typename T>
inline T clamp(T value, T min, T max) {
    return std::max(min, std::min(max, value));
}

/**
 * @brief Linear interpolation
 */
template<typename T>
inline T lerp(T a, T b, float t) {
    return a + (b - a) * t;
}

/**
 * @brief Convert dBm to watts
 */
inline double dBmToWatts(double dbm) {
    return 0.001 * std::pow(10.0, dbm / 10.0);
}

/**
 * @brief Convert watts to dBm
 */
inline double wattsTodBm(double watts) {
    return 10.0 * std::log10(watts / 0.001);
}

/**
 * @brief Calculate free space path loss
 */
inline double freeSpacePathLoss(double freq_hz, double distance_m) {
    const double c = 299792458.0;  // Speed of light
    return 20.0 * std::log10(4.0 * M_PI * distance_m * freq_hz / c);
}

/**
 * @brief Convert between frequency and wavelength
 */
inline double frequencyToWavelength(double freq_hz) {
    const double c = 299792458.0;  // Speed of light
    return c / freq_hz;
}

inline double wavelengthToFrequency(double wavelength_m) {
    const double c = 299792458.0;  // Speed of light
    return c / wavelength_m;
}

} // namespace utils
} // namespace ares

#endif // ARES_COMMON_UTILS_H