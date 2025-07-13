/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Core Utilities
 */

#ifndef ARES_CORE_UTILS_H
#define ARES_CORE_UTILS_H

#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

namespace ares {
namespace core {

/**
 * @brief Time utilities
 */
class TimeUtils {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;
    
    static TimePoint now() {
        return Clock::now();
    }
    
    static double elapsedSeconds(const TimePoint& start) {
        return Duration(now() - start).count();
    }
    
    static uint64_t timestampMicros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now().time_since_epoch()).count();
    }
};

/**
 * @brief Memory utilities
 */
class MemoryUtils {
public:
    /**
     * @brief Securely erase memory
     */
    static void secureErase(void* ptr, size_t size) {
        if (ptr && size > 0) {
            volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
            while (size--) {
                *p++ = 0;
            }
        }
    }
    
    /**
     * @brief Align size to boundary
     */
    static size_t alignSize(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    /**
     * @brief Check if pointer is aligned
     */
    static bool isAligned(const void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
    }
};

/**
 * @brief Cryptographic utilities
 */
class CryptoUtils {
public:
    /**
     * @brief Generate cryptographically secure random bytes
     */
    static std::vector<uint8_t> generateRandomBytes(size_t length);
    
    /**
     * @brief Compute SHA-256 hash
     */
    static std::vector<uint8_t> sha256(const uint8_t* data, size_t length);
    
    /**
     * @brief Compute SHA-3 hash
     */
    static std::vector<uint8_t> sha3_256(const uint8_t* data, size_t length);
    
    /**
     * @brief Constant-time memory comparison
     */
    static bool constantTimeCompare(const uint8_t* a, const uint8_t* b, size_t length) {
        uint8_t result = 0;
        for (size_t i = 0; i < length; ++i) {
            result |= a[i] ^ b[i];
        }
        return result == 0;
    }
};

/**
 * @brief Network utilities
 */
class NetworkUtils {
public:
    /**
     * @brief Convert frequency to wavelength
     */
    static double frequencyToWavelength(double freq_hz) {
        const double c = 299792458.0;  // Speed of light in m/s
        return c / freq_hz;
    }
    
    /**
     * @brief Convert dBm to watts
     */
    static double dBmToWatts(double dbm) {
        return 0.001 * std::pow(10.0, dbm / 10.0);
    }
    
    /**
     * @brief Convert watts to dBm
     */
    static double wattsTodBm(double watts) {
        return 10.0 * std::log10(watts / 0.001);
    }
    
    /**
     * @brief Calculate free space path loss
     */
    static double freeSpacePathLoss(double freq_hz, double distance_m) {
        const double c = 299792458.0;
        return 20.0 * std::log10(4.0 * M_PI * distance_m * freq_hz / c);
    }
};

/**
 * @brief SIMD utilities for performance optimization
 */
class SIMDUtils {
public:
    /**
     * @brief Check CPU features
     */
    static bool hasAVX2();
    static bool hasAVX512();
    static bool hasNEON();  // ARM
    
    /**
     * @brief Vectorized operations
     */
    static void vectorAdd(const float* a, const float* b, float* result, size_t count);
    static void vectorMultiply(const float* a, const float* b, float* result, size_t count);
    static float vectorDotProduct(const float* a, const float* b, size_t count);
};

/**
 * @brief Error handling utilities
 */
class ErrorUtils {
public:
    enum class ErrorCode : uint32_t {
        SUCCESS = 0,
        INVALID_PARAMETER = 1,
        OUT_OF_MEMORY = 2,
        NOT_INITIALIZED = 3,
        CUDA_ERROR = 4,
        NETWORK_ERROR = 5,
        CRYPTO_ERROR = 6,
        TIMEOUT = 7,
        PERMISSION_DENIED = 8,
        UNKNOWN_ERROR = 9999
    };
    
    static std::string errorToString(ErrorCode code);
    static void logError(ErrorCode code, const std::string& context);
};

} // namespace core
} // namespace ares

#endif // ARES_CORE_UTILS_H