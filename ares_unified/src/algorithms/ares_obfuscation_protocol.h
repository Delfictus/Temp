/**
 * @file ares_obfuscation_protocol.h
 * @brief Ares Obfuscation Protocol (AOP) - Chaos Engine
 */

#pragma once

#include <vector>
#include <memory>

namespace ares {
namespace algorithms {

namespace hardware { class CudaAcceleration; }

class AresObfuscationProtocol {
private:
    hardware::CudaAcceleration* cuda_accel_;
    bool initialized_;
    
public:
    explicit AresObfuscationProtocol(hardware::CudaAcceleration* cuda = nullptr);
    ~AresObfuscationProtocol();
    
    bool initialize();
    void shutdown();
    
    std::vector<uint8_t> signatureSwapping(const std::vector<uint8_t>& data, uint32_t new_signature);
    std::vector<uint8_t> dataScrambling(const std::vector<uint8_t>& data, uint32_t key);
    std::vector<uint8_t> temporalDistortion(const std::vector<uint8_t>& data, uint32_t jitter_ms);
    std::vector<uint8_t> fieldLevelEncryption(const std::vector<uint8_t>& data);
};

} // namespace algorithms
} // namespace ares