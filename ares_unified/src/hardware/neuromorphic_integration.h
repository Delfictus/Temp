#pragma once

namespace ares {
namespace hardware {

class NeuromorphicIntegration {
public:
    NeuromorphicIntegration() = default;
    ~NeuromorphicIntegration() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace hardware
} // namespace ares