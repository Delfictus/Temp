#pragma once

namespace ares {
namespace hardware {

class FPGAInterface {
public:
    FPGAInterface() = default;
    ~FPGAInterface() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

class NeuromorphicIntegration {
public:
    NeuromorphicIntegration() = default;
    ~NeuromorphicIntegration() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace hardware
} // namespace ares