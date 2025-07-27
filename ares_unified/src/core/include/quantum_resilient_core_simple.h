#pragma once

// Simple compatibility wrapper for main.cpp
namespace ares {
namespace core {

class QuantumResilientCore {
public:
    QuantumResilientCore() = default;
    ~QuantumResilientCore() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace core
} // namespace ares