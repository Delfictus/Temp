#pragma once

namespace ares {
namespace security {

class FIPSWrapper {
public:
    FIPSWrapper() = default;
    ~FIPSWrapper() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace security
} // namespace ares