#pragma once

namespace ares {
namespace api {

class ZeusAPI {
public:
    ZeusAPI() = default;
    ~ZeusAPI() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace api
} // namespace ares