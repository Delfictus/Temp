#pragma once
#include <chrono>

namespace ares {
namespace api {

class PrometheusInterface {
public:
    PrometheusInterface() = default;
    ~PrometheusInterface() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
    
    void ingestSensorData(const std::string& name, float value, std::chrono::system_clock::time_point timestamp) {}
};

class ZeusAPI {
public:
    ZeusAPI() = default;
    ~ZeusAPI() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
};

} // namespace api
} // namespace ares