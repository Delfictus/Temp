#pragma once

namespace ares {
namespace database {

class RelationalDB {
public:
    RelationalDB() = default;
    ~RelationalDB() = default;
    
    bool initialize() { return true; }
    void shutdown() {}
    
    void storeEvent(const std::string& type, const std::string& data, std::chrono::system_clock::time_point timestamp) {}
};

} // namespace database
} // namespace ares