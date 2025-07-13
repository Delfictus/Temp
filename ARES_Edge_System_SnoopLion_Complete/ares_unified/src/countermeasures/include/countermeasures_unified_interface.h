#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace ares::countermeasures {

// Unified countermeasures interface for active defense
class CountermeasuresSystem {
public:
    CountermeasuresSystem();
    ~CountermeasuresSystem();
    
    // System initialization
    bool initialize(const CountermeasuresConfig& config);
    void shutdown();
    
    // Chaos induction operations
    struct ChaosPattern {
        std::vector<float> frequencies;  // Hz
        std::vector<float> amplitudes;
        std::vector<float> phases;
        float modulationRate;
        float spreadFactor;
    };
    
    bool activateChaosField(const ChaosPattern& pattern);
    void modulateChaosParameters(float intensity, float entropy);
    void disableChaosField();
    
    // Self-destruct protocols
    enum class DestructMode {
        SECURE_WIPE,           // Overwrite all memory
        THERMITE_MELT,         // Physical destruction
        EMP_DISCHARGE,         // Electromagnetic pulse
        QUANTUM_SCRAMBLE,      // Quantum state randomization
        CASCADING_FAILURE      // Induced system failure
    };
    
    bool armSelfDestruct(DestructMode mode, std::chrono::seconds delay);
    bool disarmSelfDestruct(const std::string& authCode);
    void executeEmergencyDestruct();
    
    // Last man standing coordination
    struct SwarmStatus {
        size_t totalNodes;
        size_t activeNodes;
        size_t compromisedNodes;
        std::vector<std::string> survivingNodeIds;
    };
    
    SwarmStatus getSwarmStatus() const;
    bool initiateLastManStanding();
    bool transferCriticalData(const std::string& targetNodeId);
    
    // Active defense mechanisms
    struct ThreatVector {
        std::string threatId;
        std::string threatType;
        float confidence;
        std::array<float, 3> direction;
        float estimatedTimeToImpact;
    };
    
    bool detectThreats(std::vector<ThreatVector>& threats);
    bool deployCountermeasure(const ThreatVector& threat);
    
    // Electronic warfare
    bool jamSignal(float centerFrequency, float bandwidth);
    bool spoofSignal(const std::vector<uint8_t>& pattern);
    bool injectFalseTargets(size_t count, float dispersion);
    
    // Deception operations
    struct DecoyProfile {
        std::string profileType;  // "drone", "vehicle", "installation"
        std::vector<float> emSignature;
        std::vector<float> thermalSignature;
        std::vector<float> radarCrossSection;
    };
    
    bool deployDecoy(const DecoyProfile& profile, const std::array<float, 3>& position);
    bool createPhantomSwarm(size_t swarmSize);
    
    // System monitoring
    struct SystemIntegrity {
        float codeIntegrity;      // 0-1
        float dataIntegrity;      // 0-1
        float hardwareStatus;     // 0-1
        bool tamperDetected;
        std::vector<std::string> anomalies;
    };
    
    SystemIntegrity checkIntegrity() const;
    
    // Performance control
    void setUseCuda(bool useCuda);
    void setAggressiveness(float level);  // 0-1
    
    // Callbacks
    using ThreatDetectedCallback = std::function<void(const ThreatVector&)>;
    using IntegrityViolationCallback = std::function<void(const SystemIntegrity&)>;
    
    void registerThreatCallback(ThreatDetectedCallback callback);
    void registerIntegrityCallback(IntegrityViolationCallback callback);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Countermeasures configuration
struct CountermeasuresConfig {
    bool enableAutoDefense = true;
    bool enableSelfDestruct = false;
    float threatDetectionThreshold = 0.7f;
    size_t maxActiveCountermeasures = 10;
    std::chrono::seconds destructDelay{30};
    std::vector<std::string> authorizedDestructCodes;
};

// Chaos field parameters
struct ChaosFieldStatus {
    bool isActive;
    float currentIntensity;
    float effectiveRadius;  // meters
    float powerConsumption; // watts
    std::chrono::steady_clock::time_point activationTime;
};

// Decoy telemetry
struct DecoyTelemetry {
    std::string decoyId;
    std::array<float, 3> position;
    std::array<float, 3> velocity;
    float batteryLevel;
    bool isCompromised;
};

} // namespace ares::countermeasures