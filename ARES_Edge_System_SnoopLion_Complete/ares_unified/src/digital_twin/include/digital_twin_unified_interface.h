#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include <functional>
#include "predictive_simulation_engine.h"
#include "realtime_state_sync.h"

namespace ares::digital_twin {

// Unified digital twin interface with physics simulation
class DigitalTwinEngine {
public:
    DigitalTwinEngine();
    ~DigitalTwinEngine();
    
    // Initialization
    bool initialize(const SimulationConfig& config);
    void shutdown();
    
    // Real-time state synchronization
    bool syncPhysicalState(const PhysicalState& state);
    PhysicalState getCurrentState() const;
    PhysicalState getPredictedState(std::chrono::milliseconds lookahead) const;
    
    // Physics simulation
    void stepSimulation(float deltaTime);
    void runSimulation(float totalTime, float timeStep);
    
    // Predictive analytics
    struct PredictionResult {
        std::vector<PhysicalState> trajectory;
        float confidenceScore;
        std::vector<std::string> warnings;
    };
    
    PredictionResult predictTrajectory(
        float timeHorizon,
        const std::vector<ControlInput>& plannedInputs = {}
    );
    
    // State validation and anomaly detection
    struct ValidationResult {
        bool isValid;
        float deviationScore;
        std::vector<std::string> anomalies;
    };
    
    ValidationResult validateState(const PhysicalState& observedState);
    
    // Performance configuration
    void setUseCuda(bool useCuda);
    void setSimulationPrecision(float precision);
    
    // Callbacks
    using StateUpdateCallback = std::function<void(const PhysicalState&)>;
    void registerStateUpdateCallback(StateUpdateCallback callback);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Configuration for digital twin simulation
struct SimulationConfig {
    size_t maxEntities = 1000;
    float spatialBounds[3] = {1000.0f, 1000.0f, 500.0f};  // meters
    float timeStep = 0.01f;  // seconds
    bool enableCollisionDetection = true;
    bool enableFluidDynamics = false;
    bool enableThermalModeling = false;
};

// Physical state representation
struct PhysicalState {
    uint64_t timestamp;
    std::vector<EntityState> entities;
    EnvironmentalConditions environment;
};

// Individual entity state
struct EntityState {
    uint32_t entityId;
    float position[3];
    float velocity[3];
    float orientation[4];  // Quaternion
    float angularVelocity[3];
    std::vector<float> sensorReadings;
};

// Environmental conditions
struct EnvironmentalConditions {
    float temperature;
    float pressure;
    float humidity;
    float windVelocity[3];
    float visibility;
};

// Control input for predictive simulation
struct ControlInput {
    uint64_t timestamp;
    uint32_t entityId;
    float thrust[3];
    float torque[3];
};

} // namespace ares::digital_twin