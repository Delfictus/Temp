/**
 * @file athena_adp.h
 * @brief Athena-ADP Adaptive Decision Potential Algorithm
 */

#pragma once

#include <vector>
#include <memory>

namespace ares {
namespace algorithms {

namespace hardware { class NeuromorphicIntegration; }

struct DecisionContext {
    float threat_level;
    float resource_availability;
    float mission_priority;
    std::vector<float> environmental_factors;
};

enum class RecommendedAction {
    PASSIVE_MONITORING = 0,
    ACTIVE_SCANNING = 1,
    DEFENSIVE_MANEUVERS = 2,
    OFFENSIVE_ACTION = 3,
    EMERGENCY_PROTOCOLS = 4
};

class AthenaADP {
private:
    hardware::NeuromorphicIntegration* neuro_integration_;
    bool initialized_;
    
public:
    explicit AthenaADP(hardware::NeuromorphicIntegration* neuro = nullptr);
    ~AthenaADP();
    
    bool initialize();
    void shutdown();
    
    float computeDecisionPotential(const DecisionContext& context);
    RecommendedAction getRecommendedAction(float decision_potential);
};

} // namespace algorithms
} // namespace ares