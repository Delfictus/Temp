/**
 * @file athena_adp.cpp
 * @brief Athena-ADP Implementation
 */

#include "athena_adp.h"
#include "../hardware/neuromorphic_integration.h"
#include <cmath>
#include <algorithm>

namespace ares {
namespace algorithms {

AthenaADP::AthenaADP(hardware::NeuromorphicIntegration* neuro)
    : neuro_integration_(neuro), initialized_(false) {
}

AthenaADP::~AthenaADP() {
    shutdown();
}

bool AthenaADP::initialize() {
    initialized_ = true;
    return true;
}

void AthenaADP::shutdown() {
    initialized_ = false;
}

float AthenaADP::computeDecisionPotential(const DecisionContext& context) {
    if (!initialized_) return 0.0f;
    
    // Weighted combination of factors
    float threat_weight = 0.4f;
    float resource_weight = 0.3f;
    float mission_weight = 0.2f;
    float env_weight = 0.1f;
    
    float decision_potential = 
        context.threat_level * threat_weight +
        context.resource_availability * resource_weight +
        context.mission_priority * mission_weight;
    
    // Add environmental factors
    if (!context.environmental_factors.empty()) {
        float env_avg = 0.0f;
        for (float factor : context.environmental_factors) {
            env_avg += factor;
        }
        env_avg /= context.environmental_factors.size();
        decision_potential += env_avg * env_weight;
    }
    
    // Apply sigmoid normalization
    decision_potential = 1.0f / (1.0f + std::exp(-decision_potential));
    
    return std::max(0.0f, std::min(1.0f, decision_potential));
}

RecommendedAction AthenaADP::getRecommendedAction(float decision_potential) {
    if (decision_potential < 0.2f) {
        return RecommendedAction::PASSIVE_MONITORING;
    } else if (decision_potential < 0.4f) {
        return RecommendedAction::ACTIVE_SCANNING;
    } else if (decision_potential < 0.6f) {
        return RecommendedAction::DEFENSIVE_MANEUVERS;
    } else if (decision_potential < 0.8f) {
        return RecommendedAction::OFFENSIVE_ACTION;
    } else {
        return RecommendedAction::EMERGENCY_PROTOCOLS;
    }
}

} // namespace algorithms
} // namespace ares