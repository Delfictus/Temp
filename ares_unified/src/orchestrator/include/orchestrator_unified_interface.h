#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include <functional>
#include <variant>

namespace ares::orchestrator {

// ChronoPath AI orchestration engine
class ChronoPathOrchestrator {
public:
    ChronoPathOrchestrator();
    ~ChronoPathOrchestrator();
    
    // System initialization
    bool initialize(const OrchestratorConfig& config);
    void shutdown();
    
    // Temporal path planning
    struct TemporalNode {
        std::string nodeId;
        std::chrono::steady_clock::time_point timestamp;
        std::array<float, 3> position;
        std::vector<float> state;
        float probability;
    };
    
    struct ChronoPath {
        std::vector<TemporalNode> nodes;
        float totalCost;
        float successProbability;
        std::chrono::milliseconds estimatedDuration;
    };
    
    ChronoPath planTemporalPath(
        const TemporalNode& start,
        const TemporalNode& goal,
        const std::vector<TemporalConstraint>& constraints
    );
    
    // Dynamic resource planning and provisioning (DRPP)
    struct ResourceRequirement {
        float computeGFLOPS;
        float memoryGB;
        float networkBandwidthMbps;
        float powerWatts;
        std::chrono::milliseconds deadline;
    };
    
    struct ResourceAllocation {
        std::vector<std::string> assignedNodes;
        std::map<std::string, float> nodeUtilization;
        float totalCost;
        bool meetsDeadline;
    };
    
    ResourceAllocation allocateResources(
        const std::vector<ResourceRequirement>& requirements
    );
    
    // Mission orchestration
    struct Mission {
        std::string missionId;
        std::vector<Task> tasks;
        std::vector<Dependency> dependencies;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::milliseconds timeout;
        PriorityLevel priority;
    };
    
    std::string deployMission(const Mission& mission);
    bool updateMission(const std::string& missionId, const MissionUpdate& update);
    MissionStatus getMissionStatus(const std::string& missionId) const;
    
    // Adaptive orchestration
    struct AdaptationTrigger {
        std::string triggerId;
        std::function<bool()> condition;
        std::function<void()> action;
        float sensitivity;
    };
    
    void registerAdaptationTrigger(const AdaptationTrigger& trigger);
    void enableAdaptiveMode(bool enable);
    
    // Multi-agent coordination
    struct AgentCapability {
        std::string agentId;
        std::vector<std::string> supportedTasks;
        ResourceCapacity capacity;
        std::array<float, 3> position;
        float reliability;
    };
    
    bool registerAgent(const AgentCapability& capability);
    std::vector<std::string> getAvailableAgents() const;
    
    // Predictive scheduling
    struct PredictedLoad {
        std::chrono::steady_clock::time_point timestamp;
        float computeLoad;
        float networkLoad;
        float storageLoad;
        float confidence;
    };
    
    std::vector<PredictedLoad> predictSystemLoad(
        std::chrono::hours horizon
    ) const;
    
    // Performance optimization
    void optimizeForLatency();
    void optimizeForThroughput();
    void optimizeForPowerEfficiency();
    
    // System monitoring
    struct OrchestratorMetrics {
        size_t activeMissions;
        size_t completedTasks;
        float averageLatency;
        float resourceUtilization;
        float missionSuccessRate;
    };
    
    OrchestratorMetrics getMetrics() const;
    
    // CUDA acceleration
    void setUseCuda(bool useCuda);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Configuration
struct OrchestratorConfig {
    size_t maxConcurrentMissions = 100;
    size_t maxAgents = 1000;
    std::chrono::milliseconds planningHorizon{3600000};  // 1 hour
    float adaptationSensitivity = 0.8f;
    bool enablePredictiveScheduling = true;
    bool enableLoadBalancing = true;
};

// Task specification
struct Task {
    std::string taskId;
    std::string taskType;
    std::vector<uint8_t> payload;
    ResourceRequirement resources;
    std::vector<std::string> requiredCapabilities;
};

// Task dependencies
struct Dependency {
    std::string fromTask;
    std::string toTask;
    DependencyType type;
    std::chrono::milliseconds delay;
};

enum class DependencyType {
    SEQUENTIAL,     // Must complete before
    PARALLEL,       // Can run simultaneously
    CONDITIONAL,    // Depends on outcome
    TEMPORAL        // Time-based dependency
};

// Priority levels
enum class PriorityLevel {
    CRITICAL = 0,
    HIGH = 1,
    NORMAL = 2,
    LOW = 3,
    BACKGROUND = 4
};

// Mission status
struct MissionStatus {
    std::string missionId;
    MissionState state;
    float progress;  // 0-1
    std::vector<TaskStatus> taskStatuses;
    std::chrono::steady_clock::time_point lastUpdate;
    std::vector<std::string> errors;
};

enum class MissionState {
    PENDING,
    SCHEDULED,
    EXECUTING,
    PAUSED,
    COMPLETED,
    FAILED,
    CANCELLED
};

struct TaskStatus {
    std::string taskId;
    TaskState state;
    std::string assignedAgent;
    float progress;
    std::optional<std::string> result;
};

enum class TaskState {
    QUEUED,
    ASSIGNED,
    RUNNING,
    COMPLETED,
    FAILED,
    TIMEOUT
};

// Mission updates
struct MissionUpdate {
    std::optional<PriorityLevel> newPriority;
    std::optional<std::vector<Task>> additionalTasks;
    std::optional<std::chrono::milliseconds> extendTimeout;
    std::vector<std::string> cancelTasks;
};

// Temporal constraints
struct TemporalConstraint {
    ConstraintType type;
    std::chrono::steady_clock::time_point time;
    std::array<float, 3> position;
    float radius;
};

enum class ConstraintType {
    AVOID_AREA,        // Don't enter area at time
    REQUIRED_WAYPOINT, // Must pass through
    TIME_WINDOW,       // Must arrive within window
    SYNCHRONIZATION    // Coordinate with other agents
};

// Resource capacity
struct ResourceCapacity {
    float maxComputeGFLOPS;
    float maxMemoryGB;
    float maxNetworkBandwidthMbps;
    float maxPowerWatts;
    float currentUtilization;  // 0-1
};

} // namespace ares::orchestrator