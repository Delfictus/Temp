#pragma once

#include <memory>
#include <vector>
#include <variant>
#include "byzantine_consensus_engine.h"
#include "distributed_task_auction.h"

namespace ares::swarm {

// Unified interface supporting both CUDA and CPU implementations
class SwarmCoordinator {
public:
    SwarmCoordinator();
    ~SwarmCoordinator();

    // Byzantine consensus operations
    bool initializeByzantineConsensus(size_t nodeCount, float byzantineThreshold = 0.33f);
    bool proposeConsensus(const std::vector<uint8_t>& data);
    bool validateConsensus(const std::string& proposalId);
    
    // Distributed task auction operations
    bool initializeTaskAuction(size_t maxTasks, size_t maxBidders);
    std::string submitTask(const TaskSpecification& task);
    bool submitBid(const std::string& taskId, const BidSpecification& bid);
    TaskAllocation getOptimalAllocation(const std::string& taskId);
    
    // Performance metrics
    struct PerformanceMetrics {
        double consensusLatencyMs;
        double auctionOptimizationTimeMs;
        size_t activeNodes;
        float networkUtilization;
    };
    
    PerformanceMetrics getMetrics() const;
    
    // Enable/disable CUDA acceleration
    void setUseCuda(bool useCuda);
    bool isCudaAvailable() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Task specification for auction system
struct TaskSpecification {
    std::string taskId;
    std::vector<float> requirements;  // CPU, GPU, memory, bandwidth
    float priority;
    uint64_t deadline;
    std::vector<uint8_t> payload;
};

// Bid specification from swarm nodes
struct BidSpecification {
    std::string nodeId;
    std::vector<float> capabilities;
    float cost;
    uint64_t estimatedCompletionTime;
};

// Task allocation result
struct TaskAllocation {
    std::string taskId;
    std::vector<std::string> assignedNodes;
    float totalCost;
    float confidenceScore;
};

} // namespace ares::swarm