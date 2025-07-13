#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <variant>

namespace ares::federated_learning {

// Unified federated learning interface with homomorphic encryption
class FederatedLearningSystem {
public:
    FederatedLearningSystem();
    ~FederatedLearningSystem();
    
    // System initialization
    bool initialize(const FederationConfig& config);
    void shutdown();
    
    // Node management
    std::string registerNode(const NodeCapabilities& capabilities);
    bool unregisterNode(const std::string& nodeId);
    std::vector<std::string> getActiveNodes() const;
    
    // Model training
    struct TrainingTask {
        std::string modelArchitecture;
        std::vector<uint8_t> initialWeights;
        HyperParameters hyperParams;
        PrivacyConstraints privacy;
    };
    
    std::string startTraining(const TrainingTask& task);
    bool submitLocalUpdate(
        const std::string& taskId,
        const std::string& nodeId,
        const EncryptedGradients& gradients
    );
    
    // Model aggregation with secure computation
    struct AggregationResult {
        std::vector<uint8_t> updatedWeights;
        float globalLoss;
        size_t contributingNodes;
        std::vector<float> nodeContributions;
    };
    
    AggregationResult performSecureAggregation(const std::string& taskId);
    
    // Homomorphic operations
    EncryptedTensor encryptTensor(const std::vector<float>& plaintext);
    std::vector<float> decryptTensor(const EncryptedTensor& ciphertext);
    EncryptedTensor homomorphicAdd(
        const EncryptedTensor& a,
        const EncryptedTensor& b
    );
    EncryptedTensor homomorphicMultiply(
        const EncryptedTensor& tensor,
        float scalar
    );
    
    // Distributed SLAM integration
    struct SLAMUpdate {
        std::string nodeId;
        std::vector<float> pose;  // x, y, z, roll, pitch, yaw
        std::vector<PointCloud> localMap;
        uint64_t timestamp;
    };
    
    bool submitSLAMUpdate(const SLAMUpdate& update);
    GlobalMap getConsensusMap() const;
    
    // Privacy and security
    bool verifyNodeAttestation(const std::string& nodeId);
    void setDifferentialPrivacyBudget(float epsilon);
    
    // Performance control
    void setUseCuda(bool useCuda);
    void setComputeBudget(const ComputeBudget& budget);
    
    // Monitoring
    struct SystemMetrics {
        float averageTrainingTime;
        float communicationOverhead;
        size_t totalDataProcessed;
        float privacyBudgetConsumed;
    };
    
    SystemMetrics getMetrics() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Federation configuration
struct FederationConfig {
    size_t minNodesForTraining = 3;
    size_t maxNodesPerRound = 100;
    float aggregationThreshold = 0.7f;  // Minimum participation
    bool enableHomomorphicEncryption = true;
    bool enableSecureMultipartyComputation = true;
    std::string encryptionScheme = "CKKS";  // or "BFV", "BGV"
};

// Node capabilities
struct NodeCapabilities {
    float computePowerTFLOPS;
    size_t memoryGB;
    float networkBandwidthMbps;
    bool hasGPU;
    bool hasTEE;  // Trusted Execution Environment
    std::vector<std::string> supportedOperations;
};

// Training hyperparameters
struct HyperParameters {
    float learningRate = 0.001f;
    size_t batchSize = 32;
    size_t localEpochs = 5;
    std::string optimizer = "SGD";  // or "Adam", "RMSprop"
    float momentum = 0.9f;
};

// Privacy constraints
struct PrivacyConstraints {
    float differentialPrivacyEpsilon = 1.0f;
    float differentialPrivacyDelta = 1e-5f;
    bool enableSecureAggregation = true;
    size_t minNodesForPrivacy = 10;
};

// Encrypted data structures
struct EncryptedTensor {
    std::vector<uint8_t> ciphertext;
    std::vector<size_t> shape;
    std::string encryptionScheme;
};

struct EncryptedGradients {
    std::vector<EncryptedTensor> layerGradients;
    float encryptedLoss;
    size_t sampleCount;
};

// SLAM data structures
struct PointCloud {
    std::vector<std::array<float, 3>> points;
    std::vector<std::array<uint8_t, 3>> colors;
    std::vector<float> confidences;
};

struct GlobalMap {
    std::vector<PointCloud> mergedClouds;
    std::vector<std::array<float, 6>> nodePoses;
    float consensusScore;
};

// Compute budget
struct ComputeBudget {
    float maxComputeTimeSeconds = 60.0f;
    size_t maxMemoryMB = 4096;
    float maxPowerWatts = 100.0f;
};

} // namespace ares::federated_learning