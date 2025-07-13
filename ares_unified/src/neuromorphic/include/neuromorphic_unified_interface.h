#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <variant>
#include <complex>

namespace ares::neuromorphic {

// Comprehensive neuromorphic computing interface
class NeuromorphicEngine {
public:
    NeuromorphicEngine();
    ~NeuromorphicEngine();
    
    // System initialization
    bool initialize(const NeuromorphicConfig& config);
    void shutdown();
    
    // Hardware backends
    enum class HardwareBackend {
        CPU_SIMULATION,
        CUDA_ACCELERATED,
        LOIHI2,
        BRAINSCALES2,
        SPINNAKER,
        TPU_NEUROMORPHIC,
        CUSTOM_ASIC
    };
    
    bool setHardwareBackend(HardwareBackend backend);
    std::vector<HardwareBackend> getAvailableBackends() const;
    
    // Spiking neural network operations
    struct SpikingNetwork {
        std::vector<NeuronGroup> neuronGroups;
        std::vector<SynapseGroup> synapseGroups;
        NetworkTopology topology;
        LearningRule learningRule;
    };
    
    std::string createNetwork(const SpikingNetwork& network);
    bool deployNetwork(const std::string& networkId);
    
    // Spike encoding/decoding
    std::vector<SpikeEvent> encodeData(
        const std::vector<float>& data,
        EncodingScheme scheme
    );
    std::vector<float> decodeSpikes(
        const std::vector<SpikeEvent>& spikes,
        DecodingScheme scheme
    );
    
    // Real-time inference
    struct InferenceResult {
        std::vector<float> outputs;
        std::vector<SpikeEvent> outputSpikes;
        float energyConsumed;  // picojoules
        std::chrono::nanoseconds latency;
    };
    
    InferenceResult runInference(
        const std::string& networkId,
        const std::vector<float>& inputs,
        std::chrono::milliseconds duration
    );
    
    // Online learning
    bool enableOnlineLearning(const std::string& networkId, bool enable);
    bool updateSynapticWeights(
        const std::string& networkId,
        const std::vector<WeightUpdate>& updates
    );
    
    // Sensor integration
    struct SensorStream {
        std::string sensorId;
        SensorType type;
        size_t channelCount;
        float samplingRate;  // Hz
        std::function<std::vector<float>()> dataCallback;
    };
    
    bool connectSensor(const SensorStream& sensor);
    bool disconnectSensor(const std::string& sensorId);
    
    // Multi-modal fusion
    struct FusionResult {
        std::vector<float> fusedFeatures;
        std::map<std::string, float> modalityContributions;
        float confidence;
    };
    
    FusionResult fuseMultiModal(
        const std::map<std::string, std::vector<float>>& modalityData
    );
    
    // Neuromorphic-specific optimizations
    void enableEventDrivenProcessing(bool enable);
    void setSparsityThreshold(float threshold);
    void optimizeForPowerEfficiency();
    
    // Monitoring and diagnostics
    struct NeuromorphicMetrics {
        size_t totalNeurons;
        size_t totalSynapses;
        float averageActivity;  // Hz
        float powerConsumption; // mW
        float memoryUsage;      // MB
        size_t spikesPerSecond;
    };
    
    NeuromorphicMetrics getMetrics() const;
    
    // Brian2 integration
    bool loadBrian2Model(const std::string& modelPath);
    bool exportToBrian2(const std::string& networkId, const std::string& outputPath);
    
    // Lava framework integration
    bool importLavaNetwork(const std::string& lavaConfig);
    bool exportToLava(const std::string& networkId, const std::string& outputPath);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Configuration
struct NeuromorphicConfig {
    size_t maxNeurons = 1000000;
    size_t maxSynapses = 10000000;
    float timeStep = 1.0f;  // milliseconds
    bool enablePlasticity = true;
    bool enableHomeostasis = true;
    std::string defaultBackend = "CUDA_ACCELERATED";
};

// Neuron group specification
struct NeuronGroup {
    std::string groupId;
    size_t neuronCount;
    NeuronModel model;
    std::vector<float> parameters;
    std::vector<float> initialStates;
};

// Neuron models
enum class NeuronModel {
    LEAKY_INTEGRATE_FIRE,
    ADAPTIVE_EXPONENTIAL,
    HODGKIN_HUXLEY,
    IZHIKEVICH,
    CUSTOM
};

// Synapse group specification
struct SynapseGroup {
    std::string sourceGroup;
    std::string targetGroup;
    SynapseModel model;
    ConnectivityPattern connectivity;
    std::vector<float> weights;
    std::vector<float> delays;  // milliseconds
};

// Synapse models
enum class SynapseModel {
    STATIC,
    STDP,  // Spike-timing dependent plasticity
    SHORT_TERM_PLASTICITY,
    TRIPLET_STDP,
    CUSTOM
};

// Connectivity patterns
struct ConnectivityPattern {
    enum Type {
        ALL_TO_ALL,
        ONE_TO_ONE,
        RANDOM,
        GAUSSIAN,
        CUSTOM
    } type;
    
    float connectionProbability = 0.1f;
    std::vector<std::pair<size_t, size_t>> customConnections;
};

// Network topology
struct NetworkTopology {
    std::vector<std::string> layerOrder;
    std::map<std::string, LayerType> layerTypes;
    bool isRecurrent;
};

enum class LayerType {
    INPUT,
    HIDDEN,
    OUTPUT,
    RESERVOIR,  // For liquid state machines
    MEMORY
};

// Learning rules
struct LearningRule {
    enum Type {
        NONE,
        STDP,
        REWARD_MODULATED_STDP,
        SUPERVISED_TEMPOTRON,
        REINFORCEMENT_LEARNING
    } type;
    
    std::vector<float> parameters;
    float learningRate = 0.01f;
};

// Spike events
struct SpikeEvent {
    size_t neuronId;
    float timestamp;  // milliseconds
    float weight = 1.0f;
};

// Encoding schemes
enum class EncodingScheme {
    RATE_CODING,
    TEMPORAL_CODING,
    PHASE_CODING,
    BURST_CODING,
    POPULATION_CODING
};

// Decoding schemes
enum class DecodingScheme {
    SPIKE_COUNT,
    FIRST_SPIKE_TIME,
    SPIKE_PATTERN,
    POPULATION_VECTOR
};

// Weight updates for online learning
struct WeightUpdate {
    size_t synapseId;
    float deltaWeight;
    float newWeight;
};

// Sensor types
enum class SensorType {
    DVS_CAMERA,      // Dynamic vision sensor
    COCHLEA,         // Neuromorphic audio
    OLFACTORY,       // Smell sensor
    TACTILE,         // Touch sensor
    IMU,             // Inertial measurement
    LIDAR,           // Neuromorphic LiDAR
    CUSTOM
};

} // namespace ares::neuromorphic