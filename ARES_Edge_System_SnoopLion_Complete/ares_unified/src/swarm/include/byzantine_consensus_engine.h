/**
 * @file byzantine_consensus_engine.h
 * @brief Byzantine fault-tolerant consensus engine for swarm coordination
 * 
 * Implements Practical Byzantine Fault Tolerance (PBFT) variant optimized
 * for real-time autonomous swarms with up to 33% Byzantine nodes
 */

#ifndef ARES_SWARM_BYZANTINE_CONSENSUS_ENGINE_H
#define ARES_SWARM_BYZANTINE_CONSENSUS_ENGINE_H

#include <cuda_runtime.h>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <memory>
#include <thread>
#include <queue>
#include <chrono>
#include <array>
#include <mutex>
#include <condition_variable>

namespace ares::swarm {

// Consensus Configuration
constexpr uint32_t MAX_SWARM_SIZE = 128;
constexpr uint32_t MAX_FAULTY_NODES = MAX_SWARM_SIZE / 3;
constexpr uint32_t CONSENSUS_TIMEOUT_MS = 100;
constexpr uint32_t VIEW_CHANGE_TIMEOUT_MS = 500;
constexpr uint32_t CHECKPOINT_INTERVAL = 100;
constexpr uint32_t MAX_MESSAGE_SIZE = 4096;
constexpr uint32_t CRYPTO_SIGNATURE_SIZE = 64;

// Message Types
enum class MessageType : uint8_t {
    REQUEST = 0,
    PRE_PREPARE = 1,
    PREPARE = 2,
    COMMIT = 3,
    REPLY = 4,
    VIEW_CHANGE = 5,
    NEW_VIEW = 6,
    CHECKPOINT = 7,
    HEARTBEAT = 8,
    STATE_TRANSFER = 9
};

// Node States
enum class NodeState : uint8_t {
    FOLLOWER = 0,
    CANDIDATE = 1,
    PRIMARY = 2,
    FAULTY = 3,
    RECOVERING = 4
};

// Request Types for Swarm Operations
enum class SwarmRequestType : uint8_t {
    FORMATION_CHANGE = 0,
    TASK_ASSIGNMENT = 1,
    WAYPOINT_UPDATE = 2,
    SENSOR_FUSION = 3,
    THREAT_RESPONSE = 4,
    RESOURCE_ALLOCATION = 5,
    EMERGENCY_ACTION = 6,
    STATE_SYNC = 7
};

// Cryptographic structures
struct Signature {
    uint8_t data[CRYPTO_SIGNATURE_SIZE];
    uint32_t signer_id;
};

struct MessageDigest {
    uint8_t hash[32];  // SHA-256
    
    bool operator==(const MessageDigest& other) const {
        return memcmp(hash, other.hash, 32) == 0;
    }
};

// Consensus Messages
struct ConsensusMessage {
    MessageType type;
    uint32_t view_number;
    uint64_t sequence_number;
    uint32_t sender_id;
    uint64_t timestamp_us;
    MessageDigest digest;
    uint8_t payload[MAX_MESSAGE_SIZE];
    uint32_t payload_size;
    Signature signature;
};

// Client Request
struct ClientRequest {
    uint64_t request_id;
    SwarmRequestType type;
    uint32_t client_id;
    uint64_t timestamp_us;
    uint8_t data[MAX_MESSAGE_SIZE];
    uint32_t data_size;
    Signature signature;
};

// Prepared Certificate
struct PreparedCertificate {
    uint32_t view_number;
    uint64_t sequence_number;
    MessageDigest digest;
    std::vector<Signature> prepare_signatures;
    bool is_complete() const { return prepare_signatures.size() >= 2 * MAX_FAULTY_NODES; }
};

// Committed Certificate
struct CommittedCertificate {
    uint32_t view_number;
    uint64_t sequence_number;
    MessageDigest digest;
    std::vector<Signature> commit_signatures;
    bool is_complete() const { return commit_signatures.size() >= 2 * MAX_FAULTY_NODES + 1; }
};

// View Change Message
struct ViewChangeMessage {
    uint32_t new_view_number;
    uint64_t last_stable_checkpoint;
    std::vector<PreparedCertificate> prepared_proofs;
    uint32_t sender_id;
    Signature signature;
};

// Checkpoint
struct Checkpoint {
    uint64_t sequence_number;
    MessageDigest state_digest;
    std::unordered_map<uint32_t, Signature> signatures;
    
    bool is_stable() const { return signatures.size() >= 2 * MAX_FAULTY_NODES + 1; }
};

// Node Information
struct NodeInfo {
    uint32_t node_id;
    NodeState state;
    std::array<float, 3> position;  // x, y, z coordinates
    float reliability_score;        // 0.0 to 1.0
    uint64_t last_heartbeat_us;
    bool is_byzantine;             // For simulation/testing
};

// Consensus State
struct ConsensusState {
    uint32_t view_number;
    uint64_t sequence_number;
    uint64_t last_executed;
    uint64_t last_stable_checkpoint;
    NodeState node_state;
    uint32_t primary_id;
    std::chrono::steady_clock::time_point view_change_timer;
};

// Performance Metrics
struct ConsensusMetrics {
    uint64_t total_consensus_rounds;
    uint64_t successful_consensus;
    uint64_t view_changes;
    uint64_t message_count;
    float average_latency_ms;
    float throughput_ops_per_sec;
    uint32_t byzantine_detections;
};

class ByzantineConsensusEngine {
public:
    ByzantineConsensusEngine(uint32_t node_id, uint32_t swarm_size);
    ~ByzantineConsensusEngine();
    
    // Initialize consensus engine
    cudaError_t initialize(
        const std::vector<NodeInfo>& initial_nodes,
        const char* crypto_key_file = nullptr
    );
    
    // Client interface
    cudaError_t submit_request(
        const ClientRequest& request,
        std::function<void(bool success, const uint8_t* reply, uint32_t size)> callback
    );
    
    // Message handling
    cudaError_t process_message(const ConsensusMessage& message);
    cudaError_t broadcast_message(const ConsensusMessage& message);
    
    // State management
    cudaError_t create_checkpoint();
    cudaError_t recover_from_checkpoint(uint64_t checkpoint_seq);
    cudaError_t transfer_state(uint32_t target_node);
    
    // View change protocol
    cudaError_t initiate_view_change();
    cudaError_t process_view_change(const ViewChangeMessage& vc_message);
    
    // Byzantine detection
    cudaError_t detect_byzantine_behavior();
    std::vector<uint32_t> get_byzantine_nodes() const;
    
    // Network management
    cudaError_t add_node(const NodeInfo& node);
    cudaError_t remove_node(uint32_t node_id);
    cudaError_t update_node_position(uint32_t node_id, const std::array<float, 3>& position);
    
    // Metrics and monitoring
    ConsensusMetrics get_metrics() const { return metrics_; }
    ConsensusState get_state() const { return state_; }
    
    // GPU acceleration for crypto operations
    cudaError_t batch_verify_signatures(
        const Signature* signatures,
        const MessageDigest* digests,
        uint32_t count,
        bool* results
    );
    
private:
    // Node identity
    uint32_t node_id_;
    uint32_t swarm_size_;
    std::vector<NodeInfo> nodes_;
    
    // Consensus state
    ConsensusState state_;
    std::mutex state_mutex_;
    
    // Message logs
    struct MessageLog {
        std::unordered_map<uint64_t, ConsensusMessage> pre_prepares;
        std::unordered_map<uint64_t, std::vector<ConsensusMessage>> prepares;
        std::unordered_map<uint64_t, std::vector<ConsensusMessage>> commits;
        std::unordered_map<uint64_t, ClientRequest> requests;
    } message_log_;
    
    // Prepared and committed certificates
    std::unordered_map<uint64_t, PreparedCertificate> prepared_certs_;
    std::unordered_map<uint64_t, CommittedCertificate> committed_certs_;
    
    // Checkpoints
    std::unordered_map<uint64_t, Checkpoint> checkpoints_;
    uint64_t last_checkpoint_seq_;
    
    // View change state
    std::vector<ViewChangeMessage> view_change_messages_;
    bool view_changing_;
    std::chrono::steady_clock::time_point view_change_start_;
    
    // Client callbacks
    std::unordered_map<uint64_t, std::function<void(bool, const uint8_t*, uint32_t)>> client_callbacks_;
    
    // Network layer (abstracted)
    std::queue<ConsensusMessage> incoming_messages_;
    std::queue<ConsensusMessage> outgoing_messages_;
    std::mutex message_queue_mutex_;
    std::condition_variable message_cv_;
    
    // Worker threads
    std::thread consensus_thread_;
    std::thread network_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> running_;
    
    // GPU resources for crypto
    struct CryptoGPU {
        uint8_t* d_signatures;
        uint8_t* d_digests;
        bool* d_verify_results;
        cudaStream_t crypto_stream;
    } crypto_gpu_;
    
    // Metrics
    ConsensusMetrics metrics_;
    
    // Internal methods
    void consensus_worker();
    void network_worker();
    void monitor_worker();
    
    // PBFT protocol phases
    cudaError_t process_request(const ClientRequest& request);
    cudaError_t send_pre_prepare(const ClientRequest& request);
    cudaError_t process_pre_prepare(const ConsensusMessage& message);
    cudaError_t send_prepare(uint64_t seq, const MessageDigest& digest);
    cudaError_t process_prepare(const ConsensusMessage& message);
    cudaError_t send_commit(uint64_t seq, const MessageDigest& digest);
    cudaError_t process_commit(const ConsensusMessage& message);
    cudaError_t execute_request(uint64_t seq);
    
    // Cryptographic operations
    MessageDigest compute_digest(const uint8_t* data, uint32_t size);
    Signature sign_message(const MessageDigest& digest);
    bool verify_signature(const Signature& sig, const MessageDigest& digest, uint32_t signer_id);
    
    // State management
    bool is_primary() const { return state_.primary_id == node_id_; }
    uint32_t get_primary(uint32_t view) const { return view % swarm_size_; }
    bool in_view(uint32_t view) const { return view == state_.view_number; }
    
    // Byzantine detection helpers
    bool is_valid_message(const ConsensusMessage& message);
    void record_byzantine_behavior(uint32_t node_id, const std::string& reason);
    
    // Utility methods
    void start_view_change_timer();
    void reset_view_change_timer();
    bool has_quorum(uint32_t count) const { return count >= 2 * MAX_FAULTY_NODES + 1; }
};

// GPU Kernels for cryptographic operations
namespace crypto_kernels {

__global__ void batch_sha256_kernel(
    const uint8_t* messages,
    const uint32_t* message_lengths,
    uint8_t* digests,
    uint32_t num_messages
);

__global__ void batch_ecdsa_verify_kernel(
    const uint8_t* signatures,
    const uint8_t* digests,
    const uint8_t* public_keys,
    bool* results,
    uint32_t num_signatures
);

__global__ void merkle_tree_kernel(
    const uint8_t* leaf_hashes,
    uint8_t* tree_nodes,
    uint32_t num_leaves
);

} // namespace crypto_kernels

// Distributed state machine for swarm operations
class SwarmStateMachine {
public:
    SwarmStateMachine();
    
    // Execute consensus decision
    bool execute(const ClientRequest& request, uint8_t* reply, uint32_t* reply_size);
    
    // State queries
    MessageDigest get_state_digest() const;
    bool restore_state(const MessageDigest& digest);
    
private:
    // Swarm state
    struct SwarmState {
        std::array<std::array<float, 3>, MAX_SWARM_SIZE> positions;
        std::array<uint32_t, MAX_SWARM_SIZE> task_assignments;
        std::array<float, MAX_SWARM_SIZE> battery_levels;
        uint32_t formation_id;
        uint64_t state_version;
    } state_;
    
    // State execution methods
    bool execute_formation_change(const uint8_t* data, uint32_t size);
    bool execute_task_assignment(const uint8_t* data, uint32_t size);
    bool execute_waypoint_update(const uint8_t* data, uint32_t size);
    bool execute_threat_response(const uint8_t* data, uint32_t size);
};

} // namespace ares::swarm

#endif // ARES_SWARM_BYZANTINE_CONSENSUS_ENGINE_H