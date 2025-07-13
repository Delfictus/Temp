/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Company: DELFICTUS I/O LLC
 * CAGE Code: 13H70
 * UEI: LXT3B9GMY4N8
 * Active DoD Contractor
 * 
 * Location: Los Angeles, California 90013 United States
 * 
 * This software contains trade secrets and proprietary information
 * of DELFICTUS I/O LLC. Unauthorized use, reproduction, or distribution
 * is strictly prohibited. This technology is subject to export controls
 * under ITAR and EAR regulations.
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * WARNING: This system is designed for authorized U.S. Department of Defense
 * use only. Misuse may result in severe criminal and civil penalties.
 */

/**
 * @file byzantine_consensus_engine.cpp
 * @brief Implementation of Byzantine fault-tolerant consensus for swarm coordination
 */

#include "../include/byzantine_consensus_engine.h"
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <cstring>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

namespace ares::swarm {

using namespace std::chrono;

// Utility function to convert digest to hex string
std::string digest_to_string(const MessageDigest& digest) {
    std::stringstream ss;
    for (int i = 0; i < 32; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest.hash[i];
    }
    return ss.str();
}

// SwarmStateMachine Implementation
SwarmStateMachine::SwarmStateMachine() {
    memset(&state_, 0, sizeof(state_));
    state_.formation_id = 1;  // Default formation
    state_.state_version = 0;
}

bool SwarmStateMachine::execute(const ClientRequest& request, uint8_t* reply, uint32_t* reply_size) {
    bool success = false;
    
    switch (request.type) {
        case SwarmRequestType::FORMATION_CHANGE:
            success = execute_formation_change(request.data, request.data_size);
            break;
            
        case SwarmRequestType::TASK_ASSIGNMENT:
            success = execute_task_assignment(request.data, request.data_size);
            break;
            
        case SwarmRequestType::WAYPOINT_UPDATE:
            success = execute_waypoint_update(request.data, request.data_size);
            break;
            
        case SwarmRequestType::THREAT_RESPONSE:
            success = execute_threat_response(request.data, request.data_size);
            break;
            
        default:
            break;
    }
    
    if (success) {
        state_.state_version++;
        
        // Prepare reply
        memcpy(reply, &state_.state_version, sizeof(uint64_t));
        *reply_size = sizeof(uint64_t);
    }
    
    return success;
}

MessageDigest SwarmStateMachine::get_state_digest() const {
    MessageDigest digest;
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, &state_, sizeof(state_));
    SHA256_Final(digest.hash, &sha256);
    return digest;
}

bool SwarmStateMachine::execute_formation_change(const uint8_t* data, uint32_t size) {
    if (size < sizeof(uint32_t)) return false;
    
    uint32_t new_formation_id;
    memcpy(&new_formation_id, data, sizeof(uint32_t));
    
    // Validate formation ID
    if (new_formation_id == 0 || new_formation_id > 10) return false;
    
    state_.formation_id = new_formation_id;
    
    // Update positions based on formation
    // This is simplified - real implementation would calculate actual positions
    for (uint32_t i = 0; i < MAX_SWARM_SIZE; ++i) {
        float angle = 2.0f * M_PI * i / MAX_SWARM_SIZE;
        float radius = 10.0f * new_formation_id;
        
        state_.positions[i][0] = radius * cosf(angle);
        state_.positions[i][1] = radius * sinf(angle);
        state_.positions[i][2] = 100.0f;  // Altitude
    }
    
    return true;
}

bool SwarmStateMachine::execute_task_assignment(const uint8_t* data, uint32_t size) {
    struct TaskAssignment {
        uint32_t node_id;
        uint32_t task_id;
    };
    
    if (size % sizeof(TaskAssignment) != 0) return false;
    
    uint32_t num_assignments = size / sizeof(TaskAssignment);
    const TaskAssignment* assignments = reinterpret_cast<const TaskAssignment*>(data);
    
    for (uint32_t i = 0; i < num_assignments; ++i) {
        if (assignments[i].node_id < MAX_SWARM_SIZE) {
            state_.task_assignments[assignments[i].node_id] = assignments[i].task_id;
        }
    }
    
    return true;
}

bool SwarmStateMachine::execute_waypoint_update(const uint8_t* data, uint32_t size) {
    struct Waypoint {
        uint32_t node_id;
        float x, y, z;
    };
    
    if (size % sizeof(Waypoint) != 0) return false;
    
    uint32_t num_waypoints = size / sizeof(Waypoint);
    const Waypoint* waypoints = reinterpret_cast<const Waypoint*>(data);
    
    for (uint32_t i = 0; i < num_waypoints; ++i) {
        if (waypoints[i].node_id < MAX_SWARM_SIZE) {
            state_.positions[waypoints[i].node_id][0] = waypoints[i].x;
            state_.positions[waypoints[i].node_id][1] = waypoints[i].y;
            state_.positions[waypoints[i].node_id][2] = waypoints[i].z;
        }
    }
    
    return true;
}

bool SwarmStateMachine::execute_threat_response(const uint8_t* data, uint32_t size) {
    struct ThreatInfo {
        float threat_x, threat_y, threat_z;
        uint32_t threat_type;
    };
    
    if (size != sizeof(ThreatInfo)) return false;
    
    const ThreatInfo* threat = reinterpret_cast<const ThreatInfo*>(data);
    
    // Simplified threat response: move all nodes away from threat
    for (uint32_t i = 0; i < MAX_SWARM_SIZE; ++i) {
        float dx = state_.positions[i][0] - threat->threat_x;
        float dy = state_.positions[i][1] - threat->threat_y;
        float dz = state_.positions[i][2] - threat->threat_z;
        
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < 100.0f && dist > 0.1f) {  // Within threat range
            // Move away from threat
            state_.positions[i][0] += 50.0f * dx / dist;
            state_.positions[i][1] += 50.0f * dy / dist;
            state_.positions[i][2] += 50.0f * dz / dist;
        }
    }
    
    return true;
}

// ByzantineConsensusEngine Implementation
ByzantineConsensusEngine::ByzantineConsensusEngine(uint32_t node_id, uint32_t swarm_size)
    : node_id_(node_id)
    , swarm_size_(swarm_size)
    , running_(false)
    , view_changing_(false)
    , last_checkpoint_seq_(0) {
    
    memset(&state_, 0, sizeof(state_));
    memset(&metrics_, 0, sizeof(metrics_));
    memset(&crypto_gpu_, 0, sizeof(crypto_gpu_));
    
    state_.node_state = NodeState::FOLLOWER;
    state_.primary_id = get_primary(0);
}

ByzantineConsensusEngine::~ByzantineConsensusEngine() {
    running_ = false;
    
    // Stop worker threads
    message_cv_.notify_all();
    if (consensus_thread_.joinable()) consensus_thread_.join();
    if (network_thread_.joinable()) network_thread_.join();
    if (monitor_thread_.joinable()) monitor_thread_.join();
    
    // Free GPU resources
    if (crypto_gpu_.d_signatures) cudaFree(crypto_gpu_.d_signatures);
    if (crypto_gpu_.d_digests) cudaFree(crypto_gpu_.d_digests);
    if (crypto_gpu_.d_verify_results) cudaFree(crypto_gpu_.d_verify_results);
    if (crypto_gpu_.crypto_stream) cudaStreamDestroy(crypto_gpu_.crypto_stream);
}

cudaError_t ByzantineConsensusEngine::initialize(
    const std::vector<NodeInfo>& initial_nodes,
    const char* crypto_key_file
) {
    nodes_ = initial_nodes;
    
    // Initialize GPU crypto resources
    cudaError_t err;
    
    err = cudaStreamCreate(&crypto_gpu_.crypto_stream);
    if (err != cudaSuccess) return err;
    
    // Allocate GPU memory for batch crypto operations
    const size_t batch_size = 1024;
    err = cudaMalloc(&crypto_gpu_.d_signatures, batch_size * CRYPTO_SIGNATURE_SIZE);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&crypto_gpu_.d_digests, batch_size * 32);  // SHA-256
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&crypto_gpu_.d_verify_results, batch_size * sizeof(bool));
    if (err != cudaSuccess) return err;
    
    // Start worker threads
    running_ = true;
    consensus_thread_ = std::thread(&ByzantineConsensusEngine::consensus_worker, this);
    network_thread_ = std::thread(&ByzantineConsensusEngine::network_worker, this);
    monitor_thread_ = std::thread(&ByzantineConsensusEngine::monitor_worker, this);
    
    // Initialize timers
    reset_view_change_timer();
    
    return cudaSuccess;
}

cudaError_t ByzantineConsensusEngine::submit_request(
    const ClientRequest& request,
    std::function<void(bool success, const uint8_t* reply, uint32_t size)> callback
) {
    // Store callback
    client_callbacks_[request.request_id] = callback;
    
    // If we're the primary, start consensus immediately
    if (is_primary()) {
        return process_request(request);
    } else {
        // Forward to primary
        ConsensusMessage msg;
        msg.type = MessageType::REQUEST;
        msg.view_number = state_.view_number;
        msg.sender_id = node_id_;
        msg.timestamp_us = duration_cast<microseconds>(
            system_clock::now().time_since_epoch()).count();
        
        memcpy(msg.payload, &request, sizeof(request));
        msg.payload_size = sizeof(request);
        
        return broadcast_message(msg);
    }
}

cudaError_t ByzantineConsensusEngine::process_request(const ClientRequest& request) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Assign sequence number
    uint64_t seq = ++state_.sequence_number;
    
    // Store request
    message_log_.requests[seq] = request;
    
    // Send pre-prepare
    return send_pre_prepare(request);
}

cudaError_t ByzantineConsensusEngine::send_pre_prepare(const ClientRequest& request) {
    ConsensusMessage msg;
    msg.type = MessageType::PRE_PREPARE;
    msg.view_number = state_.view_number;
    msg.sequence_number = state_.sequence_number;
    msg.sender_id = node_id_;
    msg.timestamp_us = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    
    // Compute digest of request
    msg.digest = compute_digest(reinterpret_cast<const uint8_t*>(&request), sizeof(request));
    
    // Include request in payload
    memcpy(msg.payload, &request, sizeof(request));
    msg.payload_size = sizeof(request);
    
    // Sign message
    msg.signature = sign_message(msg.digest);
    
    // Store in log
    message_log_.pre_prepares[msg.sequence_number] = msg;
    
    // Broadcast to all replicas
    return broadcast_message(msg);
}

cudaError_t ByzantineConsensusEngine::process_pre_prepare(const ConsensusMessage& message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Validate message
    if (!is_valid_message(message)) {
        record_byzantine_behavior(message.sender_id, "Invalid pre-prepare");
        return cudaErrorInvalidValue;
    }
    
    // Check view number
    if (!in_view(message.view_number)) {
        return cudaErrorInvalidValue;
    }
    
    // Check if we already have a pre-prepare for this sequence
    if (message_log_.pre_prepares.count(message.sequence_number) > 0) {
        return cudaErrorInvalidValue;
    }
    
    // Verify signature
    if (!verify_signature(message.signature, message.digest, message.sender_id)) {
        record_byzantine_behavior(message.sender_id, "Invalid signature");
        return cudaErrorInvalidValue;
    }
    
    // Store pre-prepare
    message_log_.pre_prepares[message.sequence_number] = message;
    
    // Extract and store request
    ClientRequest request;
    memcpy(&request, message.payload, sizeof(request));
    message_log_.requests[message.sequence_number] = request;
    
    // Send prepare message
    return send_prepare(message.sequence_number, message.digest);
}

cudaError_t ByzantineConsensusEngine::send_prepare(uint64_t seq, const MessageDigest& digest) {
    ConsensusMessage msg;
    msg.type = MessageType::PREPARE;
    msg.view_number = state_.view_number;
    msg.sequence_number = seq;
    msg.sender_id = node_id_;
    msg.timestamp_us = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    msg.digest = digest;
    msg.payload_size = 0;
    
    // Sign message
    msg.signature = sign_message(digest);
    
    // Broadcast to all replicas
    return broadcast_message(msg);
}

cudaError_t ByzantineConsensusEngine::process_prepare(const ConsensusMessage& message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Validate message
    if (!is_valid_message(message)) {
        record_byzantine_behavior(message.sender_id, "Invalid prepare");
        return cudaErrorInvalidValue;
    }
    
    // Verify signature
    if (!verify_signature(message.signature, message.digest, message.sender_id)) {
        record_byzantine_behavior(message.sender_id, "Invalid signature");
        return cudaErrorInvalidValue;
    }
    
    // Add to prepare log
    message_log_.prepares[message.sequence_number].push_back(message);
    
    // Check if we have enough prepares (2f)
    if (message_log_.prepares[message.sequence_number].size() >= 2 * MAX_FAULTY_NODES) {
        // Create prepared certificate
        PreparedCertificate cert;
        cert.view_number = message.view_number;
        cert.sequence_number = message.sequence_number;
        cert.digest = message.digest;
        
        for (const auto& prep : message_log_.prepares[message.sequence_number]) {
            cert.prepare_signatures.push_back(prep.signature);
        }
        
        prepared_certs_[message.sequence_number] = cert;
        
        // Send commit message
        return send_commit(message.sequence_number, message.digest);
    }
    
    return cudaSuccess;
}

cudaError_t ByzantineConsensusEngine::send_commit(uint64_t seq, const MessageDigest& digest) {
    ConsensusMessage msg;
    msg.type = MessageType::COMMIT;
    msg.view_number = state_.view_number;
    msg.sequence_number = seq;
    msg.sender_id = node_id_;
    msg.timestamp_us = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    msg.digest = digest;
    msg.payload_size = 0;
    
    // Sign message
    msg.signature = sign_message(digest);
    
    // Broadcast to all replicas
    return broadcast_message(msg);
}

cudaError_t ByzantineConsensusEngine::process_commit(const ConsensusMessage& message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Validate message
    if (!is_valid_message(message)) {
        record_byzantine_behavior(message.sender_id, "Invalid commit");
        return cudaErrorInvalidValue;
    }
    
    // Verify signature
    if (!verify_signature(message.signature, message.digest, message.sender_id)) {
        record_byzantine_behavior(message.sender_id, "Invalid signature");
        return cudaErrorInvalidValue;
    }
    
    // Add to commit log
    message_log_.commits[message.sequence_number].push_back(message);
    
    // Check if we have enough commits (2f + 1)
    if (message_log_.commits[message.sequence_number].size() >= 2 * MAX_FAULTY_NODES + 1) {
        // Create committed certificate
        CommittedCertificate cert;
        cert.view_number = message.view_number;
        cert.sequence_number = message.sequence_number;
        cert.digest = message.digest;
        
        for (const auto& commit : message_log_.commits[message.sequence_number]) {
            cert.commit_signatures.push_back(commit.signature);
        }
        
        committed_certs_[message.sequence_number] = cert;
        
        // Execute request
        return execute_request(message.sequence_number);
    }
    
    return cudaSuccess;
}

cudaError_t ByzantineConsensusEngine::execute_request(uint64_t seq) {
    // Execute in order
    while (state_.last_executed + 1 <= seq) {
        uint64_t exec_seq = state_.last_executed + 1;
        
        // Check if we have the request
        auto req_it = message_log_.requests.find(exec_seq);
        if (req_it == message_log_.requests.end()) {
            break;  // Wait for missing request
        }
        
        // Check if committed
        if (committed_certs_.count(exec_seq) == 0) {
            break;  // Wait for commit
        }
        
        // Execute on state machine
        static SwarmStateMachine state_machine;
        uint8_t reply[MAX_MESSAGE_SIZE];
        uint32_t reply_size = 0;
        
        bool success = state_machine.execute(req_it->second, reply, &reply_size);
        
        // Send reply to client
        auto callback_it = client_callbacks_.find(req_it->second.request_id);
        if (callback_it != client_callbacks_.end()) {
            callback_it->second(success, reply, reply_size);
            client_callbacks_.erase(callback_it);
        }
        
        // Update state
        state_.last_executed = exec_seq;
        
        // Check if we need a checkpoint
        if (exec_seq % CHECKPOINT_INTERVAL == 0) {
            create_checkpoint();
        }
        
        // Update metrics
        metrics_.successful_consensus++;
    }
    
    return cudaSuccess;
}

cudaError_t ByzantineConsensusEngine::broadcast_message(const ConsensusMessage& message) {
    std::lock_guard<std::mutex> lock(message_queue_mutex_);
    outgoing_messages_.push(message);
    message_cv_.notify_one();
    
    metrics_.message_count++;
    
    return cudaSuccess;
}

cudaError_t ByzantineConsensusEngine::process_message(const ConsensusMessage& message) {
    // Reset view change timer on valid message
    reset_view_change_timer();
    
    switch (message.type) {
        case MessageType::REQUEST:
            if (is_primary()) {
                ClientRequest request;
                memcpy(&request, message.payload, sizeof(request));
                return process_request(request);
            }
            break;
            
        case MessageType::PRE_PREPARE:
            return process_pre_prepare(message);
            
        case MessageType::PREPARE:
            return process_prepare(message);
            
        case MessageType::COMMIT:
            return process_commit(message);
            
        case MessageType::VIEW_CHANGE:
            // Handle view change
            break;
            
        case MessageType::CHECKPOINT:
            // Handle checkpoint
            break;
            
        default:
            break;
    }
    
    return cudaSuccess;
}

MessageDigest ByzantineConsensusEngine::compute_digest(const uint8_t* data, uint32_t size) {
    MessageDigest digest;
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, size);
    SHA256_Final(digest.hash, &sha256);
    return digest;
}

Signature ByzantineConsensusEngine::sign_message(const MessageDigest& digest) {
    Signature sig;
    sig.signer_id = node_id_;
    
    // Simplified - would use actual ECDSA signing
    memcpy(sig.data, digest.hash, 32);
    memset(sig.data + 32, node_id_, 32);
    
    return sig;
}

bool ByzantineConsensusEngine::verify_signature(
    const Signature& sig, 
    const MessageDigest& digest, 
    uint32_t signer_id
) {
    // Simplified - would use actual ECDSA verification
    if (sig.signer_id != signer_id) return false;
    
    return memcmp(sig.data, digest.hash, 32) == 0;
}

void ByzantineConsensusEngine::consensus_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(message_queue_mutex_);
        message_cv_.wait_for(lock, milliseconds(10), [this] {
            return !incoming_messages_.empty() || !running_;
        });
        
        while (!incoming_messages_.empty()) {
            ConsensusMessage msg = incoming_messages_.front();
            incoming_messages_.pop();
            lock.unlock();
            
            // Process message
            auto start = steady_clock::now();
            process_message(msg);
            auto end = steady_clock::now();
            
            // Update metrics
            float latency = duration_cast<microseconds>(end - start).count() / 1000.0f;
            metrics_.average_latency_ms = 0.95f * metrics_.average_latency_ms + 0.05f * latency;
            
            lock.lock();
        }
        
        // Check view change timer
        if (!view_changing_) {
            auto now = steady_clock::now();
            if (now > state_.view_change_timer) {
                initiate_view_change();
            }
        }
    }
}

void ByzantineConsensusEngine::network_worker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(message_queue_mutex_);
        message_cv_.wait_for(lock, milliseconds(10), [this] {
            return !outgoing_messages_.empty() || !running_;
        });
        
        while (!outgoing_messages_.empty()) {
            ConsensusMessage msg = outgoing_messages_.front();
            outgoing_messages_.pop();
            lock.unlock();
            
            // In real implementation, would send over network
            // For now, just loop back for testing
            lock.lock();
            incoming_messages_.push(msg);
            lock.unlock();
            
            lock.lock();
        }
    }
}

void ByzantineConsensusEngine::monitor_worker() {
    while (running_) {
        std::this_thread::sleep_for(seconds(1));
        
        // Calculate throughput
        uint64_t current_consensus = metrics_.successful_consensus;
        static uint64_t last_consensus = 0;
        
        metrics_.throughput_ops_per_sec = 
            static_cast<float>(current_consensus - last_consensus);
        last_consensus = current_consensus;
        
        // Detect Byzantine nodes
        detect_byzantine_behavior();
        
        // Update node heartbeats
        uint64_t now_us = duration_cast<microseconds>(
            system_clock::now().time_since_epoch()).count();
        
        for (auto& node : nodes_) {
            if (now_us - node.last_heartbeat_us > 5000000) {  // 5 seconds
                node.reliability_score *= 0.9f;
            }
        }
    }
}

bool ByzantineConsensusEngine::is_valid_message(const ConsensusMessage& message) {
    // Basic validation
    if (message.sender_id >= swarm_size_) return false;
    if (message.view_number > state_.view_number + 10) return false;  // Too far ahead
    if (message.payload_size > MAX_MESSAGE_SIZE) return false;
    
    // Check timestamp (not too old, not in future)
    uint64_t now = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    
    if (message.timestamp_us > now + 1000000) return false;  // 1 second in future
    if (message.timestamp_us < now - 60000000) return false;  // 1 minute old
    
    return true;
}

void ByzantineConsensusEngine::record_byzantine_behavior(
    uint32_t node_id, 
    const std::string& reason
) {
    if (node_id < nodes_.size()) {
        nodes_[node_id].reliability_score *= 0.5f;
        if (nodes_[node_id].reliability_score < 0.1f) {
            nodes_[node_id].is_byzantine = true;
            metrics_.byzantine_detections++;
        }
    }
}

void ByzantineConsensusEngine::start_view_change_timer() {
    state_.view_change_timer = steady_clock::now() + milliseconds(VIEW_CHANGE_TIMEOUT_MS);
}

void ByzantineConsensusEngine::reset_view_change_timer() {
    state_.view_change_timer = steady_clock::now() + milliseconds(CONSENSUS_TIMEOUT_MS);
}

cudaError_t ByzantineConsensusEngine::initiate_view_change() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    view_changing_ = true;
    view_change_start_ = steady_clock::now();
    
    // Increment view number
    uint32_t new_view = state_.view_number + 1;
    
    // Create view change message
    ViewChangeMessage vc_msg;
    vc_msg.new_view_number = new_view;
    vc_msg.last_stable_checkpoint = last_checkpoint_seq_;
    vc_msg.sender_id = node_id_;
    
    // Include prepared certificates
    for (const auto& [seq, cert] : prepared_certs_) {
        if (seq > last_checkpoint_seq_) {
            vc_msg.prepared_proofs.push_back(cert);
        }
    }
    
    // Sign and broadcast
    MessageDigest vc_digest = compute_digest(
        reinterpret_cast<const uint8_t*>(&vc_msg), 
        sizeof(vc_msg) - sizeof(Signature)
    );
    vc_msg.signature = sign_message(vc_digest);
    
    // Store our own view change
    view_change_messages_.push_back(vc_msg);
    
    // Broadcast to all nodes
    ConsensusMessage msg;
    msg.type = MessageType::VIEW_CHANGE;
    msg.view_number = new_view;
    msg.sender_id = node_id_;
    memcpy(msg.payload, &vc_msg, sizeof(vc_msg));
    msg.payload_size = sizeof(vc_msg);
    
    metrics_.view_changes++;
    
    return broadcast_message(msg);
}

cudaError_t ByzantineConsensusEngine::detect_byzantine_behavior() {
    // Check for conflicting messages
    for (const auto& [seq, prepares] : message_log_.prepares) {
        std::unordered_map<uint32_t, MessageDigest> node_digests;
        
        for (const auto& prepare : prepares) {
            if (node_digests.count(prepare.sender_id) > 0) {
                // Node sent multiple prepares for same sequence
                if (!(node_digests[prepare.sender_id] == prepare.digest)) {
                    record_byzantine_behavior(prepare.sender_id, "Conflicting prepares");
                }
            }
            node_digests[prepare.sender_id] = prepare.digest;
        }
    }
    
    return cudaSuccess;
}

std::vector<uint32_t> ByzantineConsensusEngine::get_byzantine_nodes() const {
    std::vector<uint32_t> byzantine_nodes;
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_byzantine) {
            byzantine_nodes.push_back(i);
        }
    }
    
    return byzantine_nodes;
}

cudaError_t ByzantineConsensusEngine::create_checkpoint() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    uint64_t checkpoint_seq = state_.last_executed;
    
    // Get state digest from state machine
    static SwarmStateMachine state_machine;
    MessageDigest state_digest = state_machine.get_state_digest();
    
    // Create checkpoint
    Checkpoint ckpt;
    ckpt.sequence_number = checkpoint_seq;
    ckpt.state_digest = state_digest;
    
    // Sign checkpoint
    Signature sig = sign_message(state_digest);
    ckpt.signatures[node_id_] = sig;
    
    checkpoints_[checkpoint_seq] = ckpt;
    
    // Broadcast checkpoint message
    ConsensusMessage msg;
    msg.type = MessageType::CHECKPOINT;
    msg.sequence_number = checkpoint_seq;
    msg.sender_id = node_id_;
    msg.digest = state_digest;
    msg.signature = sig;
    
    return broadcast_message(msg);
}

cudaError_t ByzantineConsensusEngine::batch_verify_signatures(
    const Signature* signatures,
    const MessageDigest* digests,
    uint32_t count,
    bool* results
) {
    // Copy to GPU
    cudaMemcpyAsync(crypto_gpu_.d_signatures, signatures, 
                    count * sizeof(Signature), 
                    cudaMemcpyHostToDevice, crypto_gpu_.crypto_stream);
    
    cudaMemcpyAsync(crypto_gpu_.d_digests, digests, 
                    count * sizeof(MessageDigest), 
                    cudaMemcpyHostToDevice, crypto_gpu_.crypto_stream);
    
    // Launch batch verification kernel
    const uint32_t block_size = 256;
    const uint32_t grid_size = (count + block_size - 1) / block_size;
    
    // Would call actual GPU kernel here
    // crypto_kernels::batch_ecdsa_verify_kernel<<<grid_size, block_size, 0, crypto_gpu_.crypto_stream>>>(...)
    
    // Copy results back
    cudaMemcpyAsync(results, crypto_gpu_.d_verify_results, 
                    count * sizeof(bool), 
                    cudaMemcpyDeviceToHost, crypto_gpu_.crypto_stream);
    
    return cudaStreamSynchronize(crypto_gpu_.crypto_stream);
}

} // namespace ares::swarm