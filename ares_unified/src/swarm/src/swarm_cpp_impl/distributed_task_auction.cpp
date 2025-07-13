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
 * @file distributed_task_auction.cpp
 * @brief Implementation of market-based task allocation for autonomous swarms
 */

#include "../include/distributed_task_auction.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>

namespace ares::swarm {

using namespace std::chrono;

// ResourceManager implementation
class ResourceManager {
public:
    ResourceManager(uint32_t num_agents) : num_agents_(num_agents) {
        // Initialize resource tracking
        for (uint32_t i = 0; i < num_agents; ++i) {
            agent_resources_[i] = {
                {ResourceType::BATTERY, 100.0f},
                {ResourceType::FUEL, 100.0f},
                {ResourceType::AMMUNITION, 100.0f},
                {ResourceType::SENSORS, 100.0f},
                {ResourceType::COMMUNICATION, 100.0f},
                {ResourceType::COMPUTATION, 100.0f},
                {ResourceType::STORAGE, 100.0f},
                {ResourceType::TIME, 86400.0f}  // 24 hours in seconds
            };
            
            reserved_resources_[i] = {};
        }
    }
    
    bool check_availability(
        uint32_t agent_id, 
        const std::unordered_map<ResourceType, float>& required
    ) {
        if (agent_id >= num_agents_) return false;
        
        auto& available = agent_resources_[agent_id];
        auto& reserved = reserved_resources_[agent_id];
        
        for (const auto& [type, amount] : required) {
            float free = available[type] - reserved[type];
            if (free < amount) return false;
        }
        
        return true;
    }
    
    bool reserve(
        uint32_t agent_id,
        const std::unordered_map<ResourceType, float>& amounts
    ) {
        if (!check_availability(agent_id, amounts)) return false;
        
        auto& reserved = reserved_resources_[agent_id];
        for (const auto& [type, amount] : amounts) {
            reserved[type] += amount;
        }
        
        return true;
    }
    
    void release(
        uint32_t agent_id,
        const std::unordered_map<ResourceType, float>& amounts
    ) {
        if (agent_id >= num_agents_) return;
        
        auto& reserved = reserved_resources_[agent_id];
        for (const auto& [type, amount] : amounts) {
            reserved[type] = std::max(0.0f, reserved[type] - amount);
        }
    }
    
    void consume(
        uint32_t agent_id,
        const std::unordered_map<ResourceType, float>& amounts
    ) {
        if (agent_id >= num_agents_) return;
        
        auto& available = agent_resources_[agent_id];
        auto& reserved = reserved_resources_[agent_id];
        
        for (const auto& [type, amount] : amounts) {
            available[type] = std::max(0.0f, available[type] - amount);
            reserved[type] = std::max(0.0f, reserved[type] - amount);
        }
    }
    
private:
    uint32_t num_agents_;
    std::unordered_map<uint32_t, std::unordered_map<ResourceType, float>> agent_resources_;
    std::unordered_map<uint32_t, std::unordered_map<ResourceType, float>> reserved_resources_;
};

// BidEvaluator implementation
class BidEvaluator {
public:
    BidEvaluator() : total_bids_evaluated_(0) {}
    
    float evaluate(
        const TaskBid& bid,
        const Task& task,
        const AgentCapabilities& agent,
        float reputation
    ) {
        total_bids_evaluated_++;
        
        // Base score from bid value
        float base_score = bid.bid_value / (task.base_reward + 1.0f);
        
        // Capability match score
        float capability_score = calculate_capability_match(agent, task);
        
        // Success probability score
        float success_score = bid.completion_probability;
        
        // Time efficiency score
        float time_score = 1.0f;
        if (bid.estimated_duration_ms < task.duration_estimate_ms) {
            time_score = 1.0f + 0.2f * (1.0f - (float)bid.estimated_duration_ms / 
                                        task.duration_estimate_ms);
        }
        
        // Reputation factor
        float reputation_factor = 0.5f + 0.5f * reputation;
        
        // Coalition bonus
        float coalition_bonus = 1.0f;
        if (bid.coalition_members.size() > 1) {
            coalition_bonus = 1.0f + 0.1f * std::min(3u, (uint32_t)bid.coalition_members.size() - 1);
        }
        
        // Combined score
        float score = base_score * capability_score * success_score * 
                     time_score * reputation_factor * coalition_bonus;
        
        // Apply penalties
        if (bid.estimated_cost > bid.bid_value * 0.8f) {
            score *= 0.9f;  // High cost penalty
        }
        
        return score;
    }
    
    float calculate_capability_match(
        const AgentCapabilities& agent,
        const Task& task
    ) {
        uint32_t matched = 0;
        uint32_t required = task.required_capabilities.size();
        
        for (uint32_t cap : task.required_capabilities) {
            if (agent.capabilities.count(cap) > 0) {
                matched++;
            }
        }
        
        if (matched < required) return 0.0f;  // Missing required capabilities
        
        // Bonus for extra capabilities
        uint32_t extra = agent.capabilities.size() - required;
        return 1.0f + 0.05f * std::min(5u, extra);
    }
    
    uint64_t get_total_evaluated() const { return total_bids_evaluated_; }
    
private:
    std::atomic<uint64_t> total_bids_evaluated_;
};

// TaskOptimizationEngine implementation
TaskOptimizationEngine::TaskOptimizationEngine() 
    : d_working_memory_(nullptr)
    , memory_size_(0)
    , compute_stream_(nullptr)
    , cublas_handle_(nullptr)
    , cusolver_handle_(nullptr) {
}

TaskOptimizationEngine::~TaskOptimizationEngine() {
    if (d_working_memory_) cudaFree(d_working_memory_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
}

cudaError_t TaskOptimizationEngine::initialize(uint32_t max_tasks, uint32_t max_agents) {
    cudaError_t err;
    
    // Create CUDA stream
    err = cudaStreamCreate(&compute_stream_);
    if (err != cudaSuccess) return err;
    
    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    
    cublasSetStream(cublas_handle_, compute_stream_);
    
    // Create cuSOLVER handle
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle_);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) return cudaErrorUnknown;
    
    cusolverDnSetStream(cusolver_handle_, compute_stream_);
    
    // Allocate working memory
    memory_size_ = max_tasks * max_agents * sizeof(float) * 10;  // Space for matrices
    err = cudaMalloc(&d_working_memory_, memory_size_);
    
    return err;
}

cudaError_t TaskOptimizationEngine::solve_assignment_problem(
    const float* cost_matrix,
    uint8_t* assignment_matrix,
    uint32_t num_tasks,
    uint32_t num_agents
) {
    // This would implement the Hungarian algorithm or similar
    // For now, using a greedy approach as placeholder
    
    // Launch optimization kernel
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_tasks + block_size - 1) / block_size;
    
    // Call combinatorial allocation kernel
    // auction_kernels::combinatorial_allocation_kernel<<<...>>>(...)
    
    return cudaSuccess;
}

// DistributedTaskAuction implementation
DistributedTaskAuction::DistributedTaskAuction(uint32_t agent_id, uint32_t swarm_size)
    : agent_id_(agent_id)
    , swarm_size_(swarm_size)
    , running_(false) {
    
    memset(&opt_gpu_, 0, sizeof(opt_gpu_));
    memset(&market_stats_, 0, sizeof(market_stats_));
    
    // Initialize components
    resource_manager_ = std::make_unique<ResourceManager>(swarm_size);
    optimizer_ = std::make_unique<TaskOptimizationEngine>();
    bid_evaluator_ = std::make_unique<BidEvaluator>();
    
    // Initialize agent reputations
    for (uint32_t i = 0; i < swarm_size; ++i) {
        agent_reputations_[i] = 0.5f;  // Start with neutral reputation
    }
}

DistributedTaskAuction::~DistributedTaskAuction() {
    running_ = false;
    
    if (auction_thread_.joinable()) auction_thread_.join();
    if (optimization_thread_.joinable()) optimization_thread_.join();
    
    // Free GPU resources
    if (opt_gpu_.d_bid_matrix) cudaFree(opt_gpu_.d_bid_matrix);
    if (opt_gpu_.d_agent_capabilities) cudaFree(opt_gpu_.d_agent_capabilities);
    if (opt_gpu_.d_task_requirements) cudaFree(opt_gpu_.d_task_requirements);
    if (opt_gpu_.d_allocation_matrix) cudaFree(opt_gpu_.d_allocation_matrix);
    if (opt_gpu_.d_utility_values) cudaFree(opt_gpu_.d_utility_values);
    if (opt_gpu_.optimization_stream) cudaStreamDestroy(opt_gpu_.optimization_stream);
    if (opt_gpu_.cublas_handle) cublasDestroy(opt_gpu_.cublas_handle);
}

cudaError_t DistributedTaskAuction::initialize(
    const std::vector<AgentCapabilities>& agent_caps,
    const char* optimization_model
) {
    agent_capabilities_ = agent_caps;
    
    cudaError_t err;
    
    // Initialize optimizer
    err = optimizer_->initialize(1000, swarm_size_);  // Max 1000 tasks
    if (err != cudaSuccess) return err;
    
    // Create optimization stream
    err = cudaStreamCreate(&opt_gpu_.optimization_stream);
    if (err != cudaSuccess) return err;
    
    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&opt_gpu_.cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    
    // Allocate GPU memory for optimization
    const size_t max_tasks = 1000;
    const size_t matrix_size = max_tasks * swarm_size_;
    
    err = cudaMalloc(&opt_gpu_.d_bid_matrix, matrix_size * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&opt_gpu_.d_agent_capabilities, swarm_size_ * 8 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&opt_gpu_.d_task_requirements, max_tasks * 8 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&opt_gpu_.d_allocation_matrix, matrix_size * sizeof(uint8_t));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&opt_gpu_.d_utility_values, matrix_size * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // Start worker threads
    running_ = true;
    auction_thread_ = std::thread(&DistributedTaskAuction::auction_worker, this);
    optimization_thread_ = std::thread(&DistributedTaskAuction::optimization_worker, this);
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::announce_task(const Task& task) {
    available_tasks_[task.task_id] = task;
    
    // Update market statistics
    market_stats_.total_tasks_auctioned++;
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::submit_bid(const TaskBid& bid) {
    // Validate bid
    if (available_tasks_.count(bid.task_id) == 0) {
        return cudaErrorInvalidValue;
    }
    
    // Check resource feasibility
    const Task& task = available_tasks_[bid.task_id];
    const AgentCapabilities& agent = agent_capabilities_[bid.bidder_id];
    
    if (!check_resource_feasibility(agent, task)) {
        return cudaErrorInvalidValue;
    }
    
    // Store bid
    submitted_bids_[bid.bid_id] = bid;
    
    // Update statistics
    if (agent_performance_.count(bid.bidder_id) == 0) {
        agent_performance_[bid.bidder_id] = AgentPerformance();
    }
    agent_performance_[bid.bidder_id].total_bids++;
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::start_auction_round(
    const AuctionRound& round_params
) {
    current_round_ = std::make_unique<AuctionRound>(round_params);
    current_round_->phase = AuctionRound::Phase::ANNOUNCEMENT;
    
    // Clear previous round data
    submitted_bids_.clear();
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::finalize_auction_round() {
    if (!current_round_) return cudaErrorInvalidValue;
    
    // Collect all bids for current round tasks
    std::vector<TaskBid> round_bids;
    for (const auto& [bid_id, bid] : submitted_bids_) {
        bool task_in_round = false;
        for (const auto& task : current_round_->tasks_offered) {
            if (task.task_id == bid.task_id) {
                task_in_round = true;
                break;
            }
        }
        
        if (task_in_round) {
            round_bids.push_back(bid);
        }
    }
    
    // Solve winner determination problem
    std::vector<TaskAllocation> new_allocations;
    cudaError_t err = solve_winner_determination(round_bids, new_allocations);
    if (err != cudaSuccess) return err;
    
    // Apply VCG mechanism for truthful bidding
    err = apply_vcg_mechanism(new_allocations);
    if (err != cudaSuccess) return err;
    
    // Store allocations
    for (const auto& alloc : new_allocations) {
        allocations_[alloc.task_id] = alloc;
        
        // Update agent performance
        for (uint32_t agent_id : alloc.assigned_agents) {
            agent_performance_[agent_id].successful_bids++;
        }
        
        // Reserve resources
        const Task& task = available_tasks_[alloc.task_id];
        for (uint32_t agent_id : alloc.assigned_agents) {
            std::unordered_map<ResourceType, float> required;
            for (const auto& req : task.resources) {
                required[req.type] = req.amount / alloc.assigned_agents.size();
            }
            resource_manager_->reserve(agent_id, required);
        }
    }
    
    // Update market statistics
    float total_competition = (float)round_bids.size() / 
                             current_round_->tasks_offered.size();
    market_stats_.average_competition_ratio = 
        0.9f * market_stats_.average_competition_ratio + 0.1f * total_competition;
    
    current_round_->phase = AuctionRound::Phase::COMPLETED;
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::solve_winner_determination(
    const std::vector<TaskBid>& bids,
    std::vector<TaskAllocation>& allocations
) {
    if (bids.empty()) return cudaSuccess;
    
    // Prepare bid matrix on GPU
    const uint32_t num_tasks = available_tasks_.size();
    float* h_bid_matrix = new float[swarm_size_ * num_tasks];
    memset(h_bid_matrix, 0, swarm_size_ * num_tasks * sizeof(float));
    
    // Fill bid matrix
    for (const auto& bid : bids) {
        uint32_t task_idx = 0;
        for (const auto& [task_id, task] : available_tasks_) {
            if (task_id == bid.task_id) break;
            task_idx++;
        }
        
        if (task_idx < num_tasks) {
            // Evaluate bid quality
            float reputation = agent_reputations_[bid.bidder_id];
            float score = bid_evaluator_->evaluate(
                bid, available_tasks_[bid.task_id], 
                agent_capabilities_[bid.bidder_id], reputation
            );
            
            h_bid_matrix[bid.bidder_id * num_tasks + task_idx] = score;
        }
    }
    
    // Copy to GPU
    cudaMemcpyAsync(opt_gpu_.d_bid_matrix, h_bid_matrix, 
                    swarm_size_ * num_tasks * sizeof(float),
                    cudaMemcpyHostToDevice, opt_gpu_.optimization_stream);
    
    // Clear allocation matrix
    cudaMemsetAsync(opt_gpu_.d_allocation_matrix, 0, 
                    swarm_size_ * num_tasks * sizeof(uint8_t),
                    opt_gpu_.optimization_stream);
    
    // Launch optimization kernel
    const uint32_t block_size = 256;
    const uint32_t grid_size = num_tasks;
    
    float total_utility = 0.0f;
    cudaMemcpyAsync(opt_gpu_.d_utility_values, &total_utility, sizeof(float),
                    cudaMemcpyHostToDevice, opt_gpu_.optimization_stream);
    
    // Would call actual optimization kernel here
    // auction_kernels::combinatorial_allocation_kernel<<<...>>>(...)
    
    // Copy results back
    uint8_t* h_allocation = new uint8_t[swarm_size_ * num_tasks];
    cudaMemcpyAsync(h_allocation, opt_gpu_.d_allocation_matrix,
                    swarm_size_ * num_tasks * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost, opt_gpu_.optimization_stream);
    
    cudaStreamSynchronize(opt_gpu_.optimization_stream);
    
    // Convert to allocations
    uint32_t task_idx = 0;
    for (const auto& [task_id, task] : available_tasks_) {
        std::vector<uint32_t> assigned_agents;
        float winning_bid = 0.0f;
        
        for (uint32_t agent = 0; agent < swarm_size_; ++agent) {
            if (h_allocation[agent * num_tasks + task_idx]) {
                assigned_agents.push_back(agent);
                winning_bid = std::max(winning_bid, 
                                     h_bid_matrix[agent * num_tasks + task_idx]);
            }
        }
        
        if (!assigned_agents.empty()) {
            TaskAllocation alloc;
            alloc.task_id = task_id;
            alloc.assigned_agents = assigned_agents;
            alloc.winning_bid = winning_bid;
            alloc.expected_utility = winning_bid * task.base_reward;
            alloc.allocation_time_us = duration_cast<microseconds>(
                system_clock::now().time_since_epoch()).count();
            
            allocations.push_back(alloc);
        }
        
        task_idx++;
    }
    
    delete[] h_bid_matrix;
    delete[] h_allocation;
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::apply_vcg_mechanism(
    std::vector<TaskAllocation>& allocations
) {
    // VCG mechanism ensures truthful bidding
    // Payment = social welfare without agent - social welfare with agent
    
    // This is a simplified implementation
    // Full VCG would require re-solving allocation for each agent removed
    
    for (auto& alloc : allocations) {
        // Simple second-price auction approximation
        alloc.winning_bid *= 0.9f;  // Pay 90% of bid
    }
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::propose_coalition(
    const std::vector<uint32_t>& members,
    const std::vector<uint64_t>& target_tasks
) {
    Coalition coalition;
    coalition.coalition_id = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    coalition.members = members;
    coalition.formation_time_us = coalition.coalition_id;
    
    // Calculate fair profit shares (equal for now)
    float share = 1.0f / members.size();
    for (uint32_t member : members) {
        coalition.profit_shares[member] = share;
    }
    
    // Check coalition stability
    coalition.stability_score = is_coalition_stable(coalition) ? 1.0f : 0.0f;
    
    coalitions_[coalition.coalition_id] = coalition;
    
    return cudaSuccess;
}

cudaError_t DistributedTaskAuction::report_task_completion(
    uint64_t task_id,
    bool success,
    float actual_duration_ms
) {
    if (allocations_.count(task_id) == 0) {
        return cudaErrorInvalidValue;
    }
    
    const TaskAllocation& alloc = allocations_[task_id];
    const Task& task = available_tasks_[task_id];
    
    // Update agent performance
    for (uint32_t agent_id : alloc.assigned_agents) {
        float performance = success ? 1.0f : 0.0f;
        
        // Time performance factor
        if (success && actual_duration_ms > 0) {
            float time_ratio = task.duration_estimate_ms / actual_duration_ms;
            performance *= std::min(2.0f, std::max(0.5f, time_ratio));
        }
        
        // Update reputation
        update_reputation(agent_id, success, performance);
        
        // Release resources
        std::unordered_map<ResourceType, float> consumed;
        for (const auto& req : task.resources) {
            consumed[req.type] = req.amount / alloc.assigned_agents.size();
        }
        resource_manager_->consume(agent_id, consumed);
    }
    
    // Update market statistics
    if (success) {
        market_stats_.total_tasks_completed++;
    }
    
    float completion_rate = (float)market_stats_.total_tasks_completed / 
                           market_stats_.total_tasks_auctioned;
    market_stats_.task_completion_rate = completion_rate;
    
    // Remove from available tasks
    available_tasks_.erase(task_id);
    
    return cudaSuccess;
}

bool DistributedTaskAuction::check_resource_feasibility(
    const AgentCapabilities& agent,
    const Task& task
) const {
    // Check capability requirements
    for (uint32_t req_cap : task.required_capabilities) {
        if (agent.capabilities.count(req_cap) == 0) {
            return false;
        }
    }
    
    // Check resource requirements
    for (const auto& req : task.resources) {
        if (agent.current_resources.count(req.type) == 0) {
            return false;
        }
        
        float available = agent.current_resources.at(req.type);
        if (available < req.amount) {
            return false;
        }
    }
    
    // Check distance/time feasibility
    float dx = task.location[0] - agent.position[0];
    float dy = task.location[1] - agent.position[1];
    float dz = task.location[2] - agent.position[2];
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);
    
    float travel_time = distance / agent.speed_mps;
    float total_time = travel_time + task.duration_estimate_ms / 1000.0f;
    
    uint64_t current_time = duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
    
    if (current_time + total_time * 1e6 > task.deadline_us) {
        return false;  // Cannot meet deadline
    }
    
    return true;
}

float DistributedTaskAuction::estimate_resource_consumption(
    const AgentCapabilities& agent,
    const Task& task
) const {
    float total_cost = 0.0f;
    
    // Task resource requirements
    for (const auto& req : task.resources) {
        total_cost += req.amount * req.rate_per_second * 
                     (task.duration_estimate_ms / 1000.0f);
    }
    
    // Travel costs
    float dx = task.location[0] - agent.position[0];
    float dy = task.location[1] - agent.position[1];
    float dz = task.location[2] - agent.position[2];
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);
    
    float fuel_consumption_rate = 0.1f;  // Units per meter
    total_cost += distance * fuel_consumption_rate;
    
    return total_cost;
}

void DistributedTaskAuction::update_reputation(
    uint32_t agent_id, 
    bool success, 
    float performance
) {
    float& reputation = agent_reputations_[agent_id];
    
    // Update using exponential moving average
    float update = success ? performance : -0.5f;
    reputation = 0.9f * reputation + 0.1f * update;
    
    // Bound to [0, 1]
    reputation = std::max(0.0f, std::min(1.0f, reputation));
}

bool DistributedTaskAuction::is_coalition_stable(const Coalition& coalition) const {
    // Check if coalition is stable (no member wants to leave)
    float coalition_value = calculate_coalition_value(coalition, 
                                                    std::vector<Task>());
    
    for (uint32_t member : coalition.members) {
        float individual_value = agent_reputations_.at(member);
        float coalition_share = coalition.profit_shares.at(member) * coalition_value;
        
        if (coalition_share < individual_value * 0.9f) {
            return false;  // Member would be better off alone
        }
    }
    
    return true;
}

float DistributedTaskAuction::calculate_coalition_value(
    const Coalition& coalition,
    const std::vector<Task>& tasks
) const {
    // Aggregate capabilities
    std::set<uint32_t> combined_capabilities;
    float combined_resources = 0.0f;
    
    for (uint32_t member : coalition.members) {
        const AgentCapabilities& agent = agent_capabilities_[member];
        combined_capabilities.insert(agent.capabilities.begin(), 
                                   agent.capabilities.end());
        
        for (const auto& [type, amount] : agent.current_resources) {
            combined_resources += amount;
        }
    }
    
    // Value based on capability diversity and resource pool
    float diversity_bonus = 1.0f + 0.1f * combined_capabilities.size();
    float size_penalty = 1.0f - 0.05f * (coalition.members.size() - 1);
    
    return combined_resources * diversity_bonus * size_penalty;
}

void DistributedTaskAuction::auction_worker() {
    while (running_) {
        std::this_thread::sleep_for(milliseconds(100));
        
        // Process auction phases
        if (current_round_) {
            auto now = steady_clock::now();
            auto current_time_us = duration_cast<microseconds>(
                now.time_since_epoch()).count();
            
            switch (current_round_->phase) {
                case AuctionRound::Phase::ANNOUNCEMENT:
                    if (current_time_us > current_round_->bid_deadline_us) {
                        current_round_->phase = AuctionRound::Phase::BIDDING;
                    }
                    break;
                    
                case AuctionRound::Phase::BIDDING:
                    if (current_time_us > current_round_->reveal_deadline_us) {
                        current_round_->phase = AuctionRound::Phase::REVEAL;
                    }
                    break;
                    
                case AuctionRound::Phase::REVEAL:
                    // Verify bid commitments
                    verify_bid_commitments(current_round_->submitted_bids);
                    current_round_->phase = AuctionRound::Phase::ALLOCATION;
                    break;
                    
                case AuctionRound::Phase::ALLOCATION:
                    // Allocation is done in finalize_auction_round
                    break;
                    
                default:
                    break;
            }
        }
    }
}

void DistributedTaskAuction::optimization_worker() {
    while (running_) {
        std::this_thread::sleep_for(milliseconds(500));
        
        // Periodic optimization tasks
        // Update market efficiency metrics
        // Recompute coalition values
        // Detect and handle market failures
    }
}

std::vector<TaskAllocation> DistributedTaskAuction::get_current_allocations() const {
    std::vector<TaskAllocation> result;
    for (const auto& [task_id, alloc] : allocations_) {
        result.push_back(alloc);
    }
    return result;
}

std::vector<uint64_t> DistributedTaskAuction::get_agent_tasks(uint32_t agent_id) const {
    std::vector<uint64_t> tasks;
    
    for (const auto& [task_id, alloc] : allocations_) {
        for (uint32_t assigned : alloc.assigned_agents) {
            if (assigned == agent_id) {
                tasks.push_back(task_id);
                break;
            }
        }
    }
    
    return tasks;
}

MarketStatistics DistributedTaskAuction::get_market_statistics() const {
    return market_stats_;
}

float DistributedTaskAuction::get_agent_reputation(uint32_t agent_id) const {
    auto it = agent_reputations_.find(agent_id);
    return (it != agent_reputations_.end()) ? it->second : 0.5f;
}

} // namespace ares::swarm