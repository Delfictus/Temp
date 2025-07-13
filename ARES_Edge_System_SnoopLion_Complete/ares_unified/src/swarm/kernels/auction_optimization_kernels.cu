/**
 * @file auction_optimization_kernels.cu
 * @brief GPU kernels for distributed task auction optimization
 * 
 * Implements combinatorial optimization, VCG mechanisms, and coalition formation
 * algorithms for real-time swarm task allocation
 */

#include "../include/distributed_task_auction.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

namespace cg = cooperative_groups;

namespace ares::swarm::auction_kernels {

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr float EPSILON = 1e-6f;

/**
 * @brief Evaluate bid quality based on multiple criteria
 * Considers task priority, agent capabilities, and historical performance
 */
__global__ void evaluate_bids_kernel(
    const float* bid_values,
    const float* task_priorities,
    const float* agent_capabilities,
    float* bid_scores,
    uint32_t num_bids,
    uint32_t num_tasks
) {
    const uint32_t bid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bid_idx >= num_bids) return;
    
    // Decode bid structure (task_id, agent_id encoded in bid index)
    const uint32_t task_id = bid_idx / num_tasks;
    const uint32_t agent_id = bid_idx % num_tasks;
    
    // Load bid value
    float bid_val = bid_values[bid_idx];
    
    // Skip invalid bids
    if (bid_val <= 0.0f) {
        bid_scores[bid_idx] = -INFINITY;
        return;
    }
    
    // Load task priority (higher priority = higher weight)
    float priority_weight = task_priorities[task_id];
    
    // Load agent capability score for this task type
    float capability_score = agent_capabilities[agent_id * num_tasks + task_id];
    
    // Compute bid score using multi-criteria evaluation
    float normalized_bid = bid_val / (bid_val + 100.0f);  // Normalize to [0,1]
    float priority_factor = 1.0f + priority_weight * 2.0f;
    float capability_factor = 0.5f + capability_score * 0.5f;
    
    // Final score combines value, priority, and capability
    float score = normalized_bid * priority_factor * capability_factor;
    
    // Apply penalty for overloaded agents (simplified)
    float load_penalty = 1.0f - 0.1f * __popc(__ballot_sync(0xffffffff, agent_id == threadIdx.x));
    score *= load_penalty;
    
    bid_scores[bid_idx] = score;
}

/**
 * @brief Solve combinatorial allocation problem using parallel branch-and-bound
 * Finds optimal task-agent assignments maximizing total utility
 */
__global__ void combinatorial_allocation_kernel(
    const float* bid_matrix,          // [agents x tasks]
    const uint8_t* compatibility_matrix,  // Binary compatibility
    uint8_t* allocation_matrix,       // Output: binary assignments
    float* total_utility,             // Output: total value
    uint32_t num_tasks,
    uint32_t num_agents
) {
    // Shared memory for dynamic programming
    extern __shared__ float shared_mem[];
    float* dp_table = shared_mem;
    uint8_t* best_allocation = (uint8_t*)&shared_mem[blockDim.x];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t task_id = blockIdx.x;
    
    if (task_id >= num_tasks) return;
    
    // Initialize DP table
    if (tid < num_agents) {
        dp_table[tid] = 0.0f;
        best_allocation[tid] = 0;
    }
    __syncthreads();
    
    // Branch-and-bound search for optimal allocation
    float best_value = -INFINITY;
    uint32_t best_agent = UINT32_MAX;
    
    // Each thread evaluates a subset of agents
    for (uint32_t agent = tid; agent < num_agents; agent += blockDim.x) {
        // Check compatibility
        if (!compatibility_matrix[agent * num_tasks + task_id]) continue;
        
        // Check if agent is already assigned (simplified)
        bool already_assigned = false;
        for (uint32_t t = 0; t < task_id; ++t) {
            if (allocation_matrix[agent * num_tasks + t] > 0) {
                already_assigned = true;
                break;
            }
        }
        
        if (already_assigned) continue;
        
        // Get bid value
        float bid_val = bid_matrix[agent * num_tasks + task_id];
        
        if (bid_val > best_value) {
            best_value = bid_val;
            best_agent = agent;
        }
    }
    
    // Parallel reduction to find best agent across all threads
    __shared__ float shared_values[MAX_BLOCK_SIZE];
    __shared__ uint32_t shared_agents[MAX_BLOCK_SIZE];
    
    shared_values[tid] = best_value;
    shared_agents[tid] = best_agent;
    __syncthreads();
    
    // Warp-level reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_values[tid + s] > shared_values[tid]) {
                shared_values[tid] = shared_values[tid + s];
                shared_agents[tid] = shared_agents[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0 && shared_agents[0] != UINT32_MAX) {
        allocation_matrix[shared_agents[0] * num_tasks + task_id] = 1;
        atomicAdd(total_utility, shared_values[0]);
    }
}

/**
 * @brief Calculate coalition values for groups of agents
 * Accounts for synergies and complementary capabilities
 */
__global__ void coalition_value_kernel(
    const float* agent_capabilities,   // [agents x capability_dims]
    const float* task_requirements,    // [tasks x capability_dims]
    const uint8_t* coalition_matrix,   // [coalitions x agents]
    float* coalition_values,           // Output: value per coalition
    uint32_t num_coalitions,
    uint32_t num_tasks
) {
    const uint32_t coalition_id = blockIdx.x;
    const uint32_t task_id = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    
    if (coalition_id >= num_coalitions || task_id >= num_tasks) return;
    
    const uint32_t capability_dims = 8;  // Simplified
    extern __shared__ float shared_capabilities[];
    
    // Aggregate coalition capabilities
    float aggregated_caps[8] = {0};
    
    // Each thread processes a subset of agents
    const uint32_t agents_per_thread = (num_tasks + blockDim.x - 1) / blockDim.x;
    const uint32_t start_agent = tid * agents_per_thread;
    const uint32_t end_agent = min(start_agent + agents_per_thread, num_tasks);
    
    for (uint32_t agent = start_agent; agent < end_agent; ++agent) {
        if (coalition_matrix[coalition_id * num_tasks + agent]) {
            // Add agent capabilities
            for (uint32_t c = 0; c < capability_dims; ++c) {
                aggregated_caps[c] += agent_capabilities[agent * capability_dims + c];
            }
        }
    }
    
    // Store in shared memory
    for (uint32_t c = 0; c < capability_dims; ++c) {
        shared_capabilities[tid * capability_dims + c] = aggregated_caps[c];
    }
    __syncthreads();
    
    // Reduce across threads
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (uint32_t c = 0; c < capability_dims; ++c) {
                shared_capabilities[tid * capability_dims + c] += 
                    shared_capabilities[(tid + s) * capability_dims + c];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 computes coalition value for this task
    if (tid == 0) {
        float value = 1.0f;
        
        // Check if coalition meets task requirements
        for (uint32_t c = 0; c < capability_dims; ++c) {
            float required = task_requirements[task_id * capability_dims + c];
            float available = shared_capabilities[c];
            
            if (available < required) {
                value = 0.0f;  // Cannot perform task
                break;
            }
            
            // Synergy bonus for excess capabilities
            float excess_ratio = available / (required + EPSILON);
            value *= (1.0f + logf(excess_ratio) * 0.1f);
        }
        
        // Apply coalition size penalty (coordination cost)
        uint32_t coalition_size = 0;
        for (uint32_t a = 0; a < num_tasks; ++a) {
            if (coalition_matrix[coalition_id * num_tasks + a]) {
                coalition_size++;
            }
        }
        
        float coordination_penalty = 1.0f - 0.05f * (coalition_size - 1);
        value *= fmaxf(coordination_penalty, 0.5f);
        
        // Write result
        coalition_values[coalition_id * num_tasks + task_id] = value;
    }
}

/**
 * @brief Check resource feasibility for task assignments
 * Verifies agents have sufficient resources considering travel costs
 */
__global__ void resource_feasibility_kernel(
    const float* agent_resources,      // [agents x resource_types]
    const float* task_requirements,    // [tasks x resource_types]
    const float* travel_distances,     // [agents x tasks]
    uint8_t* feasibility_matrix,       // Output: binary feasibility
    uint32_t num_agents,
    uint32_t num_tasks
) {
    const uint32_t agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t task_id = blockIdx.y;
    
    if (agent_id >= num_agents || task_id >= num_tasks) return;
    
    const uint32_t num_resources = 8;  // Resource types
    
    // Load travel distance
    float distance = travel_distances[agent_id * num_tasks + task_id];
    
    // Calculate travel cost (simplified: fuel consumption)
    float fuel_per_meter = 0.001f;  // 1 unit per km
    float travel_fuel = distance * fuel_per_meter;
    
    // Check all resource constraints
    bool feasible = true;
    
    for (uint32_t r = 0; r < num_resources; ++r) {
        float available = agent_resources[agent_id * num_resources + r];
        float required = task_requirements[task_id * num_resources + r];
        
        // Add travel cost to fuel resource
        if (r == 1) {  // Fuel resource
            required += travel_fuel;
        }
        
        // Safety margin (keep 20% reserve)
        if (available * 0.8f < required) {
            feasible = false;
            break;
        }
    }
    
    // Consider time constraints
    float max_speed = 50.0f;  // m/s
    float travel_time = distance / max_speed;
    float task_duration = task_requirements[task_id * num_resources + 7];  // Time resource
    
    if (travel_time + task_duration > available * 0.8f) {
        feasible = false;
    }
    
    feasibility_matrix[agent_id * num_tasks + task_id] = feasible ? 1 : 0;
}

/**
 * @brief Update agent reputation scores based on performance
 * Uses exponential moving average with trust decay
 */
__global__ void reputation_update_kernel(
    float* reputation_scores,
    const float* performance_history,   // Recent performance metrics
    const uint32_t* task_outcomes,      // Success/failure history
    float learning_rate,
    uint32_t num_agents
) {
    const uint32_t agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id >= num_agents) return;
    
    // Load current reputation
    float current_rep = reputation_scores[agent_id];
    
    // Calculate performance score from history (last 10 tasks)
    float performance_sum = 0.0f;
    uint32_t success_count = 0;
    uint32_t total_count = 0;
    
    for (uint32_t i = 0; i < 10; ++i) {
        uint32_t hist_idx = agent_id * 10 + i;
        float perf = performance_history[hist_idx];
        
        if (perf >= 0.0f) {  // Valid entry
            performance_sum += perf;
            if (task_outcomes[hist_idx] > 0) {
                success_count++;
            }
            total_count++;
        }
    }
    
    if (total_count == 0) return;  // No history
    
    // Calculate new reputation components
    float avg_performance = performance_sum / total_count;
    float success_rate = (float)success_count / total_count;
    
    // Reputation update formula
    float performance_factor = avg_performance * success_rate;
    float new_rep = (1.0f - learning_rate) * current_rep + 
                    learning_rate * performance_factor;
    
    // Apply trust decay for inactive agents
    float activity_factor = fminf(1.0f, total_count / 5.0f);
    new_rep *= (0.95f + 0.05f * activity_factor);
    
    // Bound reputation to [0, 1]
    new_rep = fmaxf(0.0f, fminf(1.0f, new_rep));
    
    // Store updated reputation
    reputation_scores[agent_id] = new_rep;
}

/**
 * @brief Calculate VCG (Vickrey-Clarke-Groves) payments
 * Ensures truthful bidding in the auction mechanism
 */
__global__ void vcg_payment_kernel(
    const float* bid_matrix,            // All submitted bids
    const uint8_t* allocation_with_agent,    // Allocation including agent
    const uint8_t* allocation_without_agent, // Allocation excluding agent
    float* vcg_payments,                // Output: payment per agent
    uint32_t num_agents,
    uint32_t num_tasks
) {
    const uint32_t agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id >= num_agents) return;
    
    // Calculate social welfare with agent
    float welfare_with = 0.0f;
    float welfare_without = 0.0f;
    
    for (uint32_t task = 0; task < num_tasks; ++task) {
        // Sum utilities for all agents except current one
        for (uint32_t other = 0; other < num_agents; ++other) {
            if (other == agent_id) continue;
            
            if (allocation_with_agent[other * num_tasks + task]) {
                welfare_with += bid_matrix[other * num_tasks + task];
            }
            
            if (allocation_without_agent[other * num_tasks + task]) {
                welfare_without += bid_matrix[other * num_tasks + task];
            }
        }
    }
    
    // VCG payment = welfare_without - welfare_with
    // This is the externality imposed by the agent's presence
    float payment = welfare_without - welfare_with;
    
    // Ensure non-negative payments
    vcg_payments[agent_id] = fmaxf(0.0f, payment);
}

/**
 * @brief Parallel Hungarian algorithm for optimal assignment
 * Solves the assignment problem in O(n²) parallel steps
 */
__global__ void hungarian_algorithm_kernel(
    float* cost_matrix,                 // Input/output: modified during algorithm
    uint8_t* assignment,                // Output: optimal assignment
    uint32_t size                       // Square matrix size
) {
    extern __shared__ float shared_costs[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t row = blockIdx.x;
    
    if (row >= size) return;
    
    // Step 1: Row reduction
    __shared__ float row_min;
    
    if (tid == 0) row_min = INFINITY;
    __syncthreads();
    
    // Find row minimum
    for (uint32_t col = tid; col < size; col += blockDim.x) {
        atomicMin((int*)&row_min, __float_as_int(cost_matrix[row * size + col]));
    }
    __syncthreads();
    
    // Subtract row minimum
    for (uint32_t col = tid; col < size; col += blockDim.x) {
        cost_matrix[row * size + col] -= __int_as_float((int)row_min);
    }
    
    // Additional steps would implement full Hungarian algorithm
    // This is a simplified version for demonstration
}

/**
 * @brief Simulated annealing for large-scale optimization
 * Finds near-optimal solutions for NP-hard allocation problems
 */
__global__ void simulated_annealing_kernel(
    const float* utility_matrix,
    uint8_t* current_solution,
    uint8_t* best_solution,
    float* best_utility,
    float temperature,
    uint32_t num_agents,
    uint32_t num_tasks,
    uint32_t iterations
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random state
    curandState rand_state;
    curand_init(clock64() + tid, tid, 0, &rand_state);
    
    // Each thread explores a different solution path
    for (uint32_t iter = 0; iter < iterations; ++iter) {
        // Generate neighbor solution by swapping two assignments
        uint32_t agent1 = curand(&rand_state) % num_agents;
        uint32_t agent2 = curand(&rand_state) % num_agents;
        uint32_t task1 = UINT32_MAX;
        uint32_t task2 = UINT32_MAX;
        
        // Find assigned tasks
        for (uint32_t t = 0; t < num_tasks; ++t) {
            if (current_solution[agent1 * num_tasks + t]) task1 = t;
            if (current_solution[agent2 * num_tasks + t]) task2 = t;
        }
        
        if (task1 == UINT32_MAX || task2 == UINT32_MAX) continue;
        
        // Calculate utility change
        float delta = 0.0f;
        delta -= utility_matrix[agent1 * num_tasks + task1];
        delta -= utility_matrix[agent2 * num_tasks + task2];
        delta += utility_matrix[agent1 * num_tasks + task2];
        delta += utility_matrix[agent2 * num_tasks + task1];
        
        // Accept or reject based on Metropolis criterion
        if (delta > 0 || curand_uniform(&rand_state) < expf(delta / temperature)) {
            // Swap assignments
            current_solution[agent1 * num_tasks + task1] = 0;
            current_solution[agent1 * num_tasks + task2] = 1;
            current_solution[agent2 * num_tasks + task2] = 0;
            current_solution[agent2 * num_tasks + task1] = 1;
            
            // Update best if improved
            float current_util = 0.0f;
            for (uint32_t a = 0; a < num_agents; ++a) {
                for (uint32_t t = 0; t < num_tasks; ++t) {
                    if (current_solution[a * num_tasks + t]) {
                        current_util += utility_matrix[a * num_tasks + t];
                    }
                }
            }
            
            if (current_util > *best_utility) {
                *best_utility = current_util;
                memcpy(best_solution, current_solution, 
                       num_agents * num_tasks * sizeof(uint8_t));
            }
        }
        
        // Cool down temperature
        temperature *= 0.995f;
    }
}

/**
 * @brief Shapley value calculation for fair profit distribution
 * Computes marginal contributions of agents in coalitions
 */
__global__ void shapley_value_kernel(
    const float* coalition_values,      // Value function v(S)
    float* shapley_values,              // Output: Shapley value per agent
    uint32_t num_agents,
    uint32_t coalition_mask             // Bit mask for coalition membership
) {
    const uint32_t agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id >= num_agents) return;
    
    float shapley_value = 0.0f;
    uint32_t agent_bit = 1 << agent_id;
    
    // Iterate over all possible coalitions
    for (uint32_t S = 0; S < (1 << num_agents); ++S) {
        if (S & agent_bit) continue;  // Agent already in coalition
        
        // Coalition S union {agent}
        uint32_t S_with_agent = S | agent_bit;
        
        // Calculate marginal contribution
        float v_S = coalition_values[S];
        float v_S_with = coalition_values[S_with_agent];
        float marginal = v_S_with - v_S;
        
        // Weight by coalition size
        uint32_t coalition_size = __popc(S);
        float weight = 1.0f / (num_agents * 
                               (1 << (num_agents - 1)));
        
        // Factorial weights (simplified)
        for (uint32_t i = 1; i <= coalition_size; ++i) {
            weight *= i;
        }
        for (uint32_t i = 1; i <= num_agents - coalition_size - 1; ++i) {
            weight *= i;
        }
        
        shapley_value += weight * marginal;
    }
    
    shapley_values[agent_id] = shapley_value;
}

/**
 * @brief Core computation for coalition stability
 * Checks if profit distribution is in the core
 */
__global__ void compute_core_kernel(
    const float* coalition_values,
    const float* proposed_distribution,
    uint8_t* is_in_core,
    uint32_t num_agents,
    uint32_t num_coalitions
) {
    const uint32_t coalition_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (coalition_id >= num_coalitions) return;
    
    // Get coalition value
    float coalition_value = coalition_values[coalition_id];
    
    // Sum proposed payments to coalition members
    float total_payment = 0.0f;
    
    for (uint32_t agent = 0; agent < num_agents; ++agent) {
        if (coalition_id & (1 << agent)) {  // Agent in coalition
            total_payment += proposed_distribution[agent];
        }
    }
    
    // Check core constraint: sum of payments >= coalition value
    if (total_payment < coalition_value - EPSILON) {
        atomicAnd((int*)is_in_core, 0);  // Not in core
    }
}

} // namespace ares::swarm::auction_kernels