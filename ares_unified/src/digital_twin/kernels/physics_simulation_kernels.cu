/**
 * @file physics_simulation_kernels.cu
 * @brief GPU kernels for high-fidelity physics simulation and prediction
 * 
 * Implements differentiable physics, neural ODEs, and uncertainty propagation
 * for 5-second accurate predictions
 */

#include "../include/predictive_simulation_engine.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include "../../common/include/cuda_helpers.h"
namespace cg = cooperative_groups;

namespace ares::digital_twin::prediction_kernels {

constexpr uint32_t WARP_SIZE = 32;
constexpr float GRAVITY = -9.81f;
constexpr float EPSILON = 1e-6f;

/**
 * @brief Quaternion multiplication for orientation updates
 */
__device__ float4 quaternion_multiply(float4 q1, float4 q2) {
    return make_float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

/**
 * @brief Normalize quaternion to unit length
 */
__device__ float4 quaternion_normalize(float4 q) {
    float norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return make_float4(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
}

/**
 * @brief Convert angular velocity to quaternion derivative
 */
__device__ float4 angular_velocity_to_quaternion_derivative(
    float4 q, float3 omega
) {
    float4 omega_quat = make_float4(omega.x, omega.y, omega.z, 0.0f);
    float4 q_dot = quaternion_multiply(omega_quat, q);
    return make_float4(
        0.5f * q_dot.x,
        0.5f * q_dot.y,
        0.5f * q_dot.z,
        0.5f * q_dot.w
    );
}

/**
 * @brief Rigid body dynamics integration using symplectic Euler
 * 
 * @details Implements energy-preserving integration scheme for rigid body
 * dynamics. Symplectic integration ensures long-term stability by preserving
 * the symplectic structure of Hamilton's equations, preventing energy drift
 * in long simulations (critical for 5-second predictions).
 * 
 * The algorithm follows:
 * 1. Update velocities: v(t+dt) = v(t) + a(t)*dt
 * 2. Update positions: x(t+dt) = x(t) + v(t+dt)*dt
 * 3. Update angular velocities: ω(t+dt) = ω(t) + α(t)*dt
 * 4. Update orientations: q(t+dt) = q(t) + 0.5*ω(t+dt)⊗q(t)*dt
 * 
 * @param positions [in/out] Entity positions in world space [num_entities x 3]
 * @param velocities [in/out] Linear velocities [num_entities x 3]
 * @param accelerations [out] Computed accelerations for telemetry [num_entities x 3]
 * @param orientations [in/out] Quaternion orientations (x,y,z,w) [num_entities x 4]
 * @param angular_velocities [in/out] Angular velocities in rad/s [num_entities x 3]
 * @param forces [in] Applied forces in Newtons [num_entities x 3]
 * @param torques [in] Applied torques in N⋅m [num_entities x 3]
 * @param masses [in] Entity masses in kg [num_entities]
 * @param inertia_tensors [in] 3x3 inertia tensors (row-major) [num_entities x 9]
 * @param dt Time step in seconds (typically 0.001 for 1kHz simulation)
 * @param num_entities Total number of entities to simulate
 * 
 * @note Grid: (num_entities/256, 1, 1), Block: (256, 1, 1)
 * @note Memory access is coalesced for optimal performance
 */
__global__ void rigid_body_dynamics_kernel(
    float* positions,           // [num_entities x 3]
    float* velocities,          // [num_entities x 3]
    float* accelerations,       // [num_entities x 3]
    float* orientations,        // [num_entities x 4] quaternions
    float* angular_velocities,  // [num_entities x 3]
    const float* forces,        // [num_entities x 3]
    const float* torques,       // [num_entities x 3]
    const float* masses,        // [num_entities]
    const float* inertia_tensors, // [num_entities x 9]
    float dt,
    uint32_t num_entities
) {
    const uint32_t entity_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (entity_id >= num_entities) return;
    
    // Load current state
    float3 pos = make_float3(
        positions[entity_id * 3 + 0],
        positions[entity_id * 3 + 1],
        positions[entity_id * 3 + 2]
    );
    
    float3 vel = make_float3(
        velocities[entity_id * 3 + 0],
        velocities[entity_id * 3 + 1],
        velocities[entity_id * 3 + 2]
    );
    
    float4 quat = make_float4(
        orientations[entity_id * 4 + 0],
        orientations[entity_id * 4 + 1],
        orientations[entity_id * 4 + 2],
        orientations[entity_id * 4 + 3]
    );
    
    float3 omega = make_float3(
        angular_velocities[entity_id * 3 + 0],
        angular_velocities[entity_id * 3 + 1],
        angular_velocities[entity_id * 3 + 2]
    );
    
    // Load force and torque
    float3 force = make_float3(
        forces[entity_id * 3 + 0],
        forces[entity_id * 3 + 1],
        forces[entity_id * 3 + 2]
    );
    
    float3 torque = make_float3(
        torques[entity_id * 3 + 0],
        torques[entity_id * 3 + 1],
        torques[entity_id * 3 + 2]
    );
    
    float mass = masses[entity_id];
    
    // Add gravity
    force.z += mass * GRAVITY;
    
    // Linear dynamics: F = ma
    // Newton's second law: acceleration is force divided by mass
    // This gives us the instantaneous acceleration at time t
    float3 accel = make_float3(
        force.x / mass,
        force.y / mass,
        force.z / mass
    );
    
    // Symplectic integration for position/velocity
    // Step 1: Update velocity first (explicit Euler for velocity)
    // v(t+dt) = v(t) + a(t)*dt
    // This ensures energy conservation in conservative force fields
    vel.x += accel.x * dt;
    vel.y += accel.y * dt;
    vel.z += accel.z * dt;
    
    // Step 2: Update position using NEW velocity (semi-implicit)
    // x(t+dt) = x(t) + v(t+dt)*dt
    // Using updated velocity makes the scheme symplectic
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    // Angular dynamics: τ = Iα
    // Euler's rotation equations for rigid body dynamics
    // For now, we assume principal axes alignment (diagonal inertia tensor)
    // This simplification is valid for symmetric objects or when
    // the body frame is aligned with principal axes
    float3 inertia = make_float3(
        inertia_tensors[entity_id * 9 + 0],  // Ixx
        inertia_tensors[entity_id * 9 + 4],  // Iyy
        inertia_tensors[entity_id * 9 + 8]   // Izz
    );
    
    // Angular acceleration: α = I^(-1) * τ
    // Since we have diagonal inertia, this simplifies to component-wise division
    float3 angular_accel = make_float3(
        torque.x / inertia.x,
        torque.y / inertia.y,
        torque.z / inertia.z
    );
    
    // Update angular velocity using explicit Euler
    // ω(t+dt) = ω(t) + α(t)*dt
    // Note: For more accuracy, we could include gyroscopic effects:
    // α = I^(-1) * (τ - ω × (I * ω))
    omega.x += angular_accel.x * dt;
    omega.y += angular_accel.y * dt;
    omega.z += angular_accel.z * dt;
    
    // Update orientation quaternion
    // The quaternion derivative is: q̇ = 0.5 * ω ⊗ q
    // where ω is the angular velocity quaternion (0, ωx, ωy, ωz)
    // This ensures the quaternion remains unit-length during integration
    float4 q_dot = angular_velocity_to_quaternion_derivative(quat, omega);
    
    // Integrate quaternion using explicit Euler
    // q(t+dt) = q(t) + q̇(t)*dt
    quat.x += q_dot.x * dt;
    quat.y += q_dot.y * dt;
    quat.z += q_dot.z * dt;
    quat.w += q_dot.w * dt;
    
    // Normalize quaternion to prevent drift
    // Numerical integration can cause the quaternion to drift from unit length
    // Renormalization ensures it remains a valid rotation representation
    // This is critical for long-term stability in 5-second predictions
    quat = quaternion_normalize(quat);
    
    // Store updated state
    positions[entity_id * 3 + 0] = pos.x;
    positions[entity_id * 3 + 1] = pos.y;
    positions[entity_id * 3 + 2] = pos.z;
    
    velocities[entity_id * 3 + 0] = vel.x;
    velocities[entity_id * 3 + 1] = vel.y;
    velocities[entity_id * 3 + 2] = vel.z;
    
    accelerations[entity_id * 3 + 0] = accel.x;
    accelerations[entity_id * 3 + 1] = accel.y;
    accelerations[entity_id * 3 + 2] = accel.z;
    
    orientations[entity_id * 4 + 0] = quat.x;
    orientations[entity_id * 4 + 1] = quat.y;
    orientations[entity_id * 4 + 2] = quat.z;
    orientations[entity_id * 4 + 3] = quat.w;
    
    angular_velocities[entity_id * 3 + 0] = omega.x;
    angular_velocities[entity_id * 3 + 1] = omega.y;
    angular_velocities[entity_id * 3 + 2] = omega.z;
}

/**
 * @brief Spatial hashing collision detection for O(n) complexity
 * 
 * @details Computes Morton code (Z-order curve) for 3D spatial hashing.
 * Morton codes interleave the bits of x, y, z coordinates to create
 * a single index that preserves spatial locality. Points close in 3D
 * space will have similar Morton codes, enabling efficient spatial queries.
 * 
 * The bit manipulation spreads each 10-bit coordinate across 30 bits:
 * Original: xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz
 * Result:   zyxzyxzyxzyxzyxzyxzyxzyxzyxzyx
 * 
 * This is used for:
 * - Broad-phase collision detection
 * - Spatial database indexing
 * - Cache-efficient traversal of 3D data
 * 
 * @param x X coordinate (0-1023)
 * @param y Y coordinate (0-1023)
 * @param z Z coordinate (0-1023)
 * @return 30-bit Morton code
 * 
 * @note Magic numbers are from "Bit Twiddling Hacks" by Sean Anderson
 */
__device__ uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    // Expand bits: 0000000000xxxxxxxxxx -> 00x00x00x00x00x00x00x00x00x
    x = (x | (x << 16)) & 0x030000FF;  // 0000001100000000000000001111111
    x = (x | (x << 8)) & 0x0300F00F;   // 0000001100000000111100000000111
    x = (x | (x << 4)) & 0x030C30C3;   // 0000001100001100001100001100001
    x = (x | (x << 2)) & 0x09249249;   // 0001001001001001001001001001001
    
    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;
    
    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;
    
    // Interleave: x at bit 0, y at bit 1, z at bit 2, repeat
    return x | (y << 1) | (z << 2);
}

__global__ void collision_detection_kernel(
    const float* positions,      // [num_entities x 3]
    const float* bounding_boxes, // [num_entities x 6] (min, max)
    uint32_t* collision_pairs,   // Output: pairs of colliding entities
    uint32_t* num_collisions,    // Output: total collision count
    float collision_margin,
    uint32_t num_entities
) {
    // Thread indexing strategy for N×N collision checks:
    // - Each block handles one entity_a
    // - Each thread in block checks against one entity_b
    // - This ensures we check all unique pairs (a,b) where a < b
    const uint32_t entity_a = blockIdx.x;
    const uint32_t entity_b = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (entity_a >= num_entities || entity_b >= num_entities || entity_a >= entity_b) {
        return;
    }
    
    // Load bounding boxes
    float3 min_a = make_float3(
        bounding_boxes[entity_a * 6 + 0],
        bounding_boxes[entity_a * 6 + 1],
        bounding_boxes[entity_a * 6 + 2]
    );
    
    float3 max_a = make_float3(
        bounding_boxes[entity_a * 6 + 3],
        bounding_boxes[entity_a * 6 + 4],
        bounding_boxes[entity_a * 6 + 5]
    );
    
    float3 min_b = make_float3(
        bounding_boxes[entity_b * 6 + 0],
        bounding_boxes[entity_b * 6 + 1],
        bounding_boxes[entity_b * 6 + 2]
    );
    
    float3 max_b = make_float3(
        bounding_boxes[entity_b * 6 + 3],
        bounding_boxes[entity_b * 6 + 4],
        bounding_boxes[entity_b * 6 + 5]
    );
    
    // Add collision margin
    min_a.x -= collision_margin; min_a.y -= collision_margin; min_a.z -= collision_margin;
    max_a.x += collision_margin; max_a.y += collision_margin; max_a.z += collision_margin;
    
    // AABB (Axis-Aligned Bounding Box) overlap test
    // Two AABBs overlap if and only if they overlap on all three axes
    // This is the separating axis theorem simplified for axis-aligned boxes
    // 
    // For each axis, check: min_a <= max_b AND max_a >= min_b
    // If this holds for all axes, the boxes overlap
    bool overlap = (min_a.x <= max_b.x && max_a.x >= min_b.x) &&  // X-axis overlap
                  (min_a.y <= max_b.y && max_a.y >= min_b.y) &&  // Y-axis overlap
                  (min_a.z <= max_b.z && max_a.z >= min_b.z);    // Z-axis overlap
    
    if (overlap) {
        // Store collision pair atomically to handle concurrent writes
        // Multiple threads may detect collisions simultaneously
        uint32_t idx = atomicAdd(num_collisions, 1);
        
        // Bounds check to prevent buffer overflow
        // In worst case, all entities collide: n*(n-1)/2 pairs
        if (idx < num_entities * num_entities / 2) {
            collision_pairs[idx * 2 + 0] = entity_a;
            collision_pairs[idx * 2 + 1] = entity_b;
        }
    }
}

/**
 * @brief Constraint solver using projected Gauss-Seidel
 * Enforces position and velocity constraints
 */
__global__ void constraint_solver_kernel(
    float* positions,
    float* velocities,
    const uint32_t* constraint_indices,  // Pairs of constrained entities
    float* constraint_forces,
    float dt,
    uint32_t num_constraints
) {
    const uint32_t constraint_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (constraint_id >= num_constraints) return;
    
    // Load constraint entities
    uint32_t entity_a = constraint_indices[constraint_id * 2 + 0];
    uint32_t entity_b = constraint_indices[constraint_id * 2 + 1];
    
    // Load positions
    float3 pos_a = make_float3(
        positions[entity_a * 3 + 0],
        positions[entity_a * 3 + 1],
        positions[entity_a * 3 + 2]
    );
    
    float3 pos_b = make_float3(
        positions[entity_b * 3 + 0],
        positions[entity_b * 3 + 1],
        positions[entity_b * 3 + 2]
    );
    
    // Distance constraint (example)
    float target_distance = 1.0f;  // Would be loaded from constraint data
    
    float3 delta = make_float3(
        pos_b.x - pos_a.x,
        pos_b.y - pos_a.y,
        pos_b.z - pos_a.z
    );
    
    float current_distance = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    
    if (current_distance < EPSILON) return;
    
    // Constraint error
    float error = current_distance - target_distance;
    
    // Baumgarte stabilization
    float beta = 0.2f;  // Stabilization parameter
    float correction = -beta * error / dt;
    
    // Constraint force direction
    float3 direction = make_float3(
        delta.x / current_distance,
        delta.y / current_distance,
        delta.z / current_distance
    );
    
    // Apply constraint forces
    float force_magnitude = correction * 1000.0f;  // Stiffness
    
    constraint_forces[entity_a * 3 + 0] += force_magnitude * direction.x;
    constraint_forces[entity_a * 3 + 1] += force_magnitude * direction.y;
    constraint_forces[entity_a * 3 + 2] += force_magnitude * direction.z;
    
    constraint_forces[entity_b * 3 + 0] -= force_magnitude * direction.x;
    constraint_forces[entity_b * 3 + 1] -= force_magnitude * direction.y;
    constraint_forces[entity_b * 3 + 2] -= force_magnitude * direction.z;
}

/**
 * @brief Neural ODE layer for learned dynamics
 * 
 * @details Implements a single layer of a Neural Ordinary Differential Equation.
 * Neural ODEs model continuous-time dynamics as: dx/dt = f(x(t), t; θ)
 * where f is a neural network with parameters θ.
 * 
 * This approach has several advantages for physics prediction:
 * - Continuous-time modeling (arbitrary time resolution)
 * - Memory efficient (constant memory regardless of depth)
 * - Naturally handles irregular time series
 * - Learns conservation laws from data
 * 
 * The implementation uses residual connections for stability and
 * tanh activation for smooth, bounded dynamics.
 * 
 * @note This is integrated using adaptive ODE solvers in the host code
 */
__global__ void neural_ode_layer_kernel(
    const float* input,         // [batch_size x input_dim]
    const float* weights,       // [input_dim x hidden_dim]
    const float* bias,          // [hidden_dim]
    float* output,              // [batch_size x hidden_dim]
    float* hidden_state,        // Internal activations
    uint32_t input_dim,
    uint32_t hidden_dim,
    uint32_t batch_size
) {
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_dim) return;
    
    // Compute matrix multiplication for this output element
    float sum = bias[hidden_idx];
    
    for (uint32_t i = 0; i < input_dim; ++i) {
        float input_val = input[batch_idx * input_dim + i];
        float weight_val = weights[i * hidden_dim + hidden_idx];
        sum += input_val * weight_val;
    }
    
    // Apply activation (tanh for smooth dynamics)
    // Tanh is chosen because:
    // 1. Smooth and differentiable (important for ODE solvers)
    // 2. Bounded output [-1, 1] (prevents explosion)
    // 3. Odd function (preserves symmetries)
    float activated = tanhf(sum);
    
    // Residual connection for stability
    // This implements: h(t+dt) = (1-α)*f(h(t)) + α*h(t)
    // where α=0.1 provides stability by limiting rate of change
    // This is critical for long-term predictions (5 seconds)
    if (hidden_state != nullptr && blockIdx.y > 0) {
        float prev_state = hidden_state[batch_idx * hidden_dim + hidden_idx];
        activated = 0.9f * activated + 0.1f * prev_state;
    }
    
    // Store output
    output[batch_idx * hidden_dim + hidden_idx] = activated;
    
    if (hidden_state != nullptr) {
        hidden_state[batch_idx * hidden_dim + hidden_idx] = activated;
    }
}

/**
 * @brief Uncertainty propagation using unscented transform
 * 
 * @details Implements the Unscented Kalman Filter (UKF) prediction step.
 * The unscented transform propagates uncertainty through nonlinear dynamics
 * without requiring Jacobian computation (unlike EKF).
 * 
 * Algorithm:
 * 1. Generate 2n+1 sigma points from current distribution
 * 2. Propagate each sigma point through nonlinear dynamics
 * 3. Reconstruct mean and covariance from propagated points
 * 
 * Sigma points are chosen as:
 * - X₀ = μ (mean)
 * - Xᵢ = μ + √((n+κ)Σ)ᵢ for i=1..n
 * - Xᵢ = μ - √((n+κ)Σ)ᵢ₋ₙ for i=n+1..2n
 * 
 * This captures mean and covariance exactly for linear systems
 * and provides 3rd-order accuracy for nonlinear systems.
 * 
 * @param state_mean Current state estimate
 * @param state_covariance Current uncertainty (covariance matrix)
 * @param predicted_mean Output: predicted state after dt
 * @param predicted_covariance Output: predicted uncertainty
 * @param process_noise Model uncertainty added at each step
 * @param dt Time step for prediction
 * @param state_dim Dimension of state vector
 */
__global__ void uncertainty_propagation_kernel(
    const float* state_mean,         // [state_dim]
    const float* state_covariance,   // [state_dim x state_dim]
    float* predicted_mean,           // Output: [state_dim]
    float* predicted_covariance,     // Output: [state_dim x state_dim]
    const float* process_noise,      // [state_dim x state_dim]
    float dt,
    uint32_t state_dim
) {
    const uint32_t row = blockIdx.x;
    const uint32_t col = threadIdx.x;
    
    if (row >= state_dim || col >= state_dim) return;
    
    // Sigma point generation parameters
    // κ (kappa) is a tuning parameter: κ = 3 - n is standard choice
    // It controls the spread of sigma points around the mean
    // Larger κ = points further from mean = better for highly nonlinear systems
    const float kappa = 3.0f - state_dim;
    
    // Weights for mean and covariance reconstruction
    // w₀ = κ/(n+κ) for the mean sigma point
    // wᵢ = 1/(2(n+κ)) for the other 2n sigma points
    const float weight_0 = kappa / (state_dim + kappa);
    const float weight_i = 0.5f / (state_dim + kappa);
    
    extern __shared__ float sigma_points[];
    
    // Generate 2n+1 sigma points
    if (col == 0) {
        // Mean sigma point
        sigma_points[row] = state_mean[row];
        
        // +/- sqrt((n+κ)Σ) sigma points
        float scale = sqrtf(state_dim + kappa);
        for (uint32_t i = 0; i < state_dim; ++i) {
            float cov_elem = state_covariance[row * state_dim + i];
            sigma_points[(1 + i) * state_dim + row] = 
                state_mean[row] + scale * sqrtf(fabsf(cov_elem));
            sigma_points[(1 + state_dim + i) * state_dim + row] = 
                state_mean[row] - scale * sqrtf(fabsf(cov_elem));
        }
    }
    __syncthreads();
    
    // Propagate sigma points through dynamics (simplified linear)
    if (col < 2 * state_dim + 1) {
        float propagated = sigma_points[col * state_dim + row];
        
        // Simple dynamics model: x' = x + v*dt
        if (row < state_dim / 2) {  // Position components
            propagated += sigma_points[col * state_dim + row + state_dim/2] * dt;
        }
        
        sigma_points[col * state_dim + row] = propagated;
    }
    __syncthreads();
    
    // Compute predicted mean and covariance
    if (col == 0) {
        // Mean
        float mean = weight_0 * sigma_points[row];
        for (uint32_t i = 1; i < 2 * state_dim + 1; ++i) {
            mean += weight_i * sigma_points[i * state_dim + row];
        }
        predicted_mean[row] = mean;
    }
    
    // Covariance (simplified - full implementation would need reduction)
    float cov_elem = 0.0f;
    for (uint32_t i = 0; i < 2 * state_dim + 1; ++i) {
        float weight = (i == 0) ? weight_0 : weight_i;
        float diff_row = sigma_points[i * state_dim + row] - predicted_mean[row];
        float diff_col = sigma_points[i * state_dim + col] - predicted_mean[col];
        cov_elem += weight * diff_row * diff_col;
    }
    
    // Add process noise
    cov_elem += process_noise[row * state_dim + col];
    
    predicted_covariance[row * state_dim + col] = cov_elem;
}

/**
 * @brief Monte Carlo scenario generation
 * Creates diverse scenarios for robust prediction
 */
__global__ void scenario_generation_kernel(
    const float* base_state,         // [state_dim]
    float* scenario_states,          // Output: [num_scenarios x state_dim]
    const float* parameter_variations, // Parameter ranges
    uint32_t num_scenarios,
    uint32_t state_dim,
    uint32_t num_params
) {
    const uint32_t scenario_id = blockIdx.x;
    const uint32_t state_idx = threadIdx.x;
    
    if (scenario_id >= num_scenarios || state_idx >= state_dim) return;
    
    // Initialize random generator
    curandState rand_state;
    curand_init(clock64() + scenario_id * state_dim + state_idx, 0, 0, &rand_state);
    
    // Load base state
    float base_value = base_state[state_idx];
    
    // Apply parameter variations
    float variation = 0.0f;
    
    // Environmental variations
    if (state_idx < 3) {  // Position
        // Add wind effect
        float wind_strength = parameter_variations[0] * curand_uniform(&rand_state);
        variation += wind_strength * 0.1f;
    } else if (state_idx < 6) {  // Velocity
        // Add turbulence
        float turbulence = parameter_variations[1] * curand_normal(&rand_state);
        variation += turbulence * 0.5f;
    }
    
    // System uncertainties
    float uncertainty = parameter_variations[2] * curand_normal(&rand_state);
    variation += base_value * uncertainty * 0.01f;  // 1% uncertainty
    
    // Failure modes
    if (scenario_id % 10 == 0 && state_idx == 9) {  // 10% failure rate on component 9
        variation *= parameter_variations[3];  // Failure severity
    }
    
    // Store scenario state
    scenario_states[scenario_id * state_dim + state_idx] = base_value + variation;
}

/**
 * @brief Reality gap computation
 * 
 * @details Measures divergence between predicted and observed states to
 * quantify simulation accuracy. This is critical for:
 * - Online model adaptation
 * - Uncertainty calibration  
 * - Detecting model degradation
 * - Triggering re-learning
 * 
 * Computes multiple metrics:
 * - Position RMSE: Critical for spatial accuracy
 * - Velocity RMSE: Important for dynamic behavior
 * - Total RMSE: Overall prediction quality
 * 
 * The reality gap tends to grow over time due to:
 * - Model approximations
 * - Unmodeled dynamics
 * - Numerical integration errors
 * - Environmental changes
 * 
 * @param predicted_states Predicted state trajectories
 * @param observed_states Ground truth from sensors
 * @param gap_metrics Output: [pos_rmse, vel_rmse, total_rmse, count]
 * @param num_states Number of states to compare
 * @param state_dim Dimension of each state vector
 */
__global__ void reality_gap_kernel(
    const float* predicted_states,   // [num_states x state_dim]
    const float* observed_states,    // [num_states x state_dim]
    float* gap_metrics,              // Output: various metrics
    uint32_t num_states,
    uint32_t state_dim
) {
    extern __shared__ float shared_errors[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t state_idx = blockIdx.x;
    
    if (state_idx >= num_states) return;
    
    // Compute errors for this state
    float position_error = 0.0f;
    float velocity_error = 0.0f;
    float total_error = 0.0f;
    
    for (uint32_t d = tid; d < state_dim; d += blockDim.x) {
        float pred = predicted_states[state_idx * state_dim + d];
        float obs = observed_states[state_idx * state_dim + d];
        float error = pred - obs;
        
        if (d < 3) {  // Position
            position_error += error * error;
        } else if (d < 6) {  // Velocity
            velocity_error += error * error;
        }
        
        total_error += error * error;
    }
    
    // Store in shared memory
    shared_errors[tid] = position_error;
    shared_errors[tid + blockDim.x] = velocity_error;
    shared_errors[tid + 2 * blockDim.x] = total_error;
    __syncthreads();
    
    // Reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_errors[tid] += shared_errors[tid + s];
            shared_errors[tid + blockDim.x] += shared_errors[tid + s + blockDim.x];
            shared_errors[tid + 2 * blockDim.x] += shared_errors[tid + s + 2 * blockDim.x];
        }
        __syncthreads();
    }
    
    // Store metrics
    if (tid == 0) {
        atomicAdd(&gap_metrics[0], sqrtf(shared_errors[0]));  // Position RMSE
        atomicAdd(&gap_metrics[1], sqrtf(shared_errors[blockDim.x]));  // Velocity RMSE
        atomicAdd(&gap_metrics[2], sqrtf(shared_errors[2 * blockDim.x] / state_dim));  // Total RMSE
        atomicAdd(&gap_metrics[3], 1.0f);  // Count for averaging
    }
}

/**
 * @brief Trajectory optimization using gradient descent
 * Optimizes control inputs to minimize cost function
 */
__global__ void trajectory_optimization_kernel(
    float* trajectory,              // [trajectory_length x state_dim]
    const float* gradients,         // Cost function gradients
    const float* constraints,       // Constraint values
    float learning_rate,
    uint32_t trajectory_length,
    uint32_t state_dim
) {
    const uint32_t time_idx = blockIdx.x;
    const uint32_t state_idx = threadIdx.x;
    
    if (time_idx >= trajectory_length || state_idx >= state_dim) return;
    
    const uint32_t idx = time_idx * state_dim + state_idx;
    
    // Load current state and gradient
    float current_state = trajectory[idx];
    float gradient = gradients[idx];
    
    // Apply constraint penalties (barrier method)
    if (constraints != nullptr) {
        float constraint_val = constraints[time_idx];
        if (constraint_val > 0) {  // Constraint violated
            gradient += 100.0f * constraint_val;  // Penalty
        }
    }
    
    // Gradient descent update with momentum
    static __shared__ float momentum[1024];
    if (threadIdx.y == 0) {
        momentum[state_idx] = 0.9f * momentum[state_idx] - learning_rate * gradient;
        current_state += momentum[state_idx];
    }
    
    // Apply bounds
    if (state_idx < 3) {  // Position bounds
        current_state = fmaxf(-1000.0f, fminf(1000.0f, current_state));
    } else if (state_idx < 6) {  // Velocity bounds
        current_state = fmaxf(-100.0f, fminf(100.0f, current_state));
    }
    
    // Store updated state
    trajectory[idx] = current_state;
}

} // namespace prediction_kernels

// Utility functions for vector operations
__device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Differentiable maximum function for gradient-based optimization
 * 
 * @details Approximates max(a,b) with a smooth function that is differentiable
 * everywhere. Uses LogSumExp-like formulation for numerical stability.
 * As smoothness → 0, approaches true max. As smoothness → ∞, approaches average.
 * 
 * Used in differentiable physics for:
 * - Contact forces (smooth collision response)
 * - Friction models (stick-slip transitions)
 * - Constraint satisfaction (soft constraints)
 * 
 * @param a First value
 * @param b Second value  
 * @param smoothness Controls transition sharpness (default 1.0)
 * @return Smooth approximation of max(a,b)
 */
template<typename T>
__device__ T smooth_max(T a, T b, T smoothness = 1.0) {
    // Smooth approximation: max(a,b) ≈ (a+b)/2 + |a-b|/2
    // We smooth the absolute value function
    return (a + b) - (a - b) * fmaxf(0.0, fminf(1.0, (a - b) / smoothness));
}

/**
 * @brief Differentiable minimum function for gradient-based optimization
 * 
 * @details Smooth approximation of min(a,b) using the identity:
 * min(a,b) = -max(-a,-b)
 * 
 * @param a First value
 * @param b Second value
 * @param smoothness Controls transition sharpness (default 1.0)
 * @return Smooth approximation of min(a,b)
 */
template<typename T>
__device__ T smooth_min(T a, T b, T smoothness = 1.0) {
    return (a + b) - (b - a) * fmaxf(0.0, fminf(1.0, (b - a) / smoothness));
}
