# Swarm Intelligence Module Documentation

## Module Overview

The Swarm module implements Byzantine fault-tolerant consensus and distributed task auction mechanisms for coordinating multiple ARES agents. It enables resilient decision-making in adversarial environments where up to 33% of agents may be compromised. The module uses advanced algorithms for leader election, task allocation, and collective intelligence.

## Functions & Classes

### `ByzantineConsensusEngine`
- **Purpose**: Achieves consensus among distributed agents despite Byzantine failures
- **Key Methods**:
  - `propose_value(value, priority)` - Propose value for consensus
  - `run_consensus_round()` - Execute PBFT consensus protocol
  - `verify_message(msg, signature)` - Cryptographic verification
  - `get_consensus_value()` - Retrieve agreed-upon value
  - `detect_byzantine_agent(agent_id)` - Identify malicious agents
- **Return Types**: ConsensusResult with value and confidence
- **External Dependencies**: Quantum-resilient signatures from Core

### `DistributedTaskAuction`
- **Purpose**: Allocates tasks optimally across swarm agents
- **Key Methods**:
  - `announce_task(task, requirements)` - Broadcast new task
  - `submit_bid(task_id, capability_score)` - Agent bidding
  - `run_auction()` - Execute sealed-bid auction
  - `assign_winners()` - Optimal task assignment
  - `monitor_progress()` - Track task completion
- **Return Types**: TaskAssignment structures
- **Side Effects**: Updates agent reputation scores

### `SwarmUnifiedInterface`
- **Purpose**: High-level API for swarm operations
- **Key Methods**:
  - `join_swarm(agent_credentials)` - Agent registration
  - `leave_swarm()` - Graceful departure
  - `broadcast_message(msg, reliability)` - Reliable multicast
  - `form_subswarm(criteria)` - Dynamic group formation
  - `execute_collective_decision()` - Democratic decisions

### Key Algorithms

#### PBFT (Practical Byzantine Fault Tolerance)
- **Phases**: Pre-prepare, Prepare, Commit
- **Fault Tolerance**: f = (n-1)/3 Byzantine agents
- **Message Complexity**: O(n²)
- **Latency**: 3 network round-trips

#### Task Auction Protocol
- **Type**: Sealed-bid second-price (Vickrey)
- **Properties**: Truthful bidding incentive
- **Optimization**: Maximizes swarm utility
- **Constraints**: Capability matching, locality

## Example Usage

```cpp
// Initialize swarm module
SwarmConfig config;
config.agent_id = "ARES-001";
config.byzantine_threshold = 0.33f;
config.consensus_timeout_ms = 1000;

SwarmUnifiedInterface swarm(config);

// Join swarm network
if (swarm.join_swarm(agent_credentials)) {
    std::cout << "Successfully joined swarm" << std::endl;
}

// Propose critical decision
ConsensusValue value;
value.type = DecisionType::ENGAGE_TARGET;
value.data = serialize_target_info(target);

auto result = swarm.propose_value(value, Priority::CRITICAL);
if (result.reached_consensus) {
    std::cout << "Consensus achieved with " 
              << result.agreement_percentage << "% agreement" << std::endl;
}

// Participate in task auction
swarm.on_task_announced([](const Task& task) {
    float my_capability = assess_capability(task);
    return submit_bid(task.id, my_capability);
});
```

## Integration Notes

- **Core Module**: Uses quantum-resilient crypto for message signing
- **CEW Module**: Coordinates jamming strategies across swarm
- **Neuromorphic**: Spike-based voting for rapid decisions
- **Orchestrator**: Receives swarm-optimized resource allocation
- **Countermeasures**: Collective defense strategies

## Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|------------|
| Max Agents | 1024 | Tested limit |
| Consensus Time | <100ms | LAN environment |
| Message Overhead | O(n²) | Per consensus round |
| Byzantine Tolerance | 33% | Theoretical maximum |
| Task Assignment | <10ms | 100 tasks, 50 agents |

## TODOs or Refactor Suggestions

1. **TODO**: Implement hierarchical consensus for >1000 agents
2. **TODO**: Add machine learning for Byzantine behavior prediction
3. **Enhancement**: Quantum-secure communication channels
4. **Optimization**: Implement gossip protocols for large swarms
5. **Research**: Explore homomorphic auction protocols
6. **Testing**: Chaos engineering test suite for Byzantine scenarios
7. **Feature**: Add support for dynamic reputation systems
8. **Security**: Implement zero-knowledge proofs for anonymous bidding