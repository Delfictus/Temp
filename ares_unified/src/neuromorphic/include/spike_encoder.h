#pragma once
#include "loihi2_spike_encoding.h"

namespace ares {
namespace neuromorphic {

using SpikeEncoder = Loihi2SpikeEncoder;

struct NetworkConfig {
    int num_neurons = 100000;
    int num_synapses = 1000000;
    float learning_rate = 0.01f;
};

} // namespace neuromorphic
} // namespace ares
