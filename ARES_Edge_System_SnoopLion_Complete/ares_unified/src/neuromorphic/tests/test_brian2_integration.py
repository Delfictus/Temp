"""
Test suite for Brian2 ARES integration
"""

import pytest
import numpy as np
from brian2 import *
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from brian2_ares_integration import (
    ThreatDetectionNetwork,
    SwarmCoordinationNetwork,
    ChaosDetectionNetwork,
    ARESNeuronModels,
    NeuronParameters,
    Brian2LoihiBridge
)


class TestNeuronModels:
    """Test neuron model definitions"""
    
    def test_lif_equations(self):
        """Test LIF neuron equation generation"""
        params = NeuronParameters()
        eqs = ARESNeuronModels.get_lif_equations(params)
        assert 'dv/dt' in eqs
        assert 'tau' in eqs
        assert 'unless refractory' in eqs
    
    def test_adex_equations(self):
        """Test AdEx neuron equation generation"""
        params = NeuronParameters()
        eqs = ARESNeuronModels.get_adex_equations(params)
        assert 'dv/dt' in eqs
        assert 'dw/dt' in eqs
        assert 'exp' in eqs
    
    def test_em_sensor_equations(self):
        """Test EM sensor neuron equations"""
        params = NeuronParameters()
        eqs = ARESNeuronModels.get_em_sensor_equations(params)
        assert 'I_em' in eqs
        assert 'em_amplitude' in eqs
        assert 'preferred_freq' in eqs
    
    def test_chaos_detector_equations(self):
        """Test chaos detector equations"""
        params = NeuronParameters()
        eqs = ARESNeuronModels.get_chaos_detector_equations(params)
        assert 'dx/dt' in eqs
        assert 'dy/dt' in eqs
        assert 'I_chaos' in eqs


class TestThreatDetectionNetwork:
    """Test threat detection neural network"""
    
    @pytest.fixture
    def small_network(self):
        """Create a small test network"""
        return ThreatDetectionNetwork(input_size=10, hidden_size=5, output_size=3)
    
    def test_network_creation(self, small_network):
        """Test network initialization"""
        assert small_network.input_size == 10
        assert small_network.hidden_size == 5
        assert small_network.output_size == 3
        assert len(small_network.input_layer) == 10
        assert len(small_network.hidden_layer) == 5
        assert len(small_network.output_layer) == 3
    
    def test_process_em_spectrum(self, small_network):
        """Test EM spectrum processing"""
        test_spectrum = np.random.random(10) * 10
        result = small_network.process_em_spectrum(test_spectrum, duration=10*ms)
        
        assert 'threat_scores' in result
        assert 'spike_count' in result
        assert 'active_neurons' in result
        assert len(result['threat_scores']) == 3
        assert np.all(result['threat_scores'] >= 0)
    
    def test_invalid_spectrum_size(self, small_network):
        """Test error handling for invalid input size"""
        wrong_spectrum = np.random.random(20)
        with pytest.raises(ValueError):
            small_network.process_em_spectrum(wrong_spectrum)
    
    def test_training(self, small_network):
        """Test network training"""
        # Create simple training data
        training_data = []
        for i in range(30):
            spectrum = np.random.random(10) * 10
            label = i % 3  # 3 classes
            training_data.append((spectrum, label))
        
        # Train for 1 epoch (fast test)
        small_network.train(training_data, epochs=1, duration_per_sample=10*ms)
        
        # Check that network still functions after training
        test_spectrum = np.random.random(10) * 10
        result = small_network.process_em_spectrum(test_spectrum)
        assert result is not None


class TestSwarmCoordinationNetwork:
    """Test swarm coordination network"""
    
    @pytest.fixture
    def small_swarm(self):
        """Create a small swarm network"""
        return SwarmCoordinationNetwork(num_agents=5, neurons_per_agent=3)
    
    def test_swarm_creation(self, small_swarm):
        """Test swarm network initialization"""
        assert small_swarm.num_agents == 5
        assert small_swarm.neurons_per_agent == 3
        assert len(small_swarm.agent_groups) == 5
        for group in small_swarm.agent_groups:
            assert len(group) == 3
    
    def test_coordinate_decision(self, small_swarm):
        """Test swarm decision coordination"""
        agent_inputs = {
            0: np.array([1.0]),
            2: np.array([2.0]),
            4: np.array([1.5])
        }
        
        decisions = small_swarm.coordinate_decision(agent_inputs, duration=50*ms)
        
        assert len(decisions) == 5
        assert np.all(np.isfinite(decisions))
    
    def test_empty_inputs(self, small_swarm):
        """Test with no agent inputs"""
        decisions = small_swarm.coordinate_decision({}, duration=10*ms)
        assert len(decisions) == 5


class TestChaosDetectionNetwork:
    """Test chaos detection network"""
    
    @pytest.fixture
    def chaos_network(self):
        """Create chaos detection network"""
        return ChaosDetectionNetwork(num_detectors=10)
    
    def test_chaos_network_creation(self, chaos_network):
        """Test chaos network initialization"""
        assert chaos_network.num_detectors == 10
        assert len(chaos_network.chaos_detectors) == 10
    
    def test_detect_periodic_signal(self, chaos_network):
        """Test detection on periodic signal"""
        t = np.linspace(0, 10, 100)
        periodic_signal = np.sin(t)
        
        result = chaos_network.detect_chaos(periodic_signal, duration=50*ms)
        
        assert 'chaos_scores' in result
        assert 'mean_chaos' in result
        assert 'chaos_detected' in result
        assert len(result['chaos_scores']) == 10
    
    def test_detect_chaotic_signal(self, chaos_network):
        """Test detection on chaotic signal"""
        t = np.linspace(0, 10, 100)
        # Lorenz-like chaotic signal
        chaotic_signal = np.sin(t) + 0.5 * np.sin(2.1 * t) + 0.3 * np.random.randn(len(t))
        
        result = chaos_network.detect_chaos(chaotic_signal, duration=50*ms)
        
        assert result['max_chaos'] > 0  # Should detect some chaos


class TestBrian2LoihiBridge:
    """Test Brian2 to Loihi conversion bridge"""
    
    @pytest.fixture
    def bridge(self):
        """Create bridge instance"""
        return Brian2LoihiBridge()
    
    def test_register_model(self, bridge):
        """Test model registration"""
        # Create a simple test network
        test_network = Network()
        neurons = NeuronGroup(10, 'dv/dt = -v/(10*ms) : volt')
        test_network.add(neurons)
        
        bridge.register_model("test_model", test_network)
        assert "test_model" in bridge.brian_models
    
    def test_convert_to_loihi(self, bridge):
        """Test Brian2 to Loihi conversion"""
        # Create and register a network
        neurons = NeuronGroup(10, 'dv/dt = -v/(10*ms) : volt', name='test_neurons')
        network = Network(neurons)
        
        bridge.register_model("conversion_test", network)
        config = bridge.convert_to_loihi("conversion_test")
        
        assert 'neurons' in config
        assert 'synapses' in config
        assert len(config['neurons']) == 1
        assert config['neurons'][0]['name'] == 'test_neurons'
        assert config['neurons'][0]['size'] == 10
    
    def test_deploy_to_loihi(self, bridge):
        """Test deployment to Loihi (simulated)"""
        neurons = NeuronGroup(5, 'dv/dt = -v/(10*ms) : volt')
        network = Network(neurons)
        
        bridge.register_model("deploy_test", network)
        success = bridge.deploy_to_loihi("deploy_test")
        
        assert success is True
        assert "deploy_test" in bridge.loihi_configs


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_threat_detection_pipeline(self):
        """Test complete threat detection pipeline"""
        # Create network
        network = ThreatDetectionNetwork(input_size=50, hidden_size=20, output_size=5)
        
        # Generate synthetic training data
        training_data = []
        for _ in range(50):
            # Create patterns for different threat types
            threat_type = np.random.randint(0, 5)
            spectrum = np.zeros(50)
            
            if threat_type == 0:  # Jamming signal
                spectrum[10:20] = np.random.random(10) * 20
            elif threat_type == 1:  # Frequency hopping
                indices = np.random.choice(50, 5, replace=False)
                spectrum[indices] = np.random.random(5) * 15
            elif threat_type == 2:  # Wideband noise
                spectrum = np.random.random(50) * 5
            elif threat_type == 3:  # Narrowband signal
                center = np.random.randint(5, 45)
                spectrum[center-2:center+3] = np.random.random(5) * 25
            else:  # No threat
                spectrum = np.random.random(50) * 0.5
            
            training_data.append((spectrum, threat_type))
        
        # Train network
        network.train(training_data, epochs=2, duration_per_sample=20*ms)
        
        # Test on new data
        test_spectrum = np.zeros(50)
        test_spectrum[15:25] = np.random.random(10) * 20  # Jamming-like signal
        
        result = network.process_em_spectrum(test_spectrum, duration=30*ms)
        
        # Verify output
        assert result['spike_count'] > 0
        assert np.argmax(result['threat_scores']) in range(5)
    
    def test_swarm_consensus(self):
        """Test swarm consensus mechanism"""
        # Create swarm
        swarm = SwarmCoordinationNetwork(num_agents=10, neurons_per_agent=5)
        
        # Simulate conflicting inputs
        agent_inputs = {
            0: np.array([2.0]),  # Agent 0 votes high
            1: np.array([2.0]),  # Agent 1 votes high
            5: np.array([0.5]),  # Agent 5 votes low
            6: np.array([0.5]),  # Agent 6 votes low
            9: np.array([1.0]),  # Agent 9 is neutral
        }
        
        # Run consensus
        decisions = swarm.coordinate_decision(agent_inputs, duration=200*ms)
        
        # Check that connected agents influence each other
        assert len(decisions) == 10
        assert np.std(decisions) > 0  # Some variation
        
        # Agents should show some consensus tendencies
        mean_decision = np.mean(decisions)
        assert 0.5 < mean_decision < 1.5  # Should be somewhere in the middle


@pytest.mark.parametrize("input_size,hidden_size,output_size", [
    (10, 5, 2),
    (100, 50, 10),
    (1000, 100, 5),
])
def test_network_scaling(input_size, hidden_size, output_size):
    """Test network creation with different sizes"""
    network = ThreatDetectionNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    assert len(network.input_layer) == input_size
    assert len(network.hidden_layer) == hidden_size
    assert len(network.output_layer) == output_size
    
    # Test processing
    test_spectrum = np.random.random(input_size) * 10
    result = network.process_em_spectrum(test_spectrum, duration=10*ms)
    assert len(result['threat_scores']) == output_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])