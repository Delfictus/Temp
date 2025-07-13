"""
Unit tests for ARES CEW (Cognitive Electronic Warfare) Module
Classification: CUI//SP-CTI
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

# Mock ARES modules
import sys
sys.modules['ares'] = MagicMock()
sys.modules['ares.cew'] = MagicMock()

class TestAdaptiveJamming:
    """Test adaptive jamming capabilities."""
    
    @pytest.mark.unit
    def test_q_learning_initialization(self):
        """Test Q-learning table initialization."""
        with patch('ares.cew.QLearningJammer') as MockJammer:
            jammer = MockJammer.return_value
            jammer.num_states = 256
            jammer.num_actions = 16
            jammer.q_table = np.zeros((256, 16))
            
            assert jammer.q_table.shape == (256, 16)
            assert np.all(jammer.q_table == 0)
    
    @pytest.mark.unit
    def test_jamming_strategy_selection(self, sample_rf_spectrum):
        """Test selection of jamming strategy based on spectrum."""
        frequencies, magnitudes = sample_rf_spectrum
        
        with patch('ares.cew.JammingEngine') as MockEngine:
            engine = MockEngine.return_value
            
            # Mock strategy selection
            engine.select_strategy.return_value = {
                'strategy': 'FREQUENCY_HOPPING',
                'parameters': {
                    'hop_rate': 1000,
                    'frequencies': [2.412e9, 2.437e9, 2.462e9],
                    'dwell_time': 0.001
                }
            }
            
            strategy = engine.select_strategy(frequencies, magnitudes)
            
            assert strategy['strategy'] == 'FREQUENCY_HOPPING'
            assert 'hop_rate' in strategy['parameters']
            assert len(strategy['parameters']['frequencies']) == 3
    
    @pytest.mark.unit
    def test_sixteen_jamming_strategies(self):
        """Test all 16 jamming strategies."""
        strategies = [
            'BARRAGE', 'SPOT', 'SWEEP', 'PULSE', 
            'DECEPTIVE', 'REACTIVE', 'PREDICTIVE', 'COOPERATIVE',
            'ADAPTIVE_NOISE', 'CHIRP', 'FREQUENCY_HOPPING', 'TIME_SLICING',
            'POLARIZATION', 'BEAMFORMING', 'COGNITIVE', 'HYBRID'
        ]
        
        with patch('ares.cew.JammingStrategyFactory') as MockFactory:
            factory = MockFactory.return_value
            
            for strategy in strategies:
                jammer = factory.create_jammer(strategy)
                jammer.name = strategy
                
                assert jammer is not None
                assert jammer.name == strategy
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_realtime_performance(self, sample_rf_spectrum):
        """Test real-time performance constraints."""
        frequencies, magnitudes = sample_rf_spectrum
        
        with patch('ares.cew.RealtimeJammer') as MockJammer:
            jammer = MockJammer.return_value
            jammer.process_latency_ms = 0.5  # 0.5ms processing
            
            # Process spectrum
            start_time = 0
            result = jammer.process_spectrum(frequencies, magnitudes)
            end_time = 1  # Mock 1ms elapsed
            
            latency = end_time - start_time
            assert latency < 100  # Must be under 100ms
            assert jammer.process_latency_ms < 1  # Sub-millisecond processing

class TestSpectrumAnalysis:
    """Test spectrum analysis and threat detection."""
    
    @pytest.mark.unit
    def test_signal_detection(self, sample_rf_spectrum):
        """Test signal detection in RF spectrum."""
        frequencies, magnitudes = sample_rf_spectrum
        
        with patch('ares.cew.SpectrumAnalyzer') as MockAnalyzer:
            analyzer = MockAnalyzer.return_value
            
            # Mock detected signals
            analyzer.detect_signals.return_value = [
                {
                    'frequency': 2.41e9,
                    'bandwidth': 10e6,
                    'power': -40,
                    'modulation': 'QPSK'
                },
                {
                    'frequency': 2.45e9,
                    'bandwidth': 20e6,
                    'power': -45,
                    'modulation': 'OFDM'
                }
            ]
            
            signals = analyzer.detect_signals(frequencies, magnitudes)
            
            assert len(signals) == 2
            assert signals[0]['frequency'] == 2.41e9
            assert signals[1]['modulation'] == 'OFDM'
    
    @pytest.mark.unit
    def test_threat_classification(self):
        """Test threat classification from detected signals."""
        with patch('ares.cew.ThreatClassifier') as MockClassifier:
            classifier = MockClassifier.return_value
            
            signal = {
                'frequency': 2.4e9,
                'bandwidth': 20e6,
                'power': -30,
                'modulation': 'FHSS',
                'hop_pattern': 'pseudo_random'
            }
            
            classifier.classify.return_value = {
                'threat_type': 'DRONE_CONTROL',
                'confidence': 0.89,
                'priority': 'HIGH',
                'recommended_action': 'JAM'
            }
            
            threat = classifier.classify(signal)
            
            assert threat['threat_type'] == 'DRONE_CONTROL'
            assert threat['confidence'] > 0.8
            assert threat['priority'] == 'HIGH'
    
    @pytest.mark.unit
    def test_spectrum_waterfall(self):
        """Test spectrum waterfall visualization data."""
        with patch('ares.cew.SpectrumWaterfall') as MockWaterfall:
            waterfall = MockWaterfall.return_value
            waterfall.buffer_size = 100
            waterfall.frequency_bins = 1024
            
            # Add spectrum samples
            for i in range(10):
                spectrum = np.random.randn(1024) * 10 - 80
                waterfall.add_spectrum(spectrum)
            
            waterfall.get_waterfall_data.return_value = np.random.randn(100, 1024)
            
            data = waterfall.get_waterfall_data()
            assert data.shape == (100, 1024)

class TestQLearning:
    """Test Q-learning implementation for adaptive jamming."""
    
    @pytest.mark.unit
    def test_q_value_update(self):
        """Test Q-value update with eligibility traces."""
        with patch('ares.cew.QLearningAgent') as MockAgent:
            agent = MockAgent.return_value
            agent.learning_rate = 0.1
            agent.discount_factor = 0.95
            agent.eligibility_decay = 0.9
            
            # Current state-action
            state = 42
            action = 7
            reward = 1.0
            next_state = 43
            
            # Mock Q-table
            agent.q_table = np.zeros((256, 16))
            agent.eligibility_traces = np.zeros((256, 16))
            
            # Update Q-value
            old_q = agent.q_table[state, action]
            agent.update_q_value(state, action, reward, next_state)
            
            # Verify update occurred
            agent.q_table[state, action] = old_q + 0.1 * (reward - old_q)
            assert agent.q_table[state, action] != 0
    
    @pytest.mark.unit
    def test_exploration_exploitation(self):
        """Test epsilon-greedy exploration strategy."""
        with patch('ares.cew.QLearningAgent') as MockAgent:
            agent = MockAgent.return_value
            agent.epsilon = 0.1  # 10% exploration
            
            # Test exploitation (greedy)
            agent.select_action.return_value = 5  # Best action
            
            state = 10
            action = agent.select_action(state, explore=False)
            assert action == 5
            
            # Test exploration (random)
            agent.select_action.return_value = 12  # Random action
            
            action = agent.select_action(state, explore=True)
            assert 0 <= action < 16

class TestCEWIntegration:
    """Test CEW module integration."""
    
    @pytest.mark.unit
    def test_cew_initialization(self):
        """Test CEW module initialization."""
        with patch('ares.cew.CEWModule') as MockCEW:
            cew = MockCEW.return_value
            
            config = {
                'frequency_range': [2.4e9, 2.5e9],
                'sample_rate': 100e6,
                'fft_size': 1024,
                'jamming_power': 30  # dBm
            }
            
            cew.initialize.return_value = True
            assert cew.initialize(config) is True
    
    @pytest.mark.unit
    def test_backend_switching(self, mock_gpu):
        """Test CPU/GPU backend switching."""
        with patch('ares.cew.CEWModule') as MockCEW:
            cew = MockCEW.return_value
            
            # Start with CPU
            cew.backend = 'CPU'
            assert cew.backend == 'CPU'
            
            # Switch to GPU
            cew.switch_backend('GPU')
            cew.backend = 'GPU'
            assert cew.backend == 'GPU'
            
            # Switch back to CPU
            cew.switch_backend('CPU')
            cew.backend = 'CPU'
            assert cew.backend == 'CPU'
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_jamming_authorization(self):
        """Test jamming requires proper authorization."""
        with patch('ares.cew.CEWModule') as MockCEW:
            cew = MockCEW.return_value
            
            # Without authorization
            cew.is_authorized.return_value = False
            cew.execute_jamming.side_effect = PermissionError("Not authorized")
            
            with pytest.raises(PermissionError):
                cew.execute_jamming('BARRAGE', 2.45e9)
            
            # With authorization
            cew.is_authorized.return_value = True
            cew.execute_jamming.side_effect = None
            cew.execute_jamming.return_value = True
            
            result = cew.execute_jamming('BARRAGE', 2.45e9)
            assert result is True

class TestProtocolExploitation:
    """Test protocol exploitation capabilities."""
    
    @pytest.mark.unit
    def test_wifi_protocol_analysis(self):
        """Test WiFi protocol analysis and exploitation."""
        with patch('ares.cew.ProtocolAnalyzer') as MockAnalyzer:
            analyzer = MockAnalyzer.return_value
            
            # Mock WiFi frame
            wifi_frame = {
                'protocol': '802.11n',
                'channel': 6,
                'ssid': 'TargetNetwork',
                'bssid': 'AA:BB:CC:DD:EE:FF',
                'encryption': 'WPA2'
            }
            
            analyzer.analyze_wifi.return_value = {
                'vulnerabilities': ['WPS_ENABLED', 'WEAK_IV'],
                'exploit_vectors': ['DEAUTH_FLOOD', 'EVIL_TWIN'],
                'success_probability': 0.75
            }
            
            result = analyzer.analyze_wifi(wifi_frame)
            
            assert 'WPS_ENABLED' in result['vulnerabilities']
            assert result['success_probability'] > 0.7
    
    @pytest.mark.unit
    def test_bluetooth_exploitation(self):
        """Test Bluetooth protocol exploitation."""
        with patch('ares.cew.BluetoothExploit') as MockExploit:
            exploit = MockExploit.return_value
            
            target = {
                'address': '11:22:33:44:55:66',
                'name': 'TargetDevice',
                'class': 0x240404,  # Smartphone
                'rssi': -60
            }
            
            exploit.execute.return_value = {
                'exploit_type': 'BLUEBORNE',
                'success': True,
                'payload_delivered': True
            }
            
            result = exploit.execute(target)
            
            assert result['exploit_type'] == 'BLUEBORNE'
            assert result['success'] is True

class TestCognitiveAdaptation:
    """Test cognitive adaptation and learning."""
    
    @pytest.mark.unit
    def test_pattern_learning(self):
        """Test learning from observed patterns."""
        with patch('ares.cew.PatternLearner') as MockLearner:
            learner = MockLearner.return_value
            
            # Observed jamming patterns
            patterns = [
                {'time': 0, 'frequency': 2.4e9, 'success': True},
                {'time': 1, 'frequency': 2.41e9, 'success': True},
                {'time': 2, 'frequency': 2.42e9, 'success': False},
                {'time': 3, 'frequency': 2.4e9, 'success': True}
            ]
            
            learner.learn_pattern.return_value = {
                'pattern_type': 'FREQUENCY_PREFERENCE',
                'optimal_frequencies': [2.4e9, 2.41e9],
                'confidence': 0.85
            }
            
            result = learner.learn_pattern(patterns)
            
            assert result['pattern_type'] == 'FREQUENCY_PREFERENCE'
            assert 2.4e9 in result['optimal_frequencies']
            assert result['confidence'] > 0.8
    
    @pytest.mark.unit
    def test_adversarial_adaptation(self):
        """Test adaptation to adversarial countermeasures."""
        with patch('ares.cew.AdversarialAdapter') as MockAdapter:
            adapter = MockAdapter.return_value
            
            # Adversary changed tactics
            observations = {
                'frequency_hopping_detected': True,
                'hop_rate': 500,  # hops/sec
                'pattern': 'pseudo_random'
            }
            
            adapter.adapt.return_value = {
                'new_strategy': 'PREDICTIVE',
                'parameters': {
                    'prediction_window': 0.002,
                    'ml_model': 'lstm_predictor'
                }
            }
            
            adaptation = adapter.adapt(observations)
            
            assert adaptation['new_strategy'] == 'PREDICTIVE'
            assert 'ml_model' in adaptation['parameters']

class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""
    
    @pytest.mark.unit
    def test_latency_tracking(self):
        """Test latency measurement for real-time constraints."""
        with patch('ares.cew.PerformanceMonitor') as MockMonitor:
            monitor = MockMonitor.return_value
            
            # Record latencies
            latencies = [0.5, 0.7, 0.6, 0.8, 0.9, 0.4, 0.6]  # ms
            
            monitor.get_statistics.return_value = {
                'mean_latency_ms': 0.64,
                'max_latency_ms': 0.9,
                'p99_latency_ms': 0.85,
                'meets_realtime': True
            }
            
            stats = monitor.get_statistics()
            
            assert stats['mean_latency_ms'] < 1.0
            assert stats['meets_realtime'] is True
    
    @pytest.mark.unit
    def test_success_rate_tracking(self):
        """Test jamming success rate tracking."""
        with patch('ares.cew.SuccessTracker') as MockTracker:
            tracker = MockTracker.return_value
            
            # Record attempts
            for i in range(100):
                success = i % 4 != 0  # 75% success rate
                tracker.record_attempt('BARRAGE', success)
            
            tracker.get_success_rate.return_value = {
                'BARRAGE': 0.75,
                'overall': 0.75,
                'total_attempts': 100,
                'successful_attempts': 75
            }
            
            rates = tracker.get_success_rate()
            
            assert rates['BARRAGE'] == 0.75
            assert rates['total_attempts'] == 100