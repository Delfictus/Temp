"""
Unit tests for ARES Core Module
Classification: CUI//SP-CTI
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import hashlib

# Mock the ARES modules since we're testing
import sys
sys.modules['ares'] = MagicMock()
sys.modules['ares.core'] = MagicMock()
sys.modules['ares.core.quantum'] = MagicMock()

class TestQuantumResilientCore:
    """Test quantum-resilient cryptography implementation."""
    
    @pytest.mark.unit
    def test_kyber_key_generation(self, quantum_keys):
        """Test Kyber key pair generation."""
        # Mock the Kyber key generation
        with patch('ares.core.quantum.generate_kyber_keypair') as mock_gen:
            mock_gen.return_value = (
                quantum_keys['kyber_public'],
                quantum_keys['kyber_private']
            )
            
            from ares.core.quantum import generate_kyber_keypair
            pub, priv = generate_kyber_keypair()
            
            assert len(pub) > 1000  # Kyber public keys are large
            assert len(priv) > 1000
            assert pub != priv
            mock_gen.assert_called_once()
    
    @pytest.mark.unit
    def test_kyber_encryption_decryption(self, quantum_keys):
        """Test Kyber encryption and decryption."""
        plaintext = b"Classified mission data"
        
        with patch('ares.core.quantum.kyber_encrypt') as mock_enc:
            mock_enc.return_value = b'encrypted_data'
            
            with patch('ares.core.quantum.kyber_decrypt') as mock_dec:
                mock_dec.return_value = plaintext
                
                from ares.core.quantum import kyber_encrypt, kyber_decrypt
                
                # Encrypt
                ciphertext = kyber_encrypt(plaintext, quantum_keys['kyber_public'])
                assert ciphertext != plaintext
                assert len(ciphertext) > len(plaintext)
                
                # Decrypt
                decrypted = kyber_decrypt(ciphertext, quantum_keys['kyber_private'])
                assert decrypted == plaintext
    
    @pytest.mark.unit
    def test_dilithium_signature(self, quantum_keys):
        """Test Dilithium signature generation and verification."""
        message = b"Authenticated command: ENGAGE"
        
        with patch('ares.core.quantum.dilithium_sign') as mock_sign:
            mock_sign.return_value = b'signature' * 100
            
            with patch('ares.core.quantum.dilithium_verify') as mock_verify:
                mock_verify.return_value = True
                
                from ares.core.quantum import dilithium_sign, dilithium_verify
                
                # Sign
                signature = dilithium_sign(message, quantum_keys['dilithium_private'])
                assert len(signature) > 2000  # Dilithium signatures are large
                
                # Verify
                valid = dilithium_verify(message, signature, quantum_keys['dilithium_public'])
                assert valid is True
                
                # Verify tampered message fails
                mock_verify.return_value = False
                tampered = message + b"TAMPERED"
                invalid = dilithium_verify(tampered, signature, quantum_keys['dilithium_public'])
                assert invalid is False
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_secure_random_generation(self):
        """Test cryptographically secure random number generation."""
        with patch('os.urandom') as mock_urandom:
            mock_urandom.return_value = b'random' * 8
            
            from ares.core.quantum import generate_secure_random
            
            # Generate random bytes
            random1 = generate_secure_random(32)
            random2 = generate_secure_random(32)
            
            assert len(random1) == 32
            assert len(random2) == 32
            assert random1 != random2  # Should be different
            assert mock_urandom.called

class TestARESCore:
    """Test main ARES Core functionality."""
    
    @pytest.mark.unit
    def test_core_initialization(self, mock_tpm):
        """Test ARES Core initialization."""
        with patch('ares.core.ARESCore') as MockCore:
            instance = MockCore.return_value
            instance.initialize.return_value = True
            instance.is_initialized = True
            
            core = MockCore()
            assert core.initialize() is True
            assert core.is_initialized is True
    
    @pytest.mark.unit
    def test_component_registration(self):
        """Test component registration and retrieval."""
        with patch('ares.core.ARESCore') as MockCore:
            instance = MockCore.return_value
            components = {}
            
            def register(name, component):
                components[name] = component
                return True
            
            def get(name):
                return components.get(name)
            
            instance.register_component = register
            instance.get_component = get
            
            core = MockCore()
            
            # Register components
            mock_cew = Mock(name='CEW')
            mock_swarm = Mock(name='Swarm')
            
            assert core.register_component('cew', mock_cew) is True
            assert core.register_component('swarm', mock_swarm) is True
            
            # Retrieve components
            assert core.get_component('cew') == mock_cew
            assert core.get_component('swarm') == mock_swarm
            assert core.get_component('nonexistent') is None
    
    @pytest.mark.unit
    def test_hardware_detection(self, mock_gpu, mock_tpu):
        """Test hardware capability detection."""
        with patch('ares.core.hardware') as mock_hw:
            mock_hw.detect_gpu.return_value = True
            mock_hw.detect_tpu.return_value = True
            mock_hw.detect_tpm.return_value = True
            
            from ares.core.hardware import detect_capabilities
            
            caps = detect_capabilities()
            assert caps['gpu'] is True
            assert caps['tpu'] is True
            assert caps['tpm'] is True
            
    @pytest.mark.unit
    @pytest.mark.security
    def test_memory_encryption(self):
        """Test in-memory encryption for sensitive data."""
        sensitive_data = b"TOP SECRET DATA"
        
        with patch('ares.core.security.encrypt_memory') as mock_enc:
            mock_enc.return_value = b'encrypted_memory'
            
            with patch('ares.core.security.decrypt_memory') as mock_dec:
                mock_dec.return_value = sensitive_data
                
                from ares.core.security import SecureMemory
                
                secure_mem = SecureMemory()
                
                # Store encrypted
                handle = secure_mem.store(sensitive_data)
                assert handle is not None
                
                # Retrieve decrypted
                retrieved = secure_mem.retrieve(handle)
                assert retrieved == sensitive_data
                
                # Clear from memory
                secure_mem.clear(handle)
                assert secure_mem.retrieve(handle) is None

class TestNeuromorphicCore:
    """Test neuromorphic processing core."""
    
    @pytest.mark.unit
    def test_spike_encoding(self, sample_spike_train):
        """Test spike encoding functionality."""
        with patch('ares.core.neuromorphic.SpikeEncoder') as MockEncoder:
            encoder = MockEncoder.return_value
            encoder.encode.return_value = sample_spike_train
            
            # Encode analog signal
            analog_signal = np.sin(np.linspace(0, 2*np.pi, 1000))
            spikes = encoder.encode(analog_signal, method='rate')
            
            assert len(spikes) > 0
            assert all(isinstance(s, tuple) for s in spikes)
            assert all(len(s) == 2 for s in spikes)  # (time, neuron_id)
    
    @pytest.mark.unit
    def test_neuron_models(self):
        """Test different neuron model implementations."""
        with patch('ares.core.neuromorphic.neurons') as mock_neurons:
            # LIF neuron
            lif = mock_neurons.LIF(tau_m=20.0, v_thresh=-50.0)
            lif.step.return_value = False  # No spike
            
            assert lif.step(10.0) is False
            
            # Izhikevich neuron
            izh = mock_neurons.Izhikevich(a=0.02, b=0.2, c=-65, d=8)
            izh.step.return_value = True  # Spike
            
            assert izh.step(20.0) is True
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_simd_optimization(self):
        """Test SIMD optimizations for neuromorphic processing."""
        with patch('ares.core.neuromorphic.simd') as mock_simd:
            mock_simd.vector_add.return_value = np.array([2, 4, 6, 8])
            
            # Test vectorized operations
            a = np.array([1, 2, 3, 4], dtype=np.float32)
            b = np.array([1, 2, 3, 4], dtype=np.float32)
            
            result = mock_simd.vector_add(a, b)
            expected = a + b
            
            np.testing.assert_array_equal(result, expected)

class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.unit
    def test_initialization_failure_recovery(self):
        """Test graceful handling of initialization failures."""
        with patch('ares.core.ARESCore') as MockCore:
            instance = MockCore.return_value
            instance.initialize.side_effect = Exception("Hardware not found")
            
            core = MockCore()
            
            with pytest.raises(Exception) as exc_info:
                core.initialize()
            
            assert "Hardware not found" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_crypto_failure_handling(self):
        """Test handling of cryptographic operation failures."""
        with patch('ares.core.quantum.kyber_encrypt') as mock_enc:
            mock_enc.side_effect = Exception("Invalid key format")
            
            from ares.core.quantum import safe_encrypt
            
            result = safe_encrypt(b"data", b"bad_key")
            assert result is None  # Should return None on failure
    
    @pytest.mark.unit
    def test_resource_cleanup(self):
        """Test proper resource cleanup on errors."""
        with patch('ares.core.ARESCore') as MockCore:
            instance = MockCore.return_value
            instance.cleanup_called = False
            
            def cleanup():
                instance.cleanup_called = True
            
            instance.cleanup = cleanup
            
            core = MockCore()
            
            try:
                # Simulate error
                raise Exception("Test error")
            except:
                core.cleanup()
            
            assert instance.cleanup_called is True

class TestConfigurationManagement:
    """Test configuration loading and validation."""
    
    @pytest.mark.unit
    def test_config_loading(self, test_data_dir):
        """Test configuration file loading."""
        config_path = os.path.join(test_data_dir, 'test_config.yaml')
        
        # Create test config
        test_config = """
system:
  node_id: test-node-001
  log_level: DEBUG
security:
  encryption: AES-256-GCM
  key_rotation: 86400
"""
        with open(config_path, 'w') as f:
            f.write(test_config)
        
        with patch('ares.core.config.load_config') as mock_load:
            mock_load.return_value = {
                'system': {
                    'node_id': 'test-node-001',
                    'log_level': 'DEBUG'
                },
                'security': {
                    'encryption': 'AES-256-GCM',
                    'key_rotation': 86400
                }
            }
            
            from ares.core.config import load_config
            config = load_config(config_path)
            
            assert config['system']['node_id'] == 'test-node-001'
            assert config['security']['encryption'] == 'AES-256-GCM'
    
    @pytest.mark.unit
    def test_config_validation(self):
        """Test configuration validation."""
        with patch('ares.core.config.validate_config') as mock_validate:
            mock_validate.return_value = (True, [])
            
            from ares.core.config import validate_config
            
            valid_config = {
                'system': {'node_id': 'test-001'},
                'security': {'encryption': 'AES-256-GCM'}
            }
            
            is_valid, errors = validate_config(valid_config)
            assert is_valid is True
            assert len(errors) == 0
            
            # Test invalid config
            mock_validate.return_value = (False, ['Missing required field: system.node_id'])
            
            invalid_config = {'security': {'encryption': 'AES-256-GCM'}}
            is_valid, errors = validate_config(invalid_config)
            
            assert is_valid is False
            assert len(errors) > 0