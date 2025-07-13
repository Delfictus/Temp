"""
ARES Edge System Test Configuration
Classification: CUI//SP-CTI
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import numpy as np

# Set test mode environment variable
os.environ['ARES_TEST_MODE'] = '1'
os.environ['ARES_DISABLE_HARDWARE'] = '1'

@pytest.fixture(scope='session')
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix='ares_test_')
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_tpm():
    """Mock TPM device for testing."""
    with patch('ares.identity.tpm') as mock:
        mock.attestation_quote.return_value = b'mock_attestation_quote'
        mock.seal_data.return_value = b'sealed_data'
        mock.unseal_data.return_value = b'unsealed_data'
        yield mock

@pytest.fixture
def mock_gpu():
    """Mock GPU for testing."""
    with patch('ares.core.cuda') as mock:
        mock.is_available.return_value = True
        mock.device_count.return_value = 1
        mock.get_device_properties.return_value = Mock(
            name='Mock GPU',
            total_memory=8589934592,  # 8GB
            major=8,
            minor=0
        )
        yield mock

@pytest.fixture
def mock_tpu():
    """Mock TPU for testing."""
    with patch('ares.neuromorphic.edgetpu') as mock:
        mock.list_edge_tpus.return_value = [
            {'name': 'Mock TPU', 'path': '/dev/mock_tpu'}
        ]
        mock.load_model.return_value = Mock()
        yield mock

@pytest.fixture
def sample_rf_spectrum():
    """Generate sample RF spectrum data."""
    frequencies = np.linspace(2.4e9, 2.5e9, 1024)
    magnitudes = np.random.randn(1024) * 10 - 80  # -80 dBm noise floor
    # Add some signals
    magnitudes[100:110] += 40  # Signal at 2.41 GHz
    magnitudes[500:520] += 35  # Signal at 2.45 GHz
    return frequencies, magnitudes

@pytest.fixture
def sample_spike_train():
    """Generate sample neuromorphic spike train."""
    num_neurons = 1000
    duration_ms = 100
    spike_rate = 10  # Hz
    
    # Generate Poisson spike train
    num_spikes = int(num_neurons * duration_ms / 1000 * spike_rate)
    spike_times = np.sort(np.random.uniform(0, duration_ms, num_spikes))
    neuron_ids = np.random.randint(0, num_neurons, num_spikes)
    
    return list(zip(spike_times, neuron_ids))

@pytest.fixture
def quantum_keys():
    """Generate mock quantum-resistant keys."""
    return {
        'kyber_public': b'mock_kyber_public_key' * 50,
        'kyber_private': b'mock_kyber_private_key' * 50,
        'dilithium_public': b'mock_dilithium_public_key' * 50,
        'dilithium_private': b'mock_dilithium_private_key' * 50
    }

@pytest.fixture
def swarm_config():
    """Swarm configuration for testing."""
    return {
        'num_agents': 10,
        'byzantine_threshold': 0.33,
        'consensus_timeout': 1000,
        'agent_prefix': 'test-agent-'
    }

@pytest.fixture
def threat_scenario():
    """Sample threat scenario for testing."""
    return {
        'threat_type': 'jamming',
        'frequency': 2.45e9,
        'bandwidth': 20e6,
        'power': -30,  # dBm
        'modulation': 'chirp',
        'duration': 5000  # ms
    }

# Skip markers for hardware-dependent tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_tpu: mark test as requiring TPU"
    )
    config.addinivalue_line(
        "markers", "requires_tpm: mark test as requiring TPM"
    )

def pytest_collection_modifyitems(config, items):
    """Skip tests based on hardware availability."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_tpu = pytest.mark.skip(reason="TPU not available")
    skip_tpm = pytest.mark.skip(reason="TPM not available")
    
    for item in items:
        if "requires_gpu" in item.keywords and not has_gpu():
            item.add_marker(skip_gpu)
        if "requires_tpu" in item.keywords and not has_tpu():
            item.add_marker(skip_tpu)
        if "requires_tpm" in item.keywords and not has_tpm():
            item.add_marker(skip_tpm)

def has_gpu():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def has_tpu():
    """Check if TPU is available."""
    try:
        import edgetpu
        return len(edgetpu.list_edge_tpus()) > 0
    except:
        return False

def has_tpm():
    """Check if TPM is available."""
    return os.path.exists('/dev/tpm0')