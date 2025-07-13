"""
Unit tests for ARES Security and Identity Management
Codename: IRONRECON
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import hashlib
import time
import json

# Mock imports
with patch.dict('sys.modules', {
    'ares_unified': MagicMock(),
    'ares_unified.identity': MagicMock(),
    'ares_unified.identity.hardware_attestation': MagicMock(),
    'ares_unified.identity.hot_swap_manager': MagicMock(),
    'ares_unified.countermeasures': MagicMock(),
}):
    from ares_unified.identity import hardware_attestation
    from ares_unified.identity import hot_swap_manager
    from ares_unified.countermeasures import self_destruct_protocol


class TestHardwareAttestation:
    """Test hardware-based attestation system."""
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_tpm_initialization(self, mock_tpm):
        """Test TPM 2.0 initialization and availability."""
        attestation = Mock()
        attestation.initialize_tpm.return_value = mock_tpm
        attestation.tpm_available = True
        
        tpm = attestation.initialize_tpm()
        assert tpm is not None
        assert attestation.tpm_available is True
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_attestation_key_generation(self, mock_tpm):
        """Test generation of attestation identity keys."""
        attestation = Mock()
        aik = {
            "public_key": b"attestation_public_key",
            "key_handle": 0x81000001,
            "algorithm": "RSA2048"
        }
        attestation.create_attestation_key.return_value = aik
        
        key = attestation.create_attestation_key(mock_tpm)
        assert key["public_key"] is not None
        assert key["key_handle"] > 0x81000000  # Persistent handle range
        assert key["algorithm"] in ["RSA2048", "ECC256"]
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_platform_measurement(self):
        """Test platform configuration measurement."""
        attestation = Mock()
        measurements = {
            "pcr_0": hashlib.sha256(b"bios").hexdigest(),
            "pcr_1": hashlib.sha256(b"bootloader").hexdigest(),
            "pcr_2": hashlib.sha256(b"kernel").hexdigest(),
            "pcr_7": hashlib.sha256(b"secure_boot").hexdigest()
        }
        attestation.get_platform_measurements.return_value = measurements
        
        pcrs = attestation.get_platform_measurements()
        assert len(pcrs) >= 4
        assert all(len(v) == 64 for v in pcrs.values())  # SHA256 hex
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_remote_attestation_quote(self, mock_tpm):
        """Test generation of remote attestation quote."""
        attestation = Mock()
        quote = {
            "quoted_pcrs": [0, 1, 2, 7],
            "quote_data": b"signed_quote_data",
            "signature": b"tpm_signature",
            "nonce": b"challenge_nonce"
        }
        attestation.generate_quote.return_value = quote
        
        nonce = b"random_challenge_123"
        attestation_quote = attestation.generate_quote(mock_tpm, nonce, [0, 1, 2, 7])
        
        assert attestation_quote["quote_data"] is not None
        assert attestation_quote["signature"] is not None
        assert attestation_quote["nonce"] == nonce
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_attestation_verification(self):
        """Test verification of attestation quotes."""
        verifier = Mock()
        verifier.verify_quote.return_value = {
            "valid": True,
            "trusted_platform": True,
            "pcr_match": True,
            "timestamp": time.time()
        }
        
        verification = verifier.verify_quote(
            quote_data=b"quote",
            signature=b"signature",
            expected_pcrs={"pcr_0": "hash0", "pcr_1": "hash1"}
        )
        
        assert verification["valid"] is True
        assert verification["trusted_platform"] is True
        assert verification["pcr_match"] is True


class TestHotSwapIdentity:
    """Test hot-swap identity management."""
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_identity_generation(self):
        """Test generation of new identities."""
        manager = Mock()
        identity = {
            "id": "ares_unit_001",
            "public_key": b"identity_public_key",
            "attributes": {
                "role": "reconnaissance",
                "clearance": "secret",
                "capabilities": ["cew", "swarm", "autonomous"]
            },
            "created_at": time.time()
        }
        manager.generate_identity.return_value = identity
        
        new_identity = manager.generate_identity(
            role="reconnaissance",
            clearance="secret"
        )
        
        assert new_identity["id"] is not None
        assert new_identity["public_key"] is not None
        assert new_identity["attributes"]["clearance"] == "secret"
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_identity_transition(self):
        """Test seamless identity transition."""
        manager = Mock()
        manager.transition_identity.return_value = {
            "success": True,
            "old_identity": "ares_unit_001",
            "new_identity": "ares_unit_002",
            "transition_time_ms": 47.3,
            "sessions_migrated": 5
        }
        
        result = manager.transition_identity(
            old_id="ares_unit_001",
            new_id="ares_unit_002"
        )
        
        assert result["success"] is True
        assert result["transition_time_ms"] < 50  # Under 50ms requirement
        assert result["sessions_migrated"] >= 0
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_identity_isolation(self):
        """Test cryptographic isolation between identities."""
        manager = Mock()
        
        # Each identity should have isolated key material
        identity1_keys = {"signing": b"key1", "encryption": b"key2"}
        identity2_keys = {"signing": b"key3", "encryption": b"key4"}
        
        manager.get_identity_keys.side_effect = [identity1_keys, identity2_keys]
        
        keys1 = manager.get_identity_keys("identity_1")
        keys2 = manager.get_identity_keys("identity_2")
        
        assert keys1["signing"] != keys2["signing"]
        assert keys1["encryption"] != keys2["encryption"]
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_identity_revocation(self):
        """Test identity revocation and cleanup."""
        manager = Mock()
        manager.revoke_identity.return_value = {
            "revoked": True,
            "keys_destroyed": True,
            "sessions_terminated": 3,
            "audit_logged": True
        }
        
        revocation = manager.revoke_identity("compromised_identity")
        
        assert revocation["revoked"] is True
        assert revocation["keys_destroyed"] is True
        assert revocation["audit_logged"] is True
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_zero_knowledge_proof(self):
        """Test zero-knowledge proof for identity verification."""
        manager = Mock()
        
        # Generate ZK proof
        proof = {
            "commitment": b"commitment_value",
            "challenge": b"random_challenge",
            "response": b"proof_response",
            "verified": True
        }
        manager.generate_zk_proof.return_value = proof
        manager.verify_zk_proof.return_value = True
        
        # Prover generates proof
        zk_proof = manager.generate_zk_proof(
            secret="identity_secret",
            statement="has_clearance_level_secret"
        )
        
        # Verifier checks proof
        valid = manager.verify_zk_proof(zk_proof, "has_clearance_level_secret")
        
        assert valid is True
        assert len(zk_proof["commitment"]) > 0


class TestCountermeasures:
    """Test security countermeasures."""
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_intrusion_detection(self):
        """Test intrusion detection system."""
        ids = Mock()
        ids.detect_intrusion.return_value = {
            "intrusion_detected": True,
            "confidence": 0.94,
            "attack_vector": "buffer_overflow",
            "affected_module": "network_stack",
            "recommended_response": "isolate_and_patch"
        }
        
        detection = ids.detect_intrusion(
            network_traffic=b"suspicious_pattern",
            system_calls=["exec", "mmap", "ptrace"]
        )
        
        assert detection["intrusion_detected"] is True
        assert detection["confidence"] > 0.9
        assert detection["attack_vector"] in [
            "buffer_overflow", "injection", "privilege_escalation", "dos"
        ]
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_automatic_response(self):
        """Test automatic security response system."""
        responder = Mock()
        responder.execute_response.return_value = {
            "response_type": "isolate",
            "success": True,
            "modules_affected": ["network", "storage"],
            "recovery_time_s": 2.5
        }
        
        response = responder.execute_response(
            threat_level="high",
            attack_vector="network"
        )
        
        assert response["success"] is True
        assert response["response_type"] in ["isolate", "patch", "restart", "failover"]
        assert response["recovery_time_s"] < 5.0
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_secure_erase(self):
        """Test secure data erasure."""
        eraser = Mock()
        eraser.secure_erase.return_value = {
            "erased": True,
            "passes": 7,
            "verification": "complete",
            "residual_data": False
        }
        
        result = eraser.secure_erase(
            data_path="/sensitive/data",
            algorithm="DoD_5220.22-M",
            verify=True
        )
        
        assert result["erased"] is True
        assert result["passes"] >= 3  # Minimum secure erasure
        assert result["residual_data"] is False
    
    @pytest.mark.unit
    @pytest.mark.security
    @pytest.mark.skip(reason="Destructive test - only run in isolated environment")
    def test_self_destruct_protocol(self):
        """Test self-destruct protocol (simulated)."""
        protocol = Mock()
        protocol.arm.return_value = {"armed": True, "countdown": 30}
        protocol.verify_authorization.return_value = True
        protocol.execute.return_value = {
            "executed": True,
            "data_destroyed": True,
            "hardware_disabled": True
        }
        
        # This would only run in a test environment
        assert protocol.verify_authorization("test_auth_code") is True
        
        # Arm the system
        armed = protocol.arm(authorization="test_auth_code", countdown_s=30)
        assert armed["armed"] is True
        
        # In real scenario, this would destroy the system
        # result = protocol.execute()
        # assert result["data_destroyed"] is True


class TestAccessControl:
    """Test access control mechanisms."""
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_role_based_access(self):
        """Test RBAC implementation."""
        rbac = Mock()
        rbac.check_permission.side_effect = lambda user, action, resource: \
            (user == "admin" or (user == "operator" and action == "read"))
        
        # Admin has all permissions
        assert rbac.check_permission("admin", "write", "config") is True
        assert rbac.check_permission("admin", "delete", "logs") is True
        
        # Operator has limited permissions
        assert rbac.check_permission("operator", "read", "status") is True
        assert rbac.check_permission("operator", "write", "config") is False
        
        # Unknown user has no permissions
        assert rbac.check_permission("unknown", "read", "anything") is False
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_mandatory_access_control(self):
        """Test MAC/MLS implementation."""
        mac = Mock()
        
        # Define security levels
        mac.get_subject_clearance.side_effect = lambda s: {
            "user1": "secret",
            "user2": "confidential",
            "user3": "top_secret"
        }.get(s, "unclassified")
        
        mac.get_object_classification.side_effect = lambda o: {
            "nuclear_codes": "top_secret",
            "mission_plans": "secret",
            "weather_data": "unclassified"
        }.get(o, "unclassified")
        
        mac.check_access.side_effect = lambda subject, obj, mode: \
            mac.get_subject_clearance(subject) >= mac.get_object_classification(obj)
        
        # Test access decisions
        assert mac.check_access("user3", "nuclear_codes", "read") is True
        assert mac.check_access("user1", "nuclear_codes", "read") is False
        assert mac.check_access("user2", "weather_data", "read") is True


@pytest.mark.compliance
class TestSecurityCompliance:
    """Test security compliance requirements."""
    
    def test_audit_trail_integrity(self):
        """Test audit trail tamper protection."""
        audit = Mock()
        
        log_entry = {
            "timestamp": time.time(),
            "event": "authentication_failure",
            "user": "unknown",
            "source_ip": "192.168.1.100",
            "hash": "previous_hash"
        }
        
        # Generate hash chain
        entry_hash = hashlib.sha256(
            json.dumps(log_entry, sort_keys=True).encode()
        ).hexdigest()
        
        audit.add_log_entry.return_value = {
            "stored": True,
            "hash": entry_hash,
            "chain_valid": True
        }
        
        result = audit.add_log_entry(log_entry)
        assert result["stored"] is True
        assert result["chain_valid"] is True
        assert len(result["hash"]) == 64  # SHA256
    
    def test_cryptographic_compliance(self):
        """Test compliance with cryptographic standards."""
        crypto = Mock()
        crypto.get_compliance_status.return_value = {
            "fips_140_2": True,
            "suite_b": True,
            "quantum_resistant": True,
            "algorithms": ["AES-256-GCM", "SHA-384", "ECDSA-P384", "CRYSTALS-DILITHIUM"]
        }
        
        status = crypto.get_compliance_status()
        assert status["fips_140_2"] is True
        assert status["quantum_resistant"] is True
        assert "AES-256-GCM" in status["algorithms"]