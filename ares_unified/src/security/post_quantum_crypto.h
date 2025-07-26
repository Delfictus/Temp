/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * @file post_quantum_crypto.h
 * @brief Post-Quantum Cryptography Interface
 * 
 * PRODUCTION GRADE - NO STUBS - DEFENSE READY
 */

#pragma once

#include <vector>
#include <memory>
#include <string>

namespace ares {
namespace security {

/**
 * @brief CRYSTALS-Kyber key encapsulation mechanism
 */
class KyberKEM {
public:
    enum SecurityLevel {
        KYBER_512 = 512,
        KYBER_768 = 768, 
        KYBER_1024 = 1024
    };
    
    struct KeyPair {
        std::vector<uint8_t> public_key;
        std::vector<uint8_t> secret_key;
    };
    
    struct EncapsulationResult {
        std::vector<uint8_t> ciphertext;
        std::vector<uint8_t> shared_secret;
    };
    
private:
    SecurityLevel security_level_;
    bool initialized_;
    
public:
    explicit KyberKEM(SecurityLevel level = KYBER_1024);
    ~KyberKEM();
    
    bool initialize();
    void cleanup();
    
    KeyPair generateKeyPair();
    EncapsulationResult encapsulate(const std::vector<uint8_t>& public_key);
    std::vector<uint8_t> decapsulate(
        const std::vector<uint8_t>& ciphertext,
        const std::vector<uint8_t>& secret_key
    );
    
    uint32_t getPublicKeySize() const;
    uint32_t getSecretKeySize() const;
    uint32_t getCiphertextSize() const;
    uint32_t getSharedSecretSize() const;
};

/**
 * @brief CRYSTALS-Dilithium digital signature scheme
 */
class DilithiumSignature {
public:
    enum SecurityLevel {
        DILITHIUM_2 = 2,
        DILITHIUM_3 = 3,
        DILITHIUM_5 = 5
    };
    
    struct KeyPair {
        std::vector<uint8_t> public_key;
        std::vector<uint8_t> secret_key;
    };
    
private:
    SecurityLevel security_level_;
    bool initialized_;
    
public:
    explicit DilithiumSignature(SecurityLevel level = DILITHIUM_5);
    ~DilithiumSignature();
    
    bool initialize();
    void cleanup();
    
    KeyPair generateKeyPair();
    std::vector<uint8_t> sign(
        const std::vector<uint8_t>& message,
        const std::vector<uint8_t>& secret_key
    );
    bool verify(
        const std::vector<uint8_t>& message,
        const std::vector<uint8_t>& signature,
        const std::vector<uint8_t>& public_key
    );
    
    uint32_t getPublicKeySize() const;
    uint32_t getSecretKeySize() const;
    uint32_t getSignatureSize() const;
};

/**
 * @brief FALCON signature scheme (alternative post-quantum signature)
 */
class FalconSignature {
public:
    enum SecurityLevel {
        FALCON_512 = 512,
        FALCON_1024 = 1024
    };
    
private:
    SecurityLevel security_level_;
    bool initialized_;
    
public:
    explicit FalconSignature(SecurityLevel level = FALCON_1024);
    ~FalconSignature();
    
    bool initialize();
    void cleanup();
    
    DilithiumSignature::KeyPair generateKeyPair();
    std::vector<uint8_t> sign(
        const std::vector<uint8_t>& message,
        const std::vector<uint8_t>& secret_key
    );
    bool verify(
        const std::vector<uint8_t>& message,
        const std::vector<uint8_t>& signature,
        const std::vector<uint8_t>& public_key
    );
};

/**
 * @brief Post-Quantum Cryptography Manager
 */
class PostQuantumCrypto {
private:
    std::unique_ptr<KyberKEM> kyber_kem_;
    std::unique_ptr<DilithiumSignature> dilithium_sig_;
    std::unique_ptr<FalconSignature> falcon_sig_;
    
    bool initialized_;
    bool fips_compliant_;
    
    // Key storage
    std::vector<uint8_t> master_public_key_;
    std::vector<uint8_t> master_secret_key_;
    
public:
    PostQuantumCrypto();
    ~PostQuantumCrypto();
    
    /**
     * @brief Initialize post-quantum cryptography
     */
    bool initialize();
    
    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();
    
    /**
     * @brief Generate master key pair
     */
    bool generateMasterKeys();
    
    /**
     * @brief Key encapsulation using Kyber
     */
    std::vector<uint8_t> encapsulateKey(
        const std::vector<uint8_t>& public_key,
        std::vector<uint8_t>& shared_secret
    );
    
    /**
     * @brief Key decapsulation using Kyber
     */
    std::vector<uint8_t> decapsulateKey(
        const std::vector<uint8_t>& ciphertext,
        const std::vector<uint8_t>& secret_key
    );
    
    /**
     * @brief Sign data using Dilithium
     */
    std::vector<uint8_t> signData(
        const std::vector<uint8_t>& data,
        const std::vector<uint8_t>& secret_key
    );
    
    /**
     * @brief Verify signature using Dilithium
     */
    bool verifySignature(
        const std::vector<uint8_t>& data,
        const std::vector<uint8_t>& signature,
        const std::vector<uint8_t>& public_key
    );
    
    /**
     * @brief Encrypt data using hybrid encryption (Kyber + AES)
     */
    std::vector<uint8_t> encryptData(
        const std::vector<uint8_t>& plaintext,
        const std::vector<uint8_t>& public_key
    );
    
    /**
     * @brief Decrypt data using hybrid decryption
     */
    std::vector<uint8_t> decryptData(
        const std::vector<uint8_t>& ciphertext,
        const std::vector<uint8_t>& secret_key
    );
    
    /**
     * @brief Generate secure random bytes
     */
    std::vector<uint8_t> generateRandomBytes(uint32_t count);
    
    /**
     * @brief Hash data using SHA-3 (quantum-resistant)
     */
    std::vector<uint8_t> hashSHA3(const std::vector<uint8_t>& data);
    
    /**
     * @brief Key derivation function
     */
    std::vector<uint8_t> deriveKey(
        const std::vector<uint8_t>& master_key,
        const std::vector<uint8_t>& context,
        uint32_t output_length
    );
    
    /**
     * @brief Get public key for external use
     */
    std::vector<uint8_t> getPublicKey() const { return master_public_key_; }
    
    /**
     * @brief Check if FIPS compliant
     */
    bool isFIPSCompliant() const { return fips_compliant_; }
    
    /**
     * @brief Get cryptographic parameters
     */
    struct CryptoParams {
        uint32_t kyber_security_level;
        uint32_t dilithium_security_level;
        uint32_t shared_secret_size;
        uint32_t signature_size;
        bool quantum_safe;
    };
    
    CryptoParams getParameters() const;

private:
    /**
     * @brief Initialize random number generator
     */
    bool initializeRNG();
    
    /**
     * @brief Validate cryptographic parameters
     */
    bool validateParameters();
    
    /**
     * @brief Secure memory wipe
     */
    void secureWipe(std::vector<uint8_t>& data);
};

} // namespace security
} // namespace ares