# Quantum-Safe Federated Learning System

A privacy-preserving federated learning system combining post-quantum cryptography, homomorphic encryption, and secure multi-party computation for decentralized machine learning.

## Table of Contents

- Overview
- Features
- Architecture
- Installation
- Quick Start
- Benchmarking
- Project Structure
- Security Analysis
- Contributing
- License
- References

---

## Overview

This project implements a production-grade federated learning system that addresses data privacy and quantum-safe security in machine learning. Our system enables multiple clients to collaboratively train a machine learning model while:

• Keeping all data local - No data leaves the client devices
• Protecting against eavesdropping - Using quantum-resistant encryption
• Preserving privacy during aggregation - Using homomorphic encryption
• Verifying authenticity - Using post-quantum digital signatures
• Ensuring integrity - Using secure channels and signature verification

### Key Problem Statement

Traditional federated learning systems are vulnerable to:

1. Quantum Attacks - Current cryptography (RSA, ECDSA) will be broken by quantum computers
2. Privacy Leakage - Model updates can leak sensitive information about training data
3. Man-in-the-Middle Attacks - Unencrypted communications can be intercepted
4. Authentication Failure - Clients can impersonate servers or vice versa

Our system solves these problems using:
• CRYSTALS-Kyber (Quantum-resistant key exchange)
• CRYSTALS-Dilithium (Quantum-resistant signatures)
• Paillier Cryptosystem (Additive homomorphic encryption)
• AES-256-GCM (Symmetric encryption for channels)

---

## Features

### Post-Quantum Cryptography (NIST Standardized)

#### CRYSTALS-Kyber (ML-KEM-768)
• Type: Key Encapsulation Mechanism (KEM)
• Security Level: 192-bit quantum resistance
• Ciphertext Size: 1088 bytes
• Shared Secret Size: 32 bytes
• Purpose: Establish quantum-safe session keys
• Standardization: NIST FIPS 203

#### CRYSTALS-Dilithium (ML-DSA-44)
• Type: Digital Signature Algorithm
• Security Level: 128-bit quantum resistance
• Signature Size: ~2420 bytes
• Public Key Size: 1312 bytes
• Purpose: Authenticate clients and verify update integrity
• Standardization: NIST FIPS 204

### Privacy-Preserving Aggregation

#### Paillier Homomorphic Encryption
• Type: Additive Homomorphic Encryption
• Key Size: 2048 bits (RSA modulus)
• Encryption Time: ~100ms per model
• Property: E(m1) + E(m2) = E(m1 + m2)

Advantage: Server never sees individual client gradients, only encrypted values

### Machine Learning Capabilities

#### Model Architecture
• Algorithm: Logistic Regression (SGD-based)
• Framework: scikit-learn
• Parameters: Weight matrix W (1 × n_features) + Bias b
• Optimizer: Stochastic Gradient Descent (SGD)
• Learning Rate: 0.01 (configurable)
• Local Epochs: 1 per round (configurable)

#### Dataset: Adult Income Census
• Type: Binary Classification
• Samples: 36,177 total (split 80/20 train/test)
• Features: 4 numeric features (Age, Capital-gain, Capital-loss, Hours-per-week)
• Target: Income > $50K (binary: 0 or 1)
• Accuracy: 79.4% (on test set)
• Convergence: 3-5 rounds

---

## Installation

### Prerequisites

• Python 3.9 or higher
• pip (Python package manager)
• Virtual environment (recommended)

### Step 1: Clone Repository

git clone https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml.git
cd quantum-safe-hybrid-ml

### Step 2: Create Virtual Environment

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### Step 3: Install Dependencies

pip install -r requirements.txt

### Required Packages

• numpy - Numerical computing
• scikit-learn - Machine learning
• pandas - Data processing
• dilithium-py - Post-quantum signatures
• kyber-py - Post-quantum key exchange
• cryptography - Symmetric encryption
• phe - Paillier homomorphic encryption
• matplotlib - Visualization

---

## Quick Start

### Run Training with Homomorphic Encryption

cd src
python main.py

### Run Training WITHOUT Homomorphic Encryption (Faster)

Edit src/main.py (line 126):
USE_HE = False

Then run:
python main.py

### Run Full Benchmark Suite

python benchmark.py

---

## Benchmarking

Run comprehensive performance benchmarks:

python benchmark.py

Metrics Collected:
• Time per round (training, encryption, aggregation)
• Communication overhead (payload sizes)
• Accuracy comparison (plaintext vs HE)
• Overhead percentages

Output Files:
```
results/
├── metrics/
│   └── benchmark_results.json
└── plots/
    └── benchmark_comparison.png
```
---

## Project Structure
```
quantum-safe-hybrid-ml/
├── src/
│   ├── main.py                      # Main training script
│   ├── benchmark.py                 # Performance benchmarking
│   ├── federated_server.py          # Server implementation
│   ├── federated_client.py          # Client implementation
│   ├── pqc_auth.py                  # Dilithium authentication
│   ├── pqc_channel.py               # Kyber secure channels
│   ├── homomorphic_encryption.py    # Paillier HE wrapper
│   └── data_utils.py                # Dataset utilities
├── results/
│   ├── metrics/                     # JSON result files
│   ├── models/                      # Trained model checkpoints
│   └── plots/                       # Visualization outputs
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Security Analysis

### Cryptographic Primitives

1. CRYSTALS-Kyber (ML-KEM-768)
   - NIST-standardized post-quantum KEM
   - Security: 192-bit quantum resistance
   - Use: Establishing session keys

2. CRYSTALS-Dilithium (ML-DSA-44)
   - NIST-standardized post-quantum signatures
   - Security: 128-bit quantum resistance
   - Use: Client authentication, update integrity

3. Paillier Homomorphic Encryption
   - Key size: 2048 bits
   - Property: Additive homomorphism
   - Use: Privacy-preserving aggregation

### Threat Model

Assumptions:
• Honest-but-curious server
• Secure local client environments
• Authenticated communication channels

Protections:
✓ Confidentiality (AES-256 + Paillier)
✓ Integrity (Dilithium signatures)
✓ Authentication (Public key infrastructure)
✓ Quantum resistance (PQC algorithms)

Limitations:
✗ Byzantine fault tolerance (no malicious client detection)
✗ Differential privacy (no formal privacy budget)
✗ Model poisoning attacks (trusts client updates)

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

git checkout -b feature/your-feature-name
make changes
python -m pytest tests/
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

### Code Style

• Follow PEP 8 conventions
• Use type hints where applicable
• Add docstrings to functions/classes
• Write unit tests for new features

### Suggested Improvements

- Add differential privacy mechanisms
- Implement Byzantine-robust aggregation
- Support for neural networks (CNN, LSTM)
- Multi-server federation
- Real-time client selection
- Adaptive security levels

---

## License

This project is licensed under the MIT License.

---

## References

### Standards & Specifications

1. NIST PQC Standardization: https://csrc.nist.gov/projects/post-quantum-cryptography
2. FIPS 204 (ML-DSA): CRYSTALS-Dilithium specification
3. FIPS 203 (ML-KEM): CRYSTALS-Kyber specification

### Related Projects

- python-paillier: https://github.com/n1analytics/python-paillier
  ---
- dilithium-py: https://github.com/GiacomoSorbi/dilithium-py
  ---
- kyber-py: https://github.com/GiacomoSorbi/kyber-py

---

## Contact & Support

• GitHub Issues: https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml/issues
• GitHub Discussions: https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml/discussions
• Email: samuel.joseph2k05@gmail.com
