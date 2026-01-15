# My Quantum-Safe Federated Learning System

I have developed this privacy-preserving federated learning system which combines post-quantum cryptography, homomorphic encryption, and secure multi-party computation for decentralized machine learning.

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

I implemented this production-grade federated learning system to address data privacy and quantum-safe security in machine learning. My system enables multiple clients to collaboratively train a machine learning model while:

• **Keeping all data local** - No data leaves the client devices
• **Protecting against eavesdropping** - I use quantum-resistant encryption
• **Preserving privacy during aggregation** - I use homomorphic encryption
• **Layered Data Protection** - I integrated Local Differential Privacy (LDP)
• **Verifying authenticity** - I use post-quantum digital signatures
• **Ensuring integrity** - I enforce secure channels and signature verification

### Key Problem Statement

I designed this to solve the vulnerabilities in traditional federated learning systems:

1. **Quantum Attacks** - Current cryptography (RSA, ECDSA) will be broken by quantum computers
2. **Privacy Leakage** - Model updates can leak sensitive information about training data
3. **Man-in-the-Middle Attacks** - Unencrypted communications can be intercepted
4. **Authentication Failure** - Clients can impersonate servers or vice versa

I solve these problems using a hybrid approach:
• **CRYSTALS-Kyber** (Quantum-resistant key exchange)
• **CRYSTALS-Dilithium** (Quantum-resistant signatures)
• **Paillier Cryptosystem** (Additive homomorphic encryption)
• **Laplace Mechanism** (Local Differential Privacy)
• **AES-256-GCM** (Symmetric encryption for channels)

---

## Features

### Post-Quantum Cryptography (NIST Standardized)

#### CRYSTALS-Kyber (ML-KEM-768)
• **Type**: Key Encapsulation Mechanism (KEM)
• **Security**: 192-bit quantum resistance
• **Purpose**: I use this to establish quantum-safe session keys
• **Standardization**: NIST FIPS 203

#### CRYSTALS-Dilithium (ML-DSA-44)
• **Type**: Digital Signature Algorithm
• **Security**: 128-bit quantum resistance
• **Purpose**: I use this to authenticate clients and verify update integrity
• **Standardization**: NIST FIPS 204

### Privacy-Preserving Aggregation

#### Paillier Homomorphic Encryption
• **Type**: Additive Homomorphic Encryption
• **Key Size**: 2048 bits
• **Property**: E(m1) + E(m2) = E(m1 + m2)

**Advantage**: My server never sees individual client gradients, only encrypted values.

### Local Differential Privacy (LDP)

#### Laplace Mechanism
• **Type**: Epsilon-Differential Privacy
• **Privacy Budget (Epsilon)**: 2.0 - 10.0 (Configurable)
• **Clipping**: L2-norm gradient clipping
• **Accounting**: I built in a privacy budget tracker (DPAnalyzer)
• **Purpose**: Protect individual data samples from membership inference attacks.

---

## Machine Learning Capabilities

### Model Architecture
• **Algorithm**: Logistic Regression (SGD-based)
• **Framework**: scikit-learn
• **Features**: **One-Hot Encoded** (38+ features)
• **Optimizer**: Stochastic Gradient Descent (SGD)
• **Regularization**: L2 (Ridge)

### Dataset: Adult Income Census
• **Type**: Binary Classification
• **Preprocessing**: I implemented robust scaling + One-Hot Encoding
• **Features**: Age, Workclass, Education, Occupation, Relationship, Race, Gender, etc.
• **Improved Accuracy**: **75% - 76%** (with privacy), **84%** (No-DP baseline)
• **Convergence**: 5-10 rounds

---

## Installation

### Prerequisites
• Python 3.9 or higher
• pip (Python package manager)
• Virtual environment (highly recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml.git
cd quantum-safe-hybrid-ml
```

### Step 2: Create Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Run My Training Loop
```bash
cd src
python main.py
```

### Run My Full Benchmark Suite
```bash
python benchmark.py
```

---

## Project Structure
```
quantum-safe-hybrid-ml/
├── src/
│   ├── main.py                      # My main training script (Tuned for ~75% Acc)
│   ├── benchmark.py                 # My performance benchmarking suite
│   ├── federated_server.py          # Server implementation with HE aggregation
│   ├── federated_client.py          # Client implementation with DP & PQC
│   ├── differential_privacy.py      # Laplace Mechanism & Privacy Accounting
│   ├── pqc_auth.py                  # Dilithium authentication wrapper
│   ├── pqc_channel.py               # Kyber-based secure channel establishment
│   ├── homomorphic_encryption.py    # Paillier HE wrapper for gradients
│   └── data_utils.py                # My robust data pipeline with OHE
├── results/
│   ├── metrics/                     # JSON result files
│   ├── models/                      # Trained model checkpoints
│   └── plots/                       # Visualization outputs
├── requirements.txt
└── README.md
```

---

## Security Analysis

### My Cryptographic Choice
1. **CRYSTALS-Kyber**: I use this for establishing session keys (NIST FIPS 203).
2. **CRYSTALS-Dilithium**: I use this for client authentication and update integrity (NIST FIPS 204).
3. **Paillier HE**: I use this for privacy-preserving aggregation without decryption.
4. **Local DP**: I implemented the Laplace mechanism for formal privacy guarantees.

### Protections
✓ Confidentiality (AES-256 + Paillier)
✓ Integrity (Dilithium signatures)
✓ Authentication (Kyber + Dilithium PKI)
✓ Quantum resistance (Standardized PQC algorithms)
✓ **Privacy Guarantees** (Local Differential Privacy)

### Current Limitations
✗ Byzantine fault tolerance (no malicious client detection)
✗ Model poisoning attacks (trusts client updates)
✗ Communication efficiency (HE increases payload size)

---

## Contributing

I welcome contributions! Please follow these guidelines:

### Development Setup
1. `git checkout -b feature/your-feature-name`
2. Make your changes
3. `python -m pytest tests/`
4. `git commit -m "Add: your feature description"`

---

## License
I have licensed this project under the MIT License.

---

## References

### Standards & Specifications
1. NIST PQC Standardization: [csrc.nist.gov](https://csrc.nist.gov/projects/post-quantum-cryptography)
2. FIPS 204 (ML-DSA): CRYSTALS-Dilithium specification
3. FIPS 203 (ML-KEM): CRYSTALS-Kyber specification

### Related Projects
- **python-paillier**: [github.com/n1analytics/python-paillier](https://github.com/n1analytics/python-paillier)
- **dilithium-py**: [github.com/GiacomoSorbi/dilithium-py](https://github.com/GiacomoSorbi/dilithium-py)
- **kyber-py**: [github.com/GiacomoSorbi/kyber-py](https://github.com/GiacomoSorbi/kyber-py)
