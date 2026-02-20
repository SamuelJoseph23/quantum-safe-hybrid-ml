# Quantum-Safe Hybrid Federated Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security Policy](https://img.shields.io/badge/Security-Policy-red.svg)](SECURITY.md)

A robust, multi-layered privacy-preserving framework for Federated Learning (FL). This project integrates **Post-Quantum Cryptography (PQC)**, **Homomorphic Encryption (HE)**, and **Differential Privacy (DP)** to provide end-to-end security for decentralized machine learning.

## 🚀 Overview

Collaborative training of machine learning models often requires sharing sensitive client data, which introduces significant privacy risks. This framework addresses these vulnerabilities by establishing a "Zero-Trust" environment where:

*   **Identities are Verified**: Using quantum-resistant Dilithium signatures.
*   **Channels are Secured**: Using Kyber-based key encapsulation to protect against quantum-era eavesdropping.
*   **Aggregation is Blind**: Using Paillier Homomorphic Encryption so the server never sees raw weights.
*   **Users are Anonymized**: Using Local Differential Privacy (LDP) to prevent individual data reconstruction.

## ✨ Key Features

- **Standardized PQC**: Implementation of NIST-standardized algorithms (ML-KEM/Kyber and ML-DSA/Dilithium).
- **Zero-Knowledge Aggregation**: Additive homomorphic encryption ensures only aggregated results are ever visible to the server.
- **Formal Privacy Guarantees**: Noise injection via the Laplace mechanism with configurable privacy budgets (Epsilon).
- **Cyber-SOC Dashboard**: An interactive React-based dashboard for real-time threat monitoring and system visualization.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Node.js (for the Web GUI)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml.git
   cd quantum-safe-hybrid-ml
   ```

2. **Backend Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup** (Optional):
   ```bash
   cd web-gui
   npm install
   ```

## 📘 Quick Start

### Core Training (CLI)
To run a headless 10-round federated training session on the Adult Income dataset:
```bash
python src/main.py
```

### Interactive Dashboard
To launch the Cyber-SOC visual demo:
1. **Start Backend**: `python src/api.py`
2. **Start Frontend**: `cd web-gui && npm run dev`
3. Access at `http://localhost:5173`

## 🗺️ Roadmap

- [ ] **Neural Network Support**: Expand from Logistic Regression to deep learning architectures.
- [ ] **Byzantine Resilience**: Implement detection for malicious or poisoned client updates.
- [ ] **Communication Optimization**: Integrate SIMD batching for Homomorphic Encryption to reduce latency.
- [ ] **TEE Integration**: Support for Trusted Execution Environments (Intel SGX) for secure server-side compute.

## 🤝 Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for setup instructions and code standards.

## 🛡️ Security

If you discover a security vulnerability, please refer to our [SECURITY.md](SECURITY.md).

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
