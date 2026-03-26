# Quantum-Safe Hybrid Federated Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI WebSockets](https://img.shields.io/badge/FastAPI-WebSockets-009688.svg)](https://fastapi.tiangolo.com/)

A robust, multi-layered privacy-preserving framework for Federated Learning (FL). This project integrates **Post-Quantum Cryptography (PQC)**, **Homomorphic Encryption (HE)**, and **Differential Privacy (DP)** to provide end-to-end security and high-performance real-time analytics for decentralized machine learning.

## Core Advancements

Following a comprehensive architectural review, this framework implements several bleeding-edge paradigms:

*   **Identities are Verified**: Using quantum-resistant ML-DSA (Dilithium) signatures.
*   **Channels are Secured**: Using Kyber-768 key encapsulation against Shor's algorithm-equipped adversaries.
*   **Aggregation is Blind (Fast)**: Utilizing parallelized `ProcessPoolExecutor` Paillier Homomorphic Encryption to leverage multi-core CPUs, preventing server access to raw updates without halting performance.
*   **Users are Anonymized**: Local Differential Privacy (LDP) is correctly scaled using mathematical bounds for true (epsilon, delta) guarantees on raw client updates.
*   **Decoupled Real-Time Architecture**: A Python `FastAPI` backend powered by `BackgroundTasks` streaming real-time simulation events to a `React` frontend via `WebSockets`.
*   **Byzantine Fault Tolerance**: Supports multidimensional median-based aggregation over plaintext domains, effectively isolating outlier clients or MITM poison attacks organically.

## Interactive Cyber-SOC Dashboard

The project includes an interactive 3-Node frontend (`web-gui/`) that connects directly into the FastAPI WebSocket event stream.

**Features of the Dashboard:**
- **Kyber Handshaking**
- **Simulation of MITM & Quantum Wavefront Attacks**
- **Real-Time Accuracy Charts**
- **Live Terminal Event Logs via WebSockets**

## Installation

### Prerequisites
- Python 3.10+
- Node.js

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

3. **Frontend Setup**:
   ```bash
   cd web-gui
   npm install
   ```

## Quick Start

### Core Training (CLI Backend)
To run a headless 10-round federated training session on the Adult Income dataset:
```bash
python src/main.py
```

### Interactive Dashboard (Real-Time WebSockets)
To launch the visual simulation:
1. **Start Backend**: 
   ```bash
   python src/api.py
   # Now runs on ws://127.0.0.1:8000/ws
   ```
2. **Start Frontend**: 
   ```bash
   cd web-gui && npm run dev
   # Access at http://localhost:5173
   ```

## Roadmap
- [ ] **Neural Network Support**: Expand from Logistic Regression to deep learning architectures.
- [ ] **TEE Integration**: Support for Trusted Execution Environments (Intel SGX) for secure server-side compute.

## Contributing
Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for setup instructions and code standards.

## Security
If you discover a security vulnerability, please refer to our [SECURITY.md](SECURITY.md).

## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
