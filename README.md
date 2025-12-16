# Quantum-Safe Federated Learning System
A privacy-preserving federated learning system with post-quantum cryptographic security and homomorphic encryption.

## Features
- CRYSTALS-Dilithium (ML-DSA-44): Digital signatures
- CRYSTALS-Kyber (ML-KEM-768): Quantum-safe key exchange
- AES-256-GCM: Symmetric encryption
- Paillier Homomorphic Encryption: Privacy-preserving aggregation

## Installation
```
git clone https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml.git
cd quantum-safe-hybrid-ml
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```
## Quick Start
cd src
python main.py

## Benchmarking
python benchmark.py

## Project Structure
```
quantum-safe-hybrid-ml/
├── src/
│   ├── main.py
│   ├── benchmark.py
│   ├── federated_server.py
│   ├── federated_client.py
│   ├── pqc_auth.py
│   ├── pqc_channel.py
│   ├── homomorphic_encryption.py
│   └── data_utils.py
├── results/
│   ├── metrics/
│   ├── models/
│   └── plots/
├── requirements.txt
└── README.md
```
## Results
- Accuracy: ~79.4% on Adult Income test set
- Training: Converges in 3-5 rounds
- Privacy: 100% - server never sees raw data

## Security
- Post-Quantum Cryptography: Protected against quantum attacks
- Homomorphic Encryption: Server aggregates without seeing plaintext
- Federated Learning: Local data remains on devices
