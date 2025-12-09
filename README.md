# Quantum‑Safe Hybrid ML
Quantum‑safe machine learning project using a hybrid encryption approach (post‑quantum homomorphic + AES) on the UCI Adult Income dataset. Includes preprocessing, baseline model, and an encrypted pipeline for privacy‑preserving training and inference on sensitive data.

- Repository: SamuelJoseph23/quantum-safe-hybrid-ml
- Language: Python (100%)
- Description: Prototype and reproducible experiments demonstrating a hybrid approach combining a post‑quantum homomorphic layer with symmetric AES encryption to enable privacy-preserving ML training and inference on tabular data.

Table of contents
- Overview
- Key features
- Repository layout
- Quick start
- Installation
- Dataset
- Preprocessing
- Baseline model (cleartext)
- Encrypted pipeline (hybrid: post‑quantum homomorphic + AES)
- Running experiments
- Evaluation & metrics
- Security considerations
- Performance & limitations
- Contributing
- License
- Citation & contact

Overview
This repository demonstrates a proof‑of‑concept workflow for training and evaluating a model on sensitive tabular data while minimizing exposure of raw features. The hybrid approach separates concerns:
- Symmetric encryption (AES) for efficient secure storage / transport of large portions of data.
- A prototype post‑quantum homomorphic layer that enables computations on encrypted features necessary for model training/inference without revealing plaintext to the compute party.

Key features
- Reproducible preprocessing pipeline for the UCI Adult Income dataset.
- Baseline cleartext model for reference (data science + ML).
- Encrypted training and inference pipelines showing integration of homomorphic and AES layers.
- Scripts to evaluate accuracy, privacy trade‑offs, and runtime/throughput.
- Configuration-driven experiments (reuse and compare multiple setups).

Repository layout
- README.md — this file
- requirements.txt — Python dependencies (generate if not present)
- src/
  - data/
    - download_data.py — download and verify the Adult dataset
    - preprocess.py — preprocessing steps and feature engineering
  - baseline/
    - train_baseline.py — trains cleartext baseline models
    - evaluate_baseline.py
  - encrypted/
    - encrypt_data.py — AES + key management helpers
    - homomorphic_layer.py — prototype post‑quantum homomorphic primitives / wrappers
    - train_encrypted.py — training pipeline using encrypted features
    - infer_encrypted.py — inference pipeline for encrypted inputs
  - utils/
    - config.py — experiment configuration utilities
    - metrics.py — evaluation metrics
  - experiments/
    - run_experiment.py — runs end‑to‑end experiments and logs results
- notebooks/ — optional EDA and demo notebooks
- docs/ — design notes and security analysis

Quick start
1. Clone the repo:
   git clone https://github.com/SamuelJoseph23/quantum-safe-hybrid-ml.git
   cd quantum-safe-hybrid-ml

2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt

3. Download and preprocess data:
   python src/data/download_data.py --output data/raw
   python src/data/preprocess.py --input data/raw --output data/processed

4. Train a baseline model (cleartext):
   python src/baseline/train_baseline.py --config configs/baseline.yaml

5. Run the encrypted pipeline (demo):
   python src/encrypted/encrypt_data.py --input data/processed/train.csv --out data/encrypted/train.enc --key keys/data.key
   python src/encrypted/train_encrypted.py --config configs/encrypted.yaml --key keys/data.key

Installation and dependencies
- Python 3.9+ (3.10+ recommended)
- It's recommended to use a virtual environment (venv, conda).
- Typical dependencies (examples — populate requirements.txt):
  - numpy, pandas, scikit-learn
  - PyCryptodome (for AES)
  - a homomorphic encryption research/prototype library or local implementation (see design)
  - tqdm, joblib, pyyaml

If you want, I can create a concrete requirements.txt with pinned versions after you confirm which HE library you intend to use (or if you're using an in-repo prototype).

Dataset
- Source: UCI Adult Income dataset (also known as "Census Income")
- Goal: Predict whether income >50K (binary classification)
- Scripts: src/data/download_data.py and src/data/preprocess.py
- Preprocessing steps:
  - Missing value handling
  - Categorical encoding (one-hot / ordinal)
  - Numeric scaling (StandardScaler or MinMax)
  - Train/validation/test splits
  - Optional feature selection

Preprocessing notes
Preprocessing is deterministic and configurable via YAML configs in configs/. Preprocessing outputs both:
- A cleartext processed CSV (for baseline experiments)
- Serialized feature transformation pipeline (scikit-learn pipeline or joblib) used by encrypted pipeline to ensure transformations match exactly between parties.

Baseline model (cleartext)
- Implemented in src/baseline/train_baseline.py
- Example models: logistic regression, random forest, simple MLP
- Use baseline results to measure privacy-utility tradeoffs introduced by encryption
- Saved artifacts: trained model, vectorizer/transformer, evaluation reports

Encrypted pipeline (hybrid approach)
High-level approach:
1. Client-side:
   - Preprocess raw personal data to feature vector.
   - Encrypt feature vector using a hybrid scheme:
     - AES for bulk encryption and storage/transfer.
     - A separate homomorphic encryption/prototype layer on selected features or summary statistics necessary for computing model updates or inference.
   - Store or transmit encrypted blobs and metadata (which features are under HE).
2. Server-side (untrusted compute):
   - Operates on homomorphically encrypted elements where necessary (e.g., inner products or linear transforms).
   - Handles AES-encrypted blobs only after authorized decryption agreement (or never, if the goal is non-colluding compute).
3. Result handling:
   - Server returns homomorphically encrypted partial results for client to decrypt and finish inference, or model updates are aggregated in encrypted space.

Important: the project intentionally uses a prototype post‑quantum homomorphic layer. Practical production use requires vetted libraries, formal security proofs, and threat modeling.

Running experiments
- Use configs in configs/ to define experiment parameters (model, training hyperparams, encryption choices).
- Example to run an experiment:
  python src/experiments/run_experiment.py --config configs/experiment_hybrid.yaml --out results/exp1
- Typical outputs:
  - results/exp1/metrics.json
  - results/exp1/model.joblib
  - results/exp1/logs/

Evaluation & metrics
- Primary metric: classification accuracy / ROC AUC
- Secondary metrics: precision, recall, F1, calibration
- Privacy/utility metrics:
  - Accuracy degradation compared to baseline
  - Additional measurements (e.g., leakage tests) can be added under docs/security_tests.md
- Performance metrics:
  - Runtime (seconds) and memory usage for cleartext vs hybrid
  - Throughput (inferences per second) for encrypted inference

Security considerations
- AES keys: key management and secure storage are outside the scope of this repo — use a secure KMS in production.
- Post‑quantum homomorphic primitives in this repo are prototypes; they should not be considered production‑ready or fully secure.
- Threat model: clear description in docs/threat_model.md — identify trusted parties, adversarial assumptions, and trust boundaries.
- Side‑channel and operational security are not fully covered here.

Performance & limitations
- Homomorphic operations are computationally expensive compared to cleartext. Expect significantly slower training and inference for HE‑protected features.
- The hybrid approach tries to limit HE usage to only the minimal necessary computations to reduce cost.
- This repo is for research/prototyping — not production deployment.

Reproducibility
- Use deterministic random seeds via config to reproduce runs.
- Save transform pipelines and random seeds alongside models in results/ to enable exact reproduction.

Contributing
Contributions, issues, and suggestions are welcome.
- Check existing issues and open a new issue for bugs/features.
- Fork the repo and create a feature branch for PRs.
- Include tests for new features where appropriate.
- Add or update docs when you modify pipeline behavior or security assumptions.

Suggested next steps I can help generate (pick any):
- A complete requirements.txt with pinned versions.
- Example config YAML files for baseline and encrypted experiments.
- A CONTRIBUTING.md or SECURITY.md with disclosure guidelines.
- Unit tests for core modules (preprocess, encrypt, homomorphic layer).

License
This project uses the MIT License — see LICENSE for details. If you prefer a different license, let me know and I’ll update the file.

Citation & contact
If you use this work in research, please cite or mention the repository. For questions or collaboration: open an issue or contact the repository owner (SamuelJoseph23).

Acknowledgements
- Based on standard datasets and homomorphic encryption literature.
- This repository is intended for research and educational exploration of privacy‑preserving ML and post‑quantum approaches.
