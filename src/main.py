"""
Real PQC-FL Implementation (Phase 2):
- Real Data (Adult Income)
- Real Model (Logistic Regression via SGD)
- PQC Authentication & Secure Channels active
- **Paillier Homomorphic Encryption for privacy-preserving aggregation**
"""

import numpy as np
import json
import os
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from federated_server import FederatedServer
from federated_client import FederatedClient
from data_utils import load_and_preprocess_data


def train_local_model_sklearn(global_weights, global_intercept, X_local, y_local):
    """Train a local SGDClassifier starting from global parameters."""
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, 
                        max_iter=1, tol=None, learning_rate='constant', eta0=0.01,
                        random_state=42)
    
    classes = np.array([0, 1])
    clf.partial_fit(X_local[0:1], y_local[0:1], classes=classes)
    
    clf.coef_ = np.array(global_weights).reshape(1, -1)
    clf.intercept_ = np.array(global_intercept).reshape(1,)
    
    clf.partial_fit(X_local, y_local)
    
    return clf.coef_, clf.intercept_


def evaluate_global_model(server_weights, server_intercept, X_test, y_test):
    """Evaluate the global model on the hold-out test set."""
    clf = SGDClassifier(loss='log_loss', random_state=42)
    clf.partial_fit(X_test[0:1], y_test[0:1], classes=np.array([0, 1]))
    
    clf.coef_ = np.array(server_weights).reshape(1, -1)
    clf.intercept_ = np.array(server_intercept).reshape(1,)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def save_results(results, filename='../results/metrics/training_results.json'):
    """Save training results to JSON file in metrics folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {filename}")


def save_model(model_params, filename='../results/models/final_model.npz'):
    """Save final model parameters to models folder"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, **model_params)
    print(f"✓ Model saved to {filename}")


def plot_results(results, filename='../results/plots/accuracy_plot.png'):
    """Create accuracy visualization and save to plots folder"""
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        rounds = [r['round'] for r in results['rounds']]
        accuracies = [r['accuracy'] for r in results['rounds']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        plt.title('Quantum-Safe Federated Learning with Paillier HE', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])
        
        for i, (r, a) in enumerate(zip(rounds, accuracies)):
            plt.annotate(f'{a:.4f}', (r, a), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {filename}")
        plt.close()
    except ImportError:
        print("⚠ matplotlib not installed. Skipping plot generation.")


def main():
    # ------------------------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------------------------
    NUM_CLIENTS = 3
    client_data, test_data = load_and_preprocess_data(NUM_CLIENTS)
    X_test, y_test = test_data
    
    n_features = X_test.shape[1]

    # ------------------------------------------------------------------
    # 2. Initialize Server with HE ENABLED
    # ------------------------------------------------------------------
    USE_HE = True  # Set to False for plaintext comparison
    server = FederatedServer(security_level=2, use_he=USE_HE)
    
    initial_model = {
        "W": np.zeros((1, n_features), dtype=float),
        "b": np.zeros((1,), dtype=float),
    }
    server.set_initial_model(initial_model)

    # ------------------------------------------------------------------
    # 3. Create and Register Clients
    # ------------------------------------------------------------------
    clients = []
    registry_info = {}

    for cid in range(NUM_CLIENTS):
        client_id = f"client_{cid+1}"
        client = FederatedClient(client_id=client_id, security_level=2)
        clients.append(client)

        # Register
        reg_data = client.register_with_server(None)
        resp = server.register_client(reg_data['client_id'], reg_data['public_key'])
        
        registry_info[client_id] = {
            "server_kyber_pk": resp["server_kyber_public_key"],
            "session_key": None
        }
        
        # Give client the HE public key if HE is enabled
        if USE_HE and "he_public_key" in resp:
            client.set_he_public_key(resp["he_public_key"])

    # ------------------------------------------------------------------
    # 4. Federated Training Loop
    # ------------------------------------------------------------------
    NUM_ROUNDS = 5
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "security_level": 2,
        "dataset": "Adult Income",
        "model": "Logistic Regression (SGD)",
        "homomorphic_encryption": USE_HE,
        "pqc_schemes": {
            "authentication": "CRYSTALS-Dilithium (ML-DSA-44)",
            "key_exchange": "CRYSTALS-Kyber (ML-KEM-768)",
            "encryption": "AES-256-GCM",
            "homomorphic": "Paillier (phe)" if USE_HE else "None"
        },
        "rounds": []
    }
    
    print(f"\nStarting training for {NUM_ROUNDS} rounds...")
    print(f"Homomorphic Encryption: {'ENABLED (Paillier)' if USE_HE else 'DISABLED'}\n")
    
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Round {rnd} ===")
        global_params = server.get_global_model()
        global_W = global_params["W"]
        global_b = global_params["b"]

        # --- Client Side ---
        for i, client in enumerate(clients):
            cid = client.client_id
            X_local, y_local = client_data[i]
            
            # Local Training
            new_W, new_b = train_local_model_sklearn(global_W, global_b, X_local, y_local)
            
            # Prepare Update
            model_update = {
                "client_id": cid,
                "encrypted_gradients": {"W": new_W, "b": new_b},
                "num_samples": len(X_local)
            }
            
            # Secure Send (PQC + HE)
            info = registry_info[cid]
            payload = client.secure_send_update(
                model_update, 
                info["server_kyber_pk"], 
                info["session_key"],
                use_he=USE_HE
            )
            
            registry_info[cid]["session_key"] = payload["session_key"]
            
            # Server Receive
            server.receive_update(payload)

        # --- Server Side ---
        new_global_params = server.finalize_round()
        
        # Evaluation
        acc = evaluate_global_model(new_global_params["W"], new_global_params["b"], X_test, y_test)
        print(f"Round {rnd} Global Accuracy: {acc:.4f}")
        
        results["rounds"].append({
            "round": rnd,
            "accuracy": float(acc)
        })

    print("\n" + "="*70)
    print("Training complete.")
    print("="*70)
    
    # Save results
    save_results(results)
    save_model(server.get_global_model())
    plot_results(results)
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Homomorphic Encryption: {'✓ ENABLED (Paillier)' if USE_HE else '✗ DISABLED'}")
    print(f"   Initial Accuracy: {results['rounds'][0]['accuracy']:.4f}")
    print(f"   Final Accuracy:   {results['rounds'][-1]['accuracy']:.4f}")
    print(f"   Improvement:      {(results['rounds'][-1]['accuracy'] - results['rounds'][0]['accuracy']):.4f}")
    print(f"   Peak Accuracy:    {max(r['accuracy'] for r in results['rounds']):.4f}")
    print(f"\n📁 Files saved:")
    print(f"   - Metrics: results/metrics/training_results.json")
    print(f"   - Model:   results/models/final_model.npz")
    print(f"   - Plot:    results/plots/accuracy_plot.png")


if __name__ == "__main__":
    main()
