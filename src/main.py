"""
Real PQC-FL Implementation (Phase 1):
- Real Data (Adult Income)
- Real Model (Logistic Regression via SGD)
- PQC Authentication & Secure Channels active
- Plaintext Aggregation (HE comes next)
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
    """
    Train a local SGDClassifier starting from global parameters.
    Returns the updated weights and intercept.
    """
    # Initialize model
    # loss='log_loss' gives Logistic Regression
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, 
                        max_iter=1, tol=None, learning_rate='constant', eta0=0.01,
                        random_state=42)
    
    # We need to call partial_fit to update existing weights.
    # But first, we must set the coefficients manually.
    # SGDClassifier expects coef_ to be shape (1, n_features) and intercept_ (1,)
    
    # Check if this is the first round (weights are all zeros)
    # If so, we just fit. If not, we set coefficients.
    classes = np.array([0, 1])
    
    # Warm-start logic:
    # Since scikit-learn doesn't easily let you set coefs BEFORE first fit,
    # we do a dummy partial_fit on the first sample to initialize shapes,
    # then overwrite the weights with global_weights.
    clf.partial_fit(X_local[0:1], y_local[0:1], classes=classes)
    
    clf.coef_ = np.array(global_weights).reshape(1, -1)
    clf.intercept_ = np.array(global_intercept).reshape(1,)
    
    # Now train on the whole local dataset
    clf.partial_fit(X_local, y_local)
    
    return clf.coef_, clf.intercept_


def evaluate_global_model(server_weights, server_intercept, X_test, y_test):
    """
    Evaluate the global model on the hold-out test set.
    """
    # Load into a temp model for prediction
    clf = SGDClassifier(loss='log_loss', random_state=42)
    clf.partial_fit(X_test[0:1], y_test[0:1], classes=np.array([0, 1])) # Init
    
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
        
        # Ensure plots directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        rounds = [r['round'] for r in results['rounds']]
        accuracies = [r['accuracy'] for r in results['rounds']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        plt.title('Quantum-Safe Federated Learning Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])
        
        # Add value labels on points
        for i, (r, a) in enumerate(zip(rounds, accuracies)):
            plt.annotate(f'{a:.4f}', (r, a), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {filename}")
        plt.close()
    except ImportError:
        print("⚠ matplotlib not installed. Skipping plot generation.")
        print("  Install with: pip install matplotlib")


def main():
    # ------------------------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------------------------
    NUM_CLIENTS = 3
    client_data, test_data = load_and_preprocess_data(NUM_CLIENTS)
    X_test, y_test = test_data
    
    # Determine number of features from data
    n_features = X_test.shape[1]

    # ------------------------------------------------------------------
    # 2. Initialize Server and Global Model
    # ------------------------------------------------------------------
    server = FederatedServer(security_level=2)
    
    # Initialize global model (Weights and Bias)
    # For Logistic Regression: W is (1, n_features), b is (1,)
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

    # ------------------------------------------------------------------
    # 4. Federated Training Loop
    # ------------------------------------------------------------------
    NUM_ROUNDS = 5
    
    # Initialize results tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "security_level": 2,
        "dataset": "Adult Income",
        "model": "Logistic Regression (SGD)",
        "pqc_schemes": {
            "authentication": "CRYSTALS-Dilithium (ML-DSA-44)",
            "key_exchange": "CRYSTALS-Kyber (ML-KEM-768)",
            "encryption": "AES-256-GCM"
        },
        "rounds": []
    }
    
    print(f"\nStarting training for {NUM_ROUNDS} rounds...")
    
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
            # In a real scenario, we'd send the *difference* (gradients), 
            # but sending new weights works for FedAvg too.
            model_update = {
                "client_id": cid,
                "encrypted_gradients": {"W": new_W, "b": new_b}, # Still plaintext here!
                "num_samples": len(X_local)
            }
            
            # Secure Send (PQC)
            info = registry_info[cid]
            payload = client.secure_send_update(
                model_update, 
                info["server_kyber_pk"], 
                info["session_key"]
            )
            
            # Update session key if it changed/was established
            registry_info[cid]["session_key"] = payload["session_key"]
            
            # Server Receive
            server.receive_update(payload)

        # --- Server Side ---
        # Aggregation
        new_global_params = server.finalize_round()
        
        # Evaluation
        acc = evaluate_global_model(new_global_params["W"], new_global_params["b"], X_test, y_test)
        print(f"Round {rnd} Global Accuracy: {acc:.4f}")
        
        # Log results
        results["rounds"].append({
            "round": rnd,
            "accuracy": float(acc)
        })

    print("\n" + "="*70)
    print("Training complete.")
    print("="*70)
    
    # ------------------------------------------------------------------
    # 5. Save All Results
    # ------------------------------------------------------------------
    
    # Save metrics
    save_results(results)
    
    # Save final model
    final_model = server.get_global_model()
    save_model(final_model)
    
    # Generate and save plot
    plot_results(results)
    
    # ------------------------------------------------------------------
    # 6. Print Summary
    # ------------------------------------------------------------------
    print("\n📊 Summary:")
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
