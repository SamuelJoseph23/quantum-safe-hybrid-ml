"""
Real PQC-FL Implementation (Phase 3A):
- Real Data (Adult Income)
- Real Model (Logistic Regression via SGD)
- PQC Authentication & Secure Channels active
- Paillier Homomorphic Encryption for privacy-preserving aggregation
- **Differential Privacy (LDP) for update protection**
"""
import numpy as np
import json
import os
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from federated_server import FederatedServer
from federated_client import FederatedClient
from differential_privacy import DPAnalyzer
from data_utils import load_and_preprocess_data

# --- Differential Privacy Configuration ---
DP_CONFIG = {
    "enabled": True,
    "epsilon": 2.0,          # Privacy Budget (Lower = More Private)
    "clipping_norm": 1.5     # Max L2 norm of update
}

def train_local_model_sklearn(global_weights, global_intercept, X_local, y_local):
    """Train a local SGDClassifier starting from global parameters."""
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, 
                        max_iter=1, tol=None, learning_rate='constant', eta0=0.01,
                        random_state=42)
    
    classes = np.array([0, 1])
    clf.partial_fit(X_local[0:1], y_local[0:1], classes=classes)
    
    # Set global weights
    clf.coef_ = np.array(global_weights).reshape(1, -1)
    clf.intercept_ = np.array(global_intercept).reshape(1,)
    
    # Train
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {filename}")

def save_model(model_params, filename='../results/models/final_model.npz'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, **model_params)
    print(f"✓ Model saved to {filename}")

def plot_results(results, filename='../results/plots/accuracy_plot.png'):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        rounds = [r['round'] for r in results['rounds']]
        accuracies = [r['accuracy'] for r in results['rounds']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        title = 'Quantum-Safe FL with Differential Privacy'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if len(accuracies) > 0:
            plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {filename}")
        plt.close()
    except ImportError:
        pass

def main():
    # ------------------------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------------------------
    NUM_CLIENTS = 3
    client_data, test_data = load_and_preprocess_data(NUM_CLIENTS)
    X_test, y_test = test_data
    n_features = X_test.shape[1]

    # ------------------------------------------------------------------
    # 2. Initialize Server
    # ------------------------------------------------------------------
    USE_HE = True
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
    
    # DP Analyzer to track privacy budget
    dp_analyzer = DPAnalyzer()
    
    for cid in range(NUM_CLIENTS):
        client_id = f"client_{cid+1}"
        client = FederatedClient(
            client_id=client_id, 
            security_level=2,
            use_dp=DP_CONFIG["enabled"],
            dp_epsilon=DP_CONFIG["epsilon"]
        )
        clients.append(client)
        
        # Register
        reg_data = client.register_with_server(None)
        resp = server.register_client(reg_data['client_id'], reg_data['public_key'])
        
        registry_info[client_id] = {
            "server_kyber_pk": resp["server_kyber_public_key"],
            "session_key": None
        }
        
        if USE_HE and "he_public_key" in resp:
            client.set_he_public_key(resp["he_public_key"])

    # ------------------------------------------------------------------
    # 4. Federated Training Loop
    # ------------------------------------------------------------------
    NUM_ROUNDS = 5
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "differential_privacy": DP_CONFIG
        },
        "rounds": []
    }
    
    print(f"\nStarting training for {NUM_ROUNDS} rounds...")
    print(f"Encryption: {'Paillier HE' if USE_HE else 'None'}")
    print(f"Differential Privacy: {'ENABLED' if DP_CONFIG['enabled'] else 'DISABLED'}")
    if DP_CONFIG['enabled']:
        print(f" > Epsilon: {DP_CONFIG['epsilon']}, Clip: {DP_CONFIG['clipping_norm']}")
    print("-" * 50)

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Round {rnd} ===")
        
        global_params = server.get_global_model()
        global_W = global_params["W"]
        global_b = global_params["b"]
        
        round_privacy_spent = 0
        
        # --- Client Side ---
        for i, client in enumerate(clients):
            cid = client.client_id
            X_local, y_local = client_data[i]
            
            # 1. Local Training
            new_W, new_b = train_local_model_sklearn(global_W, global_b, X_local, y_local)
            
            # 2. Apply Differential Privacy (Phase 3A)
            if DP_CONFIG["enabled"]:
                # Weights
                final_W = client.apply_differential_privacy(
                    new_W, global_W, DP_CONFIG["clipping_norm"]
                )
                # Bias (using same clip norm)
                final_b = client.apply_differential_privacy(
                    new_b, global_b, DP_CONFIG["clipping_norm"]
                )
                # Account once per update (not per tensor)
                client.account_privacy_step()
                
                # Track privacy spent (just from one client as representative)
                if i == 0:
                    round_privacy_spent = client.dp_engine.privacy_spent
            else:
                final_W, final_b = new_W, new_b
            
            # 3. Prepare Update
            model_update = {
                "client_id": cid,
                "model_params": {"W": final_W, "b": final_b},
                "num_samples": len(X_local)
            }
            
            # 4. Secure Send
            info = registry_info[cid]
            info.setdefault("counter", 0)
            info["counter"] += 1
            payload = client.secure_send_update(
                model_update, 
                info["server_kyber_pk"], 
                info["session_key"],
                use_he=USE_HE,
                msg_counter=info["counter"],
            )
            registry_info[cid]["session_key"] = payload["session_key"]
            server.receive_update(payload)
            
        # --- Server Side ---
        new_global_params = server.finalize_round()
        
        # Evaluation
        acc = evaluate_global_model(new_global_params["W"], new_global_params["b"], X_test, y_test)
        print(f"Round {rnd} Global Accuracy: {acc:.4f}")
        
        # Record stats
        results["rounds"].append({"round": rnd, "accuracy": float(acc)})
        
        if DP_CONFIG["enabled"]:
            dp_analyzer.record_round(
                rnd, DP_CONFIG["epsilon"], acc, round_privacy_spent, 0.0
            )

    print("\n" + "="*70)
    print("Training complete.")
    
    save_results(results)
    save_model(server.get_global_model())
    plot_results(results)
    
    if DP_CONFIG["enabled"]:
        dp_analyzer.save_report()
        print(f"Final Privacy Spent: {round_privacy_spent:.2f}")

if __name__ == "__main__":
    main()
