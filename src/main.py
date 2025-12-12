"""
Real PQC-FL Implementation (Phase 1):
- Real Data (Adult Income)
- Real Model (Logistic Regression via SGD)
- PQC Authentication & Secure Channels active
- Plaintext Aggregation (HE comes next)
"""

import numpy as np
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

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
