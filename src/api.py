"""
Cyber-SOC Backend API
FastAPI app serving the REAL Quantum-Safe Federated Learning pipeline.
Loads actual data, trains a real model, and reports live accuracy.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import os
import time

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager
from differential_privacy import DifferentialPrivacy
from federated_server import FederatedServer
from federated_client import FederatedClient
from data_utils import load_and_preprocess_data
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

app = FastAPI(title="Quantum-Safe FL API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
class PipelineState:
    """Holds the entire FL pipeline state across API calls."""
    def __init__(self):
        self.initialized = False
        self.server = None
        self.clients = []
        self.client_data = []
        self.test_data = None
        self.n_features = 0
        self.registry_info = {}

        # Training state
        self.current_round = 0
        self.accuracy_history = []

        # DP config
        self.dp_config = {
            "enabled": True,
            "epsilon": 10.0,
            "clipping_norm": 2.0,
        }

        # Security / UI state
        self.threat_level = "LOW"
        self.handshake_done = False

        # PQC modules (for handshake demo)
        self.pqc_channel = PQCSecureChannel(security_level=2)
        self.pqc_auth = PQCAuthenticator(security_level=2)


pipeline = PipelineState()


# ---------------------------------------------------------------------------
# Helper: sklearn train + evaluate (same logic as main.py)
# ---------------------------------------------------------------------------
def train_local_model(global_W, global_b, X_local, y_local):
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.0001,
        max_iter=1, tol=None, learning_rate="constant", eta0=0.01,
        random_state=42,
    )
    classes = np.array([0, 1])
    clf.partial_fit(X_local[0:1], y_local[0:1], classes=classes)
    clf.coef_ = np.array(global_W).reshape(1, -1)
    clf.intercept_ = np.array(global_b).reshape(1,)
    clf.partial_fit(X_local, y_local)
    return clf.coef_, clf.intercept_


def evaluate_model(weights, intercept, X_test, y_test):
    clf = SGDClassifier(loss="log_loss", random_state=42)
    clf.partial_fit(X_test[0:1], y_test[0:1], classes=np.array([0, 1]))
    clf.coef_ = np.array(weights).reshape(1, -1)
    clf.intercept_ = np.array(intercept).reshape(1,)
    return float(accuracy_score(y_test, clf.predict(X_test)))


# ---------------------------------------------------------------------------
# Startup: Load data + initialize FL pipeline
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_load_pipeline():
    """Load dataset and initialize server + clients on startup."""
    NUM_CLIENTS = 3

    print("[API] Loading Adult Income dataset...")
    pipeline.client_data, pipeline.test_data = load_and_preprocess_data(NUM_CLIENTS)
    X_test, _ = pipeline.test_data
    pipeline.n_features = X_test.shape[1]

    # Initialize server with HE
    pipeline.server = FederatedServer(security_level=2, use_he=True)
    initial_model = {
        "W": np.zeros((1, pipeline.n_features), dtype=float),
        "b": np.zeros((1,), dtype=float),
    }
    pipeline.server.set_initial_model(initial_model)

    # Create and register clients
    for cid in range(NUM_CLIENTS):
        client_id = f"client_{cid + 1}"
        client = FederatedClient(
            client_id=client_id,
            security_level=2,
            use_dp=pipeline.dp_config["enabled"],
            dp_epsilon=pipeline.dp_config["epsilon"],
        )
        pipeline.clients.append(client)

        reg_data = client.register_with_server(None)
        resp = pipeline.server.register_client(reg_data["client_id"], reg_data["public_key"])

        pipeline.registry_info[client_id] = {
            "server_kyber_pk": resp["server_kyber_public_key"],
            "session_key": None,
            "counter": 0,
        }

        if "he_public_key" in resp:
            client.set_he_public_key(resp["he_public_key"])

    pipeline.initialized = True
    print(f"[API] Pipeline ready. {NUM_CLIENTS} clients, {pipeline.n_features} features.")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class HandshakeResponse(BaseModel):
    public_key: str
    ciphertext: str
    session_key: str
    status: str


class PrivacyConfig(BaseModel):
    epsilon: float


class AttackRequest(BaseModel):
    attack_type: str
    parameter: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    return {
        "status": "online" if pipeline.initialized else "loading",
        "threat_level": pipeline.threat_level,
        "round": pipeline.current_round,
        "accuracy_history": pipeline.accuracy_history,
        "epsilon": pipeline.dp_config["epsilon"],
        "num_clients": len(pipeline.clients),
        "num_features": pipeline.n_features,
    }


@app.post("/api/reset")
async def reset_pipeline():
    """Reset training state (keeps data loaded)."""
    pipeline.current_round = 0
    pipeline.accuracy_history = []
    pipeline.threat_level = "LOW"
    pipeline.handshake_done = False

    # Re-initialize server model
    initial_model = {
        "W": np.zeros((1, pipeline.n_features), dtype=float),
        "b": np.zeros((1,), dtype=float),
    }
    pipeline.server.set_initial_model(initial_model)

    # Reset client DP budgets and counters
    for client in pipeline.clients:
        if client.use_dp:
            client.dp_engine.privacy_spent = 0.0
            client.dp_engine.rounds_executed = 0
        pipeline.registry_info[client.client_id]["counter"] = 0
        pipeline.registry_info[client.client_id]["session_key"] = None

    return {"msg": "Pipeline reset. Ready for new training."}


@app.get("/api/handshake", response_model=HandshakeResponse)
async def run_handshake():
    """Perform a real Kyber-768 Key Exchange."""
    server_keys = pipeline.pqc_channel.server_generate_keypair()
    encaps = pipeline.pqc_channel.client_encapsulate(server_keys["public_key"])

    pipeline.handshake_done = True

    return {
        "public_key": server_keys["public_key"][:64] + "...",
        "ciphertext": encaps["ciphertext"][:64] + "...",
        "session_key": encaps["session_key"].hex(),
        "status": "SECURE_KYBER_768",
    }


@app.get("/api/train/round")
async def train_one_round():
    """Execute one REAL federated learning training round."""
    if not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    pipeline.current_round += 1
    rnd = pipeline.current_round

    global_params = pipeline.server.get_global_model()
    global_W = global_params["W"]
    global_b = global_params["b"]

    client_logs = []
    X_test, y_test = pipeline.test_data

    for i, client in enumerate(pipeline.clients):
        cid = client.client_id
        X_local, y_local = pipeline.client_data[i]

        # 1. Local Training
        new_W, new_b = train_local_model(global_W, global_b, X_local, y_local)

        # 2. Apply Differential Privacy
        if pipeline.dp_config["enabled"]:
            # Properly scale DP delta by the number of samples so noise doesn't overpower the signal in multi-round runs
            raw_delta_W = new_W - global_W
            raw_delta_b = new_b - global_b
            
            # Clip delta
            norm_W = np.linalg.norm(raw_delta_W)
            if norm_W > pipeline.dp_config["clipping_norm"]:
                raw_delta_W = raw_delta_W * (pipeline.dp_config["clipping_norm"] / norm_W)
                
            norm_b = np.linalg.norm(raw_delta_b)
            if norm_b > pipeline.dp_config["clipping_norm"]:
                raw_delta_b = raw_delta_b * (pipeline.dp_config["clipping_norm"] / norm_b)
                
            # Add noise and scale it down by sample size for Federated Averaging
            client.dp_engine.set_sensitivity(pipeline.dp_config["clipping_norm"])
            noisy_delta_W = client.dp_engine.add_noise(raw_delta_W, account=False)
            noisy_delta_b = client.dp_engine.add_noise(raw_delta_b, account=False)
            
            final_W = global_W + (noisy_delta_W / len(X_local))
            final_b = global_b + (noisy_delta_b / len(X_local))
            
            client.account_privacy_step()
        else:
            final_W, final_b = new_W, new_b

        # 3. Prepare update
        model_update = {
            "client_id": cid,
            "model_params": {"W": final_W, "b": final_b},
            "num_samples": len(X_local),
        }

        # 4. Secure send (PQC + HE)
        info = pipeline.registry_info[cid]
        info["counter"] += 1
        payload = client.secure_send_update(
            model_update,
            info["server_kyber_pk"],
            info["session_key"],
            use_he=True,
            msg_counter=info["counter"],
        )
        info["session_key"] = payload["session_key"]
        pipeline.server.receive_update(payload)

        client_logs.append({
            "client_id": cid,
            "num_samples": len(X_local),
            "dp_noise_applied": pipeline.dp_config["enabled"],
            "privacy_spent": round(client.dp_engine.privacy_spent, 2) if client.use_dp else 0,
            "local_accuracy": round(evaluate_model(final_W, final_b, X_test, y_test), 4),
            "status": "SENT",
        })

    # Server aggregation
    new_global = pipeline.server.finalize_round()
    X_test, y_test = pipeline.test_data
    accuracy = evaluate_model(new_global["W"], new_global["b"], X_test, y_test)

    pipeline.accuracy_history.append({
        "round": rnd,
        "accuracy": round(accuracy, 4),
    })

    # Get a ciphertext snippet for display (encrypt a small sample)
    sample_weights = new_global["W"].flatten()[:3]
    he_mgr = pipeline.server.he_manager
    enc_sample = he_mgr.encrypt_vector(sample_weights)
    ct_snippet = hex(enc_sample[0].ciphertext())[:80] + "..."

    return {
        "round": rnd,
        "accuracy": round(accuracy, 4),
        "clients": client_logs,
        "encrypted_snippet": ct_snippet,
        "global_weights_sample": sample_weights.tolist(),
        "accuracy_history": pipeline.accuracy_history,
    }


@app.get("/api/train/auto")
async def train_all_rounds():
    """Run 10 more training rounds and return full results."""
    if not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    results = []
    target_round = pipeline.current_round + 10
    while pipeline.current_round < target_round:
        rnd_result = await train_one_round()
        results.append(rnd_result)

    return {
        "msg": "Batch training complete",
        "total_rounds": len(pipeline.accuracy_history),
        "final_accuracy": pipeline.accuracy_history[-1]["accuracy"],
        "accuracy_history": pipeline.accuracy_history,
    }


@app.post("/api/privacy/config")
async def set_privacy_config(config: PrivacyConfig):
    pipeline.dp_config["epsilon"] = config.epsilon
    # Update epsilon on all client DP engines
    for client in pipeline.clients:
        if client.use_dp:
            client.dp_engine.epsilon = config.epsilon
            client.dp_engine.scale = client.dp_engine._calculate_scale()
    return {"epsilon": config.epsilon, "msg": "Privacy budget updated on all clients"}


@app.post("/api/attack/simulate")
async def trigger_attack(req: AttackRequest):
    if req.attack_type == "quantum":
        pipeline.threat_level = "CRITICAL"
        return {
            "msg": "QUANTUM WAVEFRONT DETECTED",
            "action": "ROTATING KEYS",
            "success": True,
        }
    elif req.attack_type == "mitm":
        return {
            "msg": "PACKET INTERCEPTED",
            "error": "DILITHIUM SIGNATURE MISMATCH",
            "success": False,
        }
    return {"msg": "Unknown attack"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
