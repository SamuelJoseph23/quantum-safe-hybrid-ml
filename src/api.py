"""
Cyber-SOC Backend API
FastAPI app serving the Quantum-Safe Federated Learning demo.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import random
import sys
import os

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager
from differential_privacy import DifferentialPrivacy

app = FastAPI(title="Quantum-Safe FL API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Demo
class DemoState:
    epsilon = 2.0
    threat_level = "LOW"
    training_round = 0
    kyber_active = True
    
state = DemoState()

# Initialize Security Models
he_manager = HEManager(key_size=1024) # 1024 for speed in demo
pqc_channel = PQCSecureChannel(security_level=2)
pqc_auth = PQCAuthenticator(security_level=2)

# --- Models ---
class InitResponse(BaseModel):
    status: str
    security_level: str
    he_algorithm: str

class HandshakeResponse(BaseModel):
    public_key: str
    ciphertext: str
    session_key: str
    status: str

class HEDataResponse(BaseModel):
    client_view: list[float]
    server_view_snippet: str
    decrypted_sum: float

class PrivacyConfig(BaseModel):
    epsilon: float

class AttackRequest(BaseModel):
    attack_type: str # "mitm" or "quantum"
    parameter: str = ""

# --- Endpoints ---

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "threat_level": state.threat_level,
        "round": state.training_round,
        "privacy_budget": 100.0 - (state.training_round * (10.0 / state.epsilon))
    }

@app.post("/api/reset")
async def reset_demo():
    state.training_round = 0
    state.threat_level = "LOW"
    state.kyber_active = True
    return {"msg": "Demo state reset"}

@app.get("/api/handshake", response_model=HandshakeResponse)
async def simulate_handshake():
    """Simulate a Kyber Key Exchange."""
    if not state.kyber_active:
        time.sleep(1) # Delay to show 'broken' state
        raise HTTPException(status_code=503, detail="Quantum Vulnerability Detected!")
    
    # 1. Server Generate
    server_keys = pqc_channel.server_generate_keypair()
    
    # 2. Client Encapsulate
    encaps = pqc_channel.client_encapsulate(server_keys['public_key'])
    
    return {
        "public_key": server_keys['public_key'][:64] + "...",
        "ciphertext": encaps['ciphertext'][:64] + "...",
        "session_key": encaps['session_key'].hex(),
        "status": "SECURE_KYBER_768"
    }

@app.post("/api/privacy/config")
async def set_privacy_config(config: PrivacyConfig):
    state.epsilon = config.epsilon
    return {"epsilon": state.epsilon, "msg": "Privacy budget updated"}

@app.get("/api/he/simulate")
async def simulate_he_round():
    """Simulate an HE aggregation round."""
    state.training_round += 1
    
    # Generate random 'weights'
    w1 = np.random.uniform(-0.5, 0.5, 3)
    w2 = np.random.uniform(-0.5, 0.5, 3)
    
    # Encrypt
    enc1 = he_manager.encrypt_vector(w1)
    enc2 = he_manager.encrypt_vector(w2)
    
    # Aggregate
    sum_enc = he_manager.add_encrypted_vectors([enc1, enc2])
    
    # Decrypt
    decrypted = he_manager.decrypt_vector(sum_enc, (3,))
    
    # Format ciphertext for display
    # Use hex representation of the first number's ciphertext
    ct_snippet = hex(enc1[0].ciphertext())[:100] + "..."
    
    return {
        "client_view": w1.tolist(),
        "client2_view": w2.tolist(),
        "server_view_snippet": ct_snippet,
        "decrypted_sum": decrypted.tolist(),
        "true_sum": (w1 + w2).tolist()
    }

@app.post("/api/attack/simulate")
async def trigger_attack(req: AttackRequest):
    if req.attack_type == "quantum":
        state.threat_level = "CRITICAL"
        return {
            "msg": "QUANTUM WAVEFRONT DETECTED",
            "action": "ROTATING KEYS",
            "success": True
        }
    elif req.attack_type == "mitm":
        # Simulate Dilithium check failure
        return {
            "msg": "PACKET INTERCEPTED",
            "error": "DILITHIUM SIGNATURE MISMATCH",
            "success": False
        }
    return {"msg": "Unknown attack"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
